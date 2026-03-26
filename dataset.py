"""Nova 数据集与数据加载

负责把原始数据转换为模型可消费的 tensor。
支持两种数据格式，分别用于预训练和微调两个训练阶段：

┌─────────────────────────────────────────────────────────────────────┐
│                     两阶段训练的数据处理                               │
│                                                                     │
│  阶段 1: 预训练（PretrainDataset）                                   │
│  ───────────────────────────────                                    │
│  数据来源:  data/pretrain/*.jsonl                                   │
│  数据格式:  {"text": "一段百科/新闻/文章..."}                        │
│  拼接方式:  <s> 纯文本 <e>                                           │
│  训练目标:  学习语言本身（"下一个词是什么"）                           │
│  调用方式:                                                           │
│    texts = load_pretrain_data("data/pretrain/")                     │
│    dataset = PretrainDataset(texts, tokenizer, max_seq_len)         │
│                                                                     │
│  阶段 2: 微调（NovaDataset）                                        │
│  ───────────────────────────                                        │
│  数据来源:  data/sft/*.jsonl                                        │
│  数据格式:  每行 {"question": "...", "answer": "..."}               │
│  拼接方式:  <s> 问题 <sep> 回答 <e>                                  │
│  训练目标:  学习"如何回答问题"                                       │
│  调用方式:                                                           │
│    qa_pairs = load_qa_pairs("data/sft/")                            │
│    dataset = NovaDataset(qa_pairs, tokenizer, max_seq_len)          │
│                                                                     │
│  两个阶段共用同一个 BPE 分词器（在预训练数据上训练生成）              │
└─────────────────────────────────────────────────────────────────────┘

源码阅读顺序（建议按编号顺序阅读）:
┌──────────────────────────────────────────────────────────────────────┐
│  ① load_pretrain_data()   — JSONL 加载预训练纯文本（最基础的 I/O）   │
│  ② load_qa_pairs()        — JSONL 加载微调问答对（与 ① 结构相似）    │
│  ③ _encode_and_pad()      — 截断/填充/构造 labels 的共用逻辑         │
│  ④ PretrainDataset        — 预训练数据集（依赖 ① ③）                │
│  ⑤ NovaDataset            — 微调数据集（依赖 ② ③）                  │
│  ⑥ create_dataloader()    — DataLoader 工厂（依赖 ④ ⑤）             │
│                                                                      │
│  推荐理由:                                                            │
│    先读 ①② 理解数据从磁盘怎么来;                                     │
│    再读 ③ 理解 input_ids / target_ids 怎么构造;                      │
│    然后读 ④⑤ 看两种数据集如何调用 ③ 完成编码;                        │
│    最后读 ⑥ 看数据怎么按 batch 喂给模型。                             │
└──────────────────────────────────────────────────────────────────────┘

一条预训练数据从原始文本到模型输入的完整变换过程:

  原始 JSONL:
    {"text": "计算机是20世纪最先进的科学技术发明之一。"}

  步骤 a — 拼接特殊标记:
    "<s>计算机是20世纪最先进的科学技术发明之一。<e>"

  步骤 b — BPE 分词器编码:
    [1, 156, 2847, 23, 891, 5, 1203, 2]
     ↑                               ↑
    <s>                              <e>

  步骤 c — 截断（超过 max_seq_len 的部分砍掉）
  步骤 d — 填充（不足 max_seq_len 的部分补 <pad>(0)）
  步骤 e — 构造 input_ids 和 labels（与微调相同）

一条微调数据从原始文本到模型输入的完整变换过程:

  原始 JSON:
    {"question": "你叫什么名字？", "answer": "我是Nova。"}

  步骤 a — 拼接特殊标记:
    "<s>你叫什么名字？<sep>我是Nova。<e>"

  步骤 b — BPE 分词器编码:
    [1, 42, 15, 3, 33, 21, 2]
     ↑          ↑          ↑
    <s>       <sep>       <e>

  步骤 c/d/e — 截断、填充、构造 labels（与预训练相同）
"""

from __future__ import annotations

import json
import os
from typing import List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from tokenizer import NovaTokenizer, BOS_ID, EOS_ID, SEP_ID, PAD_ID


# ======================================================================
# ③ _encode_and_pad —— 内部辅助函数（阅读顺序第 3）
#
# PretrainDataset 和 NovaDataset 都需要做同样的事：
#   token_ids → 截断 → 填充 → 构造 input_ids / labels
#
# 把这段逻辑提取成共用函数，避免代码重复。
#
# 调用时机: PretrainDataset.__init__ 和 NovaDataset.__init__ 内部调用
# ======================================================================
def _encode_and_pad(
    token_ids: List[int],
    max_seq_len: int,
) -> Dict[str, torch.Tensor]:
    """将 token ID 列表截断、填充、构造 input_ids / labels。

    执行步骤:
      1. 截断: 超过 max_seq_len 的部分砍掉
      2. 填充: 不足 max_seq_len 的部分补 PAD_ID (0)
      3. 构造 labels: input_ids 左移一位（下一个 token 预测目标）
         - labels[i] = token_ids[i+1]
         - 最后一个有效位和所有 padding 位设为 -100

    参数:
      token_ids   — 已编码的 token ID 列表（含特殊标记）
      max_seq_len — 目标序列长度

    返回:
      {"input_ids": Tensor[max_seq_len], "target_ids": Tensor[max_seq_len]}
    """
    # ── 截断 ──
    if len(token_ids) > max_seq_len:
        token_ids = token_ids[:max_seq_len]

    # ── 填充 ──
    valid_len = len(token_ids)
    padding_len = max_seq_len - valid_len
    token_ids = token_ids + [PAD_ID] * padding_len

    # ── 构造 input_ids 和 labels ──
    #   labels[i] = token_ids[i+1]（"下一个 token 预测"目标）
    #   最后有效位和 padding 位设为 -100（cross_entropy 会忽略）
    #
    #   举例 (max_seq_len=8):
    #     token_ids  = [1, 42, 15, 33, 2, 0, 0, 0]
    #     valid_len  = 5
    #     input_ids  = [1, 42, 15, 33, 2, 0, 0, 0]
    #     labels     = [42, 15, 33, 2, -100, -100, -100, -100]
    input_ids = token_ids[:]
    labels = token_ids[1:] + [PAD_ID]
    for i in range(valid_len - 1, max_seq_len):
        labels[i] = -100

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "target_ids": torch.tensor(labels, dtype=torch.long),
    }


# ======================================================================
# ④ PretrainDataset —— 预训练阶段数据集（阅读顺序第 4）
#
# 典型调用顺序（在 train.py 中）:
#
#   ① 加载预训练数据
#      texts = load_pretrain_data("data/pretrain/")
#
#   ② 训练 BPE 分词器
#      tokenizer = NovaTokenizer()
#      tokenizer.train_from_texts(texts, vocab_size=8000)
#      tokenizer.save("data/tokenizer.json")
#
#   ③ 构建 PretrainDataset
#      dataset = PretrainDataset(texts, tokenizer, config.max_seq_len)
#
#   ④ 创建 DataLoader
#      dataloader = create_dataloader(dataset, batch_size=config.batch_size)
#
#   ⑤ 训练循环
#      for batch in dataloader:
#          input_ids  = batch["input_ids"]
#          target_ids = batch["target_ids"]
#          ...
# ======================================================================
class PretrainDataset(Dataset):
    """预训练数据集：将纯文本转换为模型可消费的 tensor。

    与 NovaDataset（微调）的区别:
    ┌──────────────────────────────────────────────────────────────┐
    │  PretrainDataset (预训练)     │  NovaDataset (微调)           │
    │  ─────────────────────────── │  ───────────────────────────  │
    │  输入: {"text": "..."}       │  输入: {"question", "answer"} │
    │  格式: <s> 纯文本 <e>        │  格式: <s> 问题 <sep> 回答 <e>│
    │  无 <sep> 标记               │  有 <sep> 标记                │
    │  目标: 学语言本身             │  目标: 学问答对话             │
    │  数据量: 大（万级以上）       │  数据量: 小（千级）           │
    └──────────────────────────────────────────────────────────────┘

    每条数据包含:
      input_ids  — 模型的输入序列       [max_seq_len]
      target_ids — 下一个 token 预测目标  [max_seq_len]
    """

    # ------------------------------------------------------------------
    # __init__ —— 惰性初始化，只保存文本引用，不预编码
    #
    #    参数:
    #      texts       — 纯文本列表，每个元素是一段完整的文本
    #      tokenizer   — 已训练好的 BPE NovaTokenizer 实例
    #      max_seq_len — 序列最大长度（来自 config.max_seq_len）
    #
    #    编码在 __getitem__ 中按需执行，避免大数据集下 OOM
    # ------------------------------------------------------------------
    def __init__(
        self,
        texts: List[str],
        tokenizer: NovaTokenizer,
        max_seq_len: int,
    ) -> None:
        bos_id = BOS_ID
        eos_id = EOS_ID

        texts = [t for t in texts if t.strip()]

        n = len(texts)
        self.input_ids = np.zeros((n, max_seq_len), dtype=np.int16)
        self.target_ids = np.full((n, max_seq_len), -100, dtype=np.int16)

        CHUNK = 10_000
        print(f"       预编码 {n} 条文本（预计 1-2 分钟）...")
        for start in range(0, n, CHUNK):
            end = min(start + CHUNK, n)
            encs = tokenizer._tokenizer.encode_batch(texts[start:end])
            for j, enc in enumerate(encs):
                i = start + j
                ids = enc.ids
                id_len = len(ids)
                self.input_ids[i, 0] = bos_id
                copy_len = min(id_len, max_seq_len - 2)
                if copy_len > 0:
                    self.input_ids[i, 1:copy_len + 1] = ids[:copy_len]
                eos_pos = min(copy_len + 1, max_seq_len - 1)
                self.input_ids[i, eos_pos] = eos_id
                vlen = eos_pos + 1
                if vlen > 1:
                    self.target_ids[i, :vlen - 1] = self.input_ids[i, 1:vlen]
            del encs
            print(f"       编码进度: {end}/{n} ({end * 100 // n}%)", flush=True)
        print(f"       预编码完成，占用 {self.input_ids.nbytes * 2 / 1024**3:.1f} GB")

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": torch.from_numpy(self.input_ids[idx]).long(),
            "target_ids": torch.from_numpy(self.target_ids[idx]).long(),
        }


# ======================================================================
# ⑤ NovaDataset —— 微调阶段数据集（阅读顺序第 5）
#
# 典型调用顺序（在 train.py 中）:
#
#   ① 加载微调数据
#      qa_pairs = load_qa_pairs("data/sft/")
#
#   ② 加载已有 BPE 分词器（预训练阶段生成的）
#      tokenizer = NovaTokenizer()
#      tokenizer.load("data/tokenizer.json")
#
#   ③ 构建 NovaDataset
#      dataset = NovaDataset(qa_pairs, tokenizer, config.max_seq_len)
#
#   ④ 创建 DataLoader
#      dataloader = create_dataloader(dataset, batch_size=config.batch_size)
#
#   ⑤ 训练循环
#      for batch in dataloader:
#          input_ids  = batch["input_ids"]
#          target_ids = batch["target_ids"]
#          ...
# ======================================================================
class NovaDataset(Dataset):
    """微调数据集：将问答对转换为模型可消费的 tensor。

    每条数据包含:
      input_ids  — 模型的输入序列       [max_seq_len]
      target_ids — 下一个 token 预测目标  [max_seq_len]
    """

    def __init__(
        self,
        qa_pairs: List[Dict[str, str]],
        tokenizer: NovaTokenizer,
        max_seq_len: int,
    ) -> None:
        bos_id = BOS_ID
        sep_id = SEP_ID
        eos_id = EOS_ID

        n = len(qa_pairs)
        self.input_ids = np.zeros((n, max_seq_len), dtype=np.int16)
        self.target_ids = np.full((n, max_seq_len), -100, dtype=np.int16)

        CHUNK = 10_000
        print(f"       预编码 {n} 条问答对（预计 1-2 分钟）...")
        for start in range(0, n, CHUNK):
            end = min(start + CHUNK, n)
            chunk_q = [qa_pairs[k]["question"] for k in range(start, end)]
            chunk_a = [qa_pairs[k]["answer"] for k in range(start, end)]
            q_encs = tokenizer._tokenizer.encode_batch(chunk_q)
            a_encs = tokenizer._tokenizer.encode_batch(chunk_a)
            del chunk_q, chunk_a
            for j, (q, a) in enumerate(zip(q_encs, a_encs)):
                i = start + j
                q_ids, a_ids = q.ids, a.ids
                pos = 0
                self.input_ids[i, pos] = bos_id; pos += 1
                qlen = min(len(q_ids), max_seq_len - pos - 2)
                if qlen > 0:
                    self.input_ids[i, pos:pos + qlen] = q_ids[:qlen]; pos += qlen
                self.input_ids[i, pos] = sep_id; pos += 1
                alen = min(len(a_ids), max_seq_len - pos - 1)
                if alen > 0:
                    self.input_ids[i, pos:pos + alen] = a_ids[:alen]; pos += alen
                if pos < max_seq_len:
                    self.input_ids[i, pos] = eos_id; pos += 1
                if pos > 1:
                    self.target_ids[i, :pos - 1] = self.input_ids[i, 1:pos]
            del q_encs, a_encs
            print(f"       编码进度: {end}/{n} ({end * 100 // n}%)", flush=True)
        print(f"       预编码完成，占用 {self.input_ids.nbytes * 2 / 1024**3:.1f} GB")

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": torch.from_numpy(self.input_ids[idx]).long(),
            "target_ids": torch.from_numpy(self.target_ids[idx]).long(),
        }


# ======================================================================
# ⑥ DataLoader 工厂函数（阅读顺序第 6）
#
# 预训练和微调阶段都使用同一个函数创建 DataLoader。
#
# 调用时机: train.py 中创建完 Dataset 之后
#
#   dataset = PretrainDataset(texts, tokenizer, max_seq_len)  # 预训练
#   # 或
#   dataset = NovaDataset(qa_pairs, tokenizer, max_seq_len)   # 微调
#
#   dataloader = create_dataloader(dataset, batch_size=config.batch_size)
# ======================================================================
def create_dataloader(
    dataset: PretrainDataset | NovaDataset,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """创建 DataLoader，将 Dataset 按 batch_size 分批。

    参数:
      dataset    — PretrainDataset 或 NovaDataset 实例
      batch_size — 每批数据的条数（来自 config.batch_size）
      shuffle    — 是否打乱顺序（训练时 True，验证时 False）

    返回:
      DataLoader 实例，迭代时产出 {"input_ids": [B, S], "target_ids": [B, S]}
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
    )


# ======================================================================
# ② 辅助函数：从 JSONL 文件/目录加载微调数据（qa_pairs）（阅读顺序第 2）
#
# 调用时机: train.py 微调阶段
#   qa_pairs = load_qa_pairs("data/sft/")                    # 加载目录下所有 .jsonl
#   qa_pairs = load_qa_pairs("data/sft/qa_pairs.jsonl")      # 单个文件
#
# JSONL 格式说明:
#   每行一个 JSON 对象，包含 "question" 和 "answer" 字段:
#     {"question": "你好", "answer": "你好！"}
#     {"question": "你叫什么名字？", "answer": "我是Nova。"}
# ======================================================================
def load_qa_pairs(path: str) -> List[Dict[str, str]]:
    """从 JSONL 文件或目录加载问答对列表。

    参数:
      path — JSONL 文件路径或包含 .jsonl 文件的目录路径

    返回:
      问答对字典列表，每个元素包含 "question" 和 "answer" 字段

    异常:
      FileNotFoundError — 路径不存在
      ValueError        — 目录中没有 .jsonl 文件
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"微调数据路径不存在: {path}")

    if os.path.isdir(path):
        jsonl_files = sorted(
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.endswith(".jsonl")
        )
        if not jsonl_files:
            raise ValueError(f"目录中没有 .jsonl 文件: {path}")
    else:
        jsonl_files = [path]

    qa_pairs: List[Dict[str, str]] = []
    for filepath in jsonl_files:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "question" in obj and "answer" in obj:
                    qa_pairs.append(obj)

    return qa_pairs


# ======================================================================
# ① 辅助函数：从 JSONL 文件/目录加载预训练数据（纯文本）（阅读顺序第 1）
#
# 调用时机: train.py 预训练阶段
#   texts = load_pretrain_data("data/pretrain/")      # 加载目录下所有 .jsonl
#   texts = load_pretrain_data("data/pretrain/train-00000-of-00300.jsonl")  # 单个文件
#
# JSONL 格式说明:
#   每行一个 JSON 对象，包含 "text" 字段:
#     {"text": "计算机是20世纪最先进的科学技术发明之一。"}
#     {"text": "由硬件系统和软件系统组成..."}
#
#   为什么用 JSONL 而不是 JSON？
#     - JSONL 可以逐行读取，内存友好（不用一次性加载整个文件）
#     - 适合大规模数据（百万行级别）
#     - 方便追加数据（直接往文件末尾加新行）
# ======================================================================
def load_pretrain_data(path: str) -> List[str]:
    """从 JSONL 文件或目录加载预训练纯文本列表。

    参数:
      path — JSONL 文件路径或包含 .jsonl 文件的目录路径

    返回:
      纯文本字符串列表（每个元素是一段完整的文本）

    异常:
      FileNotFoundError — 路径不存在
      ValueError        — 目录中没有 .jsonl 文件
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"预训练数据路径不存在: {path}")

    # ── 收集所有 .jsonl 文件路径 ──
    if os.path.isdir(path):
        jsonl_files = sorted(
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.endswith(".jsonl")
        )
        if not jsonl_files:
            raise ValueError(f"目录中没有 .jsonl 文件: {path}")
    else:
        jsonl_files = [path]

    # ── 逐行读取每个文件 ──
    texts: List[str] = []
    for filepath in jsonl_files:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get("text", "").strip()
                if text:
                    texts.append(text)

    return texts
