"""Nova 数据集与数据加载

负责把原始数据转换为模型可消费的tensor，分别用于预训练和微调两个训练阶段。

PyTorch中的张量 跟 数学上的张量高度相关，但并不完全等价，可以将PyTorch中的张量理解为数学张量在计算机科学或者深度学习领域中的一个特定的简化实现，
它的本质是一个数据容器，强调的是计算功能，而非数学上的几何本质。

PyTorch 本质上是一个深度学习框架，它在底层集成并统一封装了 CPU、CUDA、MPS 等算子，对上提供统一且标准的张量与算子接口。开发者只需要把数据按照约定组织成 Tensor，
并调用统一的前向、反向与优化 API；而 PyTorch 会依据 Tensor 的 shape、dtype、device 等元信息，自动将计算分发到对应的底层后端，
完成 CPU/GPU 上的高性能并行矩阵运算、自动求导与参数更新，从而屏蔽不同硬件平台和底层实现之间的差异。

Tensor数据结构：
PyTorch 张量 = 数据 + 形状 + 数据类型 + 设备

                  ┌─ data:   [1, 156, 2847, 23, 2, 0, 0, 0]  （一块连续内存）
                  │
tensor ───────────├─ shape:  (8,)  或 [8, 128] 等             （怎么解读这块内存）
                  │
                  ├─ dtype:  torch.long                       （每个数字的精度）
                  │
                  └─ device: mps                              （存在哪，谁来算）
"""

from __future__ import annotations

import json
import os
from typing import List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from tokenizer import NovaTokenizer, BOS_ID, EOS_ID, SEP_ID


# 预训练阶段将语料批量预编码、截断、填充、生成模型tensor输入和交叉熵tensor输入
class PretrainDataset(Dataset):
    def __init__(
        self,
        texts: List[str],  # load_pretrain_data函数解析后的语料纯文本数据
        tokenizer: NovaTokenizer,
        max_seq_len: int,
    ) -> None:
        bos_id = BOS_ID
        eos_id = EOS_ID

        # 过滤掉空白行
        texts = [t for t in texts if t.strip()]
        # 计算文本总行数
        n = len(texts)

        # NumPy是Python中高性能数值数组运算的基础库
        # input_ids是模型的输入，target_ids是交叉熵的输入，CrossEntropyLoss有个参数叫ignore_index，默认值就是-100损失函数遇到 -100 就跳过
        # 预先创建input_ids的全0矩阵，以及target_ids的全-100矩阵，提前完成了填充
        # 最终填充后的效果：
        # 位置索引:     0    1    2    3    4    5    6    7
        # input_ids:  [1,   42,  15,  33,  2,   0,   0,   0  ]
        # target_ids: [42,  15,  33,  2,   -100,-100,-100,-100]，注意个问题，<s>是起点不需要预测，而<e>是结尾必须保留
        self.input_ids = np.zeros((n, max_seq_len), dtype=np.int16)
        self.target_ids = np.full((n, max_seq_len), -100, dtype=np.int16)

        # 每次处理1万条，防止一次性编码全部文本导致内存爆炸
        CHUNK = 10_000
        for start in range(0, n, CHUNK):
            end = min(start + CHUNK, n)
            # 批量编码1万条文本，就是将纯文本批量编码为token_ids列表
            encs = tokenizer._tokenizer.encode_batch(texts[start:end])
            for j, enc in enumerate(encs):
                i = start + j
                # ========= 生成模型输入相关 =========
                # ids是token_ids列表中的其中一行（不含<s>和<e>）

                # texts = ["今天天气真好", "你好世界", "机器学习很有趣"]
                # encs = tokenizer.encode_batch(texts)
                # encs[0].ids → [42, 15, 33]       ← "今天天气真好"的token ids
                # encs[1].ids → [55, 60]           ← "你好世界"的token ids
                # encs[2].ids → [70, 80, 90, 12]   ← "机器学习很有趣"的token ids
                ids = enc.ids
                id_len = len(ids)  # 取某一行token_ids的长度

                # 位置0设置为<s>=1
                self.input_ids[i, 0] = bos_id

                # 单条token_ids超出max_seq_len - 2长度时截断，反之则全部保留
                copy_len = min(id_len, max_seq_len - 2)

                # 把编码后的单条token_ids的内容从模型输入input_ids中的位置1开始填充
                if copy_len > 0:
                    self.input_ids[i, 1 : copy_len + 1] = ids[
                        0:copy_len
                    ]  # 从0开始取copy_len个token_ids

                # 在正文末尾追加<e>=2结尾,当然如果长度是max_seq_len则覆盖最后一位
                eos_pos = min(copy_len + 1, max_seq_len - 1)
                self.input_ids[i, eos_pos] = eos_id

                # ========= 生成交叉熵输入相关 =========
                vlen = eos_pos + 1
                # 把模型输入的<s>之后的正文内容复制到交叉熵输入中
                if vlen > 1:
                    self.target_ids[i, 0 : vlen - 1] = self.input_ids[i, 1:vlen]

            # 释放这批编码结果的内存
            del encs
            print(f"       编码进度: {end}/{n} ({end * 100 // n}%)", flush=True)
        print(f"       预编码完成，占用 {self.input_ids.nbytes * 2 / 1024**3:.1f} GB")

    def __len__(self) -> int:
        return len(self.input_ids)

    # PyTorch回调函数，分别转换为模型tensor输入和交叉熵tensor输入
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": torch.from_numpy(self.input_ids[idx]).long(),
            "target_ids": torch.from_numpy(self.target_ids[idx]).long(),
        }


# 微调阶段将问答对批量预编码、截断、填充、生成模型tensor输入和交叉熵tensor输入
class NovaDataset(Dataset):
    def __init__(
        self,
        qa_pairs: List[Dict[str, str]],  # load_qa_pairs函数解析后的问答对列表
        tokenizer: NovaTokenizer,
        max_seq_len: int,
    ) -> None:
        bos_id = BOS_ID
        sep_id = SEP_ID
        eos_id = EOS_ID

        # 计算问答对总条数
        n = len(qa_pairs)

        # 预先创建input_ids的全0矩阵，以及target_ids的全-100矩阵，提前完成了填充
        # 微调数据的拼接格式: <s> 问题 <sep> 回答 <e>
        # 最终填充后的效果（假设 max_seq_len=10）：
        # 位置索引:     0    1    2    3    4    5    6    7    8    9
        # input_ids:  [1,   42,  15,  3,   33,  21,  2,   0,   0,   0  ]
        #              ↑    ←问题→    ↑    ←回答→    ↑    ←pad填充→
        #             <s>           <sep>           <e>
        # target_ids: [42,  15,  3,   33,  21,  2,   -100,-100,-100,-100]
        self.input_ids = np.zeros((n, max_seq_len), dtype=np.int16)
        self.target_ids = np.full((n, max_seq_len), -100, dtype=np.int16)

        # 每次处理1万条，防止一次性编码全部问答对导致内存爆炸
        CHUNK = 10_000
        for start in range(0, n, CHUNK):
            end = min(start + CHUNK, n)
            # 把这一批问答对的question和answer分别提取出来
            chunk_q = [qa_pairs[k]["question"] for k in range(start, end)]
            chunk_a = [qa_pairs[k]["answer"] for k in range(start, end)]
            # 分别批量编码question和answer为token_ids列表
            # qa_pairs = [{"question": "你叫什么名字？", "answer": "我是Nova。"}, ...]
            # q_encs[0].ids → [42, 15]       ← "你叫什么名字？"的token ids
            # a_encs[0].ids → [33, 21]       ← "我是Nova。"的token ids
            q_encs = tokenizer._tokenizer.encode_batch(chunk_q)
            a_encs = tokenizer._tokenizer.encode_batch(chunk_a)
            # 编码完成后立即释放原始文本，节省内存
            del chunk_q, chunk_a
            for j, (q, a) in enumerate(zip(q_encs, a_encs)):
                i = start + j
                # 取出当前这条问答对的question和answer的token_ids（不含特殊标记）
                q_ids, a_ids = q.ids, a.ids

                # 位置0设置为<s>=1
                pos = 0
                self.input_ids[i, pos] = bos_id
                pos += 1  # pos=1，指向下一个待填充位置

                # 填充问题token_ids，预留2个位置给后面的<sep>和<e>
                qlen = min(len(q_ids), max_seq_len - pos - 2)
                if qlen > 0:
                    self.input_ids[i, pos : pos + qlen] = q_ids[0:qlen]
                    pos += qlen  # pos移动到问题末尾的下一个位置

                # 在问题末尾追加<sep>=3分隔符
                self.input_ids[i, pos] = sep_id
                pos += 1  # pos移动到回答开始的位置

                # 填充回答token_ids，预留1个位置给后面的<e>
                alen = min(len(a_ids), max_seq_len - pos - 1)
                if alen > 0:
                    self.input_ids[i, pos : pos + alen] = a_ids[0:alen]
                    pos += alen  # pos移动到回答末尾的下一个位置

                # 在回答末尾追加<e>=2结尾
                if pos < max_seq_len:
                    self.input_ids[i, pos] = eos_id
                    pos += 1  # pos现在等于有效内容的总长度

                # ========= 生成交叉熵输入相关 =========
                # 把模型输入的<s>之后的正文内容复制到交叉熵输入中
                # pos此时就是有效长度vlen，等价于PretrainDataset中的vlen = eos_pos + 1
                if pos > 1:
                    self.target_ids[i, 0 : pos - 1] = self.input_ids[i, 1:pos]

            # 释放这批编码结果的内存
            del q_encs, a_encs
            print(f"       编码进度: {end}/{n} ({end * 100 // n}%)", flush=True)
        print(f"       预编码完成，占用 {self.input_ids.nbytes * 2 / 1024**3:.1f} GB")

    def __len__(self) -> int:
        return len(self.input_ids)

    # PyTorch回调函数，分别转换为模型tensor输入和交叉熵tensor输入
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": torch.from_numpy(self.input_ids[idx]).long(),
            "target_ids": torch.from_numpy(self.target_ids[idx]).long(),
        }


# DataLoader本质是一个数据分批迭代器，本质是把Dataset包装成可以按batch取数据的迭代器
# PretrainDataset和NovaDataset都继承自Dataset，在编码结束后将Dataset包装好训练时取出Tensor进行训练
def create_dataloader(
    dataset: PretrainDataset | NovaDataset,  # PretrainDataset 或 NovaDataset 实例
    batch_size: int,  # 每批数据的条数（来自 config.batch_size）
    shuffle: bool = True,  # 每个epoch开始前是否打乱数据顺序，防止模型记住数据的出场顺序
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
    )


# ======================================================================
# 加载并解析微调语料包
# ======================================================================
def load_qa_pairs(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"微调数据路径不存在: {path}")

    # 把目标目录下的所有jsonl文件记录到jsonl_files中
    if os.path.isdir(path):
        jsonl_files = sorted(
            os.path.join(path, f) for f in os.listdir(path) if f.endswith(".jsonl")
        )
        if not jsonl_files:
            raise ValueError(f"目录中没有 .jsonl 文件: {path}")
    else:
        # 加载指定的jsonl微调语料包文件
        jsonl_files = [path]

    # 解析jsonl文件，读取每一行json，将question和answer的value解析成纯文本序列
    # Python中的Dict类似Java的Map
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
    # 返回微调阶段的纯文本序列
    return qa_pairs


# ======================================================================
# 加载并解析预训练语料包
# ======================================================================
def load_pretrain_data(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"预训练数据路径不存在: {path}")

    # 把目标目录下的所有jsonl文件记录到jsonl_files中
    if os.path.isdir(path):
        jsonl_files = sorted(
            os.path.join(path, f) for f in os.listdir(path) if f.endswith(".jsonl")
        )
        if not jsonl_files:
            raise ValueError(f"目录中没有 .jsonl 文件: {path}")
    else:
        # 加载指定的jsonl语料文件
        jsonl_files = [path]

    # 解析jsonl中的每一行json，并提取text的value
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
    # 返回json解析后的所有预训练语料 纯文本 内容序列
    return texts
