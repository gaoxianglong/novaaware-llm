"""Nova BPE 分词器

负责"文字 ↔ 数字 ID"的双向转换。这是 Transformer 流水线的第一站——
模型不认字，只认数字，分词器就是那本"字典"。

本版本使用 BPE（Byte Pair Encoding，字节对编码）替代字符级分词。
BPE 能把高频词/词组合并为单个 token，大幅提升编码效率。

┌─────────────────────────────────────────────────────────────────┐
│                       BPE vs 字符级分词                           │
│                                                                 │
│  字符级: "人工智能是未来"  → 7 个 token (每字一个)               │
│  BPE:   "人工智能是未来"  → 3-5 个 token (常见词被合并)          │
│                                                                 │
│  BPE 的优势:                                                     │
│    1. 同样的 max_seq_len 能装下更多语义信息                       │
│    2. 模型不需要浪费容量去学"字怎么组成词"                        │
│    3. 能处理未见过的词（拆成已知的子词片段）                       │
│                                                                 │
│  BPE 训练过程:                                                   │
│    1. 初始词表 = 所有单字符（UTF-8 字节或 Unicode 字符）          │
│    2. 统计相邻 token 对的出现频率                                 │
│    3. 把最高频的 token 对合并为新 token，加入词表                  │
│    4. 重复步骤 2-3，直到词表达到指定大小                          │
│                                                                 │
│  例: "人工" 出现 1000 次 → 合并为单个 token "人工"                │
│      "智能" 出现 800 次  → 合并为单个 token "智能"                │
│      "人工智能" 出现 500 次 → 进一步合并为 "人工智能"             │
└─────────────────────────────────────────────────────────────────┘

典型调用顺序:

  ① 训练 BPE 分词器（训练前调用一次）
     tokenizer = NovaTokenizer()
     tokenizer.train_from_texts(all_texts, vocab_size=8000)
     tokenizer.save("data/tokenizer.json")

  ② 编码（训练 / 推理时调用）
     ids = tokenizer.encode("人工智能是未来的趋势")
     → [156, 2847, 23, 891, 5, 1203]  (6 个 token，而非 9 个字符)

  ③ 解码（推理时调用）
     text = tokenizer.decode([156, 2847])
     → "人工智能"

  ④ 加载已有分词器（推理时直接加载，跳过 ①）
     tokenizer = NovaTokenizer()
     tokenizer.load("data/tokenizer.json")
     text = tokenizer.decode(model_output_ids)

特殊 token 及其固定 ID（永远不变）:
  <pad> = 0   填充符，训练时把短序列补齐到统一长度
  <s>   = 1   序列开始标记
  <e>   = 2   序列结束标记（模型生成到这个 token 就停下来）
  <sep> = 3   问题和回答之间的分隔符
  <unk> = 4   未知字符（分词器无法处理的输入用这个代替）
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import List

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


# 5 个特殊 token 的名称和固定 ID
PAD_TOKEN, PAD_ID = "<pad>", 0
BOS_TOKEN, BOS_ID = "<s>",   1
EOS_TOKEN, EOS_ID = "<e>",   2
SEP_TOKEN, SEP_ID = "<sep>", 3
UNK_TOKEN, UNK_ID = "<unk>", 4

SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, SEP_TOKEN, UNK_TOKEN]
NUM_SPECIAL = len(SPECIAL_TOKENS)  # 5，普通子词从 ID=5 开始编号


class NovaTokenizer:
    """BPE 子词分词器：使用 HuggingFace tokenizers 库实现。

    内部封装一个 tokenizers.Tokenizer 实例，提供与旧版字符级分词器
    完全兼容的 API 接口（encode / decode / save / load / vocab_size）。

    ┌──────────────────────────────────────────────────────────────────┐
    │  内部结构                                                        │
    │                                                                  │
    │  self._tokenizer : tokenizers.Tokenizer                          │
    │    ├── model     : BPE           ← 核心 BPE 算法                 │
    │    ├── pre_tokenizer : ByteLevel ← 编码前的预处理                │
    │    └── decoder   : ByteLevel     ← 解码时还原空格等              │
    │                                                                  │
    │  self.vocab_size : int           ← 词表总大小                    │
    │                                                                  │
    │  兼容属性:                                                       │
    │    self.char_to_id : dict        ← 向后兼容，token→ID 映射       │
    │    self.id_to_char : dict        ← 向后兼容，ID→token 映射       │
    └──────────────────────────────────────────────────────────────────┘
    """

    def __init__(self) -> None:
        self._tokenizer: Tokenizer | None = None
        self.vocab_size: int = 0
        # 兼容旧接口: 让 dataset.py / chat.py 的
        #   tokenizer.char_to_id[BOS_TOKEN] 等代码继续工作
        self.char_to_id: dict[str, int] = {}
        self.id_to_char: dict[int, str] = {}

    # ------------------------------------------------------------------
    # ① train_from_texts —— 从文本列表训练 BPE 分词器
    #
    #    调用时机: 拿到训练数据之后、训练开始之前（替代旧版 build_vocab）
    #    输入:     texts — 所有训练文本的列表
    #              vocab_size — 目标词表大小（包含特殊 token）
    #    输出:     无返回值；内部构建 self._tokenizer 并填充映射表
    #
    #    执行流程:
    #      1. 创建 BPE 模型实例（空词表，指定 <unk> 作为未知 token）
    #      2. 配置预分词器（按 Unicode 字符 + 数字 + 标点分割）
    #      3. 创建训练器，指定特殊 token 和目标词表大小
    #      4. 将文本列表写入临时文件（tokenizers 库要求文件输入）
    #      5. 调用 tokenizers 库的 train 方法执行 BPE 训练
    #      6. 训练完成后，填充 char_to_id / id_to_char / vocab_size
    #
    #    旧版 build_vocab 的迁移说明:
    #      旧: tokenizer.build_vocab(["你好", "世界"])
    #      新: tokenizer.train_from_texts(["你好", "世界"], vocab_size=8000)
    # ------------------------------------------------------------------
    def train_from_texts(
        self,
        texts: List[str],
        vocab_size: int = 16000,
    ) -> None:
        """从文本列表训练 BPE 分词器。

        BPE 训练过程:
        ─────────────
        1. 初始词表包含所有出现过的单字符
        2. 统计所有相邻 token 对的共现频率
           例: "人工" 出现 1000 次，"工智" 出现 300 次
        3. 把最高频的 token 对合并为新 token
           "人" + "工" → "人工"（加入词表，ID 递增）
        4. 重复 2-3，直到词表大小达到 vocab_size
        5. 最终词表包含: 特殊 token + 单字符 + 高频子词/词组

        参数:
          texts      — 训练文本列表
                       预训练: ["一段长文本...", "另一段文本...", ...]
                       微调:   ["问题1回答1", "问题2回答2", ...]
          vocab_size — 目标词表大小（含 5 个特殊 token），默认 8000
                       建议范围: 4000~16000（对 50M 参数模型）
                       太小: 高频词无法合并，退化为字符级
                       太大: Embedding 层参数过多，训练数据不够学好每个 token
        """
        # ── 步骤 1: 创建 BPE 模型 ──
        # unk_token 指定未知 token 的文本表示
        tokenizer = Tokenizer(models.BPE(unk_token=UNK_TOKEN))

        # ── 步骤 2: 配置预分词器 ──
        # 按 Unicode 脚本边界 + 数字 + 标点符号进行初步切分
        # 这样中文字符、英文单词、数字、标点会被分到不同的组
        # BPE 合并只在同一组内发生，避免跨类别的无意义合并
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.UnicodeScripts(),
            pre_tokenizers.Whitespace(),
        ])

        # ── 步骤 3: 配置解码器 ──
        # 解码时自动处理子词拼接（去掉 BPE 切分产生的特殊前缀标记 "▁" 等）
        tokenizer.decoder = decoders.Fuse()

        # ── 步骤 4: 创建 BPE 训练器 ──
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=SPECIAL_TOKENS,
            show_progress=False,
            # min_frequency=2: 至少出现 2 次的字符对才会被考虑合并
            min_frequency=2,
        )

        # ── 步骤 5: 将文本写入临时文件 ──
        # tokenizers 库的 train 方法要求文件路径列表作为输入
        # （这是出于性能考虑：库用 Rust 实现，直接读文件比 Python 传数据更快）
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8"
            ) as f:
                tmp_path = f.name
                for text in texts:
                    f.write(text + "\n")

            # ── 步骤 6: 执行 BPE 训练 ──
            tokenizer.train([tmp_path], trainer=trainer)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        # ── 步骤 7: 保存内部引用并填充兼容映射表 ──
        self._tokenizer = tokenizer
        self._sync_vocab()

    # ------------------------------------------------------------------
    # build_vocab —— 兼容旧接口的别名
    #
    #    调用时机: 向后兼容，供 dataset.py / train.py / 测试代码使用
    #    行为:     等价于 train_from_texts(texts, vocab_size)
    #
    #    旧代码: tokenizer.build_vocab(all_texts)
    #    新代码: 无需修改，build_vocab 会自动调用 train_from_texts
    # ------------------------------------------------------------------
    def build_vocab(self, texts: List[str], vocab_size: int = 16000) -> None:
        """向后兼容接口：等价于 train_from_texts。

        保留此方法是为了让 dataset.py、train.py、test_*.py 中的
        tokenizer.build_vocab(all_texts) 调用无需修改。
        """
        self.train_from_texts(texts, vocab_size=vocab_size)

    # ------------------------------------------------------------------
    # _sync_vocab —— 从 _tokenizer 同步映射表到兼容属性
    #
    #    调用时机: train_from_texts / load 完成后内部调用
    #    作用:     填充 char_to_id / id_to_char / vocab_size
    #              让 dataset.py / chat.py 中的
    #                tokenizer.char_to_id[BOS_TOKEN]
    #                tokenizer.char_to_id[SEP_TOKEN]
    #              等代码继续正常工作
    # ------------------------------------------------------------------
    def _sync_vocab(self) -> None:
        """从底层 tokenizer 同步词表到兼容属性。"""
        if self._tokenizer is None:
            return
        vocab = self._tokenizer.get_vocab()
        self.char_to_id = dict(vocab)
        self.id_to_char = {v: k for k, v in vocab.items()}
        self.vocab_size = self._tokenizer.get_vocab_size()

    # ------------------------------------------------------------------
    # ② encode —— 把文本变成 ID 列表（训练和推理都要用）
    #
    #    前置条件: 必须先调用 train_from_texts() 或 load() 初始化分词器。
    #
    #    调用时机与上下游:
    #    ┌──────────────────────────────────────────────────────────────┐
    #    │  训练阶段 (dataset.py):                                      │
    #    │    raw_text = "你叫什么名字？"                                │
    #    │    ids = tokenizer.encode(raw_text)                          │
    #    │    → [42, 15, 88, 7]  (BPE 子词 ID，可能少于字符数)          │
    #    │                                                              │
    #    │  推理阶段 (chat.py):                                         │
    #    │    user_input = "你叫什么名字？"                              │
    #    │    ids = tokenizer.encode(user_input)                        │
    #    │    → [42, 15, 88, 7]                                        │
    #    └──────────────────────────────────────────────────────────────┘
    #
    #    与旧版的区别:
    #      旧版（字符级）: len(encode("人工智能")) == 4（恒等于字符数）
    #      新版（BPE）:    len(encode("人工智能")) <= 4（高频词被合并）
    #
    #    输入:  text — 一段纯文本（不含特殊标记）
    #    输出:  List[int] — BPE token ID 列表
    #
    #    职责边界: encode 只做"分词 + 查表"，不负责添加 <s>/<sep>/<e>。
    #             特殊标记的拼接由上层调用方（dataset.py / chat.py）控制。
    # ------------------------------------------------------------------
    def encode(self, text: str) -> List[int]:
        """将文本编码为 BPE token ID 列表。

        参数:
          text — 待编码的纯文本字符串

        返回:
          BPE token ID 列表。长度 <= len(text)（高频词组被合并后更短）

        示例:
          encode("人工智能")  → [156, 2847]   (2 个 token)
          encode("你好")      → [42, 15]      (可能 2 个，也可能 1 个)
          encode("")          → []            (空字符串返回空列表)
        """
        if not text:
            return []
        if self._tokenizer is None:
            raise RuntimeError("分词器未初始化，请先调用 train_from_texts() 或 load()")
        return self._tokenizer.encode(text).ids

    # ------------------------------------------------------------------
    # ③ decode —— 把 ID 列表还原成文本（推理时用）
    #
    #    前置条件: 分词器已初始化（与 encode 时使用同一份分词器）。
    #
    #    调用时机与上下游:
    #    ┌──────────────────────────────────────────────────────────────┐
    #    │  推理阶段 (chat.py):                                        │
    #    │    模型自回归生成一串 token ID:                                │
    #    │      generated_ids = [33, 21, 67, 12, 55]                   │
    #    │    tokenizer.decode(generated_ids)                           │
    #    │    → "我是Nova。"                                            │
    #    └──────────────────────────────────────────────────────────────┘
    #
    #    过滤规则:
    #      <pad>(0), <s>(1), <e>(2), <sep>(3) → 静默跳过，不出现在输出中
    #      <unk>(4) → 保留为 "<unk>" 字符串
    #      其他正常 ID → 由 BPE 解码器还原为对应文本
    #
    #    输入:  ids — 一串 token ID（可能包含特殊标记的 ID）
    #    输出:  str — 人类可读的文本
    # ------------------------------------------------------------------
    def decode(self, ids: List[int]) -> str:
        """将 token ID 列表解码为文本。

        参数:
          ids — token ID 列表（可能包含特殊标记 ID）

        返回:
          过滤掉 <pad>/<s>/<e>/<sep> 后的可读文本

        过滤逻辑:
          1. 过滤掉 PAD/BOS/EOS/SEP 的 ID（它们是控制信号，不是内容）
          2. 把过滤后的 ID 列表交给 BPE 解码器还原文本
          3. BPE 解码器会自动处理子词拼接
        """
        if not ids:
            return ""
        if self._tokenizer is None:
            raise RuntimeError("分词器未初始化，请先调用 train_from_texts() 或 load()")

        # 过滤特殊标记（PAD/BOS/EOS/SEP 是控制信号，不应出现在输出文本中）
        skip_ids = {PAD_ID, BOS_ID, EOS_ID, SEP_ID}
        filtered = [tid for tid in ids if tid not in skip_ids]

        if not filtered:
            return ""

        return self._tokenizer.decode(filtered)

    # ------------------------------------------------------------------
    # ④ save —— 将分词器持久化到磁盘（训练完成后调用）
    #
    #    前置条件: 已调用 train_from_texts()，分词器已训练完成。
    #
    #    调用时机与上下游:
    #    ┌──────────────────────────────────────────────────────────────┐
    #    │  训练阶段 (train.py):                                       │
    #    │    tokenizer = NovaTokenizer()                               │
    #    │    tokenizer.train_from_texts(all_texts, vocab_size=8000)    │
    #    │    tokenizer.save("data/tokenizer.json")                    │
    #    │    → 磁盘上生成 tokenizer.json（包含完整的 BPE 模型和词表）  │
    #    └──────────────────────────────────────────────────────────────┘
    #
    #    保存内容: 完整的 BPE 分词器状态（词表、合并规则、配置）
    #    文件格式: JSON（HuggingFace tokenizers 库的标准格式）
    #
    #    输入:  path — 保存的文件路径（如 "data/tokenizer.json"）
    #    输出:  无返回值；在 path 处生成 JSON 文件
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """将 BPE 分词器保存到磁盘。

        保存的 JSON 文件包含:
          - BPE 模型的所有合并规则（merges）
          - 完整词表（vocab）
          - 预分词器配置
          - 解码器配置
          - 特殊 token 定义

        参数:
          path — 保存路径（如 "data/tokenizer.json"）
        """
        if self._tokenizer is None:
            raise RuntimeError("分词器未初始化，无法保存")
        self._tokenizer.save(path)

    # ------------------------------------------------------------------
    # ⑤ load —— 从磁盘加载已有分词器（推理时调用，跳过 train_from_texts）
    #
    #    前置条件: path 指向的文件必须存在，且是 save() 生成的合法 JSON。
    #
    #    调用时机与上下游:
    #    ┌──────────────────────────────────────────────────────────────┐
    #    │  推理阶段 (chat.py):                                        │
    #    │    tokenizer = NovaTokenizer()                               │
    #    │    tokenizer.load("data/tokenizer.json")                    │
    #    │    → 从磁盘恢复完整的 BPE 分词器                             │
    #    │    → encode/decode 行为与训练时完全一致                       │
    #    └──────────────────────────────────────────────────────────────┘
    #
    #    输入:  path — 分词器文件路径（如 "data/tokenizer.json"）
    #    输出:  无返回值；内部恢复 _tokenizer 并填充映射表
    #
    #    注意: load 会覆盖当前实例的所有状态。如果之前调用过
    #          train_from_texts，load 之后旧分词器会被完全替换。
    # ------------------------------------------------------------------
    def load(self, path: str) -> None:
        """从磁盘加载 BPE 分词器。

        参数:
          path — 分词器文件路径（save() 生成的 JSON 文件）

        加载后的状态:
          - self._tokenizer 恢复为训练时的完整 BPE 分词器
          - self.char_to_id / id_to_char / vocab_size 全部同步更新
          - encode / decode 行为与训练时完全一致
        """
        self._tokenizer = Tokenizer.from_file(path)
        self._sync_vocab()
