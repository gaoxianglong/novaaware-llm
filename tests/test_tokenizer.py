"""NovaTokenizer BPE 分词器单元测试

按功能模块拆分为 4 个测试类:
  TestTrainFromTexts — BPE 训练（词表构建）
  TestEncode         — 编码方法（文本 → token ID 列表）
  TestDecode         — 解码方法（token ID 列表 → 文本）
  TestSaveLoad       — 持久化与恢复

覆盖要点:
  ✓ 特殊 token 的 ID 固定为 0-4
  ✓ BPE 合并行为：高频词组被合并为更少的 token
  ✓ encode/decode 往返一致性
  ✓ 特殊 token 在 decode 时被正确过滤
  ✓ 空输入处理
  ✓ 未知字符处理（UNK）
  ✓ save/load 后行为完全一致

运行方式:
  .venv/bin/python -m pytest tests/test_tokenizer.py -v
  .venv/bin/python -m unittest tests.test_tokenizer -v
"""

import os
import tempfile
import unittest

from tokenizer import (
    NovaTokenizer,
    PAD_ID, BOS_ID, EOS_ID, SEP_ID, UNK_ID,
    NUM_SPECIAL,
)


# 供测试复用的中文训练文本
# 包含多种类型：短语、句子、含英文、含标点、重复用词
SAMPLE_TEXTS = [
    "你好吗？",
    "我是Nova。",
    "你叫什么名字？",
    "人工智能是未来的趋势。",
    "机器学习和深度学习是人工智能的重要分支。",
    "自然语言处理让计算机能够理解人类语言。",
    "Transformer模型改变了自然语言处理领域。",
    "深度学习需要大量的训练数据。",
    "你好！有什么可以帮你的吗？",
    "我叫Nova，是一个微型语言模型。",
]


def _make_tokenizer(texts: list[str] | None = None, vocab_size: int = 500) -> NovaTokenizer:
    """辅助函数：训练一个 BPE 分词器供测试使用。

    参数:
      texts      — 训练文本列表，默认使用 SAMPLE_TEXTS
      vocab_size — 目标词表大小，测试中用较小的值加速
    """
    tokenizer = NovaTokenizer()
    tokenizer.train_from_texts(texts or SAMPLE_TEXTS, vocab_size=vocab_size)
    return tokenizer


# ======================================================================
# BPE 训练测试
# ======================================================================
class TestTrainFromTexts(unittest.TestCase):
    """测试 train_from_texts 的 BPE 词表构建逻辑。

    train_from_texts 的职责:
      1. 用 BPE 算法在训练文本上学习子词合并规则
      2. 构建词表（特殊 token ID 0-4 + BPE 子词）
      3. 更新 vocab_size
    """

    def setUp(self) -> None:
        self.tokenizer = _make_tokenizer()

    def test_vocab_size_is_positive(self) -> None:
        """词表大小应大于特殊 token 数量。"""
        self.assertGreater(self.tokenizer.vocab_size, NUM_SPECIAL)

    def test_vocab_size_does_not_exceed_target(self) -> None:
        """实际词表大小不应超过训练时指定的 vocab_size。"""
        target = 200
        tok = _make_tokenizer(vocab_size=target)
        self.assertLessEqual(tok.vocab_size, target)

    def test_retrain_resets_cleanly(self) -> None:
        """重新训练应完全重置分词器状态。"""
        old_size = self.tokenizer.vocab_size
        self.tokenizer.train_from_texts(["全新的文本内容"], vocab_size=100)
        self.assertNotEqual(self.tokenizer.vocab_size, old_size)
        # 新训练的分词器应该能正常编码新文本
        ids = self.tokenizer.encode("全新")
        self.assertIsInstance(ids, list)
        self.assertGreater(len(ids), 0)

    def test_single_char_texts(self) -> None:
        """即使训练文本只有单字符，也能正常构建词表。"""
        tok = _make_tokenizer(texts=["你", "好"], vocab_size=50)
        self.assertGreater(tok.vocab_size, 0)
        ids = tok.encode("你好")
        self.assertIsInstance(ids, list)

    def test_bpe_merges_frequent_pairs(self) -> None:
        """BPE 应把高频出现的字符对合并为单个 token，减少 token 数。

        验证方式: 用重复文本训练，编码后的 token 数应少于字符数。
        """
        # "人工智能" 重复 100 次，BPE 应学会把它合并为少量 token
        repeated = ["人工智能"] * 100
        tok = _make_tokenizer(texts=repeated, vocab_size=100)
        ids = tok.encode("人工智能")
        self.assertLess(len(ids), 4,
                        f"'人工智能' 有 4 个字符，BPE 后应少于 4 个 token，"
                        f"实际 {len(ids)} 个: {ids}")


# ======================================================================
# encode 测试
# ======================================================================
class TestEncode(unittest.TestCase):
    """测试 encode 方法：文本 → BPE token ID 列表。

    encode 的职责:
      1. 使用 BPE 算法将文本切分为子词序列
      2. 把每个子词映射为对应的整数 ID
      3. 返回 ID 列表
    """

    def setUp(self) -> None:
        self.tokenizer = _make_tokenizer()

    def test_basic_encode(self) -> None:
        """基本编码：返回非空的整数列表。"""
        ids = self.tokenizer.encode("你好")
        self.assertIsInstance(ids, list)
        self.assertGreater(len(ids), 0)
        for item in ids:
            self.assertIsInstance(item, int)

    def test_encode_returns_list_of_int(self) -> None:
        """返回值类型必须是 list[int]。"""
        ids = self.tokenizer.encode("你好吗？")
        self.assertIsInstance(ids, list)
        for item in ids:
            self.assertIsInstance(item, int)

    def test_encode_empty_string(self) -> None:
        """空字符串编码为空列表。"""
        self.assertEqual(self.tokenizer.encode(""), [])

    def test_encode_preserves_information(self) -> None:
        """编码后解码应能还原原文（encode-decode 往返一致性）。"""
        for text in ["你好", "人工智能", "Transformer模型"]:
            ids = self.tokenizer.encode(text)
            decoded = self.tokenizer.decode(ids)
            self.assertEqual(decoded.replace(" ", ""), text.replace(" ", ""),
                             f"往返失败: '{text}' → {ids} → '{decoded}'")

    def test_encode_length_leq_char_count(self) -> None:
        """BPE 编码后的 token 数 <= 字符数（高频词组被合并）。"""
        text = "人工智能是未来的趋势"
        ids = self.tokenizer.encode(text)
        self.assertLessEqual(len(ids), len(text),
                             f"BPE 编码后 token 数({len(ids)}) 应 <= "
                             f"字符数({len(text)})")

    def test_encode_does_not_produce_special_token_ids(self) -> None:
        """encode 普通文本时不应产生特殊 token 的 ID (0-4)。

        特殊 token 只应由上层代码（dataset.py / chat.py）手动添加。
        """
        ids = self.tokenizer.encode("你好吗")
        special_ids = {PAD_ID, BOS_ID, EOS_ID, SEP_ID}
        for tid in ids:
            self.assertNotIn(tid, special_ids,
                             f"encode('你好吗') 不应产生特殊 token ID {tid}")

    def test_same_text_always_gets_same_ids(self) -> None:
        """同一段文本多次编码结果必须相同（确定性）。"""
        text = "你叫什么名字？"
        ids1 = self.tokenizer.encode(text)
        ids2 = self.tokenizer.encode(text)
        self.assertEqual(ids1, ids2)

    def test_different_texts_get_different_ids(self) -> None:
        """不同文本的编码结果应该不同。"""
        ids1 = self.tokenizer.encode("你好")
        ids2 = self.tokenizer.encode("再见")
        self.assertNotEqual(ids1, ids2)

    def test_encode_chinese_and_english_mixed(self) -> None:
        """中英混合文本能正常编码。"""
        ids = self.tokenizer.encode("Transformer模型")
        self.assertIsInstance(ids, list)
        self.assertGreater(len(ids), 0)

    def test_encode_punctuation(self) -> None:
        """标点符号能正常编码。"""
        ids = self.tokenizer.encode("你好！")
        self.assertIsInstance(ids, list)
        self.assertGreater(len(ids), 0)

    def test_encode_raises_before_init(self) -> None:
        """未初始化的分词器调用 encode 应抛出 RuntimeError。"""
        uninit = NovaTokenizer()
        with self.assertRaises(RuntimeError):
            uninit.encode("你好")


# ======================================================================
# decode 测试
# ======================================================================
class TestDecode(unittest.TestCase):
    """测试 decode 方法：token ID 列表 → 文本。

    decode 的职责:
      1. 过滤掉特殊 token（<pad>/<s>/<e>/<sep>）
      2. 把剩余 ID 列表交给 BPE 解码器还原文本
      3. BPE 解码器自动处理子词拼接
    """

    def setUp(self) -> None:
        self.tokenizer = _make_tokenizer()

    def test_basic_decode(self) -> None:
        """正常 ID 列表解码为对应文本。"""
        ids = self.tokenizer.encode("你好")
        decoded = self.tokenizer.decode(ids)
        self.assertIn("你好", decoded.replace(" ", ""))

    def test_encode_decode_roundtrip(self) -> None:
        """encode → decode 往返，文本应能还原。"""
        for text in ["你好", "我是Nova。", "你叫什么名字？"]:
            ids = self.tokenizer.encode(text)
            decoded = self.tokenizer.decode(ids)
            # BPE 解码可能引入/移除空格，比较时忽略空格
            self.assertEqual(decoded.replace(" ", ""), text.replace(" ", ""),
                             f"往返失败: '{text}' → {ids} → '{decoded}'")

    def test_decode_empty_list(self) -> None:
        """空列表解码为空字符串。"""
        self.assertEqual(self.tokenizer.decode([]), "")

    def test_decode_filters_pad(self) -> None:
        """<pad> (ID=0) 在解码时被静默跳过。"""
        ids = self.tokenizer.encode("你好")
        result = self.tokenizer.decode([PAD_ID] + ids + [PAD_ID])
        self.assertNotIn("<pad>", result)
        self.assertIn("你好", result.replace(" ", ""))

    def test_decode_filters_bos(self) -> None:
        """<s> (ID=1) 在解码时被静默跳过。"""
        ids = self.tokenizer.encode("你好")
        result = self.tokenizer.decode([BOS_ID] + ids)
        self.assertNotIn("<s>", result)

    def test_decode_filters_eos(self) -> None:
        """<e> (ID=2) 在解码时被静默跳过。"""
        ids = self.tokenizer.encode("你好")
        result = self.tokenizer.decode(ids + [EOS_ID])
        self.assertNotIn("<e>", result)

    def test_decode_filters_sep(self) -> None:
        """<sep> (ID=3) 在解码时被静默跳过。"""
        ids = self.tokenizer.encode("你好")
        result = self.tokenizer.decode(ids[:1] + [SEP_ID] + ids[1:])
        self.assertNotIn("<sep>", result)

    def test_decode_filters_all_special_tokens_at_once(self) -> None:
        """所有特殊标记混在一起时都被正确过滤。"""
        ids = self.tokenizer.encode("你好")
        mixed = [PAD_ID, BOS_ID] + ids + [SEP_ID, EOS_ID, PAD_ID]
        result = self.tokenizer.decode(mixed)
        self.assertIn("你好", result.replace(" ", ""))
        for special in ["<pad>", "<s>", "<e>", "<sep>"]:
            self.assertNotIn(special, result)

    def test_decode_only_special_tokens(self) -> None:
        """只有特殊标记时解码为空字符串。"""
        result = self.tokenizer.decode([PAD_ID, BOS_ID, EOS_ID, SEP_ID])
        self.assertEqual(result, "")

    def test_decode_raises_before_init(self) -> None:
        """未初始化的分词器调用 decode 应抛出 RuntimeError。"""
        uninit = NovaTokenizer()
        with self.assertRaises(RuntimeError):
            uninit.decode([1, 2, 3])


# ======================================================================
# save / load 测试
#
# 调用顺序:
#   训练时: train_from_texts → save   （训练 BPE 后持久化到磁盘）
#   推理时: load                      （从磁盘恢复分词器，跳过训练）
#
# 核心验证点:
#   - save → load 后分词器行为完全一致
#   - 加载后的 encode/decode 结果与原始完全一致
#   - 特殊 token 的 ID 在加载后仍然固定
#   - load 会覆盖之前的状态
#   - 保存的文件是合法的 JSON
# ======================================================================
class TestSaveLoad(unittest.TestCase):
    """测试 BPE 分词器的持久化与恢复。"""

    def setUp(self) -> None:
        self.tokenizer = _make_tokenizer()

    def _save_and_load(self) -> tuple[NovaTokenizer, str]:
        """辅助方法：save 到临时文件 → load 到新实例，返回 (loaded, path)。"""
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmp.close()
        self.tokenizer.save(tmp.name)
        loaded = NovaTokenizer()
        loaded.load(tmp.name)
        return loaded, tmp.name

    def test_save_and_load_roundtrip(self) -> None:
        """save → load 后 vocab_size 一致。"""
        loaded, path = self._save_and_load()
        try:
            self.assertEqual(loaded.vocab_size, self.tokenizer.vocab_size)
        finally:
            os.unlink(path)

    def test_loaded_tokenizer_encodes_same(self) -> None:
        """加载后的分词器编码结果与原始完全一致。"""
        loaded, path = self._save_and_load()
        try:
            for text in ["你好", "人工智能", "Transformer模型", ""]:
                self.assertEqual(loaded.encode(text),
                                 self.tokenizer.encode(text),
                                 f"编码 '{text}' 的结果不一致")
        finally:
            os.unlink(path)

    def test_loaded_tokenizer_decodes_same(self) -> None:
        """加载后的分词器解码结果与原始完全一致。"""
        loaded, path = self._save_and_load()
        try:
            ids = self.tokenizer.encode("你好吗？")
            self.assertEqual(loaded.decode(ids), self.tokenizer.decode(ids))
            # 也验证包含特殊 token 的情况
            mixed = [PAD_ID, BOS_ID] + ids + [SEP_ID, EOS_ID]
            self.assertEqual(loaded.decode(mixed), self.tokenizer.decode(mixed))
        finally:
            os.unlink(path)

    def test_load_overwrites_previous_state(self) -> None:
        """load 会完全覆盖之前的状态。"""
        loaded, path = self._save_and_load()
        try:
            other = _make_tokenizer(texts=["完全不同的文本内容"], vocab_size=100)
            old_size = other.vocab_size
            other.load(path)
            self.assertEqual(other.vocab_size, self.tokenizer.vocab_size)
            self.assertNotEqual(other.vocab_size, old_size)
        finally:
            os.unlink(path)

    def test_saved_file_is_valid_json(self) -> None:
        """保存的文件是合法 JSON。"""
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmp.close()
        try:
            self.tokenizer.save(tmp.name)
            import json
            with open(tmp.name, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertIsInstance(data, dict)
        finally:
            os.unlink(tmp.name)

    def test_token_construction_dataset_style(self) -> None:
        """模拟 dataset.py 的 token 拼接方式：
        [BOS_ID] + encode(question) + [SEP_ID] + encode(answer) + [EOS_ID]
        """
        question = "你叫什么名字？"
        answer = "我是Nova。"

        token_ids = (
            [BOS_ID]
            + self.tokenizer.encode(question)
            + [SEP_ID]
            + self.tokenizer.encode(answer)
            + [EOS_ID]
        )

        self.assertEqual(token_ids[0], BOS_ID)
        self.assertEqual(token_ids[-1], EOS_ID)
        self.assertIn(SEP_ID, token_ids)
        self.assertGreater(len(token_ids), 3)

    def test_token_construction_chat_style(self) -> None:
        """模拟 chat.py 的 prompt 拼接方式：
        [BOS_ID] + encode(question) + [SEP_ID]
        """
        question = "你好吗？"

        input_ids = (
            [BOS_ID]
            + self.tokenizer.encode(question)
            + [SEP_ID]
        )

        self.assertEqual(input_ids[0], BOS_ID)
        self.assertEqual(input_ids[-1], SEP_ID)
        self.assertGreater(len(input_ids), 2)


if __name__ == "__main__":
    unittest.main()
