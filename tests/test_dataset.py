"""NovaDataset / PretrainDataset 单元测试

按功能模块拆分为 6 个测试类:
  TestPretrainDataEncoding  — 预训练数据编码逻辑
  TestPretrainGetItem       — 预训练 __getitem__ / __len__
  TestDataEncoding          — 微调数据编码逻辑
  TestGetItem               — 微调 __getitem__ / __len__
  TestDataLoader            — DataLoader 工厂函数
  TestLoadFunctions         — 数据加载辅助函数

运行方式:
  .venv/bin/pytest tests/test_dataset.py -v
  .venv/bin/python -m unittest tests.test_dataset -v
"""

import json
import os
import tempfile
import unittest

import torch

from tokenizer import NovaTokenizer, PAD_ID, BOS_ID, EOS_ID, SEP_ID
from dataset import (
    NovaDataset,
    PretrainDataset,
    create_dataloader,
    load_qa_pairs,
    load_pretrain_data,
)


# ======================================================================
# 辅助函数
# ======================================================================

def _make_tokenizer(texts: list[str] | None = None) -> NovaTokenizer:
    """训练一个 BPE 分词器供测试使用。"""
    if texts is None:
        texts = ["你好", "世界", "你叫什么名字", "我是Nova", "计算机是科技发明"]
    tokenizer = NovaTokenizer()
    tokenizer.build_vocab(texts, vocab_size=500)
    return tokenizer


def _make_pretrain_dataset(
    texts: list[str] | None = None,
    max_seq_len: int = 64,
) -> tuple[NovaTokenizer, PretrainDataset, list[str]]:
    """构建 tokenizer + PretrainDataset。"""
    if texts is None:
        texts = [
            "计算机是20世纪最先进的科学技术发明之一。",
            "人工智能是未来的趋势。",
            "深度学习需要大量的训练数据。",
        ]
    tokenizer = _make_tokenizer(texts)
    dataset = PretrainDataset(texts, tokenizer, max_seq_len)
    return tokenizer, dataset, texts


def _make_finetune_dataset(
    qa_pairs: list | None = None,
    max_seq_len: int = 64,
) -> tuple[NovaTokenizer, NovaDataset, list]:
    """构建 tokenizer + NovaDataset（微调）。"""
    if qa_pairs is None:
        qa_pairs = [
            {"question": "你好", "answer": "你好！"},
            {"question": "你叫什么名字？", "answer": "我是Nova。"},
        ]
    all_texts = [p["question"] + p["answer"] for p in qa_pairs]
    tokenizer = _make_tokenizer(all_texts)
    dataset = NovaDataset(qa_pairs, tokenizer, max_seq_len)
    return tokenizer, dataset, qa_pairs


# ======================================================================
# PretrainDataset 编码逻辑测试
# ======================================================================
class TestPretrainDataEncoding(unittest.TestCase):
    """验证 PretrainDataset 的编码流程: 拼接 → 编码 → 截断 → 填充。"""

    def setUp(self) -> None:
        self.tokenizer, self.dataset, self.texts = _make_pretrain_dataset()

    def test_input_ids_starts_with_bos(self) -> None:
        """每条预训练数据的 input_ids 以 <s> (BOS_ID=1) 开头。"""
        for i in range(len(self.dataset)):
            item = self.dataset[i]
            self.assertEqual(item["input_ids"][0].item(), BOS_ID)

    def test_input_ids_contains_eos(self) -> None:
        """预训练数据中包含 <e> (EOS_ID=2)（未截断时）。"""
        item = self.dataset[0]
        self.assertIn(EOS_ID, item["input_ids"].tolist())

    def test_no_sep_token_in_pretrain(self) -> None:
        """预训练数据不应包含 <sep> 标记（那是微调专用的）。"""
        for i in range(len(self.dataset)):
            item = self.dataset[i]
            ids = item["input_ids"].tolist()
            # 过滤掉 padding 区域（可能包含 0 即 PAD_ID，不会包含 SEP_ID=3 的有效含义）
            eos_pos = ids.index(EOS_ID) if EOS_ID in ids else len(ids)
            valid_ids = ids[:eos_pos + 1]
            self.assertNotIn(SEP_ID, valid_ids,
                             f"预训练数据不应包含 <sep> (ID={SEP_ID})")

    def test_sequence_order_is_bos_text_eos(self) -> None:
        """token 顺序: <s> → 文本 token → <e> → <pad>..."""
        item = self.dataset[0]
        ids = item["input_ids"].tolist()
        bos_pos = ids.index(BOS_ID)
        eos_pos = ids.index(EOS_ID)
        self.assertEqual(bos_pos, 0)
        self.assertGreater(eos_pos, bos_pos)

    def test_padding_fills_remaining_positions(self) -> None:
        """不足 max_seq_len 的部分用 PAD_ID (0) 填充。"""
        item = self.dataset[0]
        ids = item["input_ids"].tolist()
        eos_pos = ids.index(EOS_ID)
        for pad_val in ids[eos_pos + 1:]:
            self.assertEqual(pad_val, PAD_ID)

    def test_input_ids_length_equals_max_seq_len(self) -> None:
        """input_ids 长度 = max_seq_len。"""
        for i in range(len(self.dataset)):
            item = self.dataset[i]
            self.assertEqual(len(item["input_ids"]), 64)

    def test_truncation_when_too_long(self) -> None:
        """序列超过 max_seq_len 时被截断。"""
        long_text = ["这是一段很长的文本" * 50]
        _, dataset, _ = _make_pretrain_dataset(texts=long_text, max_seq_len=20)
        item = dataset[0]
        self.assertEqual(len(item["input_ids"]), 20)

    def test_empty_text_skipped(self) -> None:
        """空文本应被跳过。"""
        texts = ["正常文本", "", "   ", "另一段正常文本"]
        _, dataset, _ = _make_pretrain_dataset(texts=texts)
        self.assertEqual(len(dataset), 2)

    def test_labels_shifted_left(self) -> None:
        """labels[i] = input_ids[i+1]（下一个 token 预测）。"""
        item = self.dataset[0]
        input_ids = item["input_ids"].tolist()
        target_ids = item["target_ids"].tolist()
        eos_pos = input_ids.index(EOS_ID)
        for i in range(eos_pos):
            self.assertEqual(target_ids[i], input_ids[i + 1],
                             f"位置 {i}: target={target_ids[i]}, "
                             f"expected input[{i+1}]={input_ids[i+1]}")


# ======================================================================
# PretrainDataset __getitem__ / __len__ 测试
# ======================================================================
class TestPretrainGetItem(unittest.TestCase):
    """验证 PretrainDataset 的 __getitem__ 和 __len__。"""

    def setUp(self) -> None:
        self.tokenizer, self.dataset, self.texts = _make_pretrain_dataset()

    def test_len_equals_non_empty_text_count(self) -> None:
        """数据集长度 = 非空文本数量。"""
        self.assertEqual(len(self.dataset), len(self.texts))

    def test_getitem_returns_dict_with_correct_keys(self) -> None:
        """__getitem__ 返回包含 input_ids 和 target_ids 的字典。"""
        item = self.dataset[0]
        self.assertIn("input_ids", item)
        self.assertIn("target_ids", item)

    def test_tensors_are_long_type(self) -> None:
        """tensor dtype 必须是 torch.long。"""
        item = self.dataset[0]
        self.assertEqual(item["input_ids"].dtype, torch.long)
        self.assertEqual(item["target_ids"].dtype, torch.long)

    def test_padding_positions_labeled_minus_100(self) -> None:
        """padding 位置的 label 必须是 -100。"""
        item = self.dataset[0]
        input_ids = item["input_ids"].tolist()
        target_ids = item["target_ids"].tolist()
        eos_pos = input_ids.index(EOS_ID)
        for i in range(eos_pos, len(target_ids)):
            self.assertEqual(target_ids[i], -100,
                             f"位置 {i} 应为 -100，实际为 {target_ids[i]}")


# ======================================================================
# NovaDataset（微调）编码逻辑测试
# ======================================================================
class TestDataEncoding(unittest.TestCase):
    """验证 NovaDataset 的编码流程: 拼接 → 编码 → 截断 → 填充。"""

    def setUp(self) -> None:
        self.tokenizer, self.dataset, self.qa_pairs = _make_finetune_dataset()

    def test_input_ids_starts_with_bos(self) -> None:
        """每条数据的 input_ids 以 <s> (BOS_ID=1) 开头。"""
        for i in range(len(self.dataset)):
            item = self.dataset[i]
            self.assertEqual(item["input_ids"][0].item(), BOS_ID)

    def test_input_ids_contains_sep(self) -> None:
        """每条数据的 input_ids 中包含 <sep> (SEP_ID=3)。"""
        for i in range(len(self.dataset)):
            item = self.dataset[i]
            self.assertIn(SEP_ID, item["input_ids"].tolist())

    def test_input_ids_contains_eos(self) -> None:
        """每条数据的 input_ids 中包含 <e> (EOS_ID=2)（未被截断时）。"""
        item = self.dataset[0]
        self.assertIn(EOS_ID, item["input_ids"].tolist())

    def test_sequence_order_is_bos_question_sep_answer_eos(self) -> None:
        """token 顺序: <s> → 问题 → <sep> → 回答 → <e> → <pad>..."""
        item = self.dataset[0]
        ids = item["input_ids"].tolist()
        bos_pos = ids.index(BOS_ID)
        sep_pos = ids.index(SEP_ID)
        eos_pos = ids.index(EOS_ID)
        self.assertEqual(bos_pos, 0)
        self.assertGreater(sep_pos, bos_pos)
        self.assertGreater(eos_pos, sep_pos)

    def test_padding_fills_remaining_positions(self) -> None:
        """不足 max_seq_len 的部分用 PAD_ID (0) 填充。"""
        item = self.dataset[0]
        ids = item["input_ids"].tolist()
        eos_pos = ids.index(EOS_ID)
        for pad_val in ids[eos_pos + 1:]:
            self.assertEqual(pad_val, PAD_ID)

    def test_input_ids_length_equals_max_seq_len(self) -> None:
        """input_ids 长度 = max_seq_len。"""
        for i in range(len(self.dataset)):
            item = self.dataset[i]
            self.assertEqual(len(item["input_ids"]), 64)

    def test_truncation_when_too_long(self) -> None:
        """序列超过 max_seq_len 时被截断到 max_seq_len。"""
        long_pair = [{"question": "你" * 50, "answer": "好" * 50}]
        _, dataset, _ = _make_finetune_dataset(qa_pairs=long_pair, max_seq_len=20)
        item = dataset[0]
        self.assertEqual(len(item["input_ids"]), 20)


# ======================================================================
# NovaDataset（微调）__getitem__ / __len__ 测试
# ======================================================================
class TestGetItem(unittest.TestCase):
    """验证 NovaDataset 的 __getitem__ 和 __len__。"""

    def setUp(self) -> None:
        self.tokenizer, self.dataset, self.qa_pairs = _make_finetune_dataset()

    def test_len_equals_qa_pairs_count(self) -> None:
        """数据集长度 = 问答对数量。"""
        self.assertEqual(len(self.dataset), len(self.qa_pairs))

    def test_getitem_returns_dict_with_correct_keys(self) -> None:
        """__getitem__ 返回包含 input_ids 和 target_ids 的字典。"""
        item = self.dataset[0]
        self.assertIn("input_ids", item)
        self.assertIn("target_ids", item)

    def test_tensors_are_long_type(self) -> None:
        """input_ids 和 target_ids 的 dtype 必须是 torch.long。"""
        item = self.dataset[0]
        self.assertEqual(item["input_ids"].dtype, torch.long)
        self.assertEqual(item["target_ids"].dtype, torch.long)

    def test_input_ids_and_target_ids_same_length(self) -> None:
        """input_ids 和 target_ids 长度相同。"""
        item = self.dataset[0]
        self.assertEqual(len(item["input_ids"]), len(item["target_ids"]))

    def test_labels_are_input_ids_shifted_left_by_one(self) -> None:
        """labels[i] = input_ids[i+1]，即"下一个 token 预测"。"""
        item = self.dataset[0]
        input_ids = item["input_ids"].tolist()
        target_ids = item["target_ids"].tolist()
        eos_pos = input_ids.index(EOS_ID)
        for i in range(eos_pos):
            self.assertEqual(
                target_ids[i], input_ids[i + 1],
                f"位置 {i}: target_ids={target_ids[i]}, "
                f"但 input_ids[{i+1}]={input_ids[i+1]}",
            )

    def test_padding_positions_labeled_minus_100(self) -> None:
        """padding 位置的 label 必须是 -100。"""
        item = self.dataset[0]
        input_ids = item["input_ids"].tolist()
        target_ids = item["target_ids"].tolist()
        eos_pos = input_ids.index(EOS_ID)
        for i in range(eos_pos, len(target_ids)):
            self.assertEqual(
                target_ids[i], -100,
                f"位置 {i} 应为 -100，实际为 {target_ids[i]}",
            )

    def test_no_minus_100_in_valid_content_region(self) -> None:
        """-100 不应出现在有效内容区域（<s> 到 <e> 之前）。"""
        item = self.dataset[0]
        input_ids = item["input_ids"].tolist()
        target_ids = item["target_ids"].tolist()
        eos_pos = input_ids.index(EOS_ID)
        for i in range(eos_pos):
            self.assertNotEqual(
                target_ids[i], -100,
                f"位置 {i} 是有效内容，target_ids 不应为 -100",
            )


# ======================================================================
# DataLoader 工厂函数测试
# ======================================================================
class TestDataLoader(unittest.TestCase):
    """验证 create_dataloader 对 PretrainDataset 和 NovaDataset 都能工作。"""

    def test_pretrain_dataloader_iterable(self) -> None:
        """PretrainDataset 的 DataLoader 可以正常迭代。"""
        _, dataset, _ = _make_pretrain_dataset()
        dataloader = create_dataloader(dataset, batch_size=2)
        batch_count = sum(1 for _ in dataloader)
        self.assertGreater(batch_count, 0)

    def test_finetune_dataloader_iterable(self) -> None:
        """NovaDataset 的 DataLoader 可以正常迭代。"""
        _, dataset, _ = _make_finetune_dataset()
        dataloader = create_dataloader(dataset, batch_size=2)
        batch_count = sum(1 for _ in dataloader)
        self.assertGreater(batch_count, 0)

    def test_batch_has_correct_keys(self) -> None:
        """batch 包含 input_ids 和 target_ids。"""
        _, dataset, _ = _make_pretrain_dataset()
        dataloader = create_dataloader(dataset, batch_size=2)
        batch = next(iter(dataloader))
        self.assertIn("input_ids", batch)
        self.assertIn("target_ids", batch)

    def test_batch_shape(self) -> None:
        """batch 的 shape 是 [batch_size, max_seq_len]。"""
        _, dataset, _ = _make_finetune_dataset()
        dataloader = create_dataloader(dataset, batch_size=2)
        batch = next(iter(dataloader))
        B, S = batch["input_ids"].shape
        self.assertEqual(B, 2)
        self.assertEqual(S, 64)

    def test_batch_dtype_is_long(self) -> None:
        """batch 中的 tensor dtype 必须是 torch.long。"""
        _, dataset, _ = _make_finetune_dataset()
        dataloader = create_dataloader(dataset, batch_size=2)
        batch = next(iter(dataloader))
        self.assertEqual(batch["input_ids"].dtype, torch.long)
        self.assertEqual(batch["target_ids"].dtype, torch.long)


# ======================================================================
# 数据加载函数测试
# ======================================================================
class TestLoadFunctions(unittest.TestCase):
    """验证 load_qa_pairs 和 load_pretrain_data 辅助函数。"""

    def test_load_qa_pairs_from_real_file(self) -> None:
        """可以正确加载 data/sft/ 目录中的 JSONL 数据。"""
        qa_pairs = load_qa_pairs("data/sft/")
        self.assertIsInstance(qa_pairs, list)
        self.assertGreater(len(qa_pairs), 0)
        self.assertIn("question", qa_pairs[0])
        self.assertIn("answer", qa_pairs[0])

    def test_load_pretrain_data_from_real_dir(self) -> None:
        """可以正确加载 data/pretrain/ 目录中的数据。"""
        if not os.path.isdir("data/pretrain"):
            self.skipTest("data/pretrain/ 目录不存在")
        texts = load_pretrain_data("data/pretrain/")
        self.assertIsInstance(texts, list)
        self.assertGreater(len(texts), 0)
        self.assertIsInstance(texts[0], str)

    def test_load_pretrain_data_from_single_file(self) -> None:
        """可以从单个 .jsonl 文件加载数据。"""
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        )
        try:
            tmp.write('{"text": "第一行文本"}\n')
            tmp.write('{"text": "第二行文本"}\n')
            tmp.write('{"text": ""}\n')
            tmp.close()

            texts = load_pretrain_data(tmp.name)
            self.assertEqual(len(texts), 2)
            self.assertEqual(texts[0], "第一行文本")
            self.assertEqual(texts[1], "第二行文本")
        finally:
            os.unlink(tmp.name)

    def test_load_pretrain_data_from_dir(self) -> None:
        """可以从包含多个 .jsonl 文件的目录加载。"""
        tmp_dir = tempfile.mkdtemp()
        try:
            with open(os.path.join(tmp_dir, "a.jsonl"), "w", encoding="utf-8") as f:
                f.write('{"text": "文件A行1"}\n')
                f.write('{"text": "文件A行2"}\n')
            with open(os.path.join(tmp_dir, "b.jsonl"), "w", encoding="utf-8") as f:
                f.write('{"text": "文件B行1"}\n')

            texts = load_pretrain_data(tmp_dir)
            self.assertEqual(len(texts), 3)
        finally:
            import shutil
            shutil.rmtree(tmp_dir)

    def test_load_pretrain_data_raises_on_missing_path(self) -> None:
        """路径不存在时应抛出 FileNotFoundError。"""
        with self.assertRaises(FileNotFoundError):
            load_pretrain_data("/nonexistent/path")

    def test_load_pretrain_data_raises_on_empty_dir(self) -> None:
        """目录中没有 .jsonl 文件时应抛出 ValueError。"""
        tmp_dir = tempfile.mkdtemp()
        try:
            with self.assertRaises(ValueError):
                load_pretrain_data(tmp_dir)
        finally:
            import shutil
            shutil.rmtree(tmp_dir)

    def test_load_pretrain_data_skips_empty_lines(self) -> None:
        """JSONL 文件中的空行应被跳过。"""
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        )
        try:
            tmp.write('{"text": "有效文本"}\n')
            tmp.write('\n')
            tmp.write('   \n')
            tmp.write('{"text": "另一段"}\n')
            tmp.close()

            texts = load_pretrain_data(tmp.name)
            self.assertEqual(len(texts), 2)
        finally:
            os.unlink(tmp.name)


if __name__ == "__main__":
    unittest.main()
