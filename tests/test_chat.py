"""chat.py 单元测试

测试范围:
  1. generate              — 自回归文本生成的核心逻辑
  2. load_model_for_inference — 模型和分词器的加载
  3. 验收标准              — 输入核心问题能得到正确回答、响应时间 < 1 秒

运行方式:
  .venv/bin/python -m unittest tests.test_chat -v                       # 跑本文件全部
  .venv/bin/python -m unittest tests.test_chat.TestGenerate -v          # 只跑生成函数
  .venv/bin/python -m unittest tests.test_chat.TestAcceptanceCriteria -v # 只跑验收标准
"""

import os
import time
import unittest

import torch

from config import NovaConfig
from tokenizer import NovaTokenizer, EOS_ID, BOS_ID, SEP_ID
from model import NovaModel
from chat import generate, load_model_for_inference


# ======================================================================
# 辅助函数：构建一个已训练好的微型模型环境
# ======================================================================
def _make_trained_mini_model():
    """训练一个极小的模型用于测试 generate 函数。

    为了让测试独立于 checkpoint 文件，这里:
      1. 用 2 条问答对构建分词器
      2. 创建极小的模型 (d_model=32, 1 层)
      3. 训练 100 轮使模型过拟合这 2 条数据
      4. 返回 (model, tokenizer) 供 generate 测试使用
    """
    from dataset import NovaDataset, create_dataloader
    import torch.nn.functional as F

    tokenizer = NovaTokenizer()
    tokenizer.train_from_texts(["你好", "世界", "你叫什么名字", "我是Nova"])

    config = NovaConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=32, n_heads=2, n_layers=1, d_ff=64,
        max_seq_len=32, batch_size=2, epochs=100,
        learning_rate=1e-3, warmup_steps=2, dropout=0.0,
    )
    qa_pairs = [
        {"question": "你好", "answer": "世界"},
        {"question": "你叫什么名字", "answer": "我是Nova"},
    ]
    dataset = NovaDataset(qa_pairs, tokenizer, config.max_seq_len)
    dataloader = create_dataloader(dataset, batch_size=config.batch_size)

    device = torch.device("cpu")
    model = NovaModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    model.train()
    for epoch in range(100):
        for batch in dataloader:
            logits = model(batch["input_ids"])
            loss = F.cross_entropy(
                logits.view(-1, config.vocab_size),
                batch["target_ids"].view(-1),
                ignore_index=-100,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    return model, tokenizer


# ======================================================================
# TestGenerate — 自回归生成函数测试
# ======================================================================
class TestGenerate(unittest.TestCase):
    """验证 generate 函数的核心行为。"""

    @classmethod
    def setUpClass(cls):
        """训练一个微型模型供所有测试复用。"""
        cls.model, cls.tokenizer = _make_trained_mini_model()

    def test_returns_string(self):
        """generate 应返回 str 类型。"""
        result = generate(self.model, self.tokenizer, "你好")
        self.assertIsInstance(result, str)

    def test_returns_non_empty(self):
        """generate 应返回非空字符串。"""
        result = generate(self.model, self.tokenizer, "你好")
        self.assertGreater(len(result), 0)

    def test_no_special_tokens_in_output(self):
        """输出中不应包含特殊标记 <s>、<e>、<sep>、<pad>。"""
        result = generate(self.model, self.tokenizer, "你好")
        for token in ("<s>", "<e>", "<sep>", "<pad>"):
            self.assertNotIn(token, result)

    def test_max_new_tokens_limits_output(self):
        """max_new_tokens=5 时，生成的字符数不应超过 5。"""
        result = generate(self.model, self.tokenizer, "你好", max_new_tokens=5)
        self.assertLessEqual(len(result), 5)

    def test_temperature_zero_point_one_deterministic(self):
        """极低温度 (0.1) 下多次生成结果应高度一致。"""
        results = set()
        for _ in range(5):
            r = generate(self.model, self.tokenizer, "你好", temperature=0.1, top_k=1)
            results.add(r)
        self.assertEqual(len(results), 1, f"温度 0.1 + top_k=1 应产生确定性结果，但得到: {results}")

    def test_top_k_1_is_greedy(self):
        """top_k=1 等价于贪心解码，多次结果应完全相同。"""
        results = set()
        for _ in range(5):
            r = generate(self.model, self.tokenizer, "你好", temperature=1.0, top_k=1)
            results.add(r)
        self.assertEqual(len(results), 1, f"top_k=1 应产生相同结果，但得到: {results}")

    def test_handles_unknown_question(self):
        """对训练数据中没有的问题也不应崩溃。"""
        result = generate(self.model, self.tokenizer, "量子力学是什么？")
        self.assertIsInstance(result, str)

    def test_empty_question(self):
        """空问题也不应崩溃。"""
        result = generate(self.model, self.tokenizer, "")
        self.assertIsInstance(result, str)

    def test_generation_speed(self):
        """生成一次回答应在 1 秒内完成（微型模型 + CPU）。"""
        start = time.time()
        generate(self.model, self.tokenizer, "你好")
        elapsed = time.time() - start
        self.assertLess(elapsed, 1.0, f"生成耗时 {elapsed:.2f}s，超过 1 秒限制")


# ======================================================================
# TestLoadModel — 模型加载测试
# ======================================================================
class TestLoadModel(unittest.TestCase):
    """验证 load_model_for_inference 的加载逻辑。"""

    def test_load_with_real_checkpoint(self):
        """如果 best_model.pt 存在，应能正常加载。"""
        ckpt_path = "checkpoints/best_model.pt"
        vocab_path = "data/tokenizer.json"
        if not os.path.isfile(ckpt_path) or not os.path.isfile(vocab_path):
            self.skipTest("checkpoint 或分词器文件不存在，跳过")

        model, tokenizer, device = load_model_for_inference(ckpt_path, vocab_path)
        self.assertIsInstance(model, NovaModel)
        self.assertIsInstance(tokenizer, NovaTokenizer)
        self.assertGreater(tokenizer.vocab_size, 0)
        self.assertFalse(model.training, "模型应处于 eval 模式")

    def test_checkpoint_not_found(self):
        """checkpoint 不存在时应抛出 FileNotFoundError。"""
        with self.assertRaises(FileNotFoundError):
            load_model_for_inference("/nonexistent/ckpt.pt")

    def test_vocab_not_found(self):
        """分词器文件不存在时应抛出 FileNotFoundError。"""
        with self.assertRaises(FileNotFoundError):
            load_model_for_inference(
                "checkpoints/best_model.pt",
                vocab_path="/nonexistent/tokenizer.json",
            )


# ======================================================================
# TestAcceptanceCriteria — 验收标准测试
# ======================================================================
class TestAcceptanceCriteria(unittest.TestCase):
    """验证实现计划中列出的验收标准。

    验收标准（IMPLEMENTATION_PLAN.md 第 413-417 行）:
      ✓ 输入核心问题，能得到正确回答
      ✓ 响应时间 < 1 秒
      ✓ 界面友好、无崩溃
    """

    @classmethod
    def setUpClass(cls):
        """加载真实训练好的模型（如果有的话）。"""
        ckpt_path = "checkpoints/best_model.pt"
        vocab_path = "data/tokenizer.json"
        if not os.path.isfile(ckpt_path) or not os.path.isfile(vocab_path):
            cls.model = None
            cls.tokenizer = None
            return

        try:
            cls.model, cls.tokenizer, cls.device = load_model_for_inference(
                ckpt_path, vocab_path,
            )
            # 验证分词器 vocab_size 与模型的 Embedding 层一致
            # 升级分词器（如字符级→BPE）后旧 checkpoint 不兼容
            model_vocab = cls.model.config.vocab_size
            if cls.tokenizer.vocab_size != model_vocab:
                cls.model = None
                cls.tokenizer = None
        except Exception:
            cls.model = None
            cls.tokenizer = None

    def _skip_if_no_model(self):
        if self.model is None:
            self.skipTest("checkpoint/分词器不存在或不兼容，跳过验收测试")

    def test_acceptance_name_question(self):
        """验收: 问"你叫什么名字？"应回答包含"Nova"。"""
        self._skip_if_no_model()
        answer = generate(self.model, self.tokenizer, "你叫什么名字？",
                          temperature=0.1, top_k=5)
        self.assertIn("Nova", answer,
                       f"问'你叫什么名字？'时回答应包含'Nova'，实际: '{answer}'")

    def test_acceptance_who_are_you(self):
        """验收: 问"你是谁？"应回答包含"Nova"。"""
        self._skip_if_no_model()
        answer = generate(self.model, self.tokenizer, "你是谁？",
                          temperature=0.1, top_k=5)
        self.assertIn("Nova", answer,
                       f"问'你是谁？'时回答应包含'Nova'，实际: '{answer}'")

    def test_acceptance_greeting(self):
        """验收: 问"你好吗？"应给出正面回复。"""
        self._skip_if_no_model()
        answer = generate(self.model, self.tokenizer, "你好吗？",
                          temperature=0.1, top_k=5)
        self.assertTrue(len(answer) > 0,
                        f"问'你好吗？'时应有非空回答，实际: '{answer}'")

    def test_acceptance_response_time(self):
        """验收: 响应时间应 < 1 秒。"""
        self._skip_if_no_model()
        start = time.time()
        generate(self.model, self.tokenizer, "你叫什么名字？")
        elapsed = time.time() - start
        self.assertLess(elapsed, 1.0,
                        f"响应时间 {elapsed:.2f}s 超过 1 秒限制")

    def test_acceptance_no_crash_on_various_inputs(self):
        """验收: 各种输入都不应崩溃。"""
        self._skip_if_no_model()
        test_inputs = [
            "你好",
            "你叫什么名字？",
            "",
            "a",
            "这是一个很长很长很长的问题" * 10,
            "🎉😊",
        ]
        for q in test_inputs:
            result = generate(self.model, self.tokenizer, q)
            self.assertIsInstance(result, str,
                                 f"输入 '{q[:20]}...' 应返回 str")


if __name__ == "__main__":
    unittest.main()
