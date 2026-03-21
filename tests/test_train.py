"""train.py 单元测试

测试范围:
  1. get_lr              — 学习率调度器的数学正确性（Warmup + Cosine Decay）
  2. setup               — 初始化流程的组件构建（模型、优化器、数据集）
  3. setup_pretrain      — 预训练阶段初始化
  4. setup_finetune      — 微调阶段初始化
  5. train               — 训练循环的基本功能（loss 下降、checkpoint 保存）
  6. format_train_log    — 训练日志格式化
  7. should_log          — 日志打印条件判断
  8. load_checkpoint     — 断点续训 checkpoint 加载、验证、恢复
  9. format_train_summary— 训练结束后汇总格式化
  10. 验收标准           — 端到端功能验收（loss 下降、checkpoint 可加载、续训、过拟合）
  11. 预训练验收         — 预训练阶段端到端功能验收

运行方式:
  .venv/bin/pytest tests/test_train.py -v
  .venv/bin/python -m unittest tests.test_train -v
"""

import json
import math
import os
import shutil
import tempfile
import unittest

import torch
import torch.nn.functional as F

from config import NovaConfig
from tokenizer import NovaTokenizer
from dataset import NovaDataset, PretrainDataset, create_dataloader
from model import NovaModel
from train import (
    get_lr, train, setup, setup_pretrain, setup_finetune,
    format_train_log, should_log,
    format_train_summary,
    load_checkpoint, CHECKPOINT_REQUIRED_KEYS,
)


# ======================================================================
# TestGetLR — 学习率调度器测试
# ======================================================================
class TestGetLR(unittest.TestCase):
    """验证 get_lr 函数在各阶段返回正确的学习率。"""

    def setUp(self):
        self.warmup_steps = 100
        self.max_steps = 500
        self.max_lr = 3e-4
        self.min_lr = 1e-6

    # -- Warmup 阶段 --

    def test_warmup_step_0_returns_zero(self):
        """step=0 时学习率为 0（训练还没开始，步幅为 0）。"""
        lr = get_lr(0, self.warmup_steps, self.max_steps, self.max_lr, self.min_lr)
        self.assertAlmostEqual(lr, 0.0)

    def test_warmup_midpoint(self):
        """step=50（warmup 中点）时，学习率应为 max_lr 的一半。"""
        lr = get_lr(50, self.warmup_steps, self.max_steps, self.max_lr, self.min_lr)
        expected = self.max_lr * 50 / self.warmup_steps
        self.assertAlmostEqual(lr, expected)

    def test_warmup_end_equals_max_lr(self):
        """step=warmup_steps 时，学习率正好达到 max_lr（热身完成）。"""
        lr = get_lr(
            self.warmup_steps, self.warmup_steps, self.max_steps,
            self.max_lr, self.min_lr,
        )
        # step == warmup_steps 刚进入 cosine 阶段，decay_ratio=0 → coeff=1 → lr=max_lr
        self.assertAlmostEqual(lr, self.max_lr, places=8)

    def test_warmup_is_linear(self):
        """Warmup 阶段的学习率应严格线性增长。"""
        lrs = [
            get_lr(s, self.warmup_steps, self.max_steps, self.max_lr, self.min_lr)
            for s in range(self.warmup_steps)
        ]
        for i in range(1, len(lrs)):
            diff = lrs[i] - lrs[i - 1]
            expected_diff = self.max_lr / self.warmup_steps
            self.assertAlmostEqual(diff, expected_diff, places=10)

    # -- Cosine Decay 阶段 --

    def test_cosine_decay_decreases(self):
        """Cosine Decay 阶段学习率应单调递减。"""
        lrs = [
            get_lr(s, self.warmup_steps, self.max_steps, self.max_lr, self.min_lr)
            for s in range(self.warmup_steps, self.max_steps + 1)
        ]
        for i in range(1, len(lrs)):
            self.assertLessEqual(lrs[i], lrs[i - 1])

    def test_cosine_decay_end_equals_min_lr(self):
        """step=max_steps 时，学习率应降至 min_lr。"""
        lr = get_lr(
            self.max_steps, self.warmup_steps, self.max_steps,
            self.max_lr, self.min_lr,
        )
        self.assertAlmostEqual(lr, self.min_lr, places=8)

    def test_cosine_decay_midpoint(self):
        """Cosine 衰减中点的学习率应为 (max_lr + min_lr) / 2。"""
        mid_step = self.warmup_steps + (self.max_steps - self.warmup_steps) // 2
        lr = get_lr(mid_step, self.warmup_steps, self.max_steps, self.max_lr, self.min_lr)
        expected = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
            1.0 + math.cos(math.pi * 0.5)
        )
        self.assertAlmostEqual(lr, expected, places=6)

    # -- 边界和特殊情况 --

    def test_all_lr_positive(self):
        """所有 step 的学习率都应 > 0（除 step=0 外）。"""
        for s in range(1, self.max_steps + 1):
            lr = get_lr(s, self.warmup_steps, self.max_steps, self.max_lr, self.min_lr)
            self.assertGreater(lr, 0.0)

    def test_lr_never_exceeds_max(self):
        """学习率在任何 step 都不应超过 max_lr。"""
        for s in range(self.max_steps + 1):
            lr = get_lr(s, self.warmup_steps, self.max_steps, self.max_lr, self.min_lr)
            self.assertLessEqual(lr, self.max_lr + 1e-12)

    def test_lr_never_below_min(self):
        """Cosine Decay 阶段的学习率不应低于 min_lr。"""
        for s in range(self.warmup_steps, self.max_steps + 1):
            lr = get_lr(s, self.warmup_steps, self.max_steps, self.max_lr, self.min_lr)
            self.assertGreaterEqual(lr, self.min_lr - 1e-12)

    def test_default_min_lr(self):
        """不传 min_lr 时使用默认值 1e-6。"""
        lr = get_lr(self.max_steps, self.warmup_steps, self.max_steps, self.max_lr)
        self.assertAlmostEqual(lr, 1e-6, places=8)

    def test_warmup_0_steps(self):
        """warmup_steps=0 时，step=0 就进入 cosine 阶段，lr 应从 max_lr 开始。"""
        lr = get_lr(0, 0, self.max_steps, self.max_lr, self.min_lr)
        self.assertAlmostEqual(lr, self.max_lr, places=8)

    def test_custom_parameters(self):
        """使用自定义参数验证公式正确性。"""
        lr = get_lr(step=200, warmup_steps=50, max_steps=400, max_lr=1e-3, min_lr=1e-5)
        decay_ratio = (200 - 50) / (400 - 50)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        expected = 1e-5 + coeff * (1e-3 - 1e-5)
        self.assertAlmostEqual(lr, expected, places=10)


# ======================================================================
# TestSetup — 初始化流程测试
# ======================================================================
class TestSetup(unittest.TestCase):
    """验证 setup 函数正确构建所有训练组件。"""

    @classmethod
    def setUpClass(cls):
        """只执行一次 setup（比较耗时），结果供所有测试复用。"""
        cls.config, cls.dataloader, cls.model, cls.optimizer, \
            cls.device, cls.start_epoch, cls.best_loss = setup()

    def test_config_vocab_size_set(self):
        """config.vocab_size 应由分词器动态设置，大于 0。"""
        self.assertGreater(self.config.vocab_size, 0)

    def test_dataloader_not_empty(self):
        """DataLoader 应至少有 1 个 batch。"""
        self.assertGreater(len(self.dataloader), 0)

    def test_dataloader_batch_shape(self):
        """DataLoader 的第一个 batch 形状应为 [batch_size, max_seq_len]。"""
        batch = next(iter(self.dataloader))
        self.assertEqual(batch["input_ids"].dim(), 2)
        self.assertEqual(batch["input_ids"].shape[1], self.config.max_seq_len)
        self.assertEqual(batch["target_ids"].shape[1], self.config.max_seq_len)

    def test_model_is_nova_model(self):
        """创建的模型应该是 NovaModel 实例。"""
        self.assertIsInstance(self.model, NovaModel)

    def test_model_on_correct_device(self):
        """模型参数应在正确的设备上。"""
        param = next(self.model.parameters())
        self.assertEqual(param.device.type, self.device.type)

    def test_optimizer_is_adamw(self):
        """优化器应该是 AdamW。"""
        self.assertIsInstance(self.optimizer, torch.optim.AdamW)

    def test_optimizer_lr(self):
        """优化器的初始学习率应等于 config.learning_rate。"""
        for pg in self.optimizer.param_groups:
            self.assertEqual(pg["lr"], self.config.learning_rate)

    def test_start_epoch_zero(self):
        """不加载 checkpoint 时，start_epoch 应为 0。"""
        self.assertEqual(self.start_epoch, 0)

    def test_best_loss_inf(self):
        """不加载 checkpoint 时，best_loss 应为正无穷。"""
        self.assertEqual(self.best_loss, float("inf"))


# ======================================================================
# TestTrain — 训练循环测试
# ======================================================================
class TestTrain(unittest.TestCase):
    """验证训练循环的核心功能。"""

    def _make_mini_training_env(self, epochs=10):
        """构建一个极小的训练环境用于快速测试。"""
        tokenizer = NovaTokenizer()
        texts = ["你好", "世界", "你叫什么名字", "我是Nova"]
        tokenizer.build_vocab(texts)

        config = NovaConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_ff=64,
            max_seq_len=32,
            batch_size=2,
            epochs=epochs,
            learning_rate=1e-3,
            warmup_steps=2,
            dropout=0.0,
        )

        qa_pairs = [
            {"question": "你好", "answer": "世界"},
            {"question": "你叫什么名字", "answer": "我是Nova"},
        ]
        dataset = NovaDataset(qa_pairs, tokenizer, config.max_seq_len)
        dataloader = create_dataloader(dataset, batch_size=config.batch_size)

        device = torch.device("cpu")
        model = NovaModel(config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        return config, dataloader, model, optimizer, device

    def test_loss_decreases(self):
        """训练若干 epoch 后，loss 应该下降。"""
        config, dataloader, model, optimizer, device = self._make_mini_training_env(
            epochs=30,
        )

        # 计算初始 loss
        model.eval()
        with torch.no_grad():
            batch = next(iter(dataloader))
            logits = model(batch["input_ids"])
            initial_loss = F.cross_entropy(
                logits.view(-1, config.vocab_size),
                batch["target_ids"].view(-1),
                ignore_index=-100,
            ).item()

        # 训练
        tmp_dir = tempfile.mkdtemp()
        try:
            train(config, dataloader, model, optimizer, device,
                  checkpoint_dir=tmp_dir)

            # 计算训练后的 loss
            model.eval()
            with torch.no_grad():
                batch = next(iter(dataloader))
                logits = model(batch["input_ids"])
                final_loss = F.cross_entropy(
                    logits.view(-1, config.vocab_size),
                    batch["target_ids"].view(-1),
                    ignore_index=-100,
                ).item()

            self.assertLess(final_loss, initial_loss)
        finally:
            shutil.rmtree(tmp_dir)

    def test_checkpoint_saved_at_interval(self):
        """训练够 50 个 epoch 时，应保存 epoch_50.pt。"""
        config, dataloader, model, optimizer, device = self._make_mini_training_env(
            epochs=50,
        )
        tmp_dir = tempfile.mkdtemp()
        try:
            train(config, dataloader, model, optimizer, device,
                  checkpoint_dir=tmp_dir)
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "epoch_50.pt")))
        finally:
            shutil.rmtree(tmp_dir)

    def test_best_model_saved(self):
        """训练后应保存 best_model.pt。"""
        config, dataloader, model, optimizer, device = self._make_mini_training_env(
            epochs=5,
        )
        tmp_dir = tempfile.mkdtemp()
        try:
            train(config, dataloader, model, optimizer, device,
                  checkpoint_dir=tmp_dir)
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "best_model.pt")))
        finally:
            shutil.rmtree(tmp_dir)

    def test_checkpoint_contents(self):
        """checkpoint 应包含所有必要字段。"""
        config, dataloader, model, optimizer, device = self._make_mini_training_env(
            epochs=50,
        )
        tmp_dir = tempfile.mkdtemp()
        try:
            train(config, dataloader, model, optimizer, device,
                  checkpoint_dir=tmp_dir)
            ckpt = torch.load(
                os.path.join(tmp_dir, "epoch_50.pt"),
                map_location="cpu",
                weights_only=False,
            )
            self.assertIn("epoch", ckpt)
            self.assertIn("model_state_dict", ckpt)
            self.assertIn("optimizer_state_dict", ckpt)
            self.assertIn("loss", ckpt)
            self.assertIn("config", ckpt)
            self.assertEqual(ckpt["epoch"], 50)
        finally:
            shutil.rmtree(tmp_dir)

    def test_resume_training(self):
        """断点续训：加载 checkpoint 后应能继续训练且 loss 继续下降。"""
        config, dataloader, model, optimizer, device = self._make_mini_training_env(
            epochs=50,
        )
        tmp_dir = tempfile.mkdtemp()
        try:
            # 第一阶段：训练 50 轮
            train(config, dataloader, model, optimizer, device,
                  checkpoint_dir=tmp_dir)

            # 加载 checkpoint
            ckpt_path = os.path.join(tmp_dir, "best_model.pt")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            loss_before_resume = ckpt["loss"]

            # 重建模型和优化器，加载状态
            model2 = NovaModel(config).to(device)
            model2.load_state_dict(ckpt["model_state_dict"])
            optimizer2 = torch.optim.AdamW(
                model2.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
            optimizer2.load_state_dict(ckpt["optimizer_state_dict"])

            # 第二阶段：继续训练
            config2 = NovaConfig(
                vocab_size=config.vocab_size,
                d_model=32, n_heads=2, n_layers=1, d_ff=64,
                max_seq_len=32, batch_size=2,
                epochs=100, learning_rate=1e-3, warmup_steps=2, dropout=0.0,
            )
            train(config2, dataloader, model2, optimizer2, device,
                  start_epoch=ckpt["epoch"], best_loss=ckpt["loss"],
                  checkpoint_dir=tmp_dir)

            # 续训后的最佳模型 loss 应 <= 续训前的 loss
            ckpt2 = torch.load(
                os.path.join(tmp_dir, "best_model.pt"),
                map_location="cpu",
                weights_only=False,
            )
            self.assertLessEqual(ckpt2["loss"], loss_before_resume)
        finally:
            shutil.rmtree(tmp_dir)

    def test_model_in_train_mode_during_training(self):
        """训练过程中模型应处于 train 模式。"""
        config, dataloader, model, optimizer, device = self._make_mini_training_env(
            epochs=1,
        )
        model.eval()  # 先设成 eval，train 函数应该切回 train
        tmp_dir = tempfile.mkdtemp()
        try:
            train(config, dataloader, model, optimizer, device,
                  checkpoint_dir=tmp_dir)
            self.assertTrue(model.training)
        finally:
            shutil.rmtree(tmp_dir)

    def test_optimizer_lr_updated(self):
        """训练一轮后，优化器的学习率应被 get_lr 更新。"""
        config, dataloader, model, optimizer, device = self._make_mini_training_env(
            epochs=1,
        )
        tmp_dir = tempfile.mkdtemp()
        try:
            train(config, dataloader, model, optimizer, device,
                  checkpoint_dir=tmp_dir)
            current_lr = optimizer.param_groups[0]["lr"]
            expected_lr = get_lr(0, config.warmup_steps, config.epochs, config.learning_rate)
            self.assertAlmostEqual(current_lr, expected_lr, places=10)
        finally:
            shutil.rmtree(tmp_dir)

    def test_no_nan_in_loss(self):
        """训练过程中不应出现 NaN loss。"""
        config, dataloader, model, optimizer, device = self._make_mini_training_env(
            epochs=5,
        )
        tmp_dir = tempfile.mkdtemp()
        try:
            train(config, dataloader, model, optimizer, device,
                  checkpoint_dir=tmp_dir)
            ckpt = torch.load(
                os.path.join(tmp_dir, "best_model.pt"),
                map_location="cpu",
                weights_only=False,
            )
            self.assertFalse(math.isnan(ckpt["loss"]))
            self.assertFalse(math.isinf(ckpt["loss"]))
        finally:
            shutil.rmtree(tmp_dir)


# ======================================================================
# TestFormatTrainLog — 训练日志格式化测试
# ======================================================================
class TestFormatTrainLog(unittest.TestCase):
    """验证 format_train_log 生成的日志字符串格式和内容。"""

    def test_basic_format(self):
        """验证日志字符串的基本格式: [Epoch 010/500] loss=3.2456 lr=2.80e-04 time=1.2s"""
        msg = format_train_log(
            epoch=10, total_epochs=500, loss=3.2456, lr=2.8e-4, epoch_time=1.2
        )
        self.assertEqual(msg, "[Epoch 010/500] loss=3.2456 lr=2.80e-04 time=1.2s")

    def test_epoch_zero_padded_to_three_digits(self):
        """epoch 编号应补零到 3 位数: 1 → 001, 10 → 010, 100 → 100。"""
        msg1 = format_train_log(1, 500, 5.0, 1e-4, 0.5)
        self.assertIn("[Epoch 001/500]", msg1)

        msg2 = format_train_log(10, 500, 5.0, 1e-4, 0.5)
        self.assertIn("[Epoch 010/500]", msg2)

        msg3 = format_train_log(100, 500, 5.0, 1e-4, 0.5)
        self.assertIn("[Epoch 100/500]", msg3)

    def test_loss_four_decimal_places(self):
        """loss 应保留 4 位小数。"""
        msg = format_train_log(1, 100, 0.1, 1e-4, 1.0)
        self.assertIn("loss=0.1000", msg)

        msg2 = format_train_log(1, 100, 3.14159, 1e-4, 1.0)
        self.assertIn("loss=3.1416", msg2)

    def test_lr_scientific_notation(self):
        """学习率应以科学计数法显示（2 位小数）。"""
        msg = format_train_log(1, 100, 1.0, 3e-4, 1.0)
        self.assertIn("lr=3.00e-04", msg)

        msg2 = format_train_log(1, 100, 1.0, 1e-6, 1.0)
        self.assertIn("lr=1.00e-06", msg2)

    def test_time_one_decimal(self):
        """耗时应保留 1 位小数。"""
        msg = format_train_log(1, 100, 1.0, 1e-4, 0.123)
        self.assertIn("time=0.1s", msg)

        msg2 = format_train_log(1, 100, 1.0, 1e-4, 12.789)
        self.assertIn("time=12.8s", msg2)

    def test_very_small_loss(self):
        """loss 接近 0 时仍正常格式化。"""
        msg = format_train_log(500, 500, 0.0001, 1e-6, 2.0)
        self.assertIn("loss=0.0001", msg)

    def test_large_epoch_numbers(self):
        """epoch 超过 3 位数时不截断。"""
        msg = format_train_log(1000, 2000, 1.0, 1e-4, 1.0)
        self.assertIn("[Epoch 1000/2000]", msg)

    def test_return_type_is_string(self):
        """返回值类型应为 str。"""
        msg = format_train_log(1, 100, 1.0, 1e-4, 1.0)
        self.assertIsInstance(msg, str)


# ======================================================================
# TestShouldLog — 日志打印条件测试
# ======================================================================
class TestShouldLog(unittest.TestCase):
    """验证 should_log 在正确的 epoch 返回 True。"""

    def test_first_epoch_always_logs(self):
        """训练的第一轮（epoch == start_epoch）一定打印。"""
        self.assertTrue(should_log(epoch=0, start_epoch=0))

    def test_first_epoch_with_resume(self):
        """断点续训时，续训的第一轮也打印。"""
        self.assertTrue(should_log(epoch=50, start_epoch=50))

    def test_every_10th_epoch(self):
        """每隔 10 轮打印: epoch 9(→轮10), 19(→轮20), 49(→轮50)。"""
        self.assertTrue(should_log(epoch=9, start_epoch=0))
        self.assertTrue(should_log(epoch=19, start_epoch=0))
        self.assertTrue(should_log(epoch=49, start_epoch=0))

    def test_non_interval_epochs_no_log(self):
        """非打印轮次应返回 False。"""
        self.assertFalse(should_log(epoch=1, start_epoch=0))
        self.assertFalse(should_log(epoch=5, start_epoch=0))
        self.assertFalse(should_log(epoch=11, start_epoch=0))

    def test_custom_interval(self):
        """自定义 log_interval=5 时，每 5 轮打印一次。"""
        self.assertTrue(should_log(epoch=4, start_epoch=0, log_interval=5))
        self.assertTrue(should_log(epoch=9, start_epoch=0, log_interval=5))
        self.assertFalse(should_log(epoch=3, start_epoch=0, log_interval=5))

    def test_interval_1_always_logs(self):
        """log_interval=1 时，每轮都打印。"""
        for e in range(20):
            self.assertTrue(should_log(epoch=e, start_epoch=0, log_interval=1))

    def test_start_epoch_overrides_interval(self):
        """即使 start_epoch 不在 interval 上，也照样打印。"""
        self.assertTrue(should_log(epoch=37, start_epoch=37))
        self.assertFalse(should_log(epoch=38, start_epoch=37))


# ======================================================================
# TestLoadCheckpoint — 断点续训 checkpoint 加载测试
# ======================================================================
class TestLoadCheckpoint(unittest.TestCase):
    """验证 load_checkpoint 函数的加载、验证、恢复逻辑。

    测试策略:
      - 先用 train() 训练几轮生成真实的 checkpoint 文件
      - 再用 load_checkpoint() 加载，验证模型/优化器状态是否正确恢复
      - 同时测试各种错误场景（文件不存在、字段缺失）
    """

    def _make_mini_env(self, epochs=10):
        """构建极小训练环境，与 TestTrain 共用同一套 helper。"""
        tokenizer = NovaTokenizer()
        texts = ["你好", "世界", "你叫什么名字", "我是Nova"]
        tokenizer.build_vocab(texts)
        config = NovaConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=32, n_heads=2, n_layers=1, d_ff=64,
            max_seq_len=32, batch_size=2, epochs=epochs,
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
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        return config, dataloader, model, optimizer, device

    def _train_and_save(self, epochs=20):
        """训练几轮并返回 (config, checkpoint_dir, device)，供后续测试使用。"""
        config, dataloader, model, optimizer, device = self._make_mini_env(epochs=epochs)
        tmp_dir = tempfile.mkdtemp()
        train(config, dataloader, model, optimizer, device, checkpoint_dir=tmp_dir)
        return config, tmp_dir, device

    # ── 正常加载 ──

    def test_load_returns_correct_start_epoch(self):
        """加载 checkpoint 后返回的 start_epoch 应等于保存时的 epoch。"""
        config, tmp_dir, device = self._train_and_save(epochs=50)
        try:
            ckpt_path = os.path.join(tmp_dir, "epoch_50.pt")
            model = NovaModel(config).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
            start_epoch, _ = load_checkpoint(ckpt_path, model, optimizer, device)
            self.assertEqual(start_epoch, 50)
        finally:
            shutil.rmtree(tmp_dir)

    def test_load_returns_correct_best_loss(self):
        """加载 checkpoint 后返回的 best_loss 应等于保存时的 loss。"""
        config, tmp_dir, device = self._train_and_save(epochs=50)
        try:
            ckpt_path = os.path.join(tmp_dir, "epoch_50.pt")
            saved_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            expected_loss = saved_ckpt["loss"]

            model = NovaModel(config).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
            _, best_loss = load_checkpoint(ckpt_path, model, optimizer, device)
            self.assertAlmostEqual(best_loss, expected_loss, places=6)
        finally:
            shutil.rmtree(tmp_dir)

    def test_model_state_restored(self):
        """加载后模型参数应与 checkpoint 中保存的完全一致。"""
        config, tmp_dir, device = self._train_and_save(epochs=50)
        try:
            ckpt_path = os.path.join(tmp_dir, "best_model.pt")
            saved_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

            model = NovaModel(config).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
            load_checkpoint(ckpt_path, model, optimizer, device)

            for name, param in model.named_parameters():
                saved_param = saved_ckpt["model_state_dict"][name]
                self.assertTrue(
                    torch.equal(param.data.cpu(), saved_param.cpu()),
                    f"参数 {name} 未正确恢复",
                )
        finally:
            shutil.rmtree(tmp_dir)

    def test_optimizer_state_restored(self):
        """加载后优化器状态（动量 m/v）应与 checkpoint 中保存的一致。"""
        config, tmp_dir, device = self._train_and_save(epochs=50)
        try:
            ckpt_path = os.path.join(tmp_dir, "best_model.pt")
            saved_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

            model = NovaModel(config).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
            load_checkpoint(ckpt_path, model, optimizer, device)

            restored_state = optimizer.state_dict()
            saved_state = saved_ckpt["optimizer_state_dict"]
            self.assertEqual(
                len(restored_state["param_groups"]),
                len(saved_state["param_groups"]),
            )
            self.assertEqual(
                len(restored_state["state"]),
                len(saved_state["state"]),
            )
        finally:
            shutil.rmtree(tmp_dir)

    def test_load_best_model(self):
        """加载 best_model.pt 也能正常工作。"""
        config, tmp_dir, device = self._train_and_save(epochs=20)
        try:
            best_path = os.path.join(tmp_dir, "best_model.pt")
            self.assertTrue(os.path.exists(best_path))

            model = NovaModel(config).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
            start_epoch, best_loss = load_checkpoint(best_path, model, optimizer, device)
            self.assertGreater(start_epoch, 0)
            self.assertGreater(best_loss, 0.0)
            self.assertFalse(math.isinf(best_loss))
        finally:
            shutil.rmtree(tmp_dir)

    # ── 错误处理 ──

    def test_file_not_found_raises(self):
        """checkpoint 文件不存在时应抛出 FileNotFoundError。"""
        config, _, _, _, device = self._make_mini_env()
        model = NovaModel(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        with self.assertRaises(FileNotFoundError):
            load_checkpoint("/nonexistent/path/ckpt.pt", model, optimizer, device)

    def test_missing_keys_raises(self):
        """checkpoint 缺少必要字段时应抛出 KeyError。"""
        config, _, _, _, device = self._make_mini_env()
        model = NovaModel(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

        tmp_dir = tempfile.mkdtemp()
        try:
            bad_ckpt_path = os.path.join(tmp_dir, "bad.pt")
            torch.save({"epoch": 10}, bad_ckpt_path)
            with self.assertRaises(KeyError):
                load_checkpoint(bad_ckpt_path, model, optimizer, device)
        finally:
            shutil.rmtree(tmp_dir)

    # ── 端到端续训 ──

    def test_resume_then_train_loss_not_worse(self):
        """加载 checkpoint 后继续训练，最终 loss 应 ≤ 加载时的 loss。"""
        config, dataloader, model, optimizer, device = self._make_mini_env(epochs=50)
        tmp_dir = tempfile.mkdtemp()
        try:
            train(config, dataloader, model, optimizer, device, checkpoint_dir=tmp_dir)

            ckpt_path = os.path.join(tmp_dir, "best_model.pt")
            model2 = NovaModel(config).to(device)
            optimizer2 = torch.optim.AdamW(
                model2.parameters(), lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
            start_epoch, best_loss = load_checkpoint(ckpt_path, model2, optimizer2, device)

            config2 = NovaConfig(
                vocab_size=config.vocab_size,
                d_model=32, n_heads=2, n_layers=1, d_ff=64,
                max_seq_len=32, batch_size=2,
                epochs=100, learning_rate=1e-3, warmup_steps=2, dropout=0.0,
            )
            train(config2, dataloader, model2, optimizer2, device,
                  start_epoch=start_epoch, best_loss=best_loss,
                  checkpoint_dir=tmp_dir)

            final_ckpt = torch.load(
                os.path.join(tmp_dir, "best_model.pt"),
                map_location="cpu", weights_only=False,
            )
            self.assertLessEqual(final_ckpt["loss"], best_loss)
        finally:
            shutil.rmtree(tmp_dir)

    def test_required_keys_constant(self):
        """CHECKPOINT_REQUIRED_KEYS 应包含所有 5 个必要字段。"""
        expected = {"epoch", "model_state_dict", "optimizer_state_dict", "loss", "config"}
        self.assertEqual(CHECKPOINT_REQUIRED_KEYS, expected)


# ======================================================================
# TestTrainLogging — 训练循环中日志输出的集成测试
# ======================================================================
class TestTrainLogging(unittest.TestCase):
    """验证 train 函数在正确的 epoch 输出格式正确的日志。"""

    def _make_mini_env(self, epochs=20):
        """构建极小训练环境。"""
        tokenizer = NovaTokenizer()
        tokenizer.build_vocab(["你好", "世界"])
        config = NovaConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=32, n_heads=2, n_layers=1, d_ff=64,
            max_seq_len=32, batch_size=2, epochs=epochs,
            learning_rate=1e-3, warmup_steps=2, dropout=0.0,
        )
        qa_pairs = [
            {"question": "你好", "answer": "世界"},
        ]
        dataset = NovaDataset(qa_pairs, tokenizer, config.max_seq_len)
        dataloader = create_dataloader(dataset, batch_size=config.batch_size)
        device = torch.device("cpu")
        model = NovaModel(config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        return config, dataloader, model, optimizer, device

    def test_log_output_contains_epoch_format(self):
        """train 函数的输出应包含 [Epoch XXX/YYY] 格式的日志。"""
        import io
        import contextlib

        config, dataloader, model, optimizer, device = self._make_mini_env(epochs=10)
        tmp_dir = tempfile.mkdtemp()
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                train(config, dataloader, model, optimizer, device,
                      checkpoint_dir=tmp_dir)
            output = buf.getvalue()
            self.assertIn("[Epoch 010/10]", output)
        finally:
            shutil.rmtree(tmp_dir)

    def test_log_contains_loss_lr_time(self):
        """日志中应包含 loss=、lr=、time= 三个字段。"""
        import io
        import contextlib

        config, dataloader, model, optimizer, device = self._make_mini_env(epochs=10)
        tmp_dir = tempfile.mkdtemp()
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                train(config, dataloader, model, optimizer, device,
                      checkpoint_dir=tmp_dir)
            output = buf.getvalue()
            self.assertIn("loss=", output)
            self.assertIn("lr=", output)
            self.assertIn("time=", output)
        finally:
            shutil.rmtree(tmp_dir)

    def test_first_epoch_logged(self):
        """训练的第一轮 (epoch 0 → Epoch 001) 一定有日志。"""
        import io
        import contextlib

        config, dataloader, model, optimizer, device = self._make_mini_env(epochs=5)
        tmp_dir = tempfile.mkdtemp()
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                train(config, dataloader, model, optimizer, device,
                      checkpoint_dir=tmp_dir)
            output = buf.getvalue()
            self.assertIn("[Epoch 001/5]", output)
        finally:
            shutil.rmtree(tmp_dir)


# ======================================================================
# TestFormatTrainSummary — 训练汇总格式化测试
# ======================================================================
class TestFormatTrainSummary(unittest.TestCase):
    """验证 format_train_summary 生成的汇总字符串格式和内容。"""

    def test_contains_final_loss(self):
        """汇总中应包含最终 loss，保留 4 位小数。"""
        s = format_train_summary(0.0823, 0.0756, 487, 142.3, "checkpoints/best_model.pt")
        self.assertIn("最终 loss:      0.0823", s)

    def test_contains_best_loss_and_epoch(self):
        """汇总中应包含最佳 loss 和对应的 epoch。"""
        s = format_train_summary(0.0823, 0.0756, 487, 142.3, "checkpoints/best_model.pt")
        self.assertIn("最佳 loss:      0.0756 (epoch 487)", s)

    def test_contains_total_time_seconds_and_minutes(self):
        """汇总中应同时显示秒和分钟。"""
        s = format_train_summary(0.1, 0.1, 100, 142.3, "ckpt/best_model.pt")
        self.assertIn("142.3s", s)
        self.assertIn("2.4min", s)

    def test_contains_best_model_path(self):
        """汇总中应包含 best_model.pt 的路径。"""
        s = format_train_summary(0.1, 0.1, 100, 60.0, "my_ckpts/best_model.pt")
        self.assertIn("最佳模型路径:   my_ckpts/best_model.pt", s)

    def test_contains_separator_lines(self):
        """汇总应有分隔线。"""
        s = format_train_summary(0.1, 0.1, 100, 60.0, "ckpt/best_model.pt")
        self.assertIn("=" * 60, s)

    def test_contains_completion_header(self):
        """汇总应包含"训练完成"标题。"""
        s = format_train_summary(0.1, 0.1, 100, 60.0, "ckpt/best_model.pt")
        self.assertIn("训练完成", s)

    def test_return_type_is_string(self):
        """返回值应为 str。"""
        s = format_train_summary(0.1, 0.1, 100, 60.0, "ckpt/best_model.pt")
        self.assertIsInstance(s, str)

    def test_very_small_loss(self):
        """极小 loss 应正常格式化。"""
        s = format_train_summary(0.0001, 0.00005, 499, 300.0, "ckpt/best_model.pt")
        self.assertIn("0.0001", s)
        self.assertIn("0.0001", s)

    def test_short_training_time(self):
        """极短训练时间应正常显示。"""
        s = format_train_summary(1.0, 1.0, 1, 0.3, "ckpt/best_model.pt")
        self.assertIn("0.3s", s)
        self.assertIn("0.0min", s)


# ======================================================================
# TestTrainSummary — train() 输出汇总的集成测试
# ======================================================================
class TestTrainSummary(unittest.TestCase):
    """验证 train() 函数结束后输出正确的汇总并返回结果字典。"""

    def _make_mini_env(self, epochs=10):
        """构建极小训练环境。"""
        tokenizer = NovaTokenizer()
        tokenizer.build_vocab(["你好", "世界", "你叫什么名字", "我是Nova"])
        config = NovaConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=32, n_heads=2, n_layers=1, d_ff=64,
            max_seq_len=32, batch_size=2, epochs=epochs,
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
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        return config, dataloader, model, optimizer, device

    def test_return_dict_has_all_keys(self):
        """train() 返回的字典应包含 5 个结果字段。"""
        config, dataloader, model, optimizer, device = self._make_mini_env(epochs=5)
        tmp_dir = tempfile.mkdtemp()
        try:
            result = train(config, dataloader, model, optimizer, device,
                           checkpoint_dir=tmp_dir)
            self.assertIn("final_loss", result)
            self.assertIn("best_loss", result)
            self.assertIn("best_epoch", result)
            self.assertIn("total_time", result)
            self.assertIn("best_model_path", result)
        finally:
            shutil.rmtree(tmp_dir)

    def test_final_loss_is_finite(self):
        """最终 loss 应为有限正数。"""
        config, dataloader, model, optimizer, device = self._make_mini_env(epochs=5)
        tmp_dir = tempfile.mkdtemp()
        try:
            result = train(config, dataloader, model, optimizer, device,
                           checkpoint_dir=tmp_dir)
            self.assertGreater(result["final_loss"], 0.0)
            self.assertFalse(math.isnan(result["final_loss"]))
            self.assertFalse(math.isinf(result["final_loss"]))
        finally:
            shutil.rmtree(tmp_dir)

    def test_best_loss_le_final_loss(self):
        """最佳 loss 应 ≤ 最终 loss（最佳是全局最小值）。"""
        config, dataloader, model, optimizer, device = self._make_mini_env(epochs=20)
        tmp_dir = tempfile.mkdtemp()
        try:
            result = train(config, dataloader, model, optimizer, device,
                           checkpoint_dir=tmp_dir)
            self.assertLessEqual(result["best_loss"], result["final_loss"])
        finally:
            shutil.rmtree(tmp_dir)

    def test_best_epoch_in_range(self):
        """最佳 epoch 应在 [1, epochs] 范围内。"""
        epochs = 10
        config, dataloader, model, optimizer, device = self._make_mini_env(epochs=epochs)
        tmp_dir = tempfile.mkdtemp()
        try:
            result = train(config, dataloader, model, optimizer, device,
                           checkpoint_dir=tmp_dir)
            self.assertGreaterEqual(result["best_epoch"], 1)
            self.assertLessEqual(result["best_epoch"], epochs)
        finally:
            shutil.rmtree(tmp_dir)

    def test_total_time_positive(self):
        """总训练时间应 > 0。"""
        config, dataloader, model, optimizer, device = self._make_mini_env(epochs=3)
        tmp_dir = tempfile.mkdtemp()
        try:
            result = train(config, dataloader, model, optimizer, device,
                           checkpoint_dir=tmp_dir)
            self.assertGreater(result["total_time"], 0.0)
        finally:
            shutil.rmtree(tmp_dir)

    def test_best_model_path_correct(self):
        """返回的 best_model_path 应指向实际存在的文件。"""
        config, dataloader, model, optimizer, device = self._make_mini_env(epochs=5)
        tmp_dir = tempfile.mkdtemp()
        try:
            result = train(config, dataloader, model, optimizer, device,
                           checkpoint_dir=tmp_dir)
            self.assertTrue(os.path.isfile(result["best_model_path"]))
        finally:
            shutil.rmtree(tmp_dir)

    def test_summary_printed_to_stdout(self):
        """train() 的 stdout 输出应包含汇总信息。"""
        import io
        import contextlib

        config, dataloader, model, optimizer, device = self._make_mini_env(epochs=5)
        tmp_dir = tempfile.mkdtemp()
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                train(config, dataloader, model, optimizer, device,
                      checkpoint_dir=tmp_dir)
            output = buf.getvalue()
            self.assertIn("训练完成", output)
            self.assertIn("最终 loss:", output)
            self.assertIn("最佳 loss:", output)
            self.assertIn("总训练时间:", output)
            self.assertIn("最佳模型路径:", output)
        finally:
            shutil.rmtree(tmp_dir)


# ======================================================================
# TestAcceptanceCriteria — 实现计划验收标准
# ======================================================================
class TestAcceptanceCriteria(unittest.TestCase):
    """验证实现计划中列出的 4 条验收标准。

    验收标准（IMPLEMENTATION_PLAN.md 第 363-368 行）:
      ✓ loss 随训练逐步下降
      ✓ checkpoint 文件正确保存且可加载
      ✓ 断点续训后 loss 能继续下降
      ✓ 训练足够多 epoch 后 loss 降到 0.1 以下
    """

    def _make_mini_env(self, epochs=10):
        """构建极小训练环境。"""
        tokenizer = NovaTokenizer()
        tokenizer.build_vocab(["你好", "世界", "你叫什么名字", "我是Nova"])
        config = NovaConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=32, n_heads=2, n_layers=1, d_ff=64,
            max_seq_len=32, batch_size=2, epochs=epochs,
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
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        return config, dataloader, model, optimizer, device

    def test_acceptance_loss_decreases(self):
        """验收标准 1: loss 随训练逐步下降。"""
        config, dataloader, model, optimizer, device = self._make_mini_env(epochs=30)

        model.eval()
        with torch.no_grad():
            batch = next(iter(dataloader))
            logits = model(batch["input_ids"])
            initial_loss = F.cross_entropy(
                logits.view(-1, config.vocab_size),
                batch["target_ids"].view(-1),
                ignore_index=-100,
            ).item()

        tmp_dir = tempfile.mkdtemp()
        try:
            result = train(config, dataloader, model, optimizer, device,
                           checkpoint_dir=tmp_dir)
            self.assertLess(result["final_loss"], initial_loss,
                            "训练后 loss 应低于初始 loss")
        finally:
            shutil.rmtree(tmp_dir)

    def test_acceptance_checkpoint_save_and_load(self):
        """验收标准 2: checkpoint 文件正确保存且可加载。"""
        config, dataloader, model, optimizer, device = self._make_mini_env(epochs=50)
        tmp_dir = tempfile.mkdtemp()
        try:
            result = train(config, dataloader, model, optimizer, device,
                           checkpoint_dir=tmp_dir)

            ckpt_path = os.path.join(tmp_dir, "epoch_50.pt")
            self.assertTrue(os.path.isfile(ckpt_path), "epoch_50.pt 应存在")
            self.assertTrue(os.path.isfile(result["best_model_path"]),
                            "best_model.pt 应存在")

            model2 = NovaModel(config).to(device)
            optimizer2 = torch.optim.AdamW(model2.parameters(), lr=config.learning_rate)
            start_epoch, best_loss = load_checkpoint(ckpt_path, model2, optimizer2, device)
            self.assertEqual(start_epoch, 50, "加载后 start_epoch 应为 50")
            self.assertFalse(math.isnan(best_loss), "加载后 best_loss 不应为 NaN")
        finally:
            shutil.rmtree(tmp_dir)

    def test_acceptance_resume_continues_training(self):
        """验收标准 3: 断点续训后 loss 能继续下降。"""
        config, dataloader, model, optimizer, device = self._make_mini_env(epochs=50)
        tmp_dir = tempfile.mkdtemp()
        try:
            result1 = train(config, dataloader, model, optimizer, device,
                            checkpoint_dir=tmp_dir)
            loss_after_phase1 = result1["best_loss"]

            model2 = NovaModel(config).to(device)
            optimizer2 = torch.optim.AdamW(
                model2.parameters(), lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
            start_epoch, best_loss = load_checkpoint(
                result1["best_model_path"], model2, optimizer2, device,
            )

            config2 = NovaConfig(
                vocab_size=config.vocab_size,
                d_model=32, n_heads=2, n_layers=1, d_ff=64,
                max_seq_len=32, batch_size=2,
                epochs=100, learning_rate=1e-3, warmup_steps=2, dropout=0.0,
            )
            result2 = train(config2, dataloader, model2, optimizer2, device,
                            start_epoch=start_epoch, best_loss=best_loss,
                            checkpoint_dir=tmp_dir)

            self.assertLessEqual(result2["best_loss"], loss_after_phase1,
                                 "续训后 best_loss 应 ≤ 续训前")
        finally:
            shutil.rmtree(tmp_dir)

    def test_acceptance_overfit_small_dataset(self):
        """验收标准 4: 数据量小时，训练足够多 epoch 后 loss 应降到 0.1 以下。

        使用 2 条问答对 + 200 个 epoch 模拟过拟合场景。
        """
        config, dataloader, model, optimizer, device = self._make_mini_env(epochs=200)
        tmp_dir = tempfile.mkdtemp()
        try:
            result = train(config, dataloader, model, optimizer, device,
                           checkpoint_dir=tmp_dir)
            self.assertLess(result["best_loss"], 0.1,
                            f"200 epoch 后 best_loss 应 < 0.1，实际 {result['best_loss']:.4f}")
        finally:
            shutil.rmtree(tmp_dir)


# ======================================================================
# TestSetupPretrain — 预训练初始化测试
# ======================================================================
class TestSetupPretrain(unittest.TestCase):
    """验证 setup_pretrain 正确初始化预训练环境。"""

    def _make_pretrain_dir(self) -> str:
        """创建临时 JSONL 预训练数据目录。"""
        tmp_dir = tempfile.mkdtemp()
        jsonl_path = os.path.join(tmp_dir, "train.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for text in [
                "计算机是现代科学技术的重要发明。",
                "人工智能正在改变世界。",
                "深度学习需要大量训练数据和算力。",
                "自然语言处理让机器能理解人类语言。",
                "Transformer模型是NLP的重要突破。",
            ]:
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        return tmp_dir

    def test_setup_pretrain_returns_all_components(self) -> None:
        """setup_pretrain 应返回 7 个组件。"""
        tmp_dir = self._make_pretrain_dir()
        try:
            config, dataloader, model, optimizer, device, start_epoch, best_loss = (
                setup_pretrain(data_path=tmp_dir)
            )
            self.assertIsInstance(config, NovaConfig)
            self.assertIsInstance(model, NovaModel)
            self.assertIsInstance(optimizer, torch.optim.AdamW)
            self.assertEqual(start_epoch, 0)
            self.assertEqual(best_loss, float("inf"))
            self.assertGreater(config.vocab_size, 0)
            self.assertGreater(len(dataloader), 0)
        finally:
            shutil.rmtree(tmp_dir)

    def test_tokenizer_saved_after_pretrain_setup(self) -> None:
        """setup_pretrain 应保存 tokenizer.json。"""
        tmp_dir = self._make_pretrain_dir()
        try:
            setup_pretrain(data_path=tmp_dir)
            self.assertTrue(os.path.isfile("data/tokenizer.json"))
        finally:
            shutil.rmtree(tmp_dir)


# ======================================================================
# TestSetupFinetune — 微调初始化测试
# ======================================================================
class TestSetupFinetune(unittest.TestCase):
    """验证 setup_finetune 正确初始化微调环境。"""

    def test_setup_finetune_returns_all_components(self) -> None:
        """setup_finetune 应返回 7 个组件。"""
        if not os.path.isfile("data/tokenizer.json"):
            self.skipTest("data/tokenizer.json 不存在，需先运行预训练")

        config, dataloader, model, optimizer, device, start_epoch, best_loss = (
            setup_finetune()
        )
        self.assertIsInstance(config, NovaConfig)
        self.assertIsInstance(model, NovaModel)
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(start_epoch, 0)
        self.assertEqual(best_loss, float("inf"))
        self.assertGreater(config.vocab_size, 0)
        self.assertGreater(len(dataloader), 0)


# ======================================================================
# TestPretrainTrain — 预训练训练循环测试
# ======================================================================
class TestPretrainTrain(unittest.TestCase):
    """验证使用 PretrainDataset 进行训练的核心功能。"""

    def _make_mini_pretrain_env(self, epochs=10):
        """构建极小的预训练环境用于快速测试。"""
        texts = [
            "计算机是现代科技的重要发明",
            "人工智能正在改变我们的生活",
            "深度学习需要训练数据",
        ]
        tokenizer = NovaTokenizer()
        tokenizer.train_from_texts(texts, vocab_size=200)

        config = NovaConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=32, n_heads=2, n_layers=1, d_ff=64,
            max_seq_len=32, batch_size=2, epochs=epochs,
            learning_rate=1e-3, warmup_steps=2, dropout=0.0,
        )
        dataset = PretrainDataset(texts, tokenizer, config.max_seq_len)
        dataloader = create_dataloader(dataset, batch_size=config.batch_size)
        device = torch.device("cpu")
        model = NovaModel(config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        return config, dataloader, model, optimizer, device, tokenizer

    def test_pretrain_loss_decreases(self) -> None:
        """预训练若干 epoch 后 loss 应下降。"""
        config, dataloader, model, optimizer, device, _ = (
            self._make_mini_pretrain_env(epochs=30)
        )

        model.eval()
        with torch.no_grad():
            batch = next(iter(dataloader))
            logits = model(batch["input_ids"])
            initial_loss = F.cross_entropy(
                logits.view(-1, config.vocab_size),
                batch["target_ids"].view(-1),
                ignore_index=-100,
            ).item()

        tmp_dir = tempfile.mkdtemp()
        try:
            result = train(config, dataloader, model, optimizer, device,
                           checkpoint_dir=tmp_dir)
            self.assertLess(result["final_loss"], initial_loss)
        finally:
            shutil.rmtree(tmp_dir)

    def test_pretrain_checkpoint_saved(self) -> None:
        """预训练应保存 best_model.pt。"""
        config, dataloader, model, optimizer, device, _ = (
            self._make_mini_pretrain_env(epochs=5)
        )
        tmp_dir = tempfile.mkdtemp()
        try:
            train(config, dataloader, model, optimizer, device,
                  checkpoint_dir=tmp_dir)
            self.assertTrue(os.path.isfile(os.path.join(tmp_dir, "best_model.pt")))
        finally:
            shutil.rmtree(tmp_dir)

    def test_pretrain_then_finetune(self) -> None:
        """端到端: 预训练 → 保存 checkpoint → 加载 → 微调，loss 应下降。"""
        # ── 预训练 ──
        config, pt_dataloader, model, optimizer, device, tokenizer = (
            self._make_mini_pretrain_env(epochs=20)
        )
        tmp_dir = tempfile.mkdtemp()
        try:
            train(config, pt_dataloader, model, optimizer, device,
                  checkpoint_dir=tmp_dir)

            # ── 构建微调环境 ──
            qa_pairs = [
                {"question": "你好", "answer": "世界"},
                {"question": "计算机", "answer": "科技发明"},
            ]
            ft_dataset = NovaDataset(qa_pairs, tokenizer, config.max_seq_len)
            ft_dataloader = create_dataloader(ft_dataset, batch_size=2)

            # ── 加载预训练 checkpoint ──
            model2 = NovaModel(config).to(device)
            optimizer2 = torch.optim.AdamW(
                model2.parameters(), lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
            ckpt_path = os.path.join(tmp_dir, "best_model.pt")
            load_checkpoint(ckpt_path, model2, optimizer2, device)

            # ── 微调 ──
            ft_config = NovaConfig(
                vocab_size=config.vocab_size,
                d_model=32, n_heads=2, n_layers=1, d_ff=64,
                max_seq_len=32, batch_size=2, epochs=30,
                learning_rate=1e-3, warmup_steps=2, dropout=0.0,
            )

            # 微调前 loss
            model2.eval()
            with torch.no_grad():
                batch = next(iter(ft_dataloader))
                logits = model2(batch["input_ids"])
                loss_before = F.cross_entropy(
                    logits.view(-1, config.vocab_size),
                    batch["target_ids"].view(-1),
                    ignore_index=-100,
                ).item()

            ft_result = train(
                ft_config, ft_dataloader, model2, optimizer2, device,
                start_epoch=0, best_loss=float("inf"),
                checkpoint_dir=tmp_dir,
            )
            self.assertLess(ft_result["final_loss"], loss_before)
        finally:
            shutil.rmtree(tmp_dir)

    def test_pretrain_no_nan_loss(self) -> None:
        """预训练过程中不应出现 NaN loss。"""
        config, dataloader, model, optimizer, device, _ = (
            self._make_mini_pretrain_env(epochs=5)
        )
        tmp_dir = tempfile.mkdtemp()
        try:
            result = train(config, dataloader, model, optimizer, device,
                           checkpoint_dir=tmp_dir)
            self.assertFalse(math.isnan(result["final_loss"]))
            self.assertFalse(math.isinf(result["final_loss"]))
        finally:
            shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    unittest.main()
