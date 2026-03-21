"""Nova 训练流程（支持预训练 + 微调两阶段）

把前面所有组件串起来，完成"从原始文本到训练好的模型"的完整过程。
这是整个项目的"总调度中心"——分词器、数据集、模型、优化器都在这里汇合。

┌──────────────────────────────────────────────────────────────────────┐
│                       两阶段训练流程                                    │
│                                                                      │
│  阶段 1: 预训练（学语言）                                              │
│  ────────────────────────                                            │
│  运行: .venv/bin/python train.py --mode pretrain                     │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │ ① 加载预训练数据 (data/pretrain/*.jsonl)                       │   │
│  │ ② 训练 BPE 分词器 → 保存 data/tokenizer.json                  │   │
│  │ ③ 创建 PretrainDataset + DataLoader                           │   │
│  │ ④ 创建模型（随机初始化）                                       │   │
│  │ ⑤ 创建优化器                                                   │   │
│  │ ⑥ 训练循环 → 保存 checkpoint                                   │   │
│  │                                                                │   │
│  │ 产出: data/tokenizer.json + checkpoints/best_model.pt          │   │
│  └───────────────────────────────────────────────────────────────┘   │
│                           │                                          │
│                           ▼                                          │
│  阶段 2: 微调（学对话）                                              │
│  ────────────────────────                                            │
│  运行: .venv/bin/python train.py --mode finetune                     │
│        --resume checkpoints/best_model.pt                            │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │ ① 加载微调数据 (data/sft/*.jsonl)                              │   │
│  │ ② 加载已有 BPE 分词器 (data/tokenizer.json)                    │   │
│  │ ③ 创建 NovaDataset + DataLoader                               │   │
│  │ ④ 创建模型                                                     │   │
│  │ ⑤ 创建优化器                                                   │   │
│  │ ⑥ 加载预训练 checkpoint（模型权重 + 优化器状态）                 │   │
│  │ ⑦ 训练循环 → 保存微调后的 checkpoint                            │   │
│  │                                                                │   │
│  │ 产出: checkpoints/best_model.pt（可用于推理）                   │   │
│  └───────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NovaConfig
from tokenizer import NovaTokenizer
from dataset import (
    NovaDataset,
    PretrainDataset,
    create_dataloader,
    load_qa_pairs,
    load_pretrain_data,
)
from model import NovaModel


# ======================================================================
# 学习率调度器（Warmup + Cosine Decay）
# ======================================================================
def get_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float = 1e-6,
) -> float:
    """计算当前 step 的学习率。

    整个学习率曲线分两个阶段:

      学习率
      ▲
      │        ╭─────╮
      │       ╱       ╲        Cosine Decay 阶段
      │      ╱         ╲       （从 max_lr 平滑下降到 min_lr）
      │     ╱           ╲
      │    ╱             ╲
      │   ╱               ╲
      │  ╱                 ╲
      │ ╱                   ╲──── min_lr
      │╱                    │
      ├─────┬───────────────┤──→ step
      0   warmup          max_steps

    阶段 1: Warmup（热身，step 0 → warmup_steps）
      学习率从 0 线性增长到 max_lr。

    阶段 2: Cosine Decay（余弦衰减，warmup_steps → max_steps）
      学习率按余弦曲线从 max_lr 平滑下降到 min_lr。

    参数:
      step         — 当前训练步数（epoch 编号）
      warmup_steps — 热身阶段的步数
      max_steps    — 总训练步数
      max_lr       — 最大学习率
      min_lr       — 最小学习率（默认 1e-6）

    返回:
      当前 step 应使用的学习率（float）

    调用时机:
      在训练循环中，每个 epoch 开始时调用一次:
        lr = get_lr(epoch, config.warmup_steps, config.epochs, config.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    """
    if step < warmup_steps:
        return max_lr * step / warmup_steps

    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# ======================================================================
# 训练日志格式化
# ======================================================================
def format_train_log(
    epoch: int,
    total_epochs: int,
    loss: float,
    lr: float,
    epoch_time: float,
) -> str:
    """将一个 epoch 的训练指标格式化为一行日志字符串。

    示例输出:
      [Epoch 010/500] loss=3.2456 lr=2.80e-04 time=1.2s

    参数:
      epoch        — 当前 epoch 编号（从 1 开始计数）
      total_epochs — 训练总轮数
      loss         — 本轮的平均损失值
      lr           — 本轮使用的学习率
      epoch_time   — 本轮耗时（秒）

    返回:
      格式化后的日志字符串
    """
    return (
        f"[Epoch {epoch:03d}/{total_epochs}] "
        f"loss={loss:.4f} lr={lr:.2e} time={epoch_time:.1f}s"
    )


def format_train_summary(
    final_loss: float,
    best_loss: float,
    best_epoch: int,
    total_time: float,
    best_model_path: str,
) -> str:
    """将训练结束后的汇总信息格式化为多行字符串。

    示例输出:
      ============================================================
        训练完成
      ============================================================
        最终 loss:      0.0823
        最佳 loss:      0.0756 (epoch 487)
        总训练时间:     142.3s (2.4min)
        最佳模型路径:   checkpoints/best_model.pt
      ============================================================

    参数:
      final_loss      — 最后一个 epoch 的平均 loss
      best_loss       — 训练全程中的最低 loss
      best_epoch      — best_loss 对应的 epoch 编号
      total_time      — 训练总耗时（秒）
      best_model_path — best_model.pt 的文件路径

    返回:
      格式化后的多行汇总字符串
    """
    separator = "=" * 60
    return (
        f"\n{separator}\n"
        f"  训练完成\n"
        f"{separator}\n"
        f"  最终 loss:      {final_loss:.4f}\n"
        f"  最佳 loss:      {best_loss:.4f} (epoch {best_epoch})\n"
        f"  总训练时间:     {total_time:.1f}s ({total_time / 60:.1f}min)\n"
        f"  最佳模型路径:   {best_model_path}\n"
        f"{separator}"
    )


def should_log(epoch: int, start_epoch: int, log_interval: int = 10) -> bool:
    """判断当前 epoch 是否需要打印训练日志。

    打印规则:
      1. (epoch + 1) 是 log_interval 的倍数 → 每隔 log_interval 轮打印一次
      2. epoch == start_epoch              → 训练的第一轮一定打印

    参数:
      epoch        — 当前 epoch 编号（从 0 开始）
      start_epoch  — 训练起始 epoch
      log_interval — 打印间隔（默认 10）

    返回:
      True 表示本轮需要打印日志
    """
    return (epoch + 1) % log_interval == 0 or epoch == start_epoch


# ======================================================================
# 主训练函数
#
# 预训练和微调共用同一个训练循环，区别只在于：
#   - 使用的 Dataset 不同（PretrainDataset vs NovaDataset）
#   - 由 setup_pretrain / setup_finetune 分别准备好 dataloader
#   - train() 本身不关心数据从哪里来，只管迭代 dataloader 训练
# ======================================================================
def train(
    config: NovaConfig,
    dataloader: torch.utils.data.DataLoader,
    model: NovaModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    start_epoch: int = 0,
    best_loss: float = float("inf"),
    checkpoint_dir: str = "checkpoints",
) -> dict:
    """主训练函数：执行完整的训练循环。

    预训练和微调阶段都使用此函数，只是传入的 dataloader 内容不同。

    参数:
      config         — NovaConfig，包含所有超参数
      dataloader     — 训练数据的 DataLoader
      model          — NovaModel 实例（已移到 device 上）
      optimizer      — AdamW 优化器实例
      device         — 训练设备（cpu / cuda / mps）
      start_epoch    — 起始 epoch（断点续训时 > 0）
      best_loss      — 当前最佳 loss（断点续训时从 checkpoint 加载）
      checkpoint_dir — checkpoint 保存目录

    返回:
      dict: {"final_loss", "best_loss", "best_epoch", "total_time", "best_model_path"}
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    train_start_time = time.time()
    best_epoch = start_epoch

    # AMP: 仅在 CUDA 上启用混合精度训练
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_status = "ON (FP16)" if use_amp else "OFF (CPU/MPS 不支持)"

    print(
        f"\n开始训练: epoch {start_epoch} → {config.epochs}, "
        f"device={device}, batch_size={config.batch_size}, AMP={amp_status}"
    )
    print(f"DataLoader 每轮 {len(dataloader)} 个 batch\n")

    if start_epoch >= config.epochs:
        raise ValueError(
            f"start_epoch({start_epoch}) >= config.epochs({config.epochs})，"
            f"没有可训练的轮次。请检查 config.epochs 设置或 checkpoint 的 epoch 值。"
        )

    total_batches = len(dataloader)
    log_every_n_batches = max(1, min(50, total_batches // 20))

    for epoch in range(start_epoch, config.epochs):
        epoch_start_time = time.time()
        model.train()

        lr = get_lr(epoch, config.warmup_steps, config.epochs, config.learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        total_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, config.vocab_size),
                    target_ids.view(-1),
                    ignore_index=-100,
                )

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config.grad_clip
            )
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            if (batch_idx + 1) % log_every_n_batches == 0 or batch_idx == 0:
                elapsed = time.time() - epoch_start_time
                batches_done = batch_idx + 1
                speed = batches_done / elapsed
                remaining_batches = total_batches - batches_done
                remaining_epoch = remaining_batches / speed if speed > 0 else 0
                remaining_epochs_after = (config.epochs - epoch - 1) * (total_batches / speed) if speed > 0 else 0
                eta_total = remaining_epoch + remaining_epochs_after

                running_loss = total_loss / batches_done

                eta_h, eta_m = divmod(int(eta_total), 3600)
                eta_m = eta_m // 60
                pct = batches_done / total_batches * 100

                print(
                    f"  [Epoch {epoch + 1}/{config.epochs}] "
                    f"batch {batches_done:>6}/{total_batches} ({pct:5.1f}%) | "
                    f"loss={running_loss:.4f} lr={lr:.2e} | "
                    f"{speed:.1f} batch/s | "
                    f"ETA: {eta_h}h{eta_m:02d}m",
                    flush=True,
                )

        avg_loss = total_loss / total_batches
        epoch_time = time.time() - epoch_start_time

        # ── 训练日志 ──
        if should_log(epoch, start_epoch):
            print(format_train_log(epoch + 1, config.epochs, avg_loss, lr, epoch_time))

        # ── Checkpoint 保存 ──
        if (epoch + 1) % 50 == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch + 1}.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                    "config": config,
                },
                ckpt_path,
            )
            print(f"  → checkpoint 已保存: {ckpt_path}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1
            best_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                    "config": config,
                },
                best_path,
            )

    # ── 训练汇总 ──
    total_time = time.time() - train_start_time
    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")

    print(
        format_train_summary(
            avg_loss, best_loss, best_epoch, total_time, best_model_path
        )
    )

    return {
        "final_loss": avg_loss,
        "best_loss": best_loss,
        "best_epoch": best_epoch,
        "total_time": total_time,
        "best_model_path": best_model_path,
    }


# ======================================================================
# 断点续训 — checkpoint 加载
# ======================================================================
CHECKPOINT_REQUIRED_KEYS = {
    "epoch",
    "model_state_dict",
    "optimizer_state_dict",
    "loss",
    "config",
}


def load_checkpoint(
    path: str,
    model: NovaModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, float]:
    """从 checkpoint 文件恢复模型和优化器的完整状态。

    参数:
      path      — checkpoint 文件路径
      model     — NovaModel 实例（参数会被覆盖）
      optimizer — 优化器实例（状态会被覆盖）
      device    — 目标设备

    返回:
      (start_epoch, best_loss) 元组

    异常:
      FileNotFoundError — checkpoint 文件不存在
      KeyError          — checkpoint 缺少必要字段
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"checkpoint 文件不存在: {path}")

    print(f"\n加载 checkpoint: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    missing_keys = CHECKPOINT_REQUIRED_KEYS - set(checkpoint.keys())
    if missing_keys:
        raise KeyError(
            f"checkpoint 缺少必要字段: {missing_keys}。"
            f"已有字段: {set(checkpoint.keys())}"
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint["epoch"]
    best_loss = checkpoint["loss"]

    print(f"  → 从 epoch {start_epoch} 继续训练，loss={best_loss:.4f}")

    return start_epoch, best_loss


# ======================================================================
# 初始化函数：setup_pretrain —— 预训练阶段初始化
#
# 完整调用链:
#   main()
#     → setup_pretrain(data_path, resume_path)
#       → ① load_pretrain_data()    加载 JSONL 纯文本
#       → ② NovaTokenizer.train_from_texts()  训练 BPE 分词器
#       → ③ PretrainDataset()       构建预训练数据集
#       → ④ NovaModel()             创建模型
#       → ⑤ AdamW()                 创建优化器
#       → ⑥ load_checkpoint()       (可选) 断点续训
#     → train()                     执行训练循环
# ======================================================================
def setup_pretrain(
    data_path: str = "data/pretrain",
    resume_path: str | None = None,
) -> tuple[
    NovaConfig,
    torch.utils.data.DataLoader,
    NovaModel,
    torch.optim.Optimizer,
    torch.device,
    int,
    float,
]:
    """预训练阶段初始化：加载纯文本数据 → 训练 BPE → 构建数据集 → 创建模型。

    与 setup_finetune 的区别:
      - 使用 load_pretrain_data() 而非 load_qa_pairs()
      - 在预训练数据上训练 BPE 分词器（微调直接加载已有的）
      - 使用 PretrainDataset 而非 NovaDataset
      - 模型默认随机初始化（可通过 --resume 断点续训）

    参数:
      data_path   — 预训练数据路径（目录或单个 .jsonl 文件）
      resume_path — 断点续训的 checkpoint 路径（可选）

    返回:
      (config, dataloader, model, optimizer, device, start_epoch, best_loss)
    """
    # ── ① 加载预训练数据 ──
    print(f"[1/5] 加载预训练数据: {data_path}")
    texts = load_pretrain_data(data_path)
    print(f"       共 {len(texts)} 条文本")

    PRETRAIN_TOKEN_BUDGET = 50_000_000
    AVG_TOKENS_PER_TEXT = 500
    max_texts = PRETRAIN_TOKEN_BUDGET // AVG_TOKENS_PER_TEXT
    if len(texts) > max_texts:
        print(f"       token 预算 {PRETRAIN_TOKEN_BUDGET // 1_000_000}M → 采样 {max_texts} 条（从 {len(texts)} 条中）")
        texts = random.sample(texts, max_texts)

    # ── ② 训练 BPE 分词器（已有则跳过） ──
    tokenizer = NovaTokenizer()
    tokenizer_path = "data/tokenizer.json"
    if os.path.isfile(tokenizer_path):
        print(f"[2/5] 检测到已有分词器，直接加载: {tokenizer_path}")
        tokenizer.load(tokenizer_path)
        print(f"       词表大小: {tokenizer.vocab_size}（跳过 BPE 训练）")
    else:
        BPE_SAMPLE_SIZE = 200_000
        bpe_sample = min(len(texts), BPE_SAMPLE_SIZE)
        if len(texts) > BPE_SAMPLE_SIZE:
            print(f"[2/5] 训练 BPE 分词器（从 {len(texts)} 条中采样 {bpe_sample} 条）...")
            bpe_texts = random.sample(texts, BPE_SAMPLE_SIZE)
        else:
            print("[2/5] 训练 BPE 分词器...")
            bpe_texts = texts
        tokenizer.train_from_texts(bpe_texts)
        tokenizer.save(tokenizer_path)
        print(f"       词表大小: {tokenizer.vocab_size}，已保存到 {tokenizer_path}")

    # ── ③ 创建数据集和 DataLoader ──
    print("[3/5] 创建预训练数据集和 DataLoader...")
    config = NovaConfig(vocab_size=tokenizer.vocab_size)
    config.max_seq_len = config.pretrain_max_seq_len
    config.batch_size = config.pretrain_batch_size
    config.learning_rate = config.pretrain_lr
    dataset = PretrainDataset(texts, tokenizer, config.max_seq_len)
    del texts
    dataloader = create_dataloader(dataset, batch_size=config.batch_size)
    print(f"       数据集大小: {len(dataset)} 条")
    print(f"       每轮 batch 数: {len(dataloader)} (batch_size={config.batch_size})")

    # ── ④ 创建模型 ──
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("======基于MPS训练...")
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"[4/5] 创建模型 (device={device})...")
    model = NovaModel(config).to(device)
    model.print_parameter_summary()

    # ── ⑤ 创建优化器 ──
    print("[5/5] 创建 AdamW 优化器...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    print(f"       lr={config.learning_rate}, weight_decay={config.weight_decay}")

    # ── ⑥ （可选）断点续训 ──
    start_epoch = 0
    best_loss = float("inf")
    if resume_path is not None:
        start_epoch, best_loss = load_checkpoint(
            path=resume_path, model=model, optimizer=optimizer, device=device,
        )

    return config, dataloader, model, optimizer, device, start_epoch, best_loss


# ======================================================================
# 初始化函数：setup_finetune —— 微调阶段初始化
#
# 完整调用链:
#   main()
#     → setup_finetune(data_path, tokenizer_path, resume_path)
#       → ① load_qa_pairs()          加载 QA 问答对
#       → ② NovaTokenizer.load()     加载预训练阶段保存的 BPE 分词器
#       → ③ NovaDataset()            构建微调数据集
#       → ④ NovaModel()              创建模型
#       → ⑤ AdamW()                  创建优化器
#       → ⑥ load_checkpoint()        加载预训练 checkpoint 权重
#     → train()                      执行微调训练循环
# ======================================================================
def setup_finetune(
    data_path: str = "data/sft/",
    tokenizer_path: str = "data/tokenizer.json",
    resume_path: str | None = None,
) -> tuple[
    NovaConfig,
    torch.utils.data.DataLoader,
    NovaModel,
    torch.optim.Optimizer,
    torch.device,
    int,
    float,
]:
    """微调阶段初始化：加载 QA 数据 → 加载 BPE 分词器 → 构建数据集 → 加载预训练权重。

    与 setup_pretrain 的区别:
      - 使用 load_qa_pairs() 加载 QA 数据
      - 直接加载已有的 BPE 分词器（预训练阶段生成的 tokenizer.json）
      - 使用 NovaDataset 而非 PretrainDataset
      - 通过 --resume 加载预训练 checkpoint 权重作为起点

    参数:
      data_path      — QA 数据 JSONL 文件或目录路径
      tokenizer_path — BPE 分词器文件路径（预训练阶段保存的）
      resume_path    — 预训练 checkpoint 路径（用于加载预训练权重）

    返回:
      (config, dataloader, model, optimizer, device, start_epoch, best_loss)
    """
    # ── ① 加载微调数据 ──
    print(f"[1/5] 加载微调数据: {data_path}")
    qa_pairs = load_qa_pairs(data_path)
    print(f"       共 {len(qa_pairs)} 条问答对")

    # 限制 SFT token 总量（默认 1 亿），按 ~500 tokens/对估算采样数
    SFT_TOKEN_BUDGET = 100_000_000
    AVG_TOKENS_PER_QA = 500
    max_pairs = SFT_TOKEN_BUDGET // AVG_TOKENS_PER_QA
    if len(qa_pairs) > max_pairs:
        print(f"       token 预算 {SFT_TOKEN_BUDGET // 1_000_000}M → 采样 {max_pairs} 条（从 {len(qa_pairs)} 条中）")
        qa_pairs = random.sample(qa_pairs, max_pairs)

    # ── ② 加载已有 BPE 分词器 ──
    print(f"[2/5] 加载 BPE 分词器: {tokenizer_path}")
    tokenizer = NovaTokenizer()
    tokenizer.load(tokenizer_path)
    print(f"       词表大小: {tokenizer.vocab_size}")

    # ── ③ 创建数据集和 DataLoader ──
    print("[3/5] 创建微调数据集和 DataLoader...")
    config = NovaConfig(vocab_size=tokenizer.vocab_size)
    config.max_seq_len = config.finetune_max_seq_len
    config.batch_size = config.finetune_batch_size
    config.learning_rate = config.finetune_lr
    dataset = NovaDataset(qa_pairs, tokenizer, config.max_seq_len)
    dataloader = create_dataloader(dataset, batch_size=config.batch_size)
    print(f"       数据集大小: {len(dataset)} 条")
    print(f"       每轮 batch 数: {len(dataloader)} (batch_size={config.batch_size})")

    # ── ④ 创建模型 ──
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("======基于MPS训练...")
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"[4/5] 创建模型 (device={device})...")
    model = NovaModel(config).to(device)
    model.print_parameter_summary()

    # ── ⑤ 创建优化器 ──
    print("[5/5] 创建 AdamW 优化器...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    print(f"       lr={config.learning_rate}, weight_decay={config.weight_decay}")

    # ── ⑥ 加载预训练 checkpoint ──
    #
    # 微调加载预训练权重时，只要模型参数（语言知识），不要 start_epoch。
    # 因为微调是一个全新的训练阶段，应该从 epoch 0 开始，
    # 而不是从预训练的 epoch 500 继续（那样 range(500, 1) 就是空循环）。
    #
    # 优化器状态也重置（预训练的动量/学习率对微调数据不适用）。
    start_epoch = 0
    best_loss = float("inf")
    if resume_path is not None:
        # 只加载模型权重，不恢复优化器状态和 start_epoch
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"checkpoint 文件不存在: {resume_path}")
        print(f"\n加载预训练权重: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        state_dict = checkpoint["model_state_dict"]
        cur_seq_len = config.max_seq_len
        ckpt_seq_len = state_dict["pos_emb.weight"].shape[0]
        if ckpt_seq_len != cur_seq_len:
            print(f"  → 位置编码调整: {ckpt_seq_len} → {cur_seq_len}")
            old_weight = state_dict["pos_emb.weight"]
            if ckpt_seq_len > cur_seq_len:
                state_dict["pos_emb.weight"] = old_weight[:cur_seq_len]
            else:
                new_weight = torch.zeros(cur_seq_len, old_weight.shape[1])
                nn.init.normal_(new_weight, mean=0.0, std=0.02)
                new_weight[:ckpt_seq_len] = old_weight
                state_dict["pos_emb.weight"] = new_weight
        model.load_state_dict(state_dict)
        print(f"  → 已加载预训练权重，微调从 epoch 0 开始")

    return config, dataloader, model, optimizer, device, start_epoch, best_loss


# ======================================================================
# 向后兼容：保留旧版 setup() 函数
#
# 旧代码（测试中）直接调用 setup()，等价于微调模式但会重新训练 BPE。
# 保留此函数避免破坏已有测试。
# ======================================================================
def setup(
    resume_path: str | None = None,
) -> tuple[
    NovaConfig,
    torch.utils.data.DataLoader,
    NovaModel,
    torch.optim.Optimizer,
    torch.device,
    int,
    float,
]:
    """向后兼容的初始化函数（等价于用 QA 数据从头训练）。

    行为与旧版完全一致:
      1. 加载 QA JSONL 数据
      2. 在 QA 数据上训练 BPE 分词器
      3. 创建 NovaDataset
      4. 创建模型和优化器
      5. (可选) 加载 checkpoint

    此函数主要供已有测试使用，新代码请使用 setup_pretrain / setup_finetune。
    """
    # ── ① 加载训练数据 ──
    data_path = "data/sft/"
    print(f"[1/5] 加载训练数据: {data_path}")
    qa_pairs = load_qa_pairs(data_path)
    print(f"       共 {len(qa_pairs)} 条问答对")

    # ── ② 构建 BPE 分词器 ──
    print("[2/5] 构建 BPE 分词器...")
    tokenizer = NovaTokenizer()
    all_texts = []
    for pair in qa_pairs:
        all_texts.append(pair["question"])
        all_texts.append(pair["answer"])
    tokenizer.build_vocab(all_texts)
    tokenizer.save("data/tokenizer.json")
    print(f"       词表大小: {tokenizer.vocab_size}，已保存到 data/tokenizer.json")

    # ── ③ 创建数据集和 DataLoader ──
    print("[3/5] 创建数据集和 DataLoader...")
    config = NovaConfig(vocab_size=tokenizer.vocab_size)
    config.max_seq_len = config.finetune_max_seq_len
    config.batch_size = config.finetune_batch_size
    config.learning_rate = config.finetune_lr
    dataset = NovaDataset(qa_pairs, tokenizer, config.max_seq_len)
    dataloader = create_dataloader(dataset, batch_size=config.batch_size)
    print(f"       数据集大小: {len(dataset)} 条")
    print(f"       每轮 batch 数: {len(dataloader)} (batch_size={config.batch_size})")

    # ── ④ 创建模型 ──
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("======基于MPS训练...")
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"[4/5] 创建模型 (device={device})...")
    model = NovaModel(config).to(device)
    model.print_parameter_summary()

    # ── ⑤ 创建优化器 ──
    print("[5/5] 创建 AdamW 优化器...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    print(f"       lr={config.learning_rate}, weight_decay={config.weight_decay}")

    # ── ⑥ （可选）断点续训 ──
    start_epoch = 0
    best_loss = float("inf")
    if resume_path is not None:
        start_epoch, best_loss = load_checkpoint(
            path=resume_path, model=model, optimizer=optimizer, device=device,
        )

    return config, dataloader, model, optimizer, device, start_epoch, best_loss


# ======================================================================
# 命令行入口
#
# 用法:
#   # 预训练（从纯文本学语言，默认 3 轮）
#   .venv/bin/python train.py --mode pretrain
#   .venv/bin/python train.py --mode pretrain --data data/pretrain/
#   .venv/bin/python train.py --mode pretrain --epochs 5
#
#   # 微调（从 QA 学对话，加载预训练权重，默认 500 轮）
#   .venv/bin/python train.py --mode finetune --resume checkpoints/best_model.pt
#   .venv/bin/python train.py --mode finetune --data data/sft/ --epochs 300
#
#   # 向后兼容（等价于旧版行为：用 QA 数据从头训练，默认 500 轮）
#   .venv/bin/python train.py
#   .venv/bin/python train.py --resume checkpoints/epoch_50.pt
# ======================================================================
def main() -> None:
    """命令行入口：支持预训练 / 微调 / 向后兼容三种模式。"""
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")
        torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser(description="Nova 模型训练")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["pretrain", "finetune"],
        default=None,
        help="训练模式: pretrain=预训练, finetune=微调（不指定则使用旧版兼容模式）",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="checkpoint 路径，用于断点续训或加载预训练权重",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="数据路径（预训练: JSONL 目录/文件，微调: JSON 文件）",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="训练轮数（预训练默认 3，微调默认 500，向后兼容默认 500）",
    )
    args = parser.parse_args()

    if args.mode == "pretrain":
        # ── 预训练模式 ──
        data_path = args.data or "data/pretrain"
        config, dataloader, model, optimizer, device, start_epoch, best_loss = (
            setup_pretrain(data_path=data_path, resume_path=args.resume)
        )
        config.epochs = args.epochs or config.pretrain_epochs
        config.warmup_steps = min(config.warmup_steps, config.epochs // 2)
    elif args.mode == "finetune":
        # ── 微调模式 ──
        data_path = args.data or "data/sft/"
        config, dataloader, model, optimizer, device, start_epoch, best_loss = (
            setup_finetune(data_path=data_path, resume_path=args.resume)
        )
        config.epochs = args.epochs or config.finetune_epochs
        config.warmup_steps = 0
    else:
        # ── 向后兼容模式（不指定 --mode） ──
        config, dataloader, model, optimizer, device, start_epoch, best_loss = (
            setup(resume_path=args.resume)
        )
        config.epochs = args.epochs or config.finetune_epochs

    print(f"\n训练轮数: {config.epochs}，warmup: {config.warmup_steps}")
    print(f"batch_size: {config.batch_size}，max_seq_len: {config.max_seq_len}，lr: {config.learning_rate}")
    train(
        config=config,
        dataloader=dataloader,
        model=model,
        optimizer=optimizer,
        device=device,
        start_epoch=start_epoch,
        best_loss=best_loss,
    )



if __name__ == "__main__":
    main()
