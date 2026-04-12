"""Nova 训练流程（支持预训练 + 微调两阶段）

用法:
   # 预训练
   .venv/bin/python train.py --mode pretrain
   .venv/bin/python train.py --mode pretrain --data data/pretrain/

   # SFT微调
   .venv/bin/python train.py --mode finetune --resume checkpoints/best_model.pt
   .venv/bin/python train.py --mode finetune --data data/sft/
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time

import torch
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
from model import NovaModel, precompute_rope_freqs


# ======================================================================
# 动态计算学习率LR值
# ======================================================================
def get_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    # 超参中的LR参数值
    max_lr: float,
    min_lr: float = 1e-6,
) -> float:
    # 1、预热阶段：epoch < warmup_steps，学习率从 0 线性增长到 max_lr
    if step < warmup_steps:
        return max_lr * step / warmup_steps

    # 2、余弦衰减阶段：epoch >= warmup_steps，lr 从 max_lr 平滑下降到 min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# ======================================================================
# 训练日志格式化，快速浏览即可
# ======================================================================
def format_train_log(
    epoch: int,
    total_epochs: int,
    loss: float,
    lr: float,
    epoch_time: float,
) -> str:
    return (
        f"[Epoch {epoch:03d}/{total_epochs}] "
        f"loss={loss:.4f} lr={lr:.2e} time={epoch_time:.1f}s"
    )


# 训练结束摘要格式化
# 示例输出:
#   ============================================================
#     训练完成
#   ============================================================
#     最终 loss:      0.0823
#     最佳 loss:      0.0756 (epoch 487)
#     总训练时间:     142.3s (2.4min)
#     最佳模型路径:   checkpoints/best_model.pt
#   ============================================================
def format_train_summary(
    final_loss: float,
    best_loss: float,
    best_epoch: int,
    total_time: float,
    best_model_path: str,
) -> str:
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


# 判断是否打印日志
def should_log(epoch: int, start_epoch: int, log_interval: int = 10) -> bool:
    return (epoch + 1) % log_interval == 0 or epoch == start_epoch


# ======================================================================
# 核心训练循环
# ======================================================================
def train(
    config: NovaConfig,
    dataloader: torch.utils.data.DataLoader,
    model: NovaModel,
    # 优化器AdamW
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    start_epoch: int = 0,
    best_loss: float = float("inf"),
    checkpoint_dir: str = "checkpoints",
) -> dict:
    # 创建 checkpoints/ 目录（如果已存在就跳过），用来存训练过程中保存的.pt文件
    os.makedirs(checkpoint_dir, exist_ok=True)
    # 记录训练开始的时间戳，训练结束后用来算总耗时
    train_start_time = time.time()
    # 训练过程中如果某一轮的 loss 创了新低，就会更新这个值，最终打印在训练摘要里
    best_epoch = start_epoch

    # 前向传播中仅在 CUDA 上启用混合精度训练
    use_amp = device.type == "cuda"
    # GradScaler 是混合精度训练的 梯度缩放器
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

    # 记录一轮（epoch）有多少个 batch
    total_batches = len(dataloader)
    # 每隔多少个 batch 打印一次进度
    log_every_n_batches = max(1, min(50, total_batches // 20))

    #  epoch 外层循环，遍历每一轮训练
    for epoch in range(start_epoch, config.epochs):
        epoch_start_time = time.time()  # 记录每一轮的训练开始时间
        # 设置模型为训练模式
        # pytorch有2种模式，分别是 训练模式 和 评估(推理)模式
        # 不同的模式计算行为不同，比如Dropout在训练模式下会丢 向量维度 而评估模式不会
        model.train()

        # 动态调整LR
        # warmup_steps是学习率预热步数
        # 整个学习率曲线分两个阶段:
        #   学习率
        #   ▲
        #   │        ╭─────╮
        #   │       ╱       ╲        余弦衰减阶段
        #   │      ╱         ╲       （从 max_lr 平滑下降到 min_lr）
        #   │     ╱           ╲
        #   │    ╱             ╲
        #   │   ╱               ╲
        #   │  ╱                 ╲
        #   │ ╱                   ╲──── min_lr
        #   │╱                    │
        #   ├─────┬───────────────┤──→ step
        #   0   warmup          max_steps

        # 1、预热阶段：epoch < warmup_steps，学习率从 0 线性增长到 max_lr
        # 2、余弦衰减阶段：epoch >= warmup_steps，lr 从 max_lr 平滑下降到 min_lr
        lr = get_lr(epoch, config.warmup_steps, config.epochs, config.learning_rate)

        # 把每一轮计算出来的LR更新进优化器
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        total_loss = 0.0
        # 按照batch size遍历DataLoader中的所有tensor进行训练
        for batch_idx, batch in enumerate(dataloader):
            # 取出模型输入tensor和标签(正确答案)tensor
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)

            # CUDA时进行混合精度训练
            with torch.amp.autocast("cuda", enabled=use_amp):
                # 回调模型的forward()函数
                # Token Embedding -> Position Embedding -> Dropout -> TransformerBlock × n_layers 层 -> Final RMSNorm -> Output Linear
                # 得到batch*seq_len个token的vocab_size预测分数
                logits = model(input_ids)

                # loss值是模型的预测结果与实际结果经过交叉熵算出来的偏差分数,取平均得到一个标量loss
                # 交叉熵函数的输入由logits和target_ids构成
                loss = F.cross_entropy(  # 交叉熵计算函数
                    # 把logits[batch, seq_len, vocab_size]拉平成[batch*seq_len, vocab_size]
                    logits.view(-1, config.vocab_size),
                    # 把target_ids[batch, seq_len]拉平成[batch*seq_len]
                    target_ids.view(-1),
                    # 损失函数遇到 -100 就跳过
                    ignore_index=-100,
                )

            # 清空上一个 batch 残留的梯度
            optimizer.zero_grad(set_to_none=True)
            # 反向过程 cross_entropy → output 投影层 → final_norm → TransformerBlock × n_layers（每层里面反向走 FFN → RMSNorm → Attention → RMSNorm）→ Dropout → Embedding
            # pytorch的自动微分引擎负责 反向传播沿着 Block 各子层按相反方向，基于复合函数求导和链式法则，逐层计算影响 loss 值的梯度值
            scaler.scale(loss).backward()
            # 把之前放大的梯度缩回原始大小，只针对CUDA
            scaler.unscale_(optimizer)
            # 如果所有参数的梯度的总范数超过 config.grad_clip（比如 1.0），就等比例缩小所有梯度，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                # 梯度裁剪阈值
                max_norm=config.grad_clip,
            )
            # 执行参数更新。优化器根据梯度和学习率，调整模型的所有权重参数。在 MPS/CPU 设备上等于直接 optimizer.step()
            scaler.step(optimizer)
            # 更新 scaler 内部的缩放因子，为下一个 batch 做准备,这个也是仅针对CUDA设备
            scaler.update()

            total_loss += (
                loss.item()
            )  # 把这个 batch 的 loss 值累加起来，一轮结束后算平均 loss

            # 计算训练进度日志，每隔多少个 batch 打印一次进度，打印当前进度、loss值、学习率、每秒处理的batch数、预计剩余时间
            #   [Epoch 1/500] batch    50/200 ( 25.0%) | loss=3.2456 lr=3.00e-04 | 12.3 batch/s | ETA: 0h42m
            if (batch_idx + 1) % log_every_n_batches == 0 or batch_idx == 0:
                elapsed = time.time() - epoch_start_time
                batches_done = batch_idx + 1
                speed = batches_done / elapsed
                remaining_batches = total_batches - batches_done
                remaining_epoch = remaining_batches / speed if speed > 0 else 0
                remaining_epochs_after = (
                    (config.epochs - epoch - 1) * (total_batches / speed)
                    if speed > 0
                    else 0
                )
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
# 预训练阶段初始化
# 加载语料数据 → 训练 BPE → 构建数据集 → 创建模型 → 创建 AdamW 优化器
# ======================================================================
def setup_pretrain(
    data_path: str = "data/pretrain",
) -> tuple[
    NovaConfig,
    torch.utils.data.DataLoader,
    NovaModel,
    torch.optim.Optimizer,
    torch.device,
    int,
    float,
]:
    # 加载预训练语料包
    print(f"[1/5] 加载预训练数据: {data_path}")
    # 调用dataset.py的load_pretrain_data函数 加载/解析预训练语料包
    texts = load_pretrain_data(data_path)
    print(f"       共 {len(texts)} 条文本")

    # 限制预训练token总量（默认 50M），按 ~500 tokens/条估算采样数
    PRETRAIN_TOKEN_BUDGET = 50_000_000
    AVG_TOKENS_PER_TEXT = 500
    max_texts = PRETRAIN_TOKEN_BUDGET // AVG_TOKENS_PER_TEXT
    if len(texts) > max_texts:
        print(
            f"       token 预算 {PRETRAIN_TOKEN_BUDGET // 1_000_000}M → 采样 {max_texts} 条（从 {len(texts)} 条中）"
        )
        texts = random.sample(texts, max_texts)

    # 训练BPE分词器，如果已有则直接加载
    tokenizer = NovaTokenizer()
    tokenizer_path = "data/tokenizer.json"
    if os.path.isfile(tokenizer_path):
        print(f"[2/5] 检测到已有分词器，直接加载: {tokenizer_path}")
        tokenizer.load(tokenizer_path)  # 加载已经训练好的BPE分词器
        print(f"       词表大小: {tokenizer.vocab_size}")
    else:
        # 文本超过 20 万条 → 随机采样 20 万条来训练 BPE分词
        # 文本不超过 20 万条 → 全部拿来训练BPE分词
        BPE_SAMPLE_SIZE = 200_000
        bpe_sample = min(len(texts), BPE_SAMPLE_SIZE)
        if len(texts) > BPE_SAMPLE_SIZE:
            print(
                f"[2/5] 训练 BPE 分词器（从 {len(texts)} 条中采样 {bpe_sample} 条）..."
            )
            bpe_texts = random.sample(texts, BPE_SAMPLE_SIZE)
        else:
            print("[2/5] 训练 BPE 分词器...")
            bpe_texts = texts
        # 执行BPE分词训练
        tokenizer.train_from_texts(bpe_texts)
        # 保存训练好的BPE分词器
        tokenizer.save(tokenizer_path)
        print(f"       词表大小: {tokenizer.vocab_size}，已保存到 {tokenizer_path}")

    # 创建数据集和 DataLoader
    print("[3/5] 创建预训练数据集和 DataLoader...")
    config = NovaConfig(vocab_size=tokenizer.vocab_size)
    config.max_seq_len = config.pretrain_max_seq_len
    config.batch_size = config.pretrain_batch_size
    config.learning_rate = config.pretrain_lr

    # 预训练阶段将语料批量预编码、截断、填充、生成模型tensor输入和标签tensor输入
    dataset = PretrainDataset(texts, tokenizer, config.max_seq_len)
    del texts
    # 创建DataLoader(数据加载迭代器)
    # 编码结束后将Dataset包装好训练时取出Tensor进行训练
    dataloader = create_dataloader(dataset, batch_size=config.batch_size)
    print(f"       数据集大小: {len(dataset)} 条")
    print(f"       每轮 batch 数: {len(dataloader)} (batch_size={config.batch_size})")

    # 选择计算设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"[4/5] 创建模型 (device={device})...")

    # 初始化Decoder-Only Transformer 模型，并初始化模型的各个参数
    model = NovaModel(config).to(device)
    model.print_parameter_summary()  # 输出模型的参数统计

    # 创建AdamW优化器
    print("[5/5] 创建 AdamW 优化器...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        # LR学习率
        lr=config.learning_rate,
        # 重衰减系数
        weight_decay=config.weight_decay,
    )
    print(f"       lr={config.learning_rate}, weight_decay={config.weight_decay}")

    return config, dataloader, model, optimizer, device, 0, float("inf")


# ======================================================================
# SFT初始化动作
# 加载语料数据 → 加载BPE分词器 → 构建数据集 → 模型初始化 → 创建 AdamW 优化器 → 加载预训练权重
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
    # 加载微调数据 ──
    print(f"[1/5] 加载微调数据: {data_path}")
    # 调用dataset.py的load_qa_pairs函数 加载/解析微调语料包
    qa_pairs = load_qa_pairs(data_path)
    print(f"       共 {len(qa_pairs)} 条问答对")

    # 限制 SFT token 总量（默认 1 亿），按 ~500 tokens/对估算采样数
    SFT_TOKEN_BUDGET = 100_000_000
    AVG_TOKENS_PER_QA = 500
    max_pairs = SFT_TOKEN_BUDGET // AVG_TOKENS_PER_QA
    if len(qa_pairs) > max_pairs:
        print(
            f"       token 预算 {SFT_TOKEN_BUDGET // 1_000_000}M → 采样 {max_pairs} 条（从 {len(qa_pairs)} 条中）"
        )
        qa_pairs = random.sample(qa_pairs, max_pairs)

    # 加载BPE分词器
    print(f"[2/5] 加载 BPE 分词器: {tokenizer_path}")
    # 初始化分词器
    tokenizer = NovaTokenizer()
    # 加载BPE分词器
    tokenizer.load(tokenizer_path)
    print(f"       词表大小: {tokenizer.vocab_size}")

    print("[3/5] 创建微调数据集和 DataLoader...")
    config = NovaConfig(vocab_size=tokenizer.vocab_size)
    config.max_seq_len = config.finetune_max_seq_len
    config.batch_size = config.finetune_batch_size
    config.learning_rate = config.finetune_lr
    # 微调阶段将问答对批量预编码、截断、填充、生成模型tensor输入和标签tensor输入
    dataset = NovaDataset(qa_pairs, tokenizer, config.max_seq_len)
    # 把Dataset包装成可以按batch取数据的迭代器
    dataloader = create_dataloader(dataset, batch_size=config.batch_size)
    print(f"       数据集大小: {len(dataset)} 条")
    print(f"       每轮 batch 数: {len(dataloader)} (batch_size={config.batch_size})")

    # 指定计算设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"[4/5] 创建模型 (device={device})...")
    # 初始化模型，并完成相关参数的初始化动作
    model = NovaModel(config).to(device)
    # 输出模型参数
    model.print_parameter_summary()

    # 创建AdamW 优化器
    print("[5/5] 创建 AdamW 优化器...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    print(f"       lr={config.learning_rate}, weight_decay={config.weight_decay}")

    # 加载预训练权重
    # 由于模型在预训练阶段已经学习过语言规律了，因此SFT微调继承了预训练的权重参数，在预训练的基础上继续进行更深层次的SFT训练
    start_epoch = 0
    best_loss = float("inf")
    if resume_path is not None:
        # 只加载模型权重，不恢复优化器状态和 start_epoch
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"checkpoint 文件不存在: {resume_path}")
        print(f"\n加载预训练权重: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        state_dict = checkpoint["model_state_dict"]

        # RoPE 使用数学公式计算，不存储可训练权重。如果微调阶段 max_seq_len 与预训练不同，
        # 只需重新计算 freqs_cis buffer，无需像旧方案那样截断/扩展 pos_emb.weight
        ckpt_config: NovaConfig = checkpoint["config"]
        if ckpt_config.max_seq_len != config.max_seq_len:
            print(f"  → RoPE 频率重算: {ckpt_config.max_seq_len} → {config.max_seq_len}")
            head_dim = config.d_model // config.n_heads
            new_freqs = precompute_rope_freqs(
                head_dim, config.max_seq_len, theta=config.rope_theta
            ).to(device)
            state_dict["freqs_cis"] = new_freqs

        model.load_state_dict(state_dict)
        print(f"  → 已加载预训练权重，微调从 epoch 0 开始")

    return config, dataloader, model, optimizer, device, start_epoch, best_loss


# ======================================================================
# 主函数
# ======================================================================
def main() -> None:
    # 如果支持cuda的情况下，允许矩阵乘法用 TF32（TensorFloat-32）精度代替完整的 FP32，速度快很多，精度损失极小
    if torch.cuda.is_available():
        # highest最高精度、high高精度、medium激进且速度优先
        # AMP把能降的降到 FP16，降不了的留 FP32，而留下的那些 FP32 矩阵乘法又被 TF32 进一步加速
        torch.set_float32_matmul_precision("medium")
        # cuDNN 是 NVIDIA 提供的深度学习底层加速库，里面同一种运算（比如矩阵乘法）有不同的算法实现
        # benchmark = True会在训练过程中自动选择最优算法，提高训练速度
        torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser(description="Nova 模型训练")
    # 指定训练模式，pretrain是预训练，finetune是微调
    parser.add_argument(
        "--mode",
        type=str,
        choices=["pretrain", "finetune"],
        required=True,
        help="训练模式: pretrain=预训练, finetune=微调",
    )
    # 微调时加载预训练权重
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="checkpoint 路径，用于微调时加载预训练权重",
    )
    # 指定训练时的语料包文件或路径
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="数据路径（预训练: JSONL 目录/文件，微调: JSON 文件）",
    )
    args = parser.parse_args()

    # 根据不同的模式执行不同的训练初始化动作
    if args.mode == "pretrain":
        data_path = args.data or "data/pretrain"
        config, dataloader, model, optimizer, device, start_epoch, best_loss = (
            # 预训练初始化
            setup_pretrain(data_path=data_path)
        )
        # 根据训练模式的不同，覆盖config中的相关超参
        config.epochs = config.pretrain_epochs
        # 计算预热步数,预训练阶段LR需要进入到预热阶段，因为预训练阶段参数是随机的，需要逐步升温
        config.warmup_steps = min(config.warmup_steps, config.epochs // 2)
    else:
        data_path = args.data or "data/sft/"
        config, dataloader, model, optimizer, device, start_epoch, best_loss = (
            # 微调训练初始化
            setup_finetune(data_path=data_path, resume_path=args.resume)
        )
        config.epochs = config.finetune_epochs
        # SFT阶段LR不需要再进入到预热阶段，而是直接进入到余弦衰减阶段，因为SFT阶段可以在预训练后的权重基础上训练，参数不再随机所以不需要逐步升温
        config.warmup_steps = 0

    print(f"\n训练轮数: {config.epochs}，warmup: {config.warmup_steps}")
    print(
        f"batch_size: {config.batch_size}，max_seq_len: {config.max_seq_len}，lr: {config.learning_rate}"
    )
    # 执行核心训练循环
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
