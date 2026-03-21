"""Nova 推理与交互式对话

加载训练好的模型，通过命令行与用户实时对话。
这是整个项目的"成果展示"——前面所有组件在这里合力完成一次推理。

┌──────────────────────────────────────────────────────────────────────┐
│                    chat.py 的完整执行流程                              │
│                                                                      │
│  运行命令:                                                            │
│    .venv/bin/python chat.py                                          │
│    .venv/bin/python chat.py --checkpoint checkpoints/epoch_200.pt    │
│    .venv/bin/python chat.py --temperature 0.5 --top_k 10             │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │               初始化阶段（只执行一次）                          │   │
│  │                                                                │   │
│  │  ① 加载 BPE 分词器                                              │   │
│  │     tokenizer = NovaTokenizer()                                │   │
│  │     tokenizer.load("data/tokenizer.json")                      │   │
│  │     → 恢复训练时的 BPE 分词器（词表 + 合并规则）                 │   │
│  │                                                                │   │
│  │  ② 加载 checkpoint                                             │   │
│  │     checkpoint = torch.load("checkpoints/best_model.pt")       │   │
│  │     config = checkpoint["config"]                               │   │
│  │     → 从 checkpoint 中取出训练时的超参数                         │   │
│  │                                                                │   │
│  │  ③ 创建模型并加载训练好的权重                                    │   │
│  │     model = NovaModel(config)                                  │   │
│  │     model.load_state_dict(checkpoint["model_state_dict"])       │   │
│  │     model.eval()                                               │   │
│  │     → 模型恢复到训练结束时的状态，关闭 Dropout                   │   │
│  │                                                                │   │
│  └────────────────────────────────────────────────────────────────┘   │
│                           │                                          │
│                           ▼                                          │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │               对话循环（持续到用户输入 quit）                    │   │
│  │                                                                │   │
│  │  while True:                                                   │   │
│  │      ④ 读取用户输入                                             │   │
│  │         question = input("你: ")                                │   │
│  │                                                                │   │
│  │      ⑤ 调用 generate() 生成回答                                 │   │
│  │         answer = generate(model, tokenizer, question, ...)      │   │
│  │                                                                │   │
│  │         generate 内部流程:                                       │   │
│  │         ⑤-a 构造 prompt: "<s>问题<sep>"                         │   │
│  │         ⑤-b encode 为 token ID 列表                              │   │
│  │         ⑤-c 自回归循环:                                          │   │
│  │             logits = model(ids)          # 前向传播               │   │
│  │             logits = logits / temperature # 温度缩放              │   │
│  │             top-k 过滤 + softmax + 采样   # 选下一个字            │   │
│  │             追加到序列，继续循环                                  │   │
│  │             遇到 <e> 或达到最大长度 → 停止                       │   │
│  │         ⑤-d decode 为文本，返回                                  │   │
│  │                                                                │   │
│  │      ⑥ 打印回答                                                 │   │
│  │         print(f"Nova: {answer}")                                │   │
│  │                                                                │   │
│  └────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
if sys.platform == "win32":
    import pyreadline3  # noqa: F401 — Windows 下替代 readline，为 input() 启用行编辑
else:
    import readline  # noqa: F401 — 导入即生效，为 input() 启用行编辑（退格、删除、方向键、历史记录）
import threading
import time

import torch
import torch.nn.functional as F

from config import NovaConfig
from tokenizer import NovaTokenizer, BOS_TOKEN, SEP_TOKEN, EOS_ID
from model import NovaModel


# ======================================================================
# Spinner — 推理等待动画（类似 ollama 的加载效果）
# ======================================================================
class Spinner:
    """在后台线程中显示旋转动画，用于推理等待期间的视觉反馈。

    用法:
        with Spinner("thinking"):
            answer = generate(...)
    """

    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, label: str = "thinking") -> None:
        self._label = label
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _spin(self) -> None:
        for frame in itertools.cycle(self.FRAMES):
            if self._stop.is_set():
                break
            sys.stdout.write(f"\r\033[36m{frame}\033[0m {self._label}...")
            sys.stdout.flush()
            time.sleep(0.08)
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

    def __enter__(self) -> "Spinner":
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()


_DEFAULTS = NovaConfig()


# ======================================================================
# 步骤 7.1：自回归文本生成
# ======================================================================
@torch.no_grad()
def generate(
    model: NovaModel,
    tokenizer: NovaTokenizer,
    question: str,
    max_new_tokens: int = 100,
    temperature: float = _DEFAULTS.temperature,
    top_k: int = _DEFAULTS.top_k,
    repetition_penalty: float = 1.3,
) -> str:
    """根据用户问题，自回归生成回答文本。

    ┌──────────────────────────────────────────────────────────────────┐
    │  generate 的完整执行过程（以"你叫什么名字？"为例）                  │
    │                                                                    │
    │  ⑤-a 构造 prompt:                                                  │
    │       "<s>你叫什么名字？<sep>"                                      │
    │       和训练时 dataset.py 的拼接格式一致:                             │
    │         训练: "<s>问题<sep>回答<e>"                                  │
    │         推理: "<s>问题<sep>"  ← 只给前半部分，让模型续写后半部分      │
    │                                                                    │
    │  ⑤-b encode 为 token ID:                                           │
    │       [1, 42, 15, 88, 7, 67, 3]                                    │
    │        ↑                     ↑                                      │
    │       <s>                   <sep>                                   │
    │                                                                    │
    │  ⑤-c 自回归循环（逐字生成）:                                        │
    │                                                                    │
    │       第 1 轮: ids = [1,42,15,88,7,67,3]                            │
    │               logits = model(ids)        # 前向传播                 │
    │               取 logits[:, -1, :]        # 最后一个位置的预测        │
    │               ÷ temperature (0.8)        # 温度缩放                 │
    │               top-k 过滤                 # 只保留概率前 20 的字      │
    │               softmax → 采样             # 按概率随机选一个字        │
    │               → 选中 "我" (ID=33)                                   │
    │               追加: ids = [1,42,15,88,7,67,3,33]                    │
    │                                                                    │
    │       第 2 轮: ids = [1,42,15,88,7,67,3,33]                         │
    │               → 选中 "是" (ID=21)                                   │
    │               追加: ids = [1,42,15,88,7,67,3,33,21]                 │
    │                                                                    │
    │       ... 继续生成 ...                                              │
    │                                                                    │
    │       第 N 轮: 选中 <e> (ID=2)                                      │
    │               → 停止生成                                            │
    │                                                                    │
    │  ⑤-d decode:                                                       │
    │       生成的 ID: [33, 21, ...]                                      │
    │       tokenizer.decode() → "我是名为Nova的微型LLM。"                │
    │       返回给调用方                                                   │
    └──────────────────────────────────────────────────────────────────┘

    参数:
      model          — NovaModel 实例（已加载权重，eval 模式）
      tokenizer      — NovaTokenizer 实例（已加载词表）
      question       — 用户输入的问题文本（纯文本，不含特殊标记）
      max_new_tokens — 最多生成多少个新 token（防止无限循环）
      temperature    — 温度参数，控制生成的随机性
                       < 1: 更确定（倾向高概率字）
                       = 1: 保持原始分布
                       > 1: 更随机（倾向探索低概率字）
      top_k          — 只从概率最高的 k 个候选字中采样

    返回:
      生成的回答文本（str）

    调用时机:
      在交互循环中，用户输入问题后调用:
        answer = generate(model, tokenizer, question, ...)
        print(f"Nova: {answer}")
    """
    device = next(model.parameters()).device

    # ── ⑤-a + ⑤-b 构造 prompt 并编码为 token ID ──
    #
    # 拼接格式: [BOS_ID] + encode(问题) + [SEP_ID]
    # 和训练时 dataset.py 的格式完全一致:
    #   训练: [BOS_ID] + encode(问题) + [SEP_ID] + encode(回答) + [EOS_ID]
    #   推理: [BOS_ID] + encode(问题) + [SEP_ID]  ← 只给前半部分
    #
    # 注意: 不能直接 encode("<s>问题<sep>")，BPE 分词器会把 "<s>" 当作
    # 普通文本进行子词切分，而不是当作 BOS 特殊标记。
    # 必须手动拼接特殊标记的 ID，和 dataset.py 保持一致。
    input_ids = (
        [tokenizer.char_to_id[BOS_TOKEN]]
        + tokenizer.encode(question)
        + [tokenizer.char_to_id[SEP_TOKEN]]
    )
    # 转为 tensor，形状 [1, seq_len]（1 是 batch_size，推理时只有一条）
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    # ── ⑤-c 自回归循环 ──
    # 记录生成的新 token（不包含 prompt 部分）
    generated_ids: list[int] = []

    for _ in range(max_new_tokens):
        # 截断到 max_seq_len 以内（位置编码表只有这么长）
        ids_cond = ids[:, -tokenizer.vocab_size:]  # 安全起见用 vocab_size，实际用 max_seq_len
        if hasattr(model, 'pos_emb'):
            max_pos = model.pos_emb.weight.shape[0]
            ids_cond = ids[:, -max_pos:]

        # (i) 前向传播: 获取模型对每个位置、每个字的预测分数
        logits = model(ids_cond)  # [1, seq_len, vocab_size]

        # (ii) 只取最后一个位置的 logits（这是模型对"下一个字"的预测）
        next_logits = logits[:, -1, :]  # [1, vocab_size]

        # (iii-a) 重复惩罚: 对已生成过的 token 降低概率，防止重复循环
        if repetition_penalty != 1.0 and generated_ids:
            seen = set(generated_ids)
            for token_id in seen:
                if next_logits[0, token_id] > 0:
                    next_logits[0, token_id] /= repetition_penalty
                else:
                    next_logits[0, token_id] *= repetition_penalty

        # (iii-b) 温度缩放
        next_logits = next_logits / temperature

        # (iv) Top-k 过滤:
        #   只保留概率最高的 k 个字，其余设为 -inf（softmax 后变成 0）
        #   防止从极低概率的"噪声"字中采样导致乱码
        if top_k > 0:
            top_k_clamped = min(top_k, next_logits.size(-1))
            # topk 返回前 k 大的值和索引
            top_values, _ = torch.topk(next_logits, top_k_clamped)
            # 第 k 大的值作为阈值，低于阈值的全部设为 -inf
            threshold = top_values[:, -1].unsqueeze(-1)
            next_logits = next_logits.masked_fill(next_logits < threshold, float('-inf'))

        # (v) Softmax → 概率分布 → 按概率采样一个 token
        probs = F.softmax(next_logits, dim=-1)  # [1, vocab_size]
        next_id = torch.multinomial(probs, num_samples=1)  # [1, 1]

        next_id_int = next_id.item()

        # (vi) 遇到 <e> 标记 → 模型认为回答结束，停止生成
        if next_id_int == EOS_ID:
            break

        # (vii) 把新生成的 token 追加到序列末尾，下一轮循环时模型能看到它
        generated_ids.append(next_id_int)
        ids = torch.cat([ids, next_id], dim=1)  # [1, seq_len + 1]

    # ── ⑤-d 解码为文本 ──
    # tokenizer.decode 会跳过特殊标记（<pad>/<s>/<e>/<sep>），只留可读字符
    return tokenizer.decode(generated_ids)


# ======================================================================
# 步骤 7.1 补充：加载推理所需的模型和分词器
# ======================================================================
def load_model_for_inference(
    checkpoint_path: str,
    vocab_path: str = "data/tokenizer.json",
) -> tuple[NovaModel, NovaTokenizer, torch.device]:
    """加载训练好的模型和分词器，准备推理。

    ┌──────────────────────────────────────────────────────────────────┐
    │  加载流程（初始化阶段）                                            │
    │                                                                    │
    │  ① 加载 BPE 分词器:                                                │
    │     tokenizer = NovaTokenizer()                                    │
    │     tokenizer.load("data/tokenizer.json")                          │
    │     → 恢复训练时的 BPE 分词器（词表 + 合并规则）                    │
    │     → 确保 encode/decode 结果与训练时一致                           │
    │                                                                    │
    │  ② 选择设备:                                                       │
    │     cuda > mps > cpu（自动选最快的）                                │
    │                                                                    │
    │  ③ 加载 checkpoint:                                                │
    │     checkpoint = torch.load(path, map_location=device)             │
    │     → 取出 config（超参数）和 model_state_dict（模型权重）          │
    │                                                                    │
    │  ④ 创建模型并加载权重:                                              │
    │     model = NovaModel(config)                                      │
    │     model.load_state_dict(checkpoint["model_state_dict"])           │
    │     → Embedding、位置表、QKV 矩阵、FFN 权重全部恢复到训练结束时的状态│
    │                                                                    │
    │  ⑤ 切换到 eval 模式:                                               │
    │     model.eval()                                                   │
    │     → 关闭 Dropout（训练时随机丢弃的行为在推理时不需要）             │
    │     → 确保每次相同输入产生一致的输出                                 │
    └──────────────────────────────────────────────────────────────────┘

    参数:
      checkpoint_path — checkpoint 文件路径（如 "checkpoints/best_model.pt"）
      vocab_path      — BPE 分词器文件路径（如 "data/tokenizer.json"）

    返回:
      (model, tokenizer, device) 元组

    异常:
      FileNotFoundError — checkpoint 或 vocab 文件不存在
    """
    # ── ① 加载 BPE 分词器 ──
    if not os.path.isfile(vocab_path):
        raise FileNotFoundError(f"分词器文件不存在: {vocab_path}")

    tokenizer = NovaTokenizer()
    tokenizer.load(vocab_path)

    # ── ② 选择设备 ──
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # ── ③ 加载 checkpoint ──
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"checkpoint 文件不存在: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config: NovaConfig = checkpoint["config"]

    # ── ④ 创建模型并加载权重 ──
    model = NovaModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # ── ⑤ 切换到 eval 模式 ──
    model.eval()

    return model, tokenizer, device


# ======================================================================
# 步骤 7.2 + 7.4：交互式对话界面
# ======================================================================
def chat_loop(
    model: NovaModel,
    tokenizer: NovaTokenizer,
    temperature: float = _DEFAULTS.temperature,
    top_k: int = _DEFAULTS.top_k,
) -> None:
    """交互式对话主循环。

    ┌──────────────────────────────────────────────────────────────────┐
    │  对话循环的执行流程                                                │
    │                                                                    │
    │  ① 打印欢迎横幅                                                    │
    │                                                                    │
    │  ② while True:                                                     │
    │     ├─ 读取用户输入 (input("你: "))                                 │
    │     │                                                              │
    │     ├─ 空输入 → 跳过，提示用户重新输入                              │
    │     │                                                              │
    │     ├─ "quit" 或 "exit" → 打印再见，退出循环                       │
    │     │                                                              │
    │     ├─ 正常问题 → 调用 generate() 生成回答                          │
    │     │    generate(model, tokenizer, question, temperature, top_k)  │
    │     │                                                              │
    │     └─ 打印回答: "Nova: {answer}"                                  │
    │                                                                    │
    │  ③ Ctrl+C → 捕获 KeyboardInterrupt，优雅退出                       │
    │  ④ EOFError → 输入流结束（管道/重定向场景），优雅退出               │
    └──────────────────────────────────────────────────────────────────┘

    参数:
      model       — NovaModel 实例（已加载权重，eval 模式）
      tokenizer   — NovaTokenizer 实例（已加载词表）
      temperature — 生成温度（默认来自 config.py）
      top_k       — Top-k 采样的 k 值（默认来自 config.py）
    """
    # ── ① 打印欢迎横幅 ──
    print()
    print("╔══════════════════════════════════════╗")
    print("║       Welcome to Nova Mini LLM       ║")
    print("╚══════════════════════════════════════╝")
    print()
    print('Nova> 你好！我是 Nova，一个微型 LLM。输入 "quit" 退出。')
    print()

    # ── ② 对话循环 ──
    while True:
        try:
            user_input = input("你: ")
        except KeyboardInterrupt:
            # Ctrl+C → 换行后优雅退出（避免 ^C 残留在提示符同行）
            print()
            print("Nova> 再见！")
            break
        except EOFError:
            # 输入流结束（管道/重定向场景）→ 优雅退出
            print()
            print("Nova> 再见！")
            break

        # 去除首尾空白
        user_input = user_input.strip()

        # 空输入 → 跳过
        if not user_input:
            continue

        # 退出命令
        if user_input.lower() in ("quit", "exit"):
            print("Nova> 再见！")
            break

        with Spinner("thinking"):
            answer = generate(
                model=model,
                tokenizer=tokenizer,
                question=user_input,
                temperature=temperature,
                top_k=top_k,
            )

        print(f"Nova: {answer}")
        print()


# ======================================================================
# 步骤 7.3：命令行入口
# ======================================================================
def main() -> None:
    """命令行入口。

    用法:
      .venv/bin/python chat.py                                          # 使用默认参数
      .venv/bin/python chat.py --checkpoint checkpoints/epoch_200.pt    # 指定 checkpoint
      .venv/bin/python chat.py --temperature 0.5 --top_k 10             # 调整生成参数

    参数说明:
      --checkpoint  模型 checkpoint 路径（默认 checkpoints/best_model.pt）
      --temperature 生成温度，越小越确定（默认来自 config.py）
      --top_k       Top-k 采样的 k 值，越小越保守（默认来自 config.py）
    """
    parser = argparse.ArgumentParser(
        description="Nova 微型 LLM - 交互式对话",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="模型 checkpoint 路径（默认 checkpoints/best_model.pt）",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=_DEFAULTS.temperature,
        help=f"生成温度，控制随机性（默认 {_DEFAULTS.temperature}）",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=_DEFAULTS.top_k,
        help=f"Top-k 采样的 k 值（默认 {_DEFAULTS.top_k}）",
    )
    args = parser.parse_args()

    # ── 加载模型（带友好错误提示）──
    try:
        model, tokenizer, device = load_model_for_inference(
            checkpoint_path=args.checkpoint,
        )
    except FileNotFoundError as e:
        print(f"\n错误: {e}")
        print("请先运行 .venv/bin/python train.py 训练模型。")
        sys.exit(1)

    print(f"模型已加载: {args.checkpoint} (device={device})")
    print(f"生成参数: temperature={args.temperature}, top_k={args.top_k}")

    # ── 启动对话 ──
    chat_loop(
        model=model,
        tokenizer=tokenizer,
        temperature=args.temperature,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
