"""Nova 推理与交互式对话

运行命令:
  .venv/bin/python chat.py
  .venv/bin/python chat.py --checkpoint checkpoints/epoch_200.pt
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
from tokenizer import NovaTokenizer, BOS_ID, SEP_ID, EOS_ID
from model import NovaModel


# 推理等待动画
class Spinner:
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


# 自回归生成回答文本
@torch.no_grad()
def generate(
    model: NovaModel,
    tokenizer: NovaTokenizer,
    question: str,
    # 重复惩罚系数，防止模型反复说同样的话
    repetition_penalty: float = 1.3,
) -> str:
    # 获取模型的计算设备
    device = next(model.parameters()).device
    # 单次回答最多生成多少个答案 token
    max_new_tokens = _DEFAULTS.max_new_tokens

    # 把用户的输入问题编码为 token ids，然后收尾拼接上<s>和<sep>
    input_ids = [BOS_ID] + tokenizer.encode(question) + [SEP_ID]
    # 转为模型的输入tensor
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    # 最终答案结果的token_ids
    generated_ids: list[int] = []
    # 自回归循环(核心循环)
    for _ in range(max_new_tokens):
        # 截断输入序列，确保不超过位置编码表的长度限制（max_seq_len），超出截断前面的内容
        ids_cond = ids[:, -model.pos_emb.weight.shape[0] :]

        # 调用模型的forward函数，进行前向传播计算
        logits = model(ids_cond)  # [1, seq_len, vocab_size]

        #         输入:  位置0    位置1    位置2    位置3
        #        <s>     你      好      吗<sep>
        #        模型输出 logits 形状 [1, 4, 16000]，意思是：
        #  位置	  输入的 token	模型给出 16000 个分数	                      含义
        # 位置 0	<s>	[2.1, -0.3, 0.5, 1.8, ...] 共 16000 个	模型预测 <s> 后面应该是哪个字
        # 位置 1	你	[0.7, 1.2, -0.4, 3.1, ...] 共 16000 个	模型预测 <s>你 后面应该是哪个字
        # 位置 2	好	[-0.1, 0.8, 2.3, 0.2, ...] 共 16000 个	模型预测 <s>你好 后面应该是哪个字
        # 位置 3	吗<sep>	[0.3, -1.0, 0.1, 4.5, ...] 共 16000 个	模型预测 <s>你好吗<sep> 后面应该是哪个字
        # 推理时只关心最后一个位置（位置 3），因为我们只需要知道整个序列之后下一个字是什么，所以只取最后一个位置的 logits
        next_logits = logits[:, -1, :]  # [1, vocab_size]

        # 重复惩罚: 对已生成过的 token 降低概率，防止重复循环
        if repetition_penalty != 1.0 and generated_ids:
            seen = set(generated_ids)
            for token_id in seen:
                if next_logits[0, token_id] > 0:
                    next_logits[0, token_id] /= repetition_penalty
                else:
                    next_logits[0, token_id] *= repetition_penalty

        # 温度缩放
        next_logits = next_logits / _DEFAULTS.temperature

        # Top-k 过滤:
        #   只保留概率最高的 k 个字，其余设为 -inf（softmax 后变成 0）
        #   防止从极低概率的"噪声"字中采样导致乱码
        top_k = _DEFAULTS.top_k
        if top_k > 0:
            top_k_clamped = min(top_k, next_logits.size(-1))
            # topk 返回前 k 大的值和索引
            top_values, _ = torch.topk(next_logits, top_k_clamped)
            # 第 k 大的值作为阈值，低于阈值的全部设为 -inf
            threshold = top_values[:, -1].unsqueeze(-1)
            next_logits = next_logits.masked_fill(
                next_logits < threshold, float("-inf")
            )

        # softmax把打分转换为概率
        probs = F.softmax(next_logits, dim=-1)  # [1, vocab_size]

        # 按概率随机采样，所以同样的问题每次可能生成不同的答案
        # [0.0, 0.0, ..., 0.45, ..., 0.35, ..., 0.20, ..., 0.0]
        #                 "我"        "她"        "他"
        next_id = torch.multinomial(probs, num_samples=1)  # [1, 1]

        # 把tensor转换为token_id
        next_id_int = next_id.item()

        # 遇到 <e> 标记 → 模型认为回答结束，停止生成
        if next_id_int == EOS_ID:
            break

        # 把新生成的 token 追加到答案序列末尾
        generated_ids.append(next_id_int)

        # torch.cat 就是 tensor 拼接
        # 采样出的新 token 拼接到输入序列末尾，作为下一轮循环新入参
        ids = torch.cat([ids, next_id], dim=1)  # [1, seq_len + 1]

    # 解码返回控制台输出答案
    # tokenizer.decode 会跳过特殊标记（<pad>/<s>/<e>/<sep>），只留可读字符
    return tokenizer.decode(generated_ids)


# 加载分词器和模型权重
def load_model_for_inference(
    checkpoint_path: str,
    vocab_path: str = "data/tokenizer.json",
) -> tuple[NovaModel, NovaTokenizer, torch.device]:
    # 加载BPE分词器
    if not os.path.isfile(vocab_path):
        raise FileNotFoundError(f"分词器文件不存在: {vocab_path}")

    tokenizer = NovaTokenizer()
    tokenizer.load(vocab_path)

    # 选择计算设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # 检查checkpoint模型权重文件是否存在
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"checkpoint 文件不存在: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config: NovaConfig = checkpoint["config"]

    # 创建模型并加载checkpoint模型权重
    model = NovaModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # 训练的时候用train模式，推理的时候用评估(推理)模式
    model.eval()

    return model, tokenizer, device


# 交互式循环对话
def chat_loop(
    model: NovaModel,
    tokenizer: NovaTokenizer,
) -> None:
    print()
    print("╔══════════════════════════════════════╗")
    print("║       Welcome to Nova Mini LLM       ║")
    print("╚══════════════════════════════════════╝")
    print()
    print('Nova> 你好！我是 Nova，一个微型 LLM。输入 "quit" 退出。')
    print()

    # 对话循环
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
            # 自回归生成回答
            answer = generate(
                model=model,
                tokenizer=tokenizer,
                question=user_input,
            )

        print(f"Nova: {answer}")
        print()


# 主函数
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nova 微型 LLM - 交互式对话",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="模型 checkpoint 路径（默认 checkpoints/best_model.pt）",
    )
    args = parser.parse_args()

    try:
        # 加载分词器、创建模型、加载checkpoint模型权重
        model, tokenizer, device = load_model_for_inference(
            checkpoint_path=args.checkpoint,
        )
    except FileNotFoundError as e:
        print(f"\n错误: {e}")
        print("请先运行 .venv/bin/python train.py 训练模型。")
        sys.exit(1)

    print(f"模型已加载: {args.checkpoint} (device={device})")
    print(f"生成参数: temperature={_DEFAULTS.temperature}, top_k={_DEFAULTS.top_k}")

    # 启动对话
    chat_loop(
        model=model,
        tokenizer=tokenizer,
    )


if __name__ == "__main__":
    main()
