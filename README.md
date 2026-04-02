# Nova — 微型 Decoder-Only Transformer LLM

Nova 架构与 GPT / LLaMA 同源，使用 Pre-RMSNorm、SwiGLU、BPE 分词、可学习位置编码等现代 LLM 标准技术。

## 环境要求

- Python 3.11+
- macOS / Linux / Windows
- 支持 CUDA / MPS / CPU

## 快速上手

### 1. 创建虚拟环境并安装依赖

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 预训练

```bash
.venv/bin/python train.py --mode pretrain --data data/pretrain/
```

### 3. SFT 微调

```bash
.venv/bin/python train.py --mode finetune --data data/sft/ --resume checkpoints/best_model.pt
```

### 4. 开始对话

```bash
.venv/bin/python chat.py
```

## 项目结构

```
novaaware-llm/
├── config.py                 # 模型超参数 & 训练配置 & 推理参数
├── tokenizer.py              # BPE 分词器（训练 + encode/decode）
├── dataset.py                # 预训练/微调数据集（预编码 + 填充 + DataLoader）
├── model.py                  # Decoder-Only Transformer 模型实现
├── train.py                  # 训练流程（预训练 & SFT 微调）
├── chat.py                   # 推理与交互式对话（自回归生成）
├── data/
│   ├── pretrain/             # 预训练语料（jsonl 格式）
│   ├── sft/                  # SFT 微调问答对（jsonl 格式）
│   └── tokenizer.json        # BPE 分词器产物（训练后自动生成）
├── checkpoints/              # 模型权重产物（训练后自动生成）
├── tests/                    # 单元测试
└── docs/                     # 文档
```

## 源码阅读顺序

建议按以下顺序阅读源码，每个文件内部都标注了阅读步骤：

| 顺序 | 文件 | 说明 |
|------|------|------|
| 1 | `config.py` | 所有超参数定义，先了解模型的"规格表" |
| 2 | `tokenizer.py` | BPE 分词器，理解文本如何变成 token ID |
| 3 | `model.py` | Transformer 核心，Embedding → Block（Attention + FFN）→ 输出 |
| 4 | `dataset.py` | 训练数据如何编码、填充、生成 input_ids 和 target_ids |
| 5 | `train.py` | 训练循环，前向传播 → 算 loss → 反向传播 → 参数更新 |
| 6 | `chat.py` | 推理，加载模型 → 自回归循环（前向 → 采样 → 拼接）→ 输出回答 |

## 模型规格

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 架构 | Decoder-Only Transformer | 与 GPT / LLaMA 同源 |
| d_model | 384 | 向量维度 / 特征空间维度 |
| n_heads | 6 | 多头注意力头数 |
| n_layers | 4 | Block 堆叠层数 |
| d_ff | 1536 | FFN 扩容维度（4 × d_model） |
| max_seq_len | 128 | 最大序列长度（上下文窗口） |
| vocab_size | 16000 | BPE 词表大小 |
| dropout | 0.1 | 训练时随机丢弃概率，推理时关闭 |
| 激活函数 | SwiGLU (SiLU) | FFN 中的非线性激活 |
| 归一化 | Pre-RMSNorm | 每个子层前做 RMSNorm |
| 位置编码 | 可学习嵌入 | nn.Embedding(max_seq_len, d_model) |

## 技术栈

- **PyTorch** — 模型定义、训练、推理
- **HuggingFace tokenizers** — BPE 分词器底层实现
- **NumPy** — 数据集预编码存储

## 作者

JohnGao
