# Nova — 从零手搓的微型 LLM

Nova 是一个从零实现的 Decoder-Only Transformer 语言模型，参数量约 1-2M。架构与 GPT / LLaMA 同源，使用 Pre-RMSNorm、SwiGLU、可学习位置编码等现代 LLM 标准技术。

项目的目标不是做一个"能用"的大模型，而是把 Transformer 的每一个零件都自己焊上去——从分词器到注意力机制到训练循环，全部手写，不调 API，不 fine-tune 别人的模型。

## 环境要求

- Python 3.11+
- macOS / Linux
- 无需 GPU（微型模型，CPU 即可训练和推理）

## 快速上手

### 1. 创建虚拟环境

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 训练模型

```bash
python train.py
```

### 4. 开始对话

```bash
python chat.py
```

## 项目结构

```
novaaware-llm/
├── requirements.txt          # 依赖列表
├── README.md                 # 本文件
├── config.py                 # 模型超参数配置
├── tokenizer.py              # 字符级分词器
├── dataset.py                # 数据集与 DataLoader
├── model.py                  # Transformer 模型实现
├── train.py                  # 训练流程
├── chat.py                   # 推理与交互式对话
├── data/
│   └── qa_pairs.json         # 训练用问答对
├── checkpoints/              # 训练产物（自动创建）
└── docs/
    ├── ARCHITECTURE.md        # 架构设计文档
    └── IMPLEMENTATION_PLAN.md # 实施计划
```

## 模型规格

| 参数 | 值 |
|------|-----|
| 架构 | Decoder-Only Transformer |
| 嵌入维度 | 128 |
| 注意力头数 | 4 |
| Decoder 层数 | 4 |
| FFN 隐藏维度 | 512 |
| 最大序列长度 | 128 |
| 激活函数 | SwiGLU |
| 归一化 | Pre-RMSNorm |
| 位置编码 | 可学习嵌入 |
| 参数量 | ~1-2M |

## 作者

高翔龙
