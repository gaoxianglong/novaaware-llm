from dataclasses import dataclass


@dataclass
class NovaConfig:
    # ========================
    # 模型架构参数
    # ========================

    # RTX 4090 D (24GB) — 目标 ~22M 参数（宽而浅，GPU 友好）
    # d_model = 384, n_heads = 6, n_layers = 4, d_ff = 1536
    # max_seq_len = 1024, batch_size = 256
    # 训练预算: 3 亿预训练 tokens + 1 亿 SFT tokens = 4 亿 (Chinchilla 最优)

    # Total = V × D + S × D + L × (2D + 4D² + 3D × F) + D + D × V
    # 其中：
    # V = vocab_size（词表大小）
    # D = d_model（嵌入维度）
    # S = max_seq_len（最大序列长度）
    # L = n_layers（Transformer 层数）
    # F = d_ff（FFN 隐藏维度）

    d_model: int = 384
    """嵌入维度（Embedding Dimension）。
    每个 token 被映射为一个 d_model 维的向量。这个向量同时承载"语义"和"位置"两层信息。
    维度越大，token可以容纳的特征越多，模型能编码的语义细节就越丰富，但计算开销更大。"""

    n_heads: int = 6
    """多头注意力的头数（Number of Attention Heads）。
    自注意力会被拆成 n_heads 个独立的"头"进行独立的并行注意力计算，每个头在 d_model / n_heads 维的子空间里独立计算注意力。
    不同的头可以关注不同的语言特征——有的头可能学会关注语法搭配，有的头可能学会关注语义关联。最后把所有头的结果拼接回 d_model 维。"""

    n_layers: int = 4
    """Transformer Decoder Block 的堆叠层数。
    每一层包含：RMSNorm → 多头自注意力计算 → 残差连接 → RMSNorm → SwiGLU FFN（前馈网络） → 残差连接。
    层数越多，模型对语义的"咀嚼"越深（第 1 层学字符搭配，第 2 层学句法，第 3 层学
    语义，第 4 层学意图）。4 层足以让微型模型在预设问答场景下表现良好。"""

    d_ff: int = 1536
    """前馈网络的隐藏维度（Feed-Forward Hidden Dimension）。
    每层 Transformer Block 中的 SwiGLU FFN 会先把向量从 d_model 维扩展到
    d_ff 维，在更大的空间里做非线性变换（"展开思考"），然后再压缩回 d_model 维。
    通常 d_ff = 4 × d_model 是经验法则。"""

    pretrain_max_seq_len: int = 1024
    """预训练阶段的最大序列长度。长上下文有助于学习长距离语言依赖。"""

    finetune_max_seq_len: int = 1024
    """微调阶段的最大序列长度。QA 对较短，缩短可大幅提速。"""

    max_seq_len: int = 1024
    """最大序列长度"""

    dropout: float = 0.1
    """Dropout 概率。
    训练时随机将 10% 的神经元输出置零，迫使模型不过度依赖某几个特征通路，从而提高
    泛化能力、减轻过拟合。推理时 Dropout 自动关闭，所有神经元正常工作。"""

    vocab_size: int = 0
    """词表大小，由BPE分词器在扫描训练数据后动态设置。
    初始为 0，在构建分词器（tokenizer.build_vocab）时会被更新为实际的字符数 +
    特殊标记数（<pad>/<s>/<e>/<sep>/<unk>）。最终值通常在 800-1500 之间。"""

    # ========================
    # 训练配置参数
    # ========================

    pretrain_batch_size: int = 64
    """预训练阶段的 batch_size。数据量大，大 batch 充分利用 GPU。"""

    finetune_batch_size: int = 16
    """微调阶段的 batch_size。数据量小，小 batch 提供更多梯度更新。"""

    batch_size: int = 64
    """当前训练使用的 batch_size（由 train.py 根据模式自动设置，无需手动修改）。"""

    pretrain_epochs: int = 1
    """预训练轮数。
    预训练数据量大（万级 JSONL），1 轮即可覆盖语言知识。"""

    finetune_epochs: int = 500
    """微调轮数。
    微调数据量相对小，需要数百轮让模型学会对话格式。"""

    epochs: int = 1
    """当前训练使用的轮数（由 train.py 根据模式自动设置，无需手动修改）。"""

    pretrain_lr: float = 5e-4
    """预训练阶段的学习率。数据量大、batch 大，可以用较高 lr 加速收敛。"""

    finetune_lr: float = 3e-4
    """微调阶段的学习率。小数据集用较低 lr，训练更稳定。"""

    learning_rate: float = 5e-4
    """当前训练使用的学习率（由 train.py 根据模式自动设置，无需手动修改）。"""

    weight_decay: float = 0.01
    """权重衰减系数（Weight Decay）。
    AdamW 的正则化手段——每次参数更新时让权重乘以 (1 - weight_decay × lr)，
    轻微缩小参数值，防止权重无限增长导致过拟合。0.01 是业界标准值。"""

    warmup_steps: int = 100
    """学习率预热步数。
    训练初期模型参数是随机的，梯度方向不可靠。前 100 步学习率从 0 线性增长到
    learning_rate，给模型一个"热身"阶段，避免一开始步子太大走偏。"""

    grad_clip: float = 1.0
    """梯度裁剪阈值（Gradient Clipping Max Norm）。
    反向传播时，如果梯度的 L2 范数超过 1.0，就按比例缩小到 1.0。防止个别 batch
    产生异常大的梯度导致参数剧烈震荡（梯度爆炸）。"""

    # ========================
    # 生成（推理）配置参数
    # ========================

    temperature: float = 0.8
    """采样温度（Temperature）。
    控制生成文本的"创造力"。模型输出的 logits 在 softmax 前会除以 temperature：
    - temperature < 1：概率分布变尖锐，模型倾向选概率最高的字（更保守、更确定）
    - temperature = 1：保持原始概率分布
    - temperature > 1：概率分布变平坦，模型更愿意尝试低概率的字（更随机、更有创意）
    0.8 略偏保守，适合问答场景追求准确性。"""

    top_k: int = 5
    """Top-k 采样的 k 值。
    生成下一个字时，只从概率最高的 k=20 个候选字中采样，其余候选字的概率直接
    设为 0。这样既保留了一定的多样性（不是贪心地只选第 1 名），又避免了从概率
    极低的"噪声"字中采样导致输出乱码。"""
