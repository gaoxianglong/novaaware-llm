"""Nova Decoder-Only Transformer 模型

从零实现完整的 Decoder-Only Transformer，遵循现代 LLM 标准（LLaMA / GPT 系列）。

本文件包含以下组件（按调用顺序排列）：

┌─────────────────────────────────────────────────────────────────────┐
│                    模型组件与调用顺序                                 │
│                                                                     │
│  输入 token IDs: [batch_size, seq_len]                              │
│       │                                                             │
│       ▼                                                             │
│  NovaModel（完整模型，⑤）                                           │
│   ├── Token Embedding    查字义表: token ID → 128维向量              │
│   ├── Position Embedding 查位置表: 位置编号 → 128维向量              │
│   ├── Dropout                                                       │
│   ├── TransformerBlock × 4 层（④）                                   │
│   │    ├── RMSNorm（①）→ MultiHeadAttention（③）→ 残差              │
│   │    └── RMSNorm（①）→ SwiGLUFFN（②）→ 残差                      │
│   ├── RMSNorm（最终归一化）                                          │
│   └── Linear 输出层: 128维 → vocab_size                             │
│       │                                                             │
│       ▼                                                             │
│  输出 logits: [batch_size, seq_len, vocab_size]                     │
└─────────────────────────────────────────────────────────────────────┘

源码阅读顺序（建议按编号顺序，由内而外阅读）:
┌──────────────────────────────────────────────────────────────────────┐
│  ① RMSNorm              — 归一化层（最小的独立组件）                 │
│  ② SwiGLUFFN            — SwiGLU 前馈网络（依赖 ①）                │
│  ③ MultiHeadAttention   — 多头自注意力（依赖 ①）                   │
│  ④ TransformerBlock     — 一层 Decoder Block（依赖 ① ② ③）        │
│  ⑤ NovaModel            — 完整模型（依赖 ④，串联所有组件）          │
│     ├── ⑤a _init_weights   — 权重初始化（在 ⑤ 的构造函数中调用）   │
│     └── ⑤b print_parameter_summary — 参数量统计（训练前打印）       │
│                                                                      │
│  推荐理由:                                                            │
│    先读 ① 理解归一化原理，它在每层 Block 中被用 2 次、最后还用 1 次; │
│    再读 ②③ 理解 Block 内部的两个核心计算（FFN 和 Attention）;       │
│    然后读 ④ 看一层 Block 如何组装这些组件;                            │
│    最后读 ⑤ 看完整模型如何串联 N 层 Block 完成从 token ID 到 logits。│
└──────────────────────────────────────────────────────────────────────┘

快速导航（在编辑器中按 Cmd+F / Ctrl+F，搜索以下标记即可跳转）:
  搜索 "① RMSNorm"            → 归一化层（最小独立组件）
  搜索 "② SwiGLUFFN"          → SwiGLU 前馈网络
  搜索 "③ MultiHeadAttention"  → 多头自注意力
  搜索 "④ TransformerBlock"    → 一层 Decoder Block
  搜索 "⑤ NovaModel"           → 完整模型
  搜索 "⑤a"                    → 权重初始化（_init_weights）
  搜索 "⑤b"                    → 参数量统计（print_parameter_summary）
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NovaConfig


# ======================================================================
# 归一化处理
# Nova这种RMSNorm 被调用了 9 次（每层 Block 调 2 次 × 4 层 + 最终 1 次）
# ======================================================================
class RMSNorm(nn.Module):
    # Nova使用RMSNorm进行归一化处理
    #   公式: RMSNorm(x) = x / RMS(x) * gamma，其中 RMS(x) = sqrt(mean(x²) + eps)

    #   RMS均方根，标准是1，如果向量空间中整体尺度(量级)远大于1，或者远小于1，那就逐元素除以同一个rms数值来做整体尺度的放大或缩小，
    #   保持向量空间的尺度始终在区间1这个标准附近。

    #   ======= 解释下，为什么需要做归一化 =======

    # token 的向量会在一层层 Decoder Block 中不断经过自注意力、线性变换、激活函数和残差连接等模块，如果没有归一化，
    # 层与层之间的数值尺度就可能越来越不稳定，前向传播时激活值容易失控；而反向传播又需要沿着这条计算链逐层回传梯度，
    # 因此梯度也更容易被连续放大或连续压缩，最终导致训练难收敛（loss 不降）、容易震荡（loss 波动不下降），甚至出现梯度爆炸或梯度消失。

    # 归一化的核心作用，就是在关键计算节点前先把输入向量的整体数值尺度校准到更稳定的范围内，让每一层都尽量工作在可控的数值区间里，从而提升模型训练和推理的稳定性。
    def __init__(
        self,
        # d_model向量维度
        dim: int,
        # 防止输入向量全是0导致除数为0
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        # Pytorch注册了一个d_model维度的gamma一维数组，初始参数全为1。
        # 在反向传播过程中，模型会根据loss值计算出gamma的梯度值，优化器会根据梯度值和config中预设的超参LR在训练过程中逐步调整gamma参数
        self.gamma = nn.Parameter(torch.ones(dim))  # Parameter是Tensor的子类

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 计算均方根，先平方、再求平均、再开方。
        rms = torch.sqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)

        # 归一化计算
        x_norm = x.float() / rms

        # 解释下为什么归一化后还需要将 输入向量 中的各维度特征进行放大或缩小，
        # 主要是为了避免整体向量尺度被拉回稳定范围后，部分输入向量的特征数据被一刀抹平
        return (x_norm * self.gamma).type_as(x)


# ======================================================================
# FFN前馈网络，Nova使用的是SwiGLU-FFN架构（带门控机制）
# 前馈网络的本质是强化token的语义特征并删除一些冗余信息，最后再压缩回原向量空间维度，以获得更好的语义特征表达
# ======================================================================
class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()

        # SwiGLU有3个通路，模型会根据loss值计算出W1~W3的梯度值，优化器会根据梯度值和config中预设的超参LR在训练过程中逐步调整W1~W3参数

        # 声明W1、W2、W3通路矩阵
        # W1: 门控通路 (d_model → d_ff)，数据结构 [d_ff, d_model]
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        # W2: 压缩通路 (d_ff → d_model)，数据结构 [d_model, d_ff]
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        # W3: 内容通路 (d_model → d_ff)，数据结构 [d_ff, d_model]
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 先解释一个概念，SiLU是激活函数(也叫做Swish激活函数)，公式：silu(x) = x × sigmoid(x)
        # sigmoid(x)是压缩函数，本质就是：大的正数保留，大的负数压到接近零，中间地带按比例衰减
        #  W1 · x  = [ 1.2,  -0.3,  0.8,  0.5, -2.0,  0.1,  0.3,  -0.7]
        #               ↓      ↓     ↓     ↓     ↓     ↓     ↓      ↓
        # SiLU    = [ 0.92, -0.13, 0.55, 0.31, -0.04, 0.05, 0.17, -0.16]

        # W1通路：非线性变换，将输入向量的d_model维度扩展到d_ff维度的高维向量后进行SiLu激活计算，确认哪些冗余信息是需要删除，哪些要保留
        gate = F.silu(self.w1(x))
        # W3通路：单纯的线性变换，将输入向量的d_model维度扩展到d_ff维度的高维向量，相当于扩展后的高维向量的原始内容
        filt = self.w3(x)

        # 门控机制，这里做W1和W3的逐元素相乘，因为W1通路的结果本质上是一堆正数和≈0的数，模型语义特征有限
        # W1和W3做逐元素相乘后，激活值的特征会更丰富，因为会存在大负数的可能性。
        gated = gate * filt

        # 压缩回 d_model 维，≈0的激活值跟W2通路矩阵参数乘法后目标位置的激活值也是≈0，没什么用，会被过滤掉
        # 压缩回d_model维度的输入向量也需要和W2通路的参数进行矩阵乘法，因为压缩方式不同会导致loss值不同
        return self.w2(gated)


# ======================================================================
# 多头自注意力
# Block中的多头自注意力计算，本质就是将 输入向量*W_Q、W_K、W_V矩阵参数，得到QKV线性投影，
# 然后进行Q·K 点积 → 缩放（÷√d） → 因果掩码 → softmax → dropout → 加权 V → 新向量，最终得到了融合当前上下文的新向量激活值。
# 完整流程图:
#     x ─┬─→ W_Q ──→ Q ─┐
#        ├─→ W_K ──→ K ─┼─→ Q@K^T/√d ─→ +mask ─→ softmax ─→ drop ─→ ×V ─→ concat ─→ W_O ─→ out
#        └─→ W_V ──→ V ─┘
# ======================================================================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model({d_model}) 必须能被 n_heads({n_heads}) 整除"
        )
        # token的向量维度
        self.d_model = d_model
        # 多头数量
        self.n_heads = n_heads
        # 每个头负责的token的维度数，我的训练参数是d_model = 384, n_heads = 6，每个头负责 384 / 6 = 64个维度向量
        self.head_dim = d_model // n_heads  # // python语法只保留整数结果

        # 声明 W_Q、W_K、W_V、W_O 四个投影矩阵
        # bias=False被设置为无偏置，这其实是LLaMA的做法，因为偏置项对模型训练影响不大，去掉可以减少参数
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        # 为了避免模型特征固化，降低过拟合风险,前向传播过程中会随机将 部分 向量空间的激活值特征置为0
        # 超参dropout = 0.1，即表示随机丢弃 10% 的数值
        # 输入:       [0.5, 0.8, 0.3, 0.7, 0.2, 0.9, 0.4, 0.6, 0.1, 0.3]
        # Dropout后: [0.5, 0.0, 0.3, 0.7, 0.2, 0.0, 0.4, 0.6, 0.1, 0.3]
        #                  ↑随机变0            ↑随机变0
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape会返回Tensor的形状，[batch, seq_len, d_model]三维数组
        batch_size, seq_len, _ = x.shape

        # 1、把输入向量全部进行 QKV线性投影
        # 输入向量中的每个 token 的 n 维向量分别和 W_Q、W_K、W_V做矩阵乘法
        # 一个token会得到 Q、K、V 3个 新向量[batch, seq_len, d_model]
        q = self.w_q(x)  # 表示 我在找什么 的[batch, seq_len, d_model]向量空间
        k = self.w_k(x)  # 表示 我能提供什么 的[batch, seq_len, d_model]向量空间
        v = self.w_v(x)  # 表示 我实际提供内容 的[batch, seq_len, d_model]向量空间

        # 2、多头拆分
        # 把QKV向量的最后一维 d_model 拆成 (n_heads, head_dim)，即 [batch, seq_len, d_model] → [batch, seq_len, n_heads, head_dim]
        # 拆之前：[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]    ← 一个8维向量
        # 拆之后：
        #   头0: [0.1, 0.2, 0.3, 0.4]    ← 前4个数字
        #   头1: [0.5, 0.6, 0.7, 0.8]    ← 后4个数字

        # 使用 transpose 让索引位1(seq_len)和2(n_heads)交换，数据结构会变成 [batch, seq_len, n_heads, head_dim] → [batch, n_heads, seq_len, head_dim]
        # 目的是让按token分组变为按注意力头分组，便于后续的并行独立计算。
        # 交换前（按 token 分组）：
        #   token0: 头0[0.1,0.2,0.3,0.4], 头1[0.5,0.6,0.7,0.8]
        #   token1: 头0[...],              头1[...]
        #   token2: 头0[...],              头1[...]

        # 交换后（按头分组）：
        #   头0: token0[0.1,0.2,0.3,0.4], token1[...], token2[...]   ← 头0看所有token
        #   头1: token0[0.5,0.6,0.7,0.8], token1[...], token2[...]   ← 头1看所有token
        # 交换后每个注意力头拿到的是所有 token 在自己负责的 n 个维度上的数据，可以独立做 Q·K 点积和加权 V
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # 3、pytorch的scaled_dot_product_attention函数会依次执行 Q·K 点积 → 缩放（÷√d） → 因果掩码 → softmax → dropout → 加权求和
        # 3.1、QK点积：每个自注意力头里，每个token的Q和所有token的K做点积，算出关联度分数，得到关联度矩阵
        # 假设 4 个 token（<s>你好吗），每个头负责token的 4 维向量：
        #   Q_好 = [0.3, 0.1, -0.2, 0.4]
        #   K_<s> = [0.1, -0.1, 0.0, 0.0]
        #   K_你  = [0.4, 0.2, -0.3, 0.3]
        #   K_好  = [0.3, 0.1, -0.2, 0.4]
        #   K_吗  = [0.2, 0.0, -0.1, 0.2]
        # "好"对"<s>"的分数 = Q_好 · K_<s> = 0.3×0.1 + 0.1×(-0.1) + (-0.2)×0.0 + 0.4×0.0 = 0.02
        # "好"对"你"的分数  = Q_好 · K_你  = 0.3×0.4 + 0.1×0.2 + (-0.2)×(-0.3) + 0.4×0.3 = 0.32
        # "好"对"好"的分数  = Q_好 · K_好  = 0.3×0.3 + 0.1×0.1 + (-0.2)×(-0.2) + 0.4×0.4 = 0.30
        # "好"对"吗"的分数  = Q_好 · K_吗  = 0.3×0.2 + 0.1×0.0 + (-0.2)×(-0.1) + 0.4×0.2 = 0.16
        # 每个 token 都这样算，得到一个 4×4 的分数矩阵

        # 3.2、缩放，点积结果 ÷ √head_dim
        # 之所以要做缩放，是因为QK点积过程中，如果向量维度越大，点积结果的绝对值就越大，容易导致后续softmax变得极端，导致模型训练不稳定
        # head_dim = 4 时：  Q·K = a₁b₁ + a₂b₂ + a₃b₃ + a₄b₄           → 4项求和
        # head_dim = 64 时： Q·K = a₁b₁ + a₂b₂ + ... + a₆₄b₆₄          → 64项求和
        # 例：head_dim=64，缩放前分数 [8.0, 3.0, 1.0, 2.0]
        #   softmax → [0.99, 0.01, 0.00, 0.00]  ← 缩放前 softmax 极端到只剩一个 token 有权重
        #   ÷√64=÷8 → [1.0, 0.375, 0.125, 0.25]
        #   softmax → [0.41, 0.22, 0.17, 0.19]  ← 缩放后权重分布会变得温和，模型能综合多个 token 的信息

        # 3.3、因果掩码，覆盖掉未来
        # 点积原始结果 scores[i][j] = Q_i · K_j：        因果掩码后（把右上角替换为 -inf）：
        # K_<s>  K_你  K_好  K_吗                       K_<s>  K_你  K_好  K_吗
        # <s>  → [ 0.5,  0.3,  0.8,  0.2 ]    →    <s>  → [ 0.5, -inf, -inf, -inf ]
        # 你   → [ 0.1,  0.6,  0.4,  0.7 ]    →    你   → [ 0.1,  0.6, -inf, -inf ]
        # 好   → [ 0.2,  0.9,  0.5,  0.3 ]    →    好   → [ 0.2,  0.9,  0.5, -inf ]
        # 吗   → [ 0.3,  0.4,  0.6,  0.8 ]    →    吗   → [ 0.3,  0.4,  0.6,  0.8 ]
        # 训练时，整个序列 <s> 你 好 吗 是一次性喂给模型的，如果不加掩码，模型在预测下一个token的时候，就能直接看到好和吗。
        # 因果掩码强制每个 token 只能看自己和前面的，以便于让模型真正学到预测能力。

        # 3.4、softmax归一化，把因果掩码后的QK关联度分数转为注意力权重(百分比权重) ，点积矩阵中的每一行的各个位置的分数都会被转换为百分比权重
        # [3.1, 0.4, 0.5, -inf] → [0.88, 0.06, 0.07, 0.00]

        # 3.5、dropout是为了避免模型训练过程中过拟合，随机把归一化的注意力权重设置为0

        # 3.6、把QK注意力权重 和 V矩阵 做矩阵乘法，得到融合了当前上下文的新向量激活值
        # "好"的注意力权重：[0.21, 0.50, 0.29, 0.00]
        #                 <s>   你    好    吗
        # V_<s> = [1.0, 0.0, 0.5, 0.2]   ← <s> 的"实际内容"
        # V_你  = [0.3, 0.8, 0.1, 0.6]   ← 你 的"实际内容"
        # V_好  = [0.5, 0.4, 0.7, 0.3]   ← 好 的"实际内容"
        # "好"的新向量 = 0.21×V_<s> + 0.50×V_你 + 0.29×V_好 = [0.51, 0.52, 0.36, 0.43]
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True,
        )

        # 4、concat，把多个头的结果拼接在一起，并转换为回Tensor的形状([batch, seq_len, d_model]三维数组)，也就是还原一个完整的token向量的d_model维的激活值
        #   头1结果: [64维] →
        #   头2结果: [64维]
        #   头3结果: [64维] →  直接拼接  →  [384维]
        #   头4结果: [64维] →
        #   头5结果: [64维] →
        #   头6结果: [64维] →
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )

        # 5、W_O投影，把QK*V矩阵的新激活值和W_O矩阵做矩阵乘法，把多头学到的内容相互融合
        return self.w_o(attn_output)


# ======================================================================
# TransformerBlock层，这里主要是串联各层Block和其各子层的计算
# ======================================================================
class TransformerBlock(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1
    ) -> None:
        super().__init__()

        # RMSNorm → 多头自注意力计算 → 残差连接 → RMSNorm → SwiGLU FFN（前馈网络） → 残差连接
        # ── 前半段子模块：归一化 + 自注意力 ──
        self.attn_norm = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)

        # ── 后半段子模块：归一化 + 前馈网络 ──
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── 前半段：归一化 → 自注意力 → 残差连接 ──
        # 步骤 A1: 归一化（拉回正常范围）
        # 步骤 A2: 多头因果自注意力（token 之间交换信息）
        # 步骤 A3: 残差连接（原始 x + 注意力输出）
        x = x + self.attn(self.attn_norm(x))

        # ── 后半段：归一化 → 前馈网络 → 残差连接 ──
        # 步骤 B1: 归一化
        # 步骤 B2: SwiGLU FFN（每个 token 独立消化信息）
        # 步骤 B3: 残差连接
        x = x + self.ffn(self.ffn_norm(x))

        return x


# ──── ④ 阅读完毕 ──── 下一步请阅读 ⑤ NovaModel ──────────────────────


# ======================================================================
# ⑤ NovaModel —— 完整的 Decoder-Only Transformer 模型（阅读顺序第 5，依赖 ④）
# ======================================================================
class NovaModel(nn.Module):
    """Nova — 完整的 Decoder-Only Transformer 模型。

    这是整个模型的"总指挥"，把前面所有子组件串起来，完成从
    "整数 token ID" 到 "每个位置对词表的概率预测（logits）" 的完整变换。

    ┌──────────────────────────────────────────────────────────────────┐
    │                    完整前向传播数据流                               │
    │                                                                  │
    │  训练阶段（train.py 调用）:                                       │
    │    input_ids = batch["input_ids"]           # [batch, seq_len]   │
    │    logits = model(input_ids)                # ← 调用本类         │
    │    loss = CrossEntropyLoss(logits, targets)                      │
    │    loss.backward()                                               │
    │                                                                  │
    │  推理阶段（chat.py 调用）:                                        │
    │    logits = model(input_ids)                # ← 调用本类         │
    │    next_token = sample(logits[:, -1, :])    # 只看最后一个位置    │
    │                                                                  │
    │  ┌──────────────────────────────────────────────────────────┐     │
    │  │                model.forward(input_ids) 内部:            │     │
    │  │                                                          │     │
    │  │  input_ids: [batch, seq_len]                             │     │
    │  │       │                                                  │     │
    │  │  ① Token Embedding: 查字义表                             │     │
    │  │       token_emb = self.token_emb(input_ids)              │     │
    │  │       → [batch, seq_len, d_model]                        │     │
    │  │       │                                                  │     │
    │  │  ② Position Embedding: 查位置表                          │     │
    │  │       pos_emb = self.pos_emb(positions)                  │     │
    │  │       → [seq_len, d_model]                               │     │
    │  │       │                                                  │     │
    │  │  ③ 相加: x = token_emb + pos_emb                        │     │
    │  │       每个 token 既知道"它是谁"也知道"它在哪"            │     │
    │  │       → [batch, seq_len, d_model]                        │     │
    │  │       │                                                  │     │
    │  │  ④ Dropout: 防过拟合                                     │     │
    │  │       │                                                  │     │
    │  │  ⑤ TransformerBlock × n_layers 层                        │     │
    │  │       每层: RMSNorm→Attention→+x→RMSNorm→FFN→+x         │     │
    │  │       → [batch, seq_len, d_model]                        │     │
    │  │       │                                                  │     │
    │  │  ⑥ Final RMSNorm: 最后一次归一化                         │     │
    │  │       → [batch, seq_len, d_model]                        │     │
    │  │       │                                                  │     │
    │  │  ⑦ Output Linear: d_model → vocab_size                  │     │
    │  │       → [batch, seq_len, vocab_size]                     │     │
    │  │       │                                                  │     │
    │  │       ▼                                                  │     │
    │  │  logits（每个位置对词表中每个字的"得分"）                 │     │
    │  └──────────────────────────────────────────────────────────┘     │
    └──────────────────────────────────────────────────────────────────┘

    Token Embedding（字义表）
    ────────────────────────
    一张 vocab_size × d_model 的查找表（例如 ~1000 × 128）。
    每一行对应一个 token（字/特殊标记），存储该 token 的 128 维语义向量。

    初始时随机填充（N(0, 0.02)），训练过程中通过反向传播不断更新——
    意思相近的字的向量会逐渐靠近，意思不同的会远离。

    查表操作: token_emb(input_ids)
      input_ids = [1, 42, 15]  →  查第 1、42、15 行
      返回 3 个 128 维向量（拼成 [3, 128] 的矩阵）

    Position Embedding（位置表）
    ──────────────────────────
    一张 max_seq_len × d_model 的查找表（128 × 128）。
    第 i 行存储"在位置 i"这个信息的 128 维向量。

    为什么需要？Transformer 的自注意力是"无序"的——它只看内容，不看顺序。
    如果不加位置信息，"你好吗" 和 "吗好你" 对模型来说是一样的。
    加上位置嵌入后，同一个字在不同位置会得到不同的向量（字义 + 位置信息），
    模型就能区分顺序了。

    两个嵌入相加
    ────────────
    x = token_emb + pos_emb

    例如 "你" 在位置 1:
      字义向量: [0.12, -0.34, 0.56, ...]   ← "你"是谁
      位置向量: [0.04, -0.01, 0.05, ...]   ← 它在第 1 位
      ────────────────────────────────────
      相加得到: [0.16, -0.35, 0.61, ...]   ← 既知道是"你"，也知道在第 1 位

    输出层
    ──────
    最终的 Linear 层把 128 维向量映射到 vocab_size 维:
      128 个数字 → ~1000 个数字（每个数字是对应字的"得分"）

    训练时用 CrossEntropyLoss 和 target_ids 计算损失。
    推理时取最后一个位置的 logits 做 softmax 采样下一个字。

    参数量总览
    ──────────
    假设 vocab_size = 1000:
      Token Embedding:    1000 × 128              = 128,000
      Position Embedding: 128 × 128               = 16,384
      TransformerBlock × 4: 4 × 262,400           = 1,049,600
      Final RMSNorm:      128                     = 128
      Output Linear:      128 × 1000              = 128,000
      ──────────────────────────────────────────────────────
      总计:               ~1,322,112（约 1.3M 参数）
    """

    def __init__(self, config: NovaConfig) -> None:
        super().__init__()
        self.config = config

        # ── ① Token Embedding: 字义表 ──
        # vocab_size 行，每行 d_model 维
        # 把整数 token ID 映射为稠密向量
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)

        # ── ② Position Embedding: 位置表 ──
        # max_seq_len 行，每行 d_model 维
        # 把位置编号 0, 1, 2, ... 映射为位置向量
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)

        # ── ③ Dropout: 嵌入层之后的正则化 ──
        self.emb_dropout = nn.Dropout(config.dropout)

        # ── ④ N 层 TransformerBlock ──
        # ModuleList 让 PyTorch 知道这些是子模块（参数会被自动注册）
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.d_model, config.n_heads, config.d_ff, config.dropout
                )
                for _ in range(config.n_layers)
            ]
        )

        # ── ⑤ Final RMSNorm: 所有 Block 之后、输出层之前的最终归一化 ──
        self.final_norm = RMSNorm(config.d_model)

        # ── ⑥ Output Linear: 输出投影层 ──
        # 把 d_model 维向量映射到 vocab_size 维（每个字一个得分）
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # ── ⑤a 权重初始化 ──
        self._init_weights()

    # ------------------------------------------------------------------
    # ⑤a 权重初始化（在 NovaModel.__init__ 末尾调用）
    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        """对所有子模块执行权重初始化。

        为什么要做初始化？
        ─────────────────
        PyTorch 默认的初始化方式（nn.Linear 用 Kaiming Uniform，nn.Embedding
        用 N(0,1)）对 Transformer 来说不一定最优。好的初始化能让：
          1. 训练初期 loss 下降更快（参数起点离最优点更近）
          2. 梯度大小合理（不暴涨也不消失）
          3. 各层输出的数值范围一致（不会出现某层输出极大/极小）

        本项目的初始化策略（与 GPT-2 / LLaMA 一致）
        ──────────────────────────────────────────────
        1. nn.Embedding（字义表 + 位置表）：
             正态分布 N(0, 0.02)
             为什么不用默认的 N(0,1)？
             → 标准差 1 太大，128 维向量的 L2 范数会很大（约 √128 ≈ 11），
               导致嵌入层输出数值过大，后续计算不稳定。
               0.02 使范数约 0.02×√128 ≈ 0.23，温和得多。

        2. nn.Linear（所有投影矩阵：Q/K/V/O、FFN 的 W1/W2/W3、输出层）：
             Xavier 均匀分布: U(-√(6/(fan_in+fan_out)), √(6/(fan_in+fan_out)))
             核心思想：让输入和输出的方差保持一致。
             例如 W_Q [128, 128]：范围约 ±0.153
             例如 FFN W1 [128, 512]：范围约 ±0.097

        3. RMSNorm 的 gamma：
             全 1 初始化（已在 RMSNorm.__init__ 中完成）。
             初始时归一化后不做任何缩放，等训练来调整。

        调用时机:
          在 NovaModel.__init__ 的最后一步:
            self._init_weights()
          即模型构造完成后、第一次 forward 之前，统一做一遍初始化。
        """
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                # 嵌入层: N(0, 0.02)
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Linear):
                # 线性层: Xavier 均匀分布
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            # RMSNorm 的 gamma 已在其 __init__ 中初始化为全 1，无需再处理

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """前向传播：从 token ID 到 logits 的完整变换。

        参数:
          input_ids : LongTensor，形状 [batch_size, seq_len]
              例如 [16, 128]，每个元素是 0~vocab_size-1 的整数
              来源: dataset.py 中 NovaDataset 生成的 "input_ids" 字段

        返回:
          Tensor，形状 [batch_size, seq_len, vocab_size]
              每个位置对词表中每个字的"得分"（未经 softmax 的 raw logits）

        调用链路:
          训练阶段 (train.py):
            for batch in dataloader:
                input_ids = batch["input_ids"]                  # [batch, seq_len]
                target_ids = batch["target_ids"]                # [batch, seq_len]
                logits = model(input_ids)                       # ← 本方法
                loss = F.cross_entropy(
                    logits.view(-1, vocab_size),
                    target_ids.view(-1),
                    ignore_index=-100,
                )
                loss.backward()

          推理阶段 (chat.py):
            logits = model(input_ids)                           # ← 本方法
            next_token_logits = logits[:, -1, :]                # 最后位置的得分
            next_token_id = top_k_sample(next_token_logits)     # 采样下一个字
        """
        batch_size, seq_len = input_ids.shape

        # ── 步骤 1: Token Embedding（查字义表）──
        # 每个整数 ID 查出对应的 128 维向量
        # input_ids: [batch, seq_len] → token_emb: [batch, seq_len, d_model]
        token_emb = self.token_emb(input_ids)

        # ── 步骤 2: Position Embedding（查位置表）──
        # 生成位置编号 [0, 1, 2, ..., seq_len-1]
        # 每个位置编号查出对应的 128 维位置向量
        positions = torch.arange(seq_len, device=input_ids.device)
        pos_emb = self.pos_emb(positions)  # [seq_len, d_model]，广播加到每个 batch

        # ── 步骤 3: 字义 + 位置 = 初始工作向量 ──
        # 从这一步开始，模型不再看原始文字了，后续全部在操作这些向量
        x = token_emb + pos_emb  # [batch, seq_len, d_model]

        # ── 步骤 4: Dropout ──
        # 训练时随机丢弃一部分嵌入维度，迫使模型不过度依赖某几个特征
        x = self.emb_dropout(x)

        # ── 步骤 5: N 层 TransformerBlock ──
        # 每层做一次"讨论 + 消化":
        #   第 0 层: 学字符搭配（"好吗"经常连在一起）
        #   第 1 层: 学句法结构（这是一个问句）
        #   第 2 层: 学语义（有人在问候我）
        #   第 3 层: 学意图（我应该用问候语回答）
        for block in self.blocks:
            x = block(x)  # [batch, seq_len, d_model] → [batch, seq_len, d_model]

        # ── 步骤 6: Final RMSNorm ──
        # 4 层 Block 处理完后，数值可能又飘了，最后归一化一次
        x = self.final_norm(x)  # [batch, seq_len, d_model]

        # ── 步骤 7: 输出投影层 ──
        # 把 128 维向量映射到 vocab_size 维（每个字一个得分）
        logits = self.output(x)  # [batch, seq_len, vocab_size]

        return logits

    # ------------------------------------------------------------------
    # ⑤b 参数量统计（训练开始前调用，打印各层参数分布）
    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        """统计总的可学习参数量。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def print_parameter_summary(self) -> None:
        """打印各层参数量的详细统计。

        输出示例:
        ┌──────────────────────────────────────────────────────────┐
        │  Nova 模型参数统计                                        │
        ├────────────────────────────────┬───────────┬─────────────┤
        │ 组件                           │ 参数量     │ 占比        │
        ├────────────────────────────────┼───────────┼─────────────┤
        │ Token Embedding                │   128,000 │   9.69%     │
        │ Position Embedding             │    16,384 │   1.24%     │
        │ Block 0 - attn_norm            │       128 │   0.01%     │
        │ Block 0 - attn                 │    65,536 │   4.96%     │
        │ ...                            │           │             │
        │ Final RMSNorm                  │       128 │   0.01%     │
        │ Output Linear                  │   128,000 │   9.69%     │
        ├────────────────────────────────┼───────────┼─────────────┤
        │ 总计                           │ 1,322,112 │ 100.00%     │
        └────────────────────────────────┴───────────┴─────────────┘

        调用时机:
          训练开始前 (train.py):
            model = NovaModel(config)
            model.print_parameter_summary()    # ← 查看参数分布
        """
        total = self.count_parameters()

        rows: list[tuple[str, int]] = []

        rows.append(("Token Embedding", self.token_emb.weight.numel()))
        rows.append(("Position Embedding", self.pos_emb.weight.numel()))

        for i, block in enumerate(self.blocks):
            rows.append(
                (
                    f"Block {i} - attn_norm",
                    sum(p.numel() for p in block.attn_norm.parameters()),
                )
            )
            rows.append(
                (
                    f"Block {i} - attn",
                    sum(p.numel() for p in block.attn.parameters()),
                )
            )
            rows.append(
                (
                    f"Block {i} - ffn_norm",
                    sum(p.numel() for p in block.ffn_norm.parameters()),
                )
            )
            rows.append(
                (
                    f"Block {i} - ffn",
                    sum(p.numel() for p in block.ffn.parameters()),
                )
            )

        rows.append(
            ("Final RMSNorm", sum(p.numel() for p in self.final_norm.parameters()))
        )
        rows.append(("Output Linear", self.output.weight.numel()))

        name_width = max(len(r[0]) for r in rows) + 2
        print()
        print("=" * (name_width + 30))
        print("  Nova 模型参数统计")
        print("=" * (name_width + 30))
        print(f"  {'组件':<{name_width}} {'参数量':>12}  {'占比':>8}")
        print("-" * (name_width + 30))
        for name, count in rows:
            pct = count / total * 100 if total > 0 else 0
            print(f"  {name:<{name_width}} {count:>12,}  {pct:>7.2f}%")
        print("-" * (name_width + 30))
        print(f"  {'总计':<{name_width}} {total:>12,}  100.00%")
        print("=" * (name_width + 30))
        print()


# ══════════════════════════════════════════════════════════════════════
# ✅ model.py 全部阅读完毕！
# 建议下一步阅读: train.py — 了解模型是如何被训练的
# ══════════════════════════════════════════════════════════════════════
