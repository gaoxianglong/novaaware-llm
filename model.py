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
│   ├── Token Embedding    查字义表: token ID → n维向量              │
│   ├── Dropout                                                       │
│   ├── TransformerBlock × 4 层（④）                                   │
│   │    ├── RMSNorm（①）→ MultiHeadAttention（③ + RoPE）→ 残差       │
│   │    └── RMSNorm（①）→ SwiGLUFFN（②）→ 残差                      │
│   ├── RMSNorm（最终归一化）                                          │
│   └── Linear 输出层: n维 → vocab_size                             │
│       │                                                             │
│       ▼                                                             │
│  输出 logits: [batch_size, seq_len, vocab_size]                     │
└─────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NovaConfig


# ======================================================================
# 归一化处理
# Nova的RMSNorm 被调用了 9 次（每层 Block 调 2 次 × 4 层 + 最终 1 次）
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

    # 归一化的核心作用，就是在关键计算节点前先把输入向量的整体数值尺度校准到一个稳定的范围内，让每一层都尽量工作在可控的数值区间里，从而提升模型训练和推理的稳定性。
    def __init__(
        self,
        # d_model向量维度
        dim: int,
        # 防止输入向量全是0导致除数为0
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        # Pytorch注册了一个d_model维度的gamma一维数组
        # 在反向传播过程中，模型会根据loss值计算出gamma的梯度值，优化器会根据梯度值和config中预设的超参LR在训练过程中逐步调整gamma参数
        self.gamma = nn.Parameter(
            # 初始参数全为1
            torch.ones(dim)
        )  # Parameter是Tensor的子类

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 计算均方根，先平方、再求平均、再开方。
        # 拿一个 token 的全部 d_model 维数值算出一个数
        # 假设 d_model=4，某个 token 的向量是 [2.0, -1.0, 3.0, 0.5]
        # 第1步 平方:     [4.0, 1.0, 9.0, 0.25]
        # 第2步 求平均:   (4.0 + 1.0 + 9.0 + 0.25) / 4 = 3.5625
        # 第3步 加 eps:   3.5625 + 0.000001 = 3.562501
        # 第4步 开方:     √3.562501 = 1.888
        # rms = 1.888  ← 一个数
        rms = torch.sqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)

        # 归一化计算
        # 用一个rms去归一化整个向量 [2.0, -1.0, 3.0, 0.5] / 1.888 = [1.06, -0.53, 1.59, 0.26]
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

        # 压缩回 d_model 维，≈0的激活值跟W2通路矩阵乘法后目标位置的激活值也是≈0，没什么用，会被过滤掉
        # 压缩回d_model维度的输入向量也需要和W2通路的参数进行矩阵乘法，因为压缩方式不同会导致loss值不同
        return self.w2(gated)


# ======================================================================
# RoPE（旋转位置编码）
# 不维护可训练的位置表，而是在 Attention 内部用数学公式对 Q 和 K 向量做旋转，
# 通过旋转角度来编码位置信息。旋转使得 Q·K 点积结果只取决于两个 token 之间的
# 相对距离（m - n），而非各自的绝对位置，因此天然支持上下文长度外推。
# ======================================================================
def precompute_rope_freqs(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    scale_factor: float | None = None,
) -> torch.Tensor:
    """预计算 RoPE 的旋转频率矩阵。

    返回 shape [max_seq_len, head_dim // 2] 的复数张量，
    每个元素是 e^(i * m * θ_k)，表示位置 m 在第 k 对维度上的旋转角度。
    """
    # 每对维度的基础频率: θ_i = 1 / (10000 ^ (2i / head_dim))
    # 低维度对 → θ 大 → 旋转快（捕捉近距离）
    # 高维度对 → θ 小 → 旋转慢（捕捉远距离）
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))

    # 位置索引 [0, 1, 2, ..., max_seq_len - 1]
    t = torch.arange(max_seq_len, dtype=torch.float32)

    # 位置插值：将位置压缩回训练范围
    # 例: scale_factor=4.0 时，[0,1,2,...,2047] → [0, 0.25, 0.5, ..., 511.75]
    if scale_factor is not None:
        t = t / scale_factor

    # 外积得到每个位置在每对维度上的旋转角度
    # angles[m, k] = m × θ_k
    angles = torch.outer(t, freqs)  # [max_seq_len, head_dim // 2]

    # 转为复数形式 e^(i×angle) = cos(angle) + i×sin(angle)
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    return freqs_cis


def apply_rotary_emb(
    q: torch.Tensor,          # [batch, n_heads, seq_len, head_dim]
    k: torch.Tensor,          # [batch, n_heads, seq_len, head_dim]
    freqs_cis: torch.Tensor,  # [seq_len, head_dim // 2]
) -> tuple[torch.Tensor, torch.Tensor]:
    """对 Q 和 K 施加旋转位置编码（RoPE）。

    将 head_dim 维向量的相邻两个维度配对，视为复数的实部和虚部，
    然后和 freqs_cis 做复数乘法实现旋转。

    例: head_dim=64 的向量被切成 32 对
        (x0,x1), (x2,x3), ..., (x62,x63)
        每对视为复数 x0 + i*x1，乘以 e^(i*angle) 实现旋转
    """
    # [batch, n_heads, seq_len, head_dim]
    # → [batch, n_heads, seq_len, head_dim//2, 2]
    # → complex [batch, n_heads, seq_len, head_dim//2]
    q_complex = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_complex = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))

    # 广播旋转频率到 [1, 1, seq_len, head_dim//2]
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)

    # 复数乘法 = 二维旋转
    q_rotated = torch.view_as_real(q_complex * freqs_cis).flatten(-2)
    k_rotated = torch.view_as_real(k_complex * freqs_cis).flatten(-2)

    return q_rotated.type_as(q), k_rotated.type_as(k)


# ======================================================================
# 多头自注意力计算
# Block中的多头自注意力计算，本质就是将 输入向量*W_Q、W_K、W_V做矩阵运算，得到QKV线性投影，
# 然后进行Q·K 点积 → 缩放（÷√d） → 因果掩码 → softmax → dropout → 加权 V → 新向量，最终得到了融合当前上下文的新向量激活值。
# 完整流程图:
#     x ─┬─→ W_Q ──→ Q ─→ RoPE(Q) ─┐
#        ├─→ W_K ──→ K ─→ RoPE(K) ─┼─→ Q@K^T/√d ─→ +mask ─→ softmax ─→ drop ─→ ×V ─→ concat ─→ W_O ─→ out
#        └─→ W_V ──→ V ────────────┘
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
        # bias=False被设置为无偏置，这其实是LLaMA的做法，因为偏置项对模型训练影响不大，去掉可以减少训练参数
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

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
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

        # 2.5、RoPE 旋转位置编码：在注意力计算之前，对 Q 和 K 施加旋转
        # 位置信息不再通过 pos_emb 加到输入向量上，而是在 Attention 内部通过旋转 Q、K 注入
        # V 不旋转，因为位置信息只需要影响"谁和谁相关"（Q·K），不需要影响"提供什么内容"（V）
        q, k = apply_rotary_emb(q, k, freqs_cis)

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
        # 因果掩码强制一句话中每个位置的 token 只能看自己和前面的，以便于让模型真正学到预测能力。

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

        # 5、O投影，把QK*V矩阵的新激活值和W_O矩阵做矩阵乘法，把多头学到的内容相互融合
        return self.w_o(attn_output)


# ======================================================================
# TransformerBlock层，基于Pre-LN的归一化放置策略
# RMSNorm → 多头自注意力计算 → 残差连接 → RMSNorm → SwiGLU FFN（前馈网络） → 残差连接
# ======================================================================
class TransformerBlock(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1
    ) -> None:
        super().__init__()

        # RMSNorm、多头自注意力计算、FFN声明
        self.attn_norm = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ff)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        # 1、先做RMSNorm -> 多头自注意力计算（含RoPE旋转） -> 残差连接
        x = x + self.attn(self.attn_norm(x), freqs_cis)

        # 2、再做RMSNorm -> FFN -> 残差连接
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ======================================================================
# Decoder-Only Transformer 模型，做工作流串联
# Token Embedding -> Dropout -> TransformerBlock × n_layers 层（含 RoPE） -> Final RMSNorm -> Output Linear
# ======================================================================
class NovaModel(nn.Module):
    def __init__(self, config: NovaConfig) -> None:
        super().__init__()
        self.config = config

        # 初始化token_embedding，长度为vocab_size
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)

        # 预计算 RoPE 旋转频率，注册为 buffer（不参与梯度计算，但会随模型保存/加载/to(device)）
        head_dim = config.d_model // config.n_heads
        freqs_cis = precompute_rope_freqs(
            head_dim,
            config.max_seq_len,
            theta=config.rope_theta,
            scale_factor=config.rope_scale_factor,
        )
        self.register_buffer("freqs_cis", freqs_cis)

        # 初始化dropout,这个和多头自注意力计算中丢的东西不同
        # 多头自注意力计算中丢的是softmax计算后的注意力权重，这里丢的是最初输入向量的部分维度
        self.emb_dropout = nn.Dropout(config.dropout)

        # 将4层TransformerBlock注册到pytorch上
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.d_model, config.n_heads, config.d_ff, config.dropout
                )
                for _ in range(config.n_layers)
            ]
        )

        # 声明RMSNorm，这里是在所有TransformerBlock之后，输出层之前执行的final归一化操作
        self.final_norm = RMSNorm(config.d_model)

        # 声明Output Linear输出投影层
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # 执行各个模型参数的初始化
        self._init_weights()

    # ------------------------------------------------------------------
    # 模型各个参数的初始化动作
    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                # nn.embedding初始化
                # 把 embedding 表里的每个值，用均值为 0、标准差为 0.02 的正态分布随机填充。大部分初始值会落在 -0.04 ~ 0.04 之间（2 倍标准差范围）：
                # 初始化前：[0, 0, 0, 0, ...]           ← 空的
                # 初始化后：[0.01, -0.03, 0.02, -0.01, ...]  ← 很小的随机数
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Linear):
                # MultiHeadAttention 里的:  w_q, w_k, w_v, w_o  → 都是 nn.Linear → Xavier 初始化
                # SwiGLUFFN 里的:          w1, w2, w3           → 都是 nn.Linear → Xavier 初始化
                # Output 输出层投影矩阵:     self.output          → 也是 nn.Linear → Xavier 初始化
                # RMSNorm 的 gamma 已在其 __init__ 中初始化为全 1
                nn.init.xavier_uniform_(
                    module.weight
                )  # 初始化会根据每层矩阵的输入和输出维度，自动计算一个合适的随机范围来填充初始权重
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # seq_len表示当前这个序列有多少个 token
        batch_size, seq_len = input_ids.shape

        # 根据input_ids，查询token_embedding表，得到每一个token的token_emb([batch, seq_len, d_model]三维数组)
        # RoPE 方案下不再查位置表、不再做 token_emb + pos_emb 相加，位置信息在 Attention 内部通过旋转注入
        x = self.token_emb(input_ids)

        # 每次训练进入Block前随机丢弃一部分嵌入维度，迫使模型不过度依赖某几个特征，避免过拟合
        x = self.emb_dropout(x)

        # 截取当前序列长度对应的旋转频率
        freqs_cis = self.freqs_cis[:seq_len]

        # 执行4层TransformerBlock计算,堆叠层数越深，模型对语义的理解会越深，训练效果越好
        for block in self.blocks:
            x = block(x, freqs_cis)  # [batch, seq_len, d_model]

        # 最终RMSNorm归一化,保持向量空间的尺度最终稳定
        x = self.final_norm(x)  # [batch, seq_len, d_model]

        # 输出投影层
        # 每一句话中的每一个token都有一个vocab_size的打分表
        # 位置2(好)的向量: [0.45, 0.22, 0.11, ..., -0.05]   ← 这个向量"知道"前面是<s>你好
        #     ↓
        # × output权重矩阵 (d_model → vocab_size)
        #     ↓
        # vocab_size个分数: [<s>:-0.8, 你:1.3, 好:0.5, 吗:9.2, ...]  → 预测下一个是"吗"
        # Block层让每个 token 融合上下文得到 n 维的新向量，然后输出投影层和每个 token 的 n 维向量做矩阵乘法，
        #   得到 vocab_size 个分数，这些分数表示该位置下一个 token 最可能的打分,这里是原始数据，后面会用softmax计算出概率分布。
        logits = self.output(x)  # [batch, seq_len, vocab_size]
        return logits

    # ------------------------------------------------------------------
    # 参数量统计（训练开始前调用，打印各层参数分布）
    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        """统计总的可学习参数量。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def print_parameter_summary(self) -> None:
        """
        ┌──────────────────────────────────────────────────────────┐
        │  Nova 模型参数统计                                        │
        ├────────────────────────────────┬───────────┬─────────────┤
        │ 组件                           │ 参数量     │ 占比        │
        ├────────────────────────────────┼───────────┼─────────────┤
        │ Token Embedding                │   128,000 │   9.69%     │
        │ RoPE (buffer, 不参与训练)       │     4,096 │     N/A     │
        │ Block 0 - attn_norm            │       128 │   0.01%     │
        │ Block 0 - attn                 │    65,536 │   4.96%     │
        │ ...                            │           │             │
        │ Final RMSNorm                  │       128 │   0.01%     │
        │ Output Linear                  │   128,000 │   9.69%     │
        ├────────────────────────────────┼───────────┼─────────────┤
        │ 总计                           │ 1,272,960 │ 100.00%     │
        └────────────────────────────────┴───────────┴─────────────┘

        调用时机:
          训练开始前 (train.py):
            model = NovaModel(config)
            model.print_parameter_summary()    # ← 查看参数分布
        """
        total = self.count_parameters()

        rows: list[tuple[str, int]] = []

        rows.append(("Token Embedding", self.token_emb.weight.numel()))

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

        rope_label = "RoPE freqs_cis (buffer)"
        rope_numel = self.freqs_cis.numel()

        name_width = max(max(len(r[0]) for r in rows), len(rope_label)) + 2
        print()
        print("=" * (name_width + 30))
        print("  Nova 模型参数统计")
        print("=" * (name_width + 30))
        print(f"  {'组件':<{name_width}} {'参数量':>12}  {'占比':>8}")
        print("-" * (name_width + 30))
        for name, count in rows:
            pct = count / total * 100 if total > 0 else 0
            print(f"  {name:<{name_width}} {count:>12,}  {pct:>7.2f}%")
        print(f"  {rope_label:<{name_width}} {rope_numel:>12,}  {'N/A':>8}")
        print("-" * (name_width + 30))
        print(f"  {'总计(可训练)':<{name_width}} {total:>12,}  100.00%")
        print("=" * (name_width + 30))
        print()
