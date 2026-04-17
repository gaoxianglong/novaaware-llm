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
# RoPE旋转位置编码
# 计算每一个位置的真实旋转角度，并生成旋转系数表
# ======================================================================
def precompute_rope_freqs(
    head_dim: int,
    max_seq_len: int,
    # 控制单位旋转角度的基数
    theta: float = 10000.0,
    # 位置插值的缩放因子
    scale_factor: float | None = None,
) -> torch.Tensor:
    # 一、单位旋转角度的计算：
    # 以 head_dim 为区间、步长为 2，算出 32 个组编号 [0, 2, 4, ..., 62]，然后每个组编号除以 head_dim 得到 0~1 之间的比例。
    # 再用基数 10000 对这个比例做幂运算，最后取倒数，得到每组的单位旋转角度。最终效果是 32 个单位角度从 1.0 指数递减到 0.000132，前面的组步长大（捕捉近距离），后面的组步长小（捕捉远距离）。

    # 计算 32 个单位旋转角度（每往后走 1 个位置，该组转多少度）
    # 计算过程：freqs = 1 / (theta ^ 比例)，分3步：
    #
    #   1、torch.arange(0, head_dim, 2).float() / head_dim算比例：64 个维度两两配对分成 32 组，组编号 [0,2,4,...,62] 除以 64 得到 0~1 之间的比例
    #      [0/64, 2/64, 4/64, ..., 62/64] = [0.0, 0.03125, 0.0625, ..., 0.96875]
    #
    #   2、算基数的幂：10000 ^ 比例，比例越大结果越大
    #      10000^0.0 = 1 → 10000^0.03 = 1.318 → ... → 10000^0.5 = 100 → ... → 10000^0.97 = 7586
    #
    #   3、取倒数得到单位旋转角度：
    #      1/1 = 1.0（大步长，近距离敏感） → 1/100 = 0.01 → 1/7586 = 0.000132（小步长，远距离敏感）
    #
    # 最终 32 个单位角度从 1.0 指数递减到 0.000132，32 组分工协作覆盖从近到远的距离感知
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))

    # 二、位置插值计算
    # 1、先算扩展位置表的长度，max_seq_len * scale_factor(缩放因子)
    extended_len = (
        int(max_seq_len * scale_factor) if scale_factor is not None else max_seq_len
    )

    # 2、生成位置编号序列t，比如[0,1,2,3...extended_len-1]
    #   假设extended_len = 256，t = [0,1,...,255] / 2.0 = [0, 0.5, ..., 127.5]，表有 256 行（覆盖 2 倍上下文），将真实旋转角度 落在训练见过的 [0, 127.5] 范围内
    t = torch.arange(
        extended_len, dtype=torch.float32
    )  # freqs_cis用float32保证精度，没有任何代价(不参与反向传播，不消耗训练显存，也不影响训练速度)

    # 每个位置的真实旋转角度 = 位置 x 单位旋转角度，夹角(角度差)的计算方式是 (m-n)θ，这意味着scale_factor越大，相邻位置的角夹越小，模型越难区分相邻token
    #   以步长最大的组（θ_0=1.0）为例，相邻位置的角度差 = 1.0 / scale_factor：
    #   scale_factor=1 → 角度差 = 1.0    → "我" 和 "爱" 转了整整 1 弧度，差异巨大，轻松区分
    #   scale_factor=2 → 角度差 = 0.5    → 差异缩小一半，还能分清
    #   scale_factor=4 → 角度差 = 0.25   → 差异只剩 1/4，开始吃力
    #   scale_factor=8 → 角度差 = 0.125  → 差异很小，模型难以分辨相邻 token 的位置
    #   而步长最小的组（θ_31≈0.000132）情况更糟，scale_factor=8 → 角度差 = 0.0000165，几乎重叠，位置信号基本消失
    #   一般 2~4 倍效果不错，超过 8 倍质量明显下降
    if scale_factor is not None:
        # 3、除以缩放因子把每个位置的 真实旋转角度 映射回训练范围
        t = t / scale_factor

    # 三、计算每个位置的真实旋转角度
    # t     = [0, 1, 2, ..., extended_len-1]  ← extended_len 个位置编号（有 scale_factor 时已缩放）
    # freqs = [1.0, 0.759, ..., 0.000132]       ← 32 个单位旋转角度（每个位置转多少度）
    angles = torch.outer(
        t, freqs
    )  # [extended_len, head_dim / 2] 每一个位置的32组真实旋转角度

    # 四、把每个位置的每组角度，预先算好对应的 cos 和 sin，生成转系数表（没有这一步QK无法旋转，角度是几何概念，旋转是数学概念）
    # 转为复数形式 e^(i×angle) = cos(angle) + i×sin(angle)    # torch.polar(1.0, 角度1.0) = 0.540 + 0.841i
    #                          ↑         ↑
    #                        系数A     系数B
    # 旋转公式（手动算）:
    #   q0' = q0 × 0.540 − q1 × 0.841
    #   q1' = q0 × 0.841 + q1 × 0.540
    # 用复数计算:
    #   (q0 + q1·i) × (0.540 + 0.841·i) = q0' + q1'·i
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    # freqs_cis是通过复数计算出来的旋转系数表
    return freqs_cis  # [extended_len, head_dim / 2] 有scale_factor时表更长，覆盖扩展后的上下文范围


# 旋转Q和K向量中的head_dim维向量
def apply_rotary_emb(
    q: torch.Tensor,  # [batch, n_heads, seq_len, head_dim]
    k: torch.Tensor,  # [batch, n_heads, seq_len, head_dim]
    freqs_cis: torch.Tensor,  # [seq_len, head_dim / 2]  预计算好的旋转系数表
) -> tuple[torch.Tensor, torch.Tensor]:

    # 1、打包：把 q, k 转成 float32，然后把最后一维两两配对，塞进 PyTorch 的复数容器
    #        [batch, n_heads, seq_len, head_dim] → [batch, n_heads, seq_len, head_dim//2]（每个元素是一个复数）
    #        例如 head_dim=4 时：[a, b, c, d] → [(a,b), (c,d)] → [a+bi, c+di]
    q_complex = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_complex = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))

    # 2、对齐：freqs_cis 只有 [seq_len, half_dim] 两个维度，q 有 [batch, n_heads, seq_len, half_dim] 四个维度
    #        给 freqs_cis 前面补两个 1，变成 [1, 1, seq_len, half_dim]，这样同一张旋转表被所有 batch 和 head 复用
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)

    # 3、旋转 + 拆包：
    #   旋转：逐元素乘法，每个位置的每组配对都乘上对应的旋转系数
    #        位置0 的系数和位置1 的不同，所以旋转后的结果就带上了位置信息
    #        旋转前：位置0 [a, b, c, d]    位置1 [a, b, c, d]  ← 假设内容一样
    #        旋转后：位置0 [a',b',c',d']   位置1 [a'',b'',c'',d'']  ← 结果不同了，位置信息就在这里
    #   拆包：把复数容器拆回普通实数 tensor，再合并回 head_dim 维度
    q_rotated = torch.view_as_real(q_complex * freqs_cis).flatten(-2)
    k_rotated = torch.view_as_real(k_complex * freqs_cis).flatten(-2)

    # 4、恢复精度：从 float32 转回原来的精度（比如 bfloat16），输出形状和输入完全一样
    # 返回旋转后的QK的head_dim维向量
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

        # 2.5、RoPE 旋转位置编码：按每个位置的旋转角度旋转 Q 和 K，编入绝对位置信息（旋转后的值）；相对位置信息在后续 QK 点积时自动涌现
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

        # 计算出每一个自注意力头计算的向量维度
        head_dim = config.d_model // config.n_heads

        # 计算出max_seq_len x head_dim/2个旋转角度，tensor的shape为[max_seq_len, head_dim/2]
        # 这里要搞清楚一件事情，一个token的维度是384，那么6个自注意力头，每个头负责64维向量空间，
        # 旋转是几何操作，一个平面需要2个数字，一个位置有head_dim/2个旋转角度，相同位置的token共用一套旋转角度
        # Q 向量的 64 个数字:
        # [q0, q1,  q2, q3,  q4, q5,  q6, q7,  ...,  q62, q63]
        #  \_____/  \_____/  \_____/  \_____/         \_______/
        #   第0组    第1组    第2组    第3组    ...     第31组
        freqs_cis = precompute_rope_freqs(
            head_dim,
            config.max_seq_len,
            # 控制单位旋转角度的基数
            theta=config.rope_theta,
            # 位置插值缩放因子，训练时不使用，推理时使用x2或x4扩展上下文长度
            scale_factor=config.rope_scale_factor,
        )
        # 把 freqs_cis 注册为模型的 buffer
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

        # 截取当前序列长度对应的旋转系数,跟实际输入长度对齐，否则维度不匹配会报错
        # 训练时对其填充或裁剪到max_seq_len长度用不上，这里主要是推理的时候用，假设seq_len=10，那么只取前10个位置的AB系数
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
