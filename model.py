"""Nova Decoder-Only Transformer 模型

从零实现完整的 Decoder-Only Transformer，遵循现代 LLM 标准（LLaMA / GPT 系列）。

本文件包含以下组件（按调用顺序排列）：

┌─────────────────────────────────────────────────────────────────────┐
│                    模型组件与调用顺序                                 │
│                                                                     │
│  输入 token IDs: [batch_size, seq_len]                              │
│       │                                                             │
│       ▼                                                             │
│  NovaModel（完整模型，步骤 5.5）                                     │
│   ├── Token Embedding    查字义表: token ID → 128维向量              │
│   ├── Position Embedding 查位置表: 位置编号 → 128维向量              │
│   ├── Dropout                                                       │
│   ├── TransformerBlock × 4 层（步骤 5.4）                            │
│   │    ├── RMSNorm（步骤 5.1）→ MultiHeadAttention（步骤 5.3）→ 残差 │
│   │    └── RMSNorm（步骤 5.1）→ SwiGLUFFN（步骤 5.2）→ 残差         │
│   ├── RMSNorm（最终归一化）                                          │
│   └── Linear 输出层: 128维 → vocab_size                             │
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
# 步骤 5.1：RMSNorm（Root Mean Square Layer Normalization）
# ======================================================================
class RMSNorm(nn.Module):
    """Nova使用RMSNorm进行归一化处理
      公式: RMSNorm(x) = x / RMS(x) * gamma，其中 RMS(x) = sqrt(mean(x²) + eps)

      RMS均方根，标准是1，如果向量空间中整体尺度(量级)远大于1，或者远小于1，那就逐元素除以一个数值来整体做尺度的放大或者缩小，
      保持向量空间的尺度始终在区间1这个标准附近。

      ======= 解释下，为什么需要做归一化 =======

    token 的向量会在一层层 Decoder Block 中不断经过自注意力、线性变换、激活函数和残差连接等模块，如果没有归一化，
    层与层之间的数值尺度就可能越来越不稳定，前向传播时激活值容易失控；而反向传播又需要沿着这条计算链逐层回传梯度，
    因此梯度也更容易被连续放大或连续压缩，最终导致训练难收敛（loss 不降）、容易震荡（loss 波动不下降），甚至出现梯度爆炸或梯度消失。

    归一化的核心作用，就是在关键计算节点前先把输入向量的整体数值尺度校准到更稳定的范围内，让每一层都尽量工作在可控的数值区间里，从而提升模型训练和推理的稳定性。
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        # gamma 是可学习参数，初始全为 1（即一开始不做缩放，等训练来调整）
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        调用时机:
          在 TransformerBlock.forward 中被调用两次：
            1. 自注意力之前: normed = self.attn_norm(x)
            2. FFN 之前:     normed = self.ffn_norm(x)
          在 NovaModel.forward 中被调用一次：
            3. 所有 Block 之后、输出层之前: x = self.final_norm(x)
        """

        # 计算均方根，先平方、再求平均、再开方。
        rms = torch.sqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)

        # 归一化
        x_norm = x.float() / rms

        # 乘以可学习参数 gamma，并转回输入的原始 dtype
        return (x_norm * self.gamma).type_as(x)


# ======================================================================
# 步骤 5.2：SwiGLU 前馈网络（SwiGLU Feed-Forward Network）
# ======================================================================
class SwiGLUFFN(nn.Module):
    """SwiGLU 前馈网络 — LLaMA / PaLM 使用的 FFN 变体。

    ┌──────────────────────────────────────────────────────────────────┐
    │  在 Transformer 中的位置                                         │
    │                                                                  │
    │  TransformerBlock 的后半段:                                       │
    │    x ──→ RMSNorm ──→ 【SwiGLUFFN】──→ + x（残差连接）           │
    │                       ^^^^^^^^^^^^^                              │
    │                                                                  │
    │  每层 Block 有 1 个 SwiGLUFFN，4 层共 4 个实例                    │
    └──────────────────────────────────────────────────────────────────┘

    SwiGLU 干什么用？
    ─────────────────
    自注意力是一场"集体讨论"——每个 token 从其他 token 那里收集了信息。
    SwiGLU FFN 则是让每个 token **独立地**"消化吸收"这些信息。

    具体做三件事：
      1. 把 128 维向量扩展到 512 维（展开，留出更多"思考空间"）
      2. 过一道"智能滤网"，决定哪些信息保留、哪些丢掉
      3. 再压缩回 128 维（恢复原始维度，传给下一层）

    注意: FFN 对每个位置的向量**独立处理**，位置之间不交互。
    交互的事在自注意力那一步已经做完了。

    SwiGLU vs 传统 FFN
    ───────────────────
    传统 FFN（GPT-2 用的）:
      output = W2 · ReLU(W1 · x)
      ReLU 是简单粗暴的滤网: 负数全砍为 0，正数原样通过。

    SwiGLU（LLaMA / Nova 用的）:
      output = W2 · (SiLU(W1 · x) ⊙ W3 · x)
      - SiLU（又叫 Swish）: 平滑版的 ReLU，负数不是直接砍掉，
        而是乘一个很小的系数"压低"，保留微弱的信号
      - W3 · x 是"门控"信号: 让网络自己学哪些维度该打开、哪些该关闭
      - ⊙ 是逐元素相乘（门控机制的核心）

    多出来的 W3 让网络有了"自主选择权"——不是机械地砍负数，
    而是根据输入内容动态决定保留什么。效果更好，但参数量多了 50%。

    三个权重矩阵
    ────────────
    W1 (d_model → d_ff):  "展开 + 激活"通路
       把 128 维扩展到 512 维，然后过 SiLU 激活函数

    W3 (d_model → d_ff):  "门控"通路
       也是 128→512，但不过激活函数，直接作为"开关系数"

    W2 (d_ff → d_model):  "压缩"通路
       把 SiLU(W1·x) ⊙ W3·x 的 512 维结果压缩回 128 维

    所有矩阵都不带偏置（bias=False），这是 LLaMA 的做法。

    参数量计算
    ──────────
    每个 SwiGLUFFN: 3 × d_model × d_ff = 3 × 128 × 512 = 196,608
    4 层共: 786,432（占总参数量的大头）

    对比传统 FFN 的 2 × d_model × d_ff = 131,072，多了 50%。

    前向传播计算步骤（以 d_model=4, d_ff=8 为例）
    ──────────────────────────────────────────────
    输入 x = [0.5, -0.3, 0.8, 1.2]  (4 维)

    步骤 1：两条通路同时计算
      gate   = W1 · x → [8个数]  →  SiLU激活  → [8个数, 负值被压低]
      filter = W3 · x → [8个数]                  [8个数, 原样]

    步骤 2：门控相乘（逐元素）
      gated = SiLU(W1·x) ⊙ W3·x → [8个数]
      含义: filter 中的每个维度被 gate 的对应值"调节"——
            gate 接近 0 的维度被关闭，gate 接近 1 的维度被保留

    步骤 3：压缩回原始维度
      output = W2 · gated → [4个数]  (回到 d_model 维)

    完整流程图:
      x ─┬─→ W1 → SiLU ─→ ⊙ ──→ W2 ──→ output
         │                 ↑
         └─→ W3 ──────────┘
    """

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        # W1: "展开 + 激活"通路 (d_model → d_ff)
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        # W2: "压缩"通路 (d_ff → d_model)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        # W3: "门控"通路 (d_model → d_ff)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：对每个位置的向量独立做 SwiGLU 变换。

        参数:
          x : Tensor，形状 [batch_size, seq_len, d_model]
              例如 [16, 128, 128]

        返回:
          Tensor，形状与输入相同 [batch_size, seq_len, d_model]

        调用时机:
          在 TransformerBlock.forward 的后半段:
            normed = self.ffn_norm(x)       # 先归一化
            ffn_out = self.ffn(normed)       # ← 在这里调用
            x = x + ffn_out                 # 残差连接
        """
        # 步骤 1: 两条通路同时计算
        # SiLU(x) = x * sigmoid(x)，平滑版 ReLU
        gate = F.silu(self.w1(x))  # [batch, seq_len, d_ff]
        filt = self.w3(x)  # [batch, seq_len, d_ff]

        # 步骤 2: 门控相乘——gate 控制 filt 中每个维度的"开关"
        gated = gate * filt  # [batch, seq_len, d_ff]

        # 步骤 3: 压缩回 d_model 维
        return self.w2(gated)  # [batch, seq_len, d_model]


# ======================================================================
# 步骤 5.3：多头自注意力（Multi-Head Self-Attention）
# ======================================================================
class MultiHeadAttention(nn.Module):
    """多头自注意力 — Transformer 的核心组件，让 token 之间互相"对话"。

    ┌──────────────────────────────────────────────────────────────────┐
    │  在 Transformer 中的位置                                         │
    │                                                                  │
    │  TransformerBlock 的前半段:                                       │
    │    x ──→ RMSNorm ──→ 【MultiHeadAttention】──→ + x（残差连接）   │
    │                       ^^^^^^^^^^^^^^^^^^^^^^                     │
    │                                                                  │
    │  每层 Block 有 1 个 MultiHeadAttention，4 层共 4 个实例           │
    └──────────────────────────────────────────────────────────────────┘

    自注意力干什么用？
    ─────────────────
    在 Embedding 之后，每个 token 的向量是"各管各的"——"吗"不知道前面是"好"，
    "？"不知道自己在一个问句里。自注意力就是让每个 token 去看看其他 token，
    收集跟自己相关的信息，改写自己的工作向量。

    打个比方：6 个员工坐在会议室里开会，每个人先看看其他人的资料，判断
    "谁跟我最相关"，然后把最相关的人的信息揉进自己的资料里。

    Q、K、V 三个角色
    ─────────────────
    模型有三张变换表（权重矩阵 W_Q、W_K、W_V），每个 token 的向量分别乘以
    这三张表，变出三个新向量:

      Q (Query，查询):   "我在找什么信息？"
      K (Key，键):       "我能提供什么信息？"
      V (Value，值):     "我实际携带的内容"

    Q 和 K 做点积 → 得到关联度分数 → softmax 归一化为百分比 →
    用百分比对 V 加权求和 → 得到融合了上下文信息的新向量。

    多头机制
    ────────
    把 d_model(128) 维拆成 n_heads(4) 个独立的"头"，每个头在
    head_dim(32) 维的子空间里独立做注意力:
      - 头 0 可能学会关注语法搭配（"好" → "吗"）
      - 头 1 可能学会关注语义关系（"名字" → "Nova"）
      - 头 2 可能学会关注位置关系（相邻的字）
      - 头 3 可能学会关注标点/结构
    4 个头的结果拼回 128 维，再过一个输出投影矩阵混合各头信息。

    因果掩码（Causal Mask）
    ───────────────────────
    Decoder-Only 模型的关键规则：每个 token 只能看自己和前面的 token，
    不能偷看后面的。因为推理时模型是逐 token 生成的，生成第 3 个 token 时
    第 4 个还不存在。

      <s>   只能看: <s>
      你    只能看: <s>, 你
      好    只能看: <s>, 你, 好
      吗    只能看: <s>, 你, 好, 吗

    实现方式：在注意力分数矩阵上，把"未来位置"填成 -inf，softmax 后
    这些位置的权重变成 0，相当于完全屏蔽。

    缩放（Scaling）
    ───────────────
    Q·K 的点积结果会随 head_dim 增大而变大（32 个数相乘再求和，数字很容易
    飙到几十甚至上百）。如果不缩放，softmax 会把最大值推向 1、其余推向 0，
    梯度消失。所以除以 √head_dim = √32 ≈ 5.66，把数值拉回合理范围。

    参数说明
    ────────
    d_model : int
        输入/输出向量维度（128）。

    n_heads : int
        注意力头数（4）。d_model 必须能被 n_heads 整除。
        head_dim = d_model / n_heads = 32。

    dropout : float
        注意力权重的 Dropout 概率（训练时随机丢弃部分注意力连接，
        防止模型过度依赖某些固定的 token 关联）。

    可学习参数
    ──────────
    W_Q : nn.Linear(d_model, d_model, bias=False)  权重 [128, 128]
    W_K : nn.Linear(d_model, d_model, bias=False)  权重 [128, 128]
    W_V : nn.Linear(d_model, d_model, bias=False)  权重 [128, 128]
    W_O : nn.Linear(d_model, d_model, bias=False)  权重 [128, 128]

    参数量: 4 × d_model² = 4 × 128² = 65,536（每层）
    4 层共: 262,144

    前向传播完整流程（d_model=8, n_heads=2, head_dim=4, seq_len=3 为例）
    ────────────────────────────────────────────────────────────────────
    输入 x: [batch, 3, 8]  （3 个 token，每个 8 维）

    步骤 1: 线性投影
      Q = x @ W_Q → [batch, 3, 8]
      K = x @ W_K → [batch, 3, 8]
      V = x @ W_V → [batch, 3, 8]

    步骤 2: 拆分多头 (reshape + transpose)
      Q → [batch, 2, 3, 4]  （2个头，每头看3个token的4维向量）
      K → [batch, 2, 3, 4]
      V → [batch, 2, 3, 4]

    步骤 3: 计算注意力分数 (Q @ K^T / √head_dim)
      scores = Q @ K^T → [batch, 2, 3, 3]  （3×3 的关联度矩阵）
      scores /= √4 = 2.0

    步骤 4: 应用因果掩码
      掩码矩阵（上三角为 -inf）:
        [[0,    -inf, -inf],
         [0,    0,    -inf],
         [0,    0,    0   ]]
      scores += mask  →  未来位置变成 -inf

    步骤 5: Softmax → 注意力权重
      weights = softmax(scores) → [batch, 2, 3, 3]
      每行加起来 = 1，-inf 位置变成 0

    步骤 6: Dropout（训练时）

    步骤 7: 加权求和
      output = weights @ V → [batch, 2, 3, 4]

    步骤 8: 合并多头 (transpose + reshape)
      output → [batch, 3, 8]  （拼回 d_model 维）

    步骤 9: 输出投影
      output = output @ W_O → [batch, 3, 8]

    完整流程图:
      x ─┬─→ W_Q ──→ Q ─┐
         ├─→ W_K ──→ K ─┼─→ Q@K^T/√d ─→ +mask ─→ softmax ─→ drop ─→ ×V ─→ concat ─→ W_O ─→ out
         └─→ W_V ──→ V ─┘
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model({d_model}) 必须能被 n_heads({n_heads}) 整除"
        )
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads  # 每个头的维度: 128/4 = 32

        # 缩放因子: 1/√head_dim，用于点积注意力的缩放
        self.scale = self.head_dim**-0.5

        # 四个投影矩阵（全部无偏置，遵循 LLaMA 风格）
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：对输入序列做多头因果自注意力。

        参数:
          x : Tensor，形状 [batch_size, seq_len, d_model]
              例如 [16, 128, 128]

        返回:
          Tensor，形状与输入相同 [batch_size, seq_len, d_model]
              每个 token 的向量已融合了上下文信息

        调用时机:
          在 TransformerBlock.forward 的前半段:
            normed = self.attn_norm(x)       # 先 RMSNorm 归一化
            attn_out = self.attn(normed)     # ← 在这里调用
            x = x + attn_out                 # 残差连接
        """
        batch_size, seq_len, _ = x.shape

        # ── 步骤 1: 线性投影 ──
        # 每个 token 的 128 维向量分别乘以 W_Q、W_K、W_V，
        # 得到 Q、K、V 三个 128 维向量
        q = self.w_q(x)  # [batch, seq_len, d_model]
        k = self.w_k(x)  # [batch, seq_len, d_model]
        v = self.w_v(x)  # [batch, seq_len, d_model]

        # ── 步骤 2: 拆分多头 ──
        # 把最后一维 d_model 拆成 (n_heads, head_dim):
        #   [batch, seq_len, d_model] → [batch, seq_len, n_heads, head_dim]
        # 再 transpose 让 n_heads 提到前面:
        #   → [batch, n_heads, seq_len, head_dim]
        # 这样每个头就是一个独立的 [seq_len, head_dim] 矩阵，可以并行计算
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        # 此时 q, k, v: [batch, n_heads, seq_len, head_dim]

        # ── 步骤 3-7: FlashAttention (缩放点积 + 因果掩码 + softmax + dropout + 加权求和) ──
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True,
        )

        # ── 步骤 8: 合并多头 ──
        # transpose 把 n_heads 移回去:
        #   [batch, n_heads, seq_len, head_dim] → [batch, seq_len, n_heads, head_dim]
        # contiguous() 保证内存连续（transpose 只改变 stride，不移动数据）
        # view 把最后两维拼成 d_model:
        #   → [batch, seq_len, d_model]
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )

        # ── 步骤 9: 输出投影 ──
        # 通过 W_O 混合各头的信息，让不同头学到的特征互相融合
        return self.w_o(attn_output)  # [batch, seq_len, d_model]


# ======================================================================
# 步骤 5.4：TransformerBlock（一层 Decoder Block）
# ======================================================================
class TransformerBlock(nn.Module):
    """一层 Transformer Decoder Block — Pre-LN（先归一化再计算）结构。

    ┌──────────────────────────────────────────────────────────────────┐
    │  在整个模型中的位置                                               │
    │                                                                  │
    │  NovaModel.forward:                                              │
    │    token_emb + pos_emb → Dropout                                 │
    │        │                                                         │
    │        ▼                                                         │
    │    【TransformerBlock 第 0 层】                                   │
    │        │                                                         │
    │        ▼                                                         │
    │    【TransformerBlock 第 1 层】                                   │
    │        │                                                         │
    │        ▼                                                         │
    │    【TransformerBlock 第 2 层】                                   │
    │        │                                                         │
    │        ▼                                                         │
    │    【TransformerBlock 第 3 层】                                   │
    │        │                                                         │
    │        ▼                                                         │
    │    final RMSNorm → 输出层 → logits                               │
    │                                                                  │
    │  4 层 Block 共享相同的结构，但各自有独立的可学习参数。              │
    │  每过一层，向量的"理解深度"加深一层：                              │
    │    第 0 层: 字符搭配 —— "好吗"经常连在一起                        │
    │    第 1 层: 句法结构 —— 这是一个问句                              │
    │    第 2 层: 语义理解 —— 有人在问候我                              │
    │    第 3 层: 意图判断 —— 我应该用问候语回答                        │
    └──────────────────────────────────────────────────────────────────┘

    Pre-LN 结构详解
    ───────────────
    "Pre-LN" 的意思是**先归一化，再做计算**。与之对应的是 "Post-LN"
    （先计算，再归一化），这是原始 Transformer 论文用的方式。

    Pre-LN（Nova / LLaMA / GPT-3 使用）:
      x → RMSNorm → Attention → + x → RMSNorm → FFN → + x
          ^^^^^^^^ 先归一化                ^^^^^^^^ 先归一化

    Post-LN（原始 Transformer 论文）:
      x → Attention → + x → LayerNorm → FFN → + x → LayerNorm
                             ^^^^^^^^^ 后归一化        ^^^^^^^^^ 后归一化

    为什么用 Pre-LN？
      - 训练更稳定：归一化后的输入数值范围可控，Attention/FFN 不会收到
        暴涨或趋零的输入
      - 梯度流更顺畅：残差连接直接把梯度传回去，不需要"穿过"归一化层
      - 不容易出现训练早期 loss 爆炸的问题
      - LLaMA、GPT-3、PaLM 等现代大模型全部采用 Pre-LN

    一层 Block 的完整数据流（以 "你好吗？" 为例）
    ──────────────────────────────────────────────
    输入 x: [batch, seq_len, 128]  — 4 个 token 的 128 维向量

    ┌─── 前半段：自注意力（token 之间互相"对话"） ───────────────────┐
    │                                                              │
    │  步骤 A1: attn_norm = RMSNorm(x)                             │
    │           把数值拉回正常范围，防止 Attention 收到极端输入        │
    │                                                              │
    │  步骤 A2: attn_out = MultiHeadAttention(attn_norm)            │
    │           4 个头各自做 Q·K·V 注意力，融合上下文信息             │
    │           "好" 从 "你" 和 "吗" 那里收集了相关信息              │
    │                                                              │
    │  步骤 A3: x = x + attn_out     （残差连接）                   │
    │           即使 Attention 算出了垃圾，原始 x 还在               │
    │           等价于 output = 原始信息 + 新增信息                  │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘

    ┌─── 后半段：前馈网络（每个 token 独立"消化"信息） ─────────────┐
    │                                                              │
    │  步骤 B1: ffn_norm = RMSNorm(x)                              │
    │           再次归一化（经过 Attention + 残差后数值可能又飘了）    │
    │                                                              │
    │  步骤 B2: ffn_out = SwiGLUFFN(ffn_norm)                      │
    │           128→512→128 的展开-过滤-压缩，每个 token 独立处理    │
    │           "好" 独自消化从上一步收集来的上下文信息              │
    │                                                              │
    │  步骤 B3: x = x + ffn_out      （残差连接）                   │
    │           再一次兜底                                          │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘

    输出 x: [batch, seq_len, 128]  — 形状不变，内容更"深层"了

    参数说明
    ────────
    d_model : int
        向量维度（128）。

    n_heads : int
        注意力头数（4）。

    d_ff : int
        FFN 隐藏层维度（512）。

    dropout : float
        Dropout 概率（0.1），用于 Attention 内部。

    子模块和参数量（每层）
    ──────────────────────
    attn_norm : RMSNorm          →  128 个参数（gamma）
    attn      : MultiHeadAttention → 4 × 128² = 65,536 个参数
    ffn_norm  : RMSNorm          →  128 个参数（gamma）
    ffn       : SwiGLUFFN        → 3 × 128 × 512 = 196,608 个参数
    ───────────────────────────────────────────────────────────
    每层合计:  262,400 个参数
    4 层合计:  1,049,600 个参数
    """

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1
    ) -> None:
        super().__init__()

        # ── 前半段子模块：归一化 + 自注意力 ──
        self.attn_norm = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)

        # ── 后半段子模块：归一化 + 前馈网络 ──
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：Pre-LN 结构的一层 Decoder Block。

        参数:
          x : Tensor，形状 [batch_size, seq_len, d_model]
              例如 [16, 128, 128]

        返回:
          Tensor，形状与输入相同 [batch_size, seq_len, d_model]
              经过一次"讨论 + 消化"后的向量

        调用时机:
          在 NovaModel.forward 中被循环调用:
            for block in self.blocks:
                x = block(x)      # ← 每层调用一次，共 4 次
        """
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


# ======================================================================
# 步骤 5.5：NovaModel（完整的 Decoder-Only Transformer 模型）
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

        # ── ⑦ 权重初始化（步骤 5.6）──
        self._init_weights()

    # ------------------------------------------------------------------
    # 步骤 5.6：权重初始化
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
    # 步骤 5.7：参数量统计
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
