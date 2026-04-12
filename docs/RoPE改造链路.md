# Nova 位置编码改造链路：可学习绝对位置嵌入 → RoPE + 位置插值

> **作者**：高翔龙

---

## 一、为什么要改？当前方案有什么问题？

Nova 当前使用**可学习绝对位置嵌入（Learned Absolute Position Embedding）**，也就是 GPT-2 的方案。它的本质是一张固定大小的查找表：

```python
# model.py — NovaModel.__init__
self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
# 等价于 nn.Embedding(128, 384)，一张 128 行 × 384 列的表
```

**这张表是"可学习"的**——和 `token_emb`、`W_Q`、`W_K` 一样，表里的数字初始化为随机值，训练过程中通过反向传播由优化器更新。训练完成后，每个位置（0~127）都有一组学到的 384 维向量，用来告诉模型"这个 token 在第几个位置"。

它的致命问题：**训练时用多长的上下文，推理时就只能用多长。** 因为表只有 128 行，位置 128 以后没有对应的行可查，推理时超出就只能截断：

```python
# chat.py — generate 函数
ids_cond = ids[:, -model.pos_emb.weight.shape[0]:]  # 超出 128 就截掉前面的
```

### 三种位置编码方案的对比

| | 正弦位置编码（原始 Transformer） | 可学习绝对位置嵌入（GPT-2 / Nova 当前） | RoPE（LLaMA / 改造目标） |
|---|---|---|---|
| 位置信息怎么来的 | 固定数学公式（sin/cos），不可学习 | 训练学出来的查找表 | 固定数学公式（旋转角度），不可学习 |
| 位置信息加在哪 | 加到输入向量上（进 Block 之前） | 加到输入向量上（进 Block 之前） | 旋转 Q 和 K（在 Attention 内部） |
| 额外参数量 | 0（公式计算） | max_seq_len × d_model | 0（公式计算） |
| 编码的是什么 | 绝对位置 | 绝对位置 | **相对距离**（两个 token 之间差多远） |
| 能否泛化到更长序列 | 理论可以，效果差 | **不行**，表没有的行查不了 | 可以，配合位置插值效果好 |
| 谁在用 | 原始 Transformer 论文 | GPT-2、Nova | LLaMA、Mistral、Qwen、DeepSeek |

---

## 二、RoPE 到底在干什么？

### 2.1 核心思想：用旋转来编码位置

RoPE（Rotary Position Embedding，旋转位置编码）不再往输入向量上加一个位置偏移量，而是在 Attention 内部，对 Q 和 K 向量按位置做**旋转**。

把 `head_dim=64` 的向量切成 32 对 `(x_0, x_1), (x_2, x_3), ..., (x_62, x_63)`，每一对看作二维平面上的一个点，然后根据当前 token 的位置旋转一个角度：

```
位置 m 的第 i 对维度的旋转角度 = m × θ_i
其中 θ_i = 1 / (10000 ^ (2i / head_dim))
```

- 低维度对（i 小）：θ 大 → 旋转快 → 捕捉近距离关系
- 高维度对（i 大）：θ 小 → 旋转慢 → 捕捉远距离关系

### 2.2 为什么旋转就能编码相对距离？

两个 token（位置 m 和位置 n）做 Q·K 点积时，RoPE 的旋转使得点积结果只取决于 `m - n`（两者的距离），而不取决于 m 和 n 各自的绝对值。

这意味着：
- "位置 3 和位置 5"的注意力分数
- "位置 100 和位置 102"的注意力分数

两者是一样的，因为距离都是 2。模型学到的是"距离为 2 的 token 之间的关系模式"，而不是"第 3 个位置和第 5 个位置的关系模式"。

这就是 RoPE 能泛化到更长序列的根本原因——它学的是**距离**，不是**坐标**。

### 2.3 和当前方案的关键区别

```
当前方案（可学习绝对位置嵌入）：
  输入向量 = token_emb(ID) + pos_emb(位置)     ← 位置信息在进 Block 之前就加上了
  Attention 内部的 Q、K 已经包含了位置信息

RoPE：
  输入向量 = token_emb(ID)                      ← 不加位置信息
  Attention 内部：Q = rotate(W_Q·x, 位置)       ← 位置信息在 Attention 内部通过旋转注入
                  K = rotate(W_K·x, 位置)
                  V = W_V·x                     ← V 不旋转
```

---

## 三、位置插值（Position Interpolation）

### 3.1 问题：RoPE 不做任何处理就能无限延长吗？

不能。虽然 RoPE 的公式对任何位置 m 都能算出旋转角度，但模型的 W_Q、W_K、FFN 等权重只在训练长度范围内优化过。直接用超出训练范围的位置，注意力模式会崩掉。

### 3.2 位置插值的做法

假设训练时 `max_seq_len=512`，推理时想用 2048（4 倍扩展）：

```
原始位置索引:     [0, 1, 2, 3, ..., 2047]      间距 = 1.0
插值后位置索引:   [0, 0.25, 0.5, 0.75, ..., 511.75]  间距 = 0.25
                  ↑ 所有位置被压缩回 [0, 512) 范围
```

代码实现就是在计算旋转角度时，把位置除以缩放因子：

```python
t = torch.arange(target_seq_len, dtype=torch.float32)
t = t / scale_factor  # scale_factor = target_len / train_len = 4.0
```

### 3.3 扩展倍数的经验上限

Meta 的位置插值论文（2023）在 LLaMA-7B（70 亿参数，训练长度 2048）上的实验结论：

| 扩展倍数 | 推理长度 | 是否需要续训 | 效果 |
|---------|---------|------------|------|
| 2x | 4096 | 不需要 | 基本无损 |
| 4x | 8192 | 不需要 | 轻微下降，可接受 |
| 8x | 16384 | 需要（~1000 步续训） | 续训后可用 |
| 16x | 32768 | 需要（~1000 步续训） | 续训后勉强可用 |

**注意**：这些数据基于 70 亿参数大模型。对于 Nova 这样的 22M 微型模型，泛化能力更弱，建议保守估计 **2-4 倍**为安全范围。

### 3.4 续训是什么？

位置插值后的续训（continued pre-training）**不是 SFT 微调**，而是一种短期的预训练：

- 用**长文本语料**（和预训练数据类似）
- 跑很短的步数（几百到一千步）
- 目的是让 W_Q、W_K 适应被压缩后的新位置间距

完整训练流程变为：

```
1. 预训练（max_seq_len=512，正常 RoPE）
2. PI 续训（把 max_seq_len 扩到 2048，开启位置插值，用长文本跑 1000 步）
3. SFT 微调（用问答对训练对话能力）
4. 推理（可处理 2048 token 的上下文）
```

---

## 四、改造涉及的文件和具体代码变更

### 4.1 总览

| 文件 | 改动内容 | 工作量 |
|------|---------|--------|
| `config.py` | 新增 `rope_theta`、`rope_scale_factor` | 新增 2 个字段 |
| `model.py` | 新增 RoPE 函数、修改 3 个类 | ~50 行变更 |
| `train.py` | 删除 `pos_emb.weight` 适配逻辑 | 删除 ~12 行 |
| `chat.py` | 修改推理截断逻辑 | 改 1 行 |

### 4.2 config.py — 新增 RoPE 配置

```python
@dataclass
class NovaConfig:
    # ... 现有字段 ...

    # RoPE 基础频率（控制旋转速度的基数，业界标准值）
    rope_theta: float = 10000.0

    # 位置插值缩放因子
    # None = 不插值（推理上下文 = 训练上下文）
    # 2.0 = 上下文扩展 2 倍
    # 4.0 = 上下文扩展 4 倍
    rope_scale_factor: float | None = None
```

### 4.3 model.py — 新增 RoPE 核心函数

#### 函数 1：预计算旋转频率矩阵

```python
def precompute_rope_freqs(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    scale_factor: float | None = None,
) -> torch.Tensor:
    """预计算 RoPE 的旋转频率矩阵。

    Args:
        head_dim:     每个注意力头的维度（d_model // n_heads）
        max_seq_len:  最大序列长度
        theta:        基础频率（默认 10000.0）
        scale_factor: 位置插值缩放因子（None 表示不插值）

    Returns:
        shape [max_seq_len, head_dim // 2] 的复数张量，
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
    # 后续用复数乘法实现旋转
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    return freqs_cis
```

#### 函数 2：对 Q 和 K 施加旋转

```python
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
        每对视为复数 x0 + i*x1，乘以 e^(i*angle) = cos+i*sin 实现旋转
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
```

### 4.4 model.py — 修改 MultiHeadAttention

```python
class MultiHeadAttention(nn.Module):
    # __init__ 不变

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        #                                ^^^^^^^^^^^^^^^^^^^^^^  新增参数
        batch_size, seq_len, _ = x.shape

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # ★ 新增：在注意力计算之前，对 Q 和 K 施加 RoPE 旋转
        q, k = apply_rotary_emb(q, k, freqs_cis)

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True,
        )

        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )
        return self.w_o(attn_output)
```

### 4.5 model.py — 修改 TransformerBlock

```python
class TransformerBlock(nn.Module):
    # __init__ 不变

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        #                                ^^^^^^^^^^^^^^^^^^^^^^  新增参数
        x = x + self.attn(self.attn_norm(x), freqs_cis)  # 透传给 Attention
        x = x + self.ffn(self.ffn_norm(x))
        return x
```

### 4.6 model.py — 修改 NovaModel

```python
class NovaModel(nn.Module):
    def __init__(self, config: NovaConfig) -> None:
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)

        # ★ 删除: self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)

        # ★ 新增: 预计算 RoPE 旋转频率，注册为 buffer
        # buffer 不参与梯度计算，但会随模型保存/加载/to(device)
        head_dim = config.d_model // config.n_heads
        freqs_cis = precompute_rope_freqs(
            head_dim,
            config.max_seq_len,
            theta=config.rope_theta,
        )
        self.register_buffer("freqs_cis", freqs_cis)

        self.emb_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([...])  # 不变
        self.final_norm = RMSNorm(config.d_model)
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self._init_weights()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape

        # ★ 只查字义表，不再查位置表、不再做 token_emb + pos_emb 相加
        x = self.token_emb(input_ids)
        x = self.emb_dropout(x)

        # ★ 截取当前序列长度对应的旋转频率
        freqs_cis = self.freqs_cis[:seq_len]

        for block in self.blocks:
            x = block(x, freqs_cis)  # ★ 传入 freqs_cis

        x = self.final_norm(x)
        logits = self.output(x)
        return logits
```

### 4.7 train.py — 删除位置编码适配逻辑

`setup_finetune` 函数中有一段处理预训练和微调 `max_seq_len` 不同时，截断或扩展 `pos_emb.weight` 的逻辑（第 507-518 行），改造后直接删除：

```python
# ★ 删除以下代码（RoPE 没有 pos_emb.weight，不需要适配）:
#
# cur_seq_len = config.max_seq_len
# ckpt_seq_len = state_dict["pos_emb.weight"].shape[0]
# if ckpt_seq_len != cur_seq_len:
#     print(f"  → 位置编码调整: {ckpt_seq_len} → {cur_seq_len}")
#     old_weight = state_dict["pos_emb.weight"]
#     if ckpt_seq_len > cur_seq_len:
#         state_dict["pos_emb.weight"] = old_weight[:cur_seq_len]
#     else:
#         new_weight = torch.zeros(cur_seq_len, old_weight.shape[1])
#         nn.init.normal_(new_weight, mean=0.0, std=0.02)
#         new_weight[:ckpt_seq_len] = old_weight
#         state_dict["pos_emb.weight"] = new_weight
```

如果微调阶段想使用不同的 `max_seq_len`，只需在加载权重后重新计算 `freqs_cis`：

```python
if config.max_seq_len != ckpt_config.max_seq_len:
    head_dim = config.d_model // config.n_heads
    model.freqs_cis = precompute_rope_freqs(
        head_dim, config.max_seq_len, theta=config.rope_theta
    ).to(device)
```

### 4.8 chat.py — 修改推理截断逻辑

```python
# 改造前:
ids_cond = ids[:, -model.pos_emb.weight.shape[0]:]

# 改造后:
ids_cond = ids[:, -model.config.max_seq_len:]
```

### 4.9 推理时扩展上下文（位置插值）

训练完成后，如果想在推理时扩展上下文窗口，只需重新计算 `freqs_cis`：

```python
# 假设训练时 max_seq_len=512，推理时想支持 2048
target_len = 2048
train_len = 512
scale_factor = target_len / train_len  # = 4.0

head_dim = config.d_model // config.n_heads
new_freqs = precompute_rope_freqs(
    head_dim,
    max_seq_len=target_len,
    theta=config.rope_theta,
    scale_factor=scale_factor,
)
model.freqs_cis = new_freqs.to(device)
model.config.max_seq_len = target_len  # 更新截断阈值
```

---

## 五、改造后的参数量变化

| 组件 | 改造前 | 改造后 | 变化 |
|------|-------|-------|------|
| Position Embedding | max_seq_len × d_model = 128 × 384 = **49,152** | 0（RoPE 无参数） | **减少 49,152** |
| freqs_cis buffer | 0 | max_seq_len × head_dim/2 = 128 × 32 = 4,096 | 不算参数（不参与训练） |
| 其他所有组件 | 不变 | 不变 | 0 |

改造后模型参数量减少约 49K，同时获得了上下文扩展能力。

---

## 六、改造后的完整数据流对比

### 改造前

```
input_ids [batch, seq_len]
    │
    ├──→ token_emb(input_ids)  → [batch, seq_len, 384]
    │
    ├──→ pos_emb([0,1,...,seq_len-1]) → [seq_len, 384]   ← 查位置表（128 行上限）
    │
    └──→ token_emb + pos_emb → Dropout → 4 层 Block → RMSNorm → Output
```

### 改造后

```
input_ids [batch, seq_len]
    │
    ├──→ token_emb(input_ids) → Dropout → [batch, seq_len, 384]
    │
    │   freqs_cis[:seq_len]   → [seq_len, 32]             ← 公式计算（无上限）
    │
    └──→ 4 层 Block:
              每层 Attention 内部:
                Q = W_Q(x)  →  rotate(Q, freqs_cis)   ← 位置信息在这里注入
                K = W_K(x)  →  rotate(K, freqs_cis)
                V = W_V(x)                              ← V 不旋转
              → softmax(Q·K^T/√d) × V → W_O
              → FFN
         → RMSNorm → Output
```

---

## 七、改造检查清单

- [ ] `config.py`: 新增 `rope_theta` 和 `rope_scale_factor` 字段
- [ ] `model.py`: 新增 `precompute_rope_freqs` 函数
- [ ] `model.py`: 新增 `apply_rotary_emb` 函数
- [ ] `model.py`: `MultiHeadAttention.forward` 新增 `freqs_cis` 参数，调用 `apply_rotary_emb`
- [ ] `model.py`: `TransformerBlock.forward` 新增 `freqs_cis` 参数，透传给 Attention
- [ ] `model.py`: `NovaModel.__init__` 删除 `self.pos_emb`，新增 `self.register_buffer("freqs_cis", ...)`
- [ ] `model.py`: `NovaModel.forward` 删除位置表查询和相加，传 `freqs_cis` 给各 Block
- [ ] `model.py`: `NovaModel._init_weights` 中 `nn.Embedding` 的初始化会自动跳过已删除的 `pos_emb`，无需改动
- [ ] `model.py`: `NovaModel.print_parameter_summary` 中 Position Embedding 的统计行需要删除或改为 RoPE buffer 信息
- [ ] `train.py`: `setup_finetune` 中删除 `pos_emb.weight` 的截断/扩展逻辑（第 507-518 行）
- [ ] `chat.py`: 推理截断逻辑从 `model.pos_emb.weight.shape[0]` 改为 `model.config.max_seq_len`
- [ ] 删除旧 checkpoint，**重新训练**（位置编码机制完全改变，旧权重不兼容）
- [ ] （可选）训练完成后，在推理侧配置 `rope_scale_factor` 扩展上下文
