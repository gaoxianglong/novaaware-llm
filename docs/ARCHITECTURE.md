# 深入剖析 | Nova 微型 LLM 架构设计

> **作者**：高翔龙

写这篇文档的起因很简单——我想从零手搓一个能对话的 LLM。不是调 API，不是 fine-tune 别人的模型，而是从第一行代码开始，把 Transformer 的每一个零件都自己焊上去。模型不大，参数量约 22M（对比 GPT-3 的 1750 亿，连零头都不算），但架构和那些大家伙一模一样。我给它起了个名字：**Nova**。

这篇文档会把 Nova 的架构设计掰开了揉碎了讲清楚。所有的数学公式，我都会用大白话翻译一遍——保证你不用去翻线性代数课本也能看明白。同时，每讲到一个核心机制，我会直接贴出 Nova 项目中对应的源码实现，让你不仅知道"原理是什么"，还能看到"代码怎么写"。

---

## 一、先说清楚一件事：什么是 Transformer？

在聊 Nova 的架构之前，我们得先把 Transformer 这个东西讲透。因为如果连 Transformer 是什么都没搞明白，后面的一切都是空中楼阁。

### 1.1 Transformer 从哪来的？

2017 年，Google 发了一篇论文，标题相当自信——《Attention Is All You Need》（注意力就是你所需要的一切）。这篇论文提出了 Transformer 架构，一把掀翻了之前统治 NLP 领域的 RNN（循环神经网络）和 LSTM（长短期记忆网络）。

在 Transformer 出现之前，处理文本靠的是 RNN。RNN 的工作方式像流水线工人：一个字一个字地读，读完第一个字才能读第二个，读完第二个才能读第三个。这有两个致命问题：

1. **慢**。串行处理，没法并行，GPU 再多也白搭。
2. **记性差**。句子一长，前面读过的内容就忘得差不多了。你让它处理"今天早上我在公司楼下的那家开了三年的咖啡馆里买了一杯拿铁"——等它读到"拿铁"的时候，"今天早上"这几个字的信息已经衰减得所剩无几了。

Transformer 的解决思路非常暴力：**不排队了，所有字同时处理，每个字直接和所有其他字建立联系**。不管句子多长，"今天早上"和"拿铁"之间的距离永远是一步。

这个机制就叫做**自注意力（Self-Attention）**，它是 Transformer 的灵魂。

### 1.2 Transformer 长什么样？

原始的 Transformer 分成两半：

- **Encoder（编码器）**：负责"读懂"输入。比如你输入一句英文，Encoder 把它理解成一组向量。
- **Decoder（解码器）**：负责"生成"输出。比如根据 Encoder 理解的内容，一个字一个字地蹦出中文翻译。

这种 Encoder-Decoder 结构适合翻译任务。但后来大家发现，如果你只是想做文本生成（给一段开头，让模型往后写），其实只需要 Decoder 就够了。

GPT 系列、LLaMA、Mistral……当今所有主流 LLM 都是**只用 Decoder**的架构，叫做 **Decoder-Only**。

Nova 也是 Decoder-Only。

### 1.3 Decoder-Only 到底在干什么？

说白了就三件事：

1. **读进去**：把你的文字变成计算机能处理的数字
2. **想明白**：通过多层 Transformer Block 反复"咀嚼"这些数字，理解上下文
3. **蹦出来**：根据理解的结果，一个字一个字地生成回答

整个过程的核心循环是——**预测下一个字**。你给模型"今天天气"，它猜下一个字可能是"很"；你把"很"接上变成"今天天气很"，它猜下一个字可能是"好"；一直这样循环下去，直到它觉得句子该结束了。

ChatGPT 打字的时候一个字一个字往外蹦，不是为了装酷，而是它真的就是一个字一个字生成的。

### 1.4 一次"预测下一个字"的完整旅程

为了把 Transformer 讲透，我们跟踪一个完整的推理过程。假设用户问"你好吗？"，模型要生成回答"我很好"。

```
模型的输入: <s>你好吗？<sep>
                              ↑ 从这里开始，一个字一个字往后猜
```

在猜第一个字的时候，这串输入会经历以下旅程：

```
第一站：分词器
  "你好吗？" → 一串数字编号 [1, 42, 15, 88, 7, 3]
                              ↑ 1是<s>  ↑ 3是<sep>

第二站：嵌入层
  每个数字 → 一个384维的向量（一串384个小数）
  同时加上"位置信息"（告诉模型谁在前谁在后）

第三站：Transformer Block × 4 层
  ┌──────────────────────────────────────────────────────┐
  │  第一步：多头自注意力                                    │
  │  每个字去"看"其他所有字，理解谁和谁有关系                    │
  │  "吗"发现自己和"好"关系很近——因为"好吗"是一个搭配             │
  │                                                      │
  │  第二步：前馈网络                                       │
  │  对每个字的信息做深度加工，提炼出更高层的语义                   │
  └──────────────────────────────────────────────────────┘
  重复4次——每重复一次，模型对语义的理解就更深一层

第四站：输出层
  最后一个位置（<sep>后面）的向量 → 变成一个概率分布
  "我"的概率 = 0.85  ← 最高！选它
  "你"的概率 = 0.03
  "他"的概率 = 0.01
  ...
```

就这样，模型输出了第一个字"我"。然后把"我"接上去，变成 `<s>你好吗？<sep>我`，再跑一遍，猜出"很"。再接上去，猜出"好"。直到猜出 `<e>`（结束标记），整个回答就生成完了。

### 1.5 为什么 Transformer 这么强？

说到底就是**自注意力机制**带来的两个碾压级优势：

**第一，全局视野。** 每个字都能"看到"输入中的所有其他字。不像 RNN 只能看到左边（还越远越模糊），Transformer 的每个字到其他字的距离都是一步之遥。句子再长，首尾之间的联系也不会丢失。

**第二，天然并行。** 所有字的注意力计算可以同时进行，完美利用 GPU 的并行算力。RNN 是串行的，一次只能处理一个字，GPU 再快也是一核有难八核围观。Transformer 是矩阵运算，GPU 上千个核心可以同时干活。

这两个优势让 Transformer 成了大模型时代的绝对基石。从 2017 年到现在，几乎所有重要的 AI 突破——GPT、BERT、LLaMA、Stable Diffusion、Sora——底层都是 Transformer。

---

## 二、Nova 的整体架构

有了上面的铺垫，现在来看 Nova 的具体架构就清晰多了。

> **关于维度数字的说明**：下文的讲解中，为了让概念更直观，部分示例会使用较小的数字（如 128 维、32 维）来举例。Nova 实际使用的维度参数是 d_model=384、n_heads=6、head_dim=64、d_ff=1536，详见第六章的规格表。原理完全一致，只是数字更大。

```
用户输入: "你叫什么名字？"
       │
       ▼
┌─────────────────────┐
│  1. BPE 分词器       │  把文字拆成子词片段，每个片段变成一个数字编号
│     (Tokenizer)      │  "人工智能" → [156, 2847], ...
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  2. 嵌入层           │  把数字编号变成384维向量
│  Token + Position    │  同时注入位置信息
│     Embedding        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│  3. Transformer Decoder × 4 层          │
│  ┌───────────────────────────────────┐   │
│  │ RMSNorm → 多头自注意力 → 残差连接   │   │
│  ├───────────────────────────────────┤   │
│  │ RMSNorm → SwiGLU FFN → 残差连接    │   │
│  └───────────────────────────────────┘   │
└──────────┬──────────────────────────────┘
           │
           ▼
┌─────────────────────┐
│  4. RMSNorm + 线性层  │  映射回词表大小，输出概率分布
└──────────┬──────────┘
           │
           ▼
输出: "我" → "是" → "名" → "为" → "N" → "o" → "v" → "a" → ...
```

整个模型大概 22M 参数。这是什么概念？一张 4K 照片大概 15-20MB，而 Nova 整个模型序列化后也就这个量级。真·微型。

上面这张图的每一个方块，在源码中都有对应的组件。打开 `model.py`，你能在 `NovaModel.__init__` 中看到它们是如何被"焊"在一起的：

```python
# model.py — NovaModel.__init__

class NovaModel(nn.Module):
    def __init__(self, config: NovaConfig) -> None:
        super().__init__()
        self.config = config

        # ① Token Embedding：字义表，把 token ID 映射为 d_model 维向量
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)

        # ② Position Embedding：位置表，把位置编号映射为 d_model 维向量
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)

        # ③ Dropout：嵌入层之后的正则化
        self.emb_dropout = nn.Dropout(config.dropout)

        # ④ N 层 TransformerBlock
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.d_model, config.n_heads, config.d_ff, config.dropout
            )
            for _ in range(config.n_layers)
        ])

        # ⑤ 最终归一化
        self.final_norm = RMSNorm(config.d_model)

        # ⑥ 输出投影层：d_model → vocab_size
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # ⑦ 权重初始化
        self._init_weights()
```

看到了吗？`token_emb`、`pos_emb` 对应嵌入层，`blocks` 是 4 层 Transformer Decoder，`final_norm` + `output` 对应最终的输出层。整个模型的骨架就这么几行代码。复杂的部分藏在每个子组件里——接下来一个一个拆。

---

## 三、跟着一句话走完 Transformer——从"你好吗"到"我"

网上讲 Transformer 的文章，十篇有九篇上来就是 Q、K、V、Multi-Head Attention 一通轰炸，看完跟没看一样。问题出在哪儿？出在他们只讲零件，不讲流水线。你拿到一堆齿轮和螺丝，不给你一张装配图，你能装出一台发动机？

所以这一章换个讲法：拿"你好吗？"这句话当活标本，从它进入模型的第一步开始，一步一步跟到模型吐出"我"这个字为止。每一步只讲三件事——**数据现在长什么样、这一步在干什么、为什么非干不可**。跟完这条线，Transformer 对你来说就不是黑盒了。

### 3.1 第一步：把文字变成数字（分词）

废话不多说，直接进正题。计算机不认字，只认数字。所以第一件事——把用户输入的文字翻译成数字。

Nova 使用 **BPE（Byte Pair Encoding，字节对编码）** 分词器。和最简单的"一个字一个编号"不同，BPE 会把高频出现的词组合并成一个 token，用更少的编号表示更多的语义。比如"人工智能"如果在训练语料中出现了几百次，BPE 就会把它合并成一两个 token，而不是拆成 4 个字。

```
用户输入: "你好吗？"

加上控制标记后变成:  <s> 你好吗？ <sep>

BPE 编码得到编号:    [1,  42, 15, 88, 7,  3]
```

- `<s>` 是"开始说话了"的标记，编号 1
- `<sep>` 是"问题结束，该你回答了"的标记，编号 3

BPE 分词器的核心训练过程很有意思：先把所有字符当作单独的 token，然后反复统计哪两个相邻 token 出现得最频繁，把它们合并成一个新 token。重复这个过程，直到词表达到目标大小。打开 `tokenizer.py`，可以看到 BPE 训练的完整实现：

```python
# tokenizer.py — NovaTokenizer.train_from_texts

def train_from_texts(self, texts: List[str], vocab_size: int = 16000) -> None:
    # 创建 BPE 模型
    tokenizer = Tokenizer(models.BPE(unk_token=UNK_TOKEN))

    # 按 Unicode 脚本边界 + 空白符做初步切分
    # 中文字符、英文单词、数字、标点会被分到不同的组
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.UnicodeScripts(),
        pre_tokenizers.Whitespace(),
    ])

    # 解码器：解码时自动处理子词拼接
    tokenizer.decoder = decoders.Fuse()

    # BPE 训练器：至少出现 2 次的字符对才会被考虑合并
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,  # <pad>, <s>, <e>, <sep>, <unk>
        min_frequency=2,
    )

    # 将文本写入临时文件，调用 tokenizers 库执行 BPE 训练
    tokenizer.train([tmp_path], trainer=trainer)
```

编码和解码就简单了——分别调用底层 BPE 引擎：

```python
# tokenizer.py — encode / decode

def encode(self, text: str) -> List[int]:
    """文字 → token ID 列表"""
    return self._tokenizer.encode(text).ids

def decode(self, ids: List[int]) -> str:
    """token ID 列表 → 文字（过滤掉特殊标记）"""
    skip_ids = {PAD_ID, BOS_ID, EOS_ID, SEP_ID}
    filtered = [tid for tid in ids if tid not in skip_ids]
    return self._tokenizer.decode(filtered)
```

推理时，`chat.py` 的 `generate` 函数负责把用户输入拼成模型能理解的格式——这个格式必须和训练时一模一样：

```python
# chat.py — generate 函数中的 prompt 构造

# 拼接格式和训练时 dataset.py 的格式完全一致:
#   训练: [BOS_ID] + encode(问题) + [SEP_ID] + encode(回答) + [EOS_ID]
#   推理: [BOS_ID] + encode(问题) + [SEP_ID]  ← 只给前半部分，让模型续写
input_ids = (
    [tokenizer.char_to_id[BOS_TOKEN]]
    + tokenizer.encode(question)
    + [tokenizer.char_to_id[SEP_TOKEN]]
)
ids = torch.tensor([input_ids], dtype=torch.long, device=device)
```

到这一步为止，数据就是 **6 个整数**。很简单，没有任何花活。

### 3.2 第二步：查两张表，把数字变成"有内涵的向量"（嵌入层）

光有编号没用。你在公司内网看到工号 42，你知道这人是谁吗？不知道。你只知道他排在 41 后面、43 前面，但他叫什么、干什么、水平怎么样，编号本身一个字都没告诉你。

所以模型里有**两张表**——这就是上面第二章提到的字义表和位置表，这里展开讲讲它们到底怎么用的：

#### 第一张表：字义表（Token Embedding Table）

这是一张大表，每个字对应一行，每行是 384 个数字。你可以把它理解成**每个字的"简历"**——用 384 个维度来描述这个字的特征。

```
字义表（部分）:
┌────────┬──────────────────────────────────────┐
│ 编号   │ 384个特征数字（简历）                   │
├────────┼──────────────────────────────────────┤
│ 42(你) │ [0.12, -0.34, 0.56, 0.01, ..., 0.78] │
│ 15(好) │ [-0.23, 0.45, 0.11, -0.09, ..., -0.67]│
│ 88(吗) │ [0.33, 0.12, -0.44, 0.05, ..., 0.21] │
│ ...    │ ...                                   │
└────────┴──────────────────────────────────────┘
```

这张表一开始是随机填的——对，你没看错，就是随机数。训练过程中模型自己把它学好（怎么学的？后面 3.11 节会专门讲）。训练完之后，**意思相近的字，表里的那行数字会越来越接近**。比如"高兴"和"开心"那两行数字会很像，"高兴"和"桌子"那两行数字会差十万八千里。

#### 第二张表：位置表（Position Embedding Table）

光知道"你"是什么意思还不够。"我打你"和"你打我"用的字完全一样，但谁先谁后决定了完全不同的意思。

位置表的作用就是告诉模型**每个字排在第几个位置**。它也是 384 个数字一行，但这里不是按"字"查，而是按"位置"查：

```
位置表:
┌────────┬──────────────────────────────────────┐
│ 位置   │ 384个数字                              │
├────────┼──────────────────────────────────────┤
│ 第0位  │ [0.01, 0.02, -0.01, ..., 0.03]       │
│ 第1位  │ [0.04, -0.01, 0.05, ..., -0.02]      │
│ 第2位  │ [-0.02, 0.06, 0.03, ..., 0.01]       │
│ ...    │ ...                                   │
└────────┴──────────────────────────────────────┘
```

#### 两张表相加 = Transformer 的真正输入

最后，把同一个位置查到的两行数字**逐个相加**：

```
第0位的"<s>":
  字义表查出来:  [0.05, 0.08, -0.11, ...]   ← "<s>"是谁
  位置表查出来:  [0.01, 0.02, -0.01, ...]   ← 它在第0位
  ─────────────────────────────────────
  相加得到:      [0.06, 0.10, -0.12, ...]   ← 既知道它是谁，也知道它在哪

第1位的"你":
  字义表查出来:  [0.12, -0.34, 0.56, ...]   ← "你"是谁
  位置表查出来:  [0.04, -0.01, 0.05, ...]   ← 它在第1位
  ─────────────────────────────────────
  相加得到:      [0.16, -0.35, 0.61, ...]   ← 既知道它是谁，也知道它在哪

...6个位置都这样做完
```

做完之后，原来的 6 个整数变成了 **6 组、每组 384 个数字**。这就是 Transformer 后面所有环节要处理的原材料。

**关键理解：从这一步开始，模型再也不看原始文字了。后面所有操作都是在反复加工这 6 组向量。**

上面这一整套"查两张表 → 相加 → Dropout"的流程，在 `NovaModel.forward` 的前半段一目了然：

```python
# model.py — NovaModel.forward（嵌入部分）

def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len = input_ids.shape

    # ① 查字义表：每个 token ID 查出对应的 384 维向量
    token_emb = self.token_emb(input_ids)      # [batch, seq_len, d_model]

    # ② 查位置表：生成 [0, 1, 2, ..., seq_len-1]，查出每个位置的 384 维向量
    positions = torch.arange(seq_len, device=input_ids.device)
    pos_emb = self.pos_emb(positions)           # [seq_len, d_model]

    # ③ 字义 + 位置 = 初始工作向量
    x = token_emb + pos_emb                     # [batch, seq_len, d_model]

    # ④ Dropout：训练时随机丢弃一部分嵌入维度，防过拟合
    x = self.emb_dropout(x)
    # ...后续进入 Transformer Block
```

`nn.Embedding` 本质就是一次查表操作——给定一个整数索引，返回对应行的向量。`self.token_emb(input_ids)` 等价于"用 input_ids 这串编号，去字义表里查对应的行"。

### 3.3 第三步：字与字之间互相"对话"（自注意力）

到目前为止，每个字的向量都是"各管各的"——"吗"不知道自己前面是"好"，"？"不知道自己在一个问句里。这还理解个锤子。

自注意力干的就是打破这个隔阂：**让每个字去看看其他字，收集跟自己相关的信息，改写自己的工作向量。**

注意这里的措辞——改写的是"工作向量"，不是字义表里的原始数据。这一点很重要，后面 3.11 节会专门掰扯。

打个比方：6 个员工坐在会议室里，每个人手里拿着一份自己的简历**复印件**（注意，是复印件，不是人事档案原件）。现在开会——每个人先看看其他人的复印件，判断"谁跟我最相关"，然后把最相关的人的信息揉进自己的复印件里。

会开完之后，每个人手里的复印件变了——不再只有自己的信息，还融入了跟自己最相关的那些人的信息。但复印件还是 384 个数字，格式没变，内容变了。人事档案（字义表）始终没人动过。

具体怎么判断"谁跟谁相关"？这里要引入三个东西：Q、K、V。先别急，我一步一步来。

#### Q、K、V 到底是什么？怎么算出来的？

首先澄清一个容易搞混的点：**Q、K、V 不是三个固定的向量，也不是从字义表里直接拿的。它们是每个字的工作向量经过矩阵乘法"变换"出来的三个新向量。**

模型里有三张**变换表**（学名叫权重矩阵），分别叫 $W_Q$、$W_K$、$W_V$。这三张表也是训练出来的参数（和字义表一样，推理时只读，训练时更新）。

每个字的工作向量（就是上一步字义表 + 位置表相加得到的那 384 个数字），要分别乘以这三张表，变出三个新向量：

```
"吗"的工作向量（384个数字）
    │
    ├── × W_Q ──→ Q向量（384个数字）："吗"在找什么信息？
    │
    ├── × W_K ──→ K向量（384个数字）："吗"能提供什么信息？
    │
    └── × W_V ──→ V向量（384个数字）："吗"实际携带的内容
```

**每个字都要做这套操作**，所以 6 个字算完之后，就有了 6 组 Q、6 组 K、6 组 V。

在源码中，这三次矩阵乘法就是三个 `nn.Linear`：

```python
# model.py — MultiHeadAttention.__init__ 和 forward（QKV 投影部分）

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads   # 384 / 6 = 64
        self.scale = self.head_dim ** -0.5   # 1/√64 ≈ 0.125

        # 四个投影矩阵（全部无偏置，遵循 LLaMA 风格）
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)  # 输出投影

        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # 线性投影：每个 token 的向量分别乘以 W_Q、W_K、W_V
        q = self.w_q(x)    # [batch, seq_len, d_model]
        k = self.w_k(x)    # [batch, seq_len, d_model]
        v = self.w_v(x)    # [batch, seq_len, d_model]
        # ...后续拆分多头
```

注意 `nn.Linear(d_model, d_model, bias=False)` 就是一个 384×384 的矩阵乘法，没有偏置项。这是 LLaMA 的做法——偏置项对效果影响不大，去掉可以减少参数。

#### 用 Q 和 K 算"谁跟谁相关"

Q 的含义是"我在找什么"，K 的含义是"我能提供什么"。两个向量越像，说明一个字正好在找另一个字能提供的东西——也就是说它们**相关**。

怎么衡量"像不像"？把 Q 和 K 做点积（对应位置相乘再求和），得到一个分数。分数越高越相关。

拿"吗"举例，它的 Q 会和每个字（包括自己）的 K 逐一配对打分：

```
"吗"的Q × "好"的K → 8.5 分（很相关，"好吗"是常见搭配）
"吗"的Q × "你"的K → 3.2 分（有点关系）
"吗"的Q × "吗"的K → 4.1 分（跟自己也会算一次）
"吗"的Q × "<s>"的K → 0.5 分（关系不大）
```

然后把这些分数转成百分比（用 softmax），决定"吗"应该从每个字那里借多少信息：

```
好=52%, 吗=25%, 你=18%, <s>=5%
```

#### 用打分结果和 V 算出新向量

最后，按上面的百分比，从每个字的 **V 向量**里按比例取信息，加在一起：

```
"吗"的新向量 = 52% × V(好) + 25% × V(吗) + 18% × V(你) + 5% × V(<s>)
```

做完之后，"吗"的工作向量里就融入了大量来自"好"的信息——模型开始"理解"到这是一个"好吗？"的问句了。

#### 整个过程用一张图串起来

```
字义表（训练好的）查出 384 维向量
       ↓  加上
位置表（训练好的）查出 384 维向量
       ↓  得到
工作向量（384 维）
       ↓  × 变换表
Q、K、V 向量（各 384 维，拆成 6 个头后每头 64 维）
       ↓  Q 和 K 做点积
关联度分数
```

#### 多头：同时从多个角度看

Nova 有 6 个注意力"头"。每个头独立做一遍上面的过程，但关注的角度不一样——有的头可能关注语法搭配，有的头可能关注语义关系。6 个头的结果最后拼在一起。

具体实现上，并不是创建 6 套独立的 QKV 矩阵，而是用一个大矩阵做完投影后，把结果**拆成 6 份**：

```python
# model.py — MultiHeadAttention.forward（多头拆分与注意力计算）

        # 把 d_model(384) 维拆成 n_heads(6) × head_dim(64) 维
        # [batch, seq_len, 384] → [batch, seq_len, 6, 64] → [batch, 6, seq_len, 64]
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # 缩放点积注意力 + 因果掩码 + softmax + 加权求和（一步到位）
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True,   # ← 因果掩码：每个 token 只能看到自己和前面的
        )

        # 合并多头：[batch, 6, seq_len, 64] → [batch, seq_len, 384]
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )

        # 输出投影：混合各头信息
        return self.w_o(attn_output)
```

这里有个很巧妙的地方：PyTorch 提供了 `F.scaled_dot_product_attention` 这个高阶函数，它把"Q@K^T 点积 → 除以 √head_dim 缩放 → 因果掩码 → softmax → Dropout → 乘以 V"这一长串操作**全部封装成一行调用**，底层还会自动使用 FlashAttention 等高效实现。

#### 因果掩码：不许偷看后面

还有一个关键规则：**每个字只能看它自己和前面的字，不能偷看后面的**。

```
<s>   只能看到: <s>
你    只能看到: <s>, 你
好    只能看到: <s>, 你, 好
吗    只能看到: <s>, 你, 好, 吗
？    只能看到: <s>, 你, 好, 吗, ？
<sep> 只能看到: <s>, 你, 好, 吗, ？, <sep>
```

为什么？因为推理的时候模型是一个字一个字生成的，生成第 3 个字时第 4 个字还不存在。训练时如果让它偷看了，推理时没得偷看就傻了。

实现上就是上面代码中 `is_causal=True` 那个参数——它会在注意力分数矩阵的"未来位置"填上 $-\infty$，softmax 之后这些位置的权重自动变成 0。

### 3.4 第四步：每个字独立"消化吸收"（前馈网络 FFN）

自注意力是一场集体讨论——大家互相交换了信息。接下来每个字需要独立"消化"一下收集到的信息。

前馈网络就是干这个的。它对每个位置的向量**单独处理**（位置之间不再交互），做的事情可以理解为：

1. 把 384 个数字扩展到 1536 个（展开，留出更多"思考空间"）
2. 过一道"智能滤网"，决定哪些信息保留、哪些丢掉
3. 再压缩回 384 个数字

Nova 用的滤网叫 **SwiGLU**，它比传统的滤网（ReLU）更精细——传统方式是"负数全砍掉"，SwiGLU 是"让网络自己学什么该砍什么该留"。

看看源码就一目了然——SwiGLU 的核心是**两条并行通路**加一次**逐元素相乘**：

```python
# model.py — SwiGLUFFN

class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)   # 展开 + 激活 通路
        self.w2 = nn.Linear(d_ff, d_model, bias=False)   # 压缩 通路
        self.w3 = nn.Linear(d_model, d_ff, bias=False)   # 门控 通路

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 两条通路同时计算
        gate = F.silu(self.w1(x))   # 通路1: 展开到 1536 维，过 SiLU 激活
        filt = self.w3(x)           # 通路2: 展开到 1536 维，不过激活（原样）

        # 门控相乘：gate 控制 filt 中每个维度的"开关"
        gated = gate * filt

        # 压缩回 384 维
        return self.w2(gated)
```

关键点在 `gate * filt` 这行——这就是"门控"机制。`gate` 经过 SiLU 激活后，值域大致在 [-0.3, +∞)，它决定了 `filt` 中每个维度"开多大"。如果 `gate` 某个维度接近 0，对应的 `filt` 值就被关掉了；如果 `gate` 某个维度是正值，对应的 `filt` 值就被放行。

和传统 FFN 比，SwiGLU 多了一个 `w3`（门控矩阵），参数量多了 50%，但效果显著更好。LLaMA、PaLM 都做了这个选择。

> **SiLU（Swish）激活函数**：$\text{SiLU}(x) = x \cdot \sigma(x)$，其中 $\sigma$ 是 sigmoid。它是 ReLU 的平滑版——负数不是直接砍为 0，而是乘以一个很小的系数"压低"，保留微弱信号。

### 3.5 第五步：加上两道保险——残差连接和归一化

做过大型系统的人都知道，链路越长越容易出幺蛾子。Transformer 也一样，注意力和 FFN 叠了好几层，如果不加保护，数据在传递过程中要么信息丢光，要么数值爆炸。所以每一步都有两道"保险"。

**保险一：残差连接**

每一步的输出 = 这一步加工的结果 + 这一步的原始输入。

用做系统的话讲就是"降级兜底"——即使某一层加工出来的结果是垃圾，原始输入还在，不至于全盘崩掉。没有这个机制，4 层叠下来信息传着传着就丢光了，模型根本训不动。

**保险二：RMSNorm（归一化）**

数据在层与层之间传递，数值会越来越不稳定——有的暴涨到几千，有的趋近于零。这跟你系统里监控指标不做归一化是一个道理：量纲差了几个数量级，根本没法一起分析。RMSNorm 就是在每一步之前把数据"拉回正常范围"，让训练保持稳定。

RMSNorm 的实现只有两行核心代码——算 RMS，除以它，再乘以可学习的缩放系数：

```python
# model.py — RMSNorm

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))  # 可学习的缩放系数，初始全1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS = sqrt(mean(x²) + eps)
        rms = torch.sqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        # 归一化后乘以 gamma
        x_norm = x.float() / rms
        return (x_norm * self.gamma).type_as(x)
```

公式写出来就是：$\text{RMSNorm}(x) = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} \cdot \gamma$

和标准 LayerNorm 比，RMSNorm 去掉了"减均值"那一步（实验证明对效果影响不大），也去掉了偏置参数 $\beta$。好处就是更快更简洁，LLaMA 实测验证了这一点。

把残差连接和归一化组合在一起，就是一层完整的 Transformer Block。看看 `TransformerBlock.forward` 有多简洁：

```python
# model.py — TransformerBlock.forward

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)            # 注意力前的归一化
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn_norm = RMSNorm(d_model)              # FFN 前的归一化
        self.ffn = SwiGLUFFN(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 前半段：归一化 → 自注意力 → 残差连接
        x = x + self.attn(self.attn_norm(x))

        # 后半段：归一化 → 前馈网络 → 残差连接
        x = x + self.ffn(self.ffn_norm(x))

        return x
```

这两行就是整个 Pre-LN（先归一化再计算）结构的完整实现。注意看 `x = x + self.attn(...)` 这个写法——`x +` 就是残差连接，`self.attn_norm(x)` 就是先归一化。整个 Block 的输入和输出形状完全相同，都是 `[batch, seq_len, d_model]`，所以可以无限堆叠。

### 3.6 第三步到第五步会重复 4 次

上面三步（注意力 → FFN → 加保险）合在一起叫做**一层 Transformer Block**。Nova 把这个结构重复 4 次——对，就是同一套逻辑跑 4 遍。你可以理解为流水线上有 4 个一模一样的工位，每个工位都做"开会讨论 → 独立消化 → 质量兜底"这套流程，但每个工位有自己的参数（也就是有自己的"工作经验"）。

```
       ┌────────────────────────────────────┐
       │         一层 Transformer Block      │
       │                                    │
输入 ──→│ 归一化 → 自注意力 → 加回输入(残差)   │
       │ 归一化 → 前馈网络 → 加回输入(残差)    │──→ 输出
       │                                    │
       └────────────────────────────────────┘
              ↑          重复4次          ↑
              第1层 → 第2层 → 第3层 → 第4层
```

每过一层，向量就被"改写"一次。可以理解为：

- **第 1 层**：学到最基础的字符搭配——"好吗"经常连在一起
- **第 2 层**：学到句法结构——这是一个问句
- **第 3 层**：学到语义——有人在问候我
- **第 4 层**：学到意图——我应该用问候语来回答

但这 4 层的格式始终没变：**都是 6 组、每组 384 个数字**。变的是数字的内容——一层比一层"理解"得更深。

对应到 `NovaModel.forward` 的后半段：

```python
# model.py — NovaModel.forward（Transformer Block 循环 + 输出层）

        # ⑤ 4 层 TransformerBlock 依次处理
        for block in self.blocks:
            x = block(x)              # [batch, seq_len, d_model] → 不变

        # ⑥ 最终归一化
        x = self.final_norm(x)        # [batch, seq_len, d_model]

        # ⑦ 输出投影：384 维 → vocab_size 维
        logits = self.output(x)       # [batch, seq_len, vocab_size]

        return logits
```

`for block in self.blocks` ——就这么一行循环，4 层 Transformer Block 就跑完了。每一层的输入输出形状都是 `[batch, seq_len, 384]`，所以 `x = block(x)` 可以无缝串联。

### 3.7 第六步：从向量变回文字（输出层）

4 层 Transformer Block 处理完，我们得到了 6 组 384 维向量。

现在要回答"下一个字是什么"。模型只需要看**最后一个位置**（也就是 `<sep>` 对应的那组向量），因为这个位置已经通过注意力"看到"了前面所有的字，信息最完整。

这 384 个数字怎么变成"下一个字是什么"？两步：

**第一步：用一个线性层把 384 个数字变成"词表大小"个数字。** 假设词表有 8000 个 token，就变成 8000 个数字，每个数字对应一个 token 的"得分"。

```
384个数字 → 线性变换 → 8000个得分
                        "我" → 5.8
                        "你" → 1.2
                        "好" → 3.1
                        "是" → 4.5
                        ...
```

**第二步：把得分转成概率。** 用 softmax 把得分归一化成概率（加起来等于 1）：

```
"我" → 62%  ← 最高！
"是" → 18%
"好" →  9%
"你" →  3%
...
```

然后从概率最高的几个字里选一个——选出了"我"。

这个"选字"的过程在 `chat.py` 的 `generate` 函数中，涉及温度缩放、Top-k 过滤、重复惩罚等采样策略：

```python
# chat.py — generate 函数（自回归采样核心循环）

@torch.no_grad()
def generate(model, tokenizer, question, max_new_tokens=100,
             temperature=0.8, top_k=5, repetition_penalty=1.3):
    # ...构造 prompt 并编码（见 3.1 节）...

    generated_ids: list[int] = []

    for _ in range(max_new_tokens):
        # (i) 前向传播
        logits = model(ids)                          # [1, seq_len, vocab_size]

        # (ii) 只取最后一个位置（模型对"下一个字"的预测）
        next_logits = logits[:, -1, :]               # [1, vocab_size]

        # (iii) 重复惩罚：已生成过的 token 降低概率
        if repetition_penalty != 1.0 and generated_ids:
            for token_id in set(generated_ids):
                if next_logits[0, token_id] > 0:
                    next_logits[0, token_id] /= repetition_penalty
                else:
                    next_logits[0, token_id] *= repetition_penalty

        # (iv) 温度缩放：temperature < 1 更确定，> 1 更随机
        next_logits = next_logits / temperature

        # (v) Top-k 过滤：只保留概率最高的 k 个候选，其余设为 -inf
        if top_k > 0:
            top_values, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            threshold = top_values[:, -1].unsqueeze(-1)
            next_logits = next_logits.masked_fill(next_logits < threshold, float('-inf'))

        # (vi) softmax → 概率分布 → 按概率采样
        probs = F.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        # (vii) 遇到 <e> 就停
        if next_id.item() == EOS_ID:
            break

        # (viii) 追加到序列，下一轮继续
        generated_ids.append(next_id.item())
        ids = torch.cat([ids, next_id], dim=1)

    return tokenizer.decode(generated_ids)
```

几个值得注意的采样技巧：
- **温度缩放**（temperature）：除以一个系数来调节概率分布的"尖锐程度"。温度 < 1 让分布更尖锐（模型更确定、更保守），温度 > 1 让分布更平坦（更随机、更有创意）。
- **Top-k 过滤**：只从得分最高的 k 个候选里选，直接把其余候选的概率设为 0。防止从概率极低的"噪声" token 里采样出乱码。
- **重复惩罚**（repetition_penalty）：对已经生成过的 token 降低得分，防止模型陷入重复循环（"的的的的……"）。

### 3.8 生成第二个字、第三个字……

"我"生成出来后，把它接到输入末尾，变成 `<s>你好吗？<sep>我`，再跑一遍整个流程，预测下一个字"很"。再接上去，预测"好"。直到生成出 `<e>`（结束标记），就停下来。

**每生成一个字都要跑一遍完整的流水线**：分词 → 查两张表 → 4 层 Transformer Block → 输出层 → 选字。

### 3.9 整条流水线的完整总结

现在把整条线串起来，从头到尾就是这么回事：

```
"你好吗？"
    │
    │ ① 分词：文字经过 BPE 变成 token 编号
    ▼
[1, 42, 15, 88, 7, 3]
    │
    │ ② 查字义表：每个编号变成384个数字（描述这个字是什么意思）
    │ ③ 查位置表：每个位置变成384个数字（描述这个字在第几位）
    │ ④ 两张表相加：得到Transformer的输入（既知道是什么字，也知道在第几位）
    ▼
6组 × 384个数字
    │
    │ ⑤ 自注意力：每个字看看其他字，收集上下文信息，改写自己的向量
    │ ⑥ 前馈网络：每个字独立消化吸收，做更深层的语义加工
    │ ⑦ 残差+归一化：两道保险，防止信息丢失和数值跑偏
    │    （⑤⑥⑦ 重复4次）
    ▼
6组 × 384个数字（内容已经被彻底改写，充满了上下文信息）
    │
    │ ⑧ 取最后一个位置的384个数字
    │ ⑨ 线性变换成 vocab_size 个得分（每个 token 一个分）
    │ ⑩ 温度缩放 → Top-k 过滤 → softmax → 采样
    ▼
"我"
```

这就是 Transformer 的全部。没有任何隐藏环节，没有任何跳过的步骤。看完这条线你会发现，Transformer 的核心思路其实就一句话：**把文字变成一堆数字，反复加工这堆数字让它们融入上下文信息，最后从加工好的数字里挑出下一个字。**

**记住这条线：文字 → 数字 → 查两张表相加 → 4层反复改写工作向量 → 最后一个位置投影回文字。**

这里再强调一遍：中间所有的"改写"都是在临时的工作向量上操作的，字义表和位置表在推理时纹丝不动。关于这些表的数字到底什么时候变，下面 3.11 节详细讲。

### 3.10 特殊标记在流程中的角色

| 标记 | 干什么用 | 在流程哪个环节起作用 |
|------|---------|-------------------|
| `<s>` | 标记序列开始 | 第①步分词时加到最前面 |
| `<sep>` | 分隔问题和回答 | 第①步分词时加在问题末尾；第⑧步取的就是它后面的位置 |
| `<e>` | 标记回答结束 | 第⑩步如果选出了它，就停止生成 |
| `<pad>` | 训练时补齐长度 | 第①步，仅训练时用，推理时不需要 |
| `<unk>` | 替代未知字符 | 第①步，分词器无法处理的输入用它代替 |

### 3.11 一个必须搞清楚的问题：字义表里的数字，什么时候变？

看完上面的流程，一定会冒出一个疑问：

> 自注意力不是把每个字的向量"改写"了吗？改写之后的新数字，要不要写回字义表（Token Embedding Table）？如果要写，什么时候写？

答案非常明确：**不写回。从头到尾都不写回。**

这是理解 Transformer 运行机制最关键的一个点，值得展开讲讲。

#### 推理时：字义表是只读字典

整个推理过程（也就是用户问问题、模型生成回答的过程），字义表就像一本**印刷好的字典**——你可以翻开查阅，但你不会往字典上面写字。

具体来说，上面流程中的每一步：

```
① 分词得到编号 [1, 42, 15, 88, 7, 3]
② 拿编号去字义表"查"对应的384个数字         ← 只读，不改表
③ 拿位置去位置表"查"对应的384个数字           ← 只读，不改表
④ 两组数字相加，得到工作向量
⑤⑥⑦ 注意力、FFN反复改写这些工作向量          ← 改的是工作向量，不是表里的数字
⑧⑨⑩ 从工作向量算出下一个字
```

注意看第⑤⑥⑦步：自注意力和 FFN 改写的是从表里查出来之后的那份**副本**（工作向量），不是表本身。

打个比方：你去图书馆借了一本书，在自己的笔记本上做了大量批注和总结。**你的笔记本变了，但图书馆里那本书一个字都没动。**

所以推理的时候，不管你问多少次"你好吗"，字义表里"你"对应的那行 384 个数字，从第一次问到第一万次问，**一模一样，纹丝不动**。

#### 那字义表里的数字到底什么时候变？——只在训练时

字义表的更新发生在**训练阶段**，而且有一套非常规矩的流程。用一个类比来讲：

想象你开了一家翻译公司，给每个员工发了一本"业务手册"（就是字义表）。日常工作中，大家照着手册干活，不会改手册。但每个月底你会做一次**复盘**：

1. **先干活（前向传播）**：拿一批翻译任务，按手册流程走一遍，得出翻译结果
2. **看结果对不对（算损失）**：把翻译结果和标准答案对比，算出"错了多少"
3. **追责（反向传播）**：从最终结果往回追——这个错误是哪个环节导致的？是手册第 42 页写得不好？还是第 15 页有问题？每一页分别该承担多少责任？
4. **改手册（参数更新）**：按照追责结果，把手册里对应的数字微调一点点——错误大的地方多改一些，错误小的地方少改一些

这就是训练的一个完整周期。字义表里的每一行数字，都是在第 4 步被改的。

用技术语言说就是：

- **前向传播**：拿训练数据走一遍模型，算出预测结果
- **计算损失**：预测结果和正确答案的差距有多大
- **反向传播**：从损失出发，算出每个参数（包括字义表的每个数字）对这个差距"贡献"了多少（梯度）
- **参数更新**：优化器（AdamW）根据梯度，把每个参数微调一点点

#### 梯度是怎么"倒推"回字义表的？

这里有必要展开讲一下"反向传播"到底干了什么。前向传播是正着走的：

```
字义表 → +位置表 → 工作向量 → ×W_Q/W_K/W_V → 注意力 → FFN → ... → 输出预测
```

反向传播就是**倒着走一遍**，从最终的错误出发，一层一层往回追"谁的责任"：

```
预测结果 vs 正确答案 → 算出"错了多少"（损失值）
                                │
              从损失倒着往回追，逐层算"梯度"
                                │
                  ┌─────────────┼─────────────┐
                  ▼             ▼             ▼
            输出层的梯度   FFN的梯度     注意力的梯度
                                              │
                                    继续往回追 │
                              ┌───────────────┼──────────┐
                              ▼               ▼          ▼
                        W_Q的梯度       W_K的梯度    W_V的梯度
                                              │
                                    继续往回追 │
                                              ▼
                                       工作向量的梯度
                                              │
                                    继续往回追 │
                              ┌───────────────┤
                              ▼               ▼
                        字义表的梯度     位置表的梯度
```

最终到达字义表的那个梯度，含义是什么？打个比方：

> 字义表里"你"那行的第 37 个数字目前是 0.56。梯度告诉你："如果你把这个 0.56 往上调一点点，最终的预测错误会减少 0.003。"

优化器收到这个信号，就把 0.56 微调成 0.561（具体调多少由学习率决定）。

**关键理解：注意力、FFN 算出来的工作向量本身不会回写字义表。但它们参与了这条"追责链"——梯度要经过注意力和 FFN 才能传回字义表。** 如果注意力这一环节算得好（提取到了有用的上下文），那传回去的梯度信号就更精准，字义表就能朝着更好的方向更新；反之则更新方向就会偏。

训练几百轮之后，字义表里的数字就从一开始的随机噪声，逐渐变成了有语义信息的向量——"高兴"和"开心"那两行趋近，"高兴"和"桌子"那两行远离。

#### 不只是字义表——所有参数都是这个套路

同样的更新机制适用于模型里的**所有参数**：

| 参数 | 什么时候查/用 | 什么时候更新 |
|------|-------------|-------------|
| 字义表（Token Embedding） | 每次推理的第②步查表 | 只在训练时通过反向传播更新 |
| 位置表（Position Embedding） | 每次推理的第③步查表 | 只在训练时通过反向传播更新 |
| Q/K/V 的变换矩阵 | 每次推理的第⑤步做矩阵乘法 | 只在训练时通过反向传播更新 |
| FFN 的权重矩阵 | 每次推理的第⑥步做矩阵乘法 | 只在训练时通过反向传播更新 |
| 输出层的权重矩阵 | 每次推理的第⑨步做矩阵乘法 | 只在训练时通过反向传播更新 |

规律很清晰：**推理时所有参数只读不写，训练时统一更新。**

训练结束后，把所有参数保存成一个文件（checkpoint），推理时加载这个文件就行了。从加载的那一刻起，所有参数就定死了——这就是为什么一个训练好的模型不管你问什么，它的"知识"是固定的，不会因为你跟它聊天而变聪明或变傻。

#### 一句话总结

**字义表在推理时是只读字典，只在训练时通过"算错误 → 追责任 → 改数字"这套流程来更新。中间层算出来的工作向量是临时的草稿纸，用完就扔，不会写回任何表里。**

---

## 四、训练——模型是怎么"学会"的

模型架构定好了，接下来是训练。Nova 采用**两阶段训练**策略——先"预训练"学通用语言知识，再"微调"学对话能力。这和大模型的训练路径一脉相承。

### 4.1 训练目标

说出来可能不信，所有 GPT 类模型的训练目标只有一个，极其简单——**预测下一个字**。

给模型"我是名为"四个字，它要预测下一个字应该是"N"。给它"我是名为N"五个字，它要预测下一个字应该是"o"。

这个任务叫 **Next Token Prediction**。看起来简单到离谱，但正是通过在大量文本上反复做这个任务，模型逐渐学会了语言的语法规律、语义关系、甚至一定程度的推理能力。

### 4.2 损失函数：交叉熵

模型每次预测下一个字的时候，会输出一个概率分布（每个字一个概率值）。交叉熵损失衡量的就是——**模型给正确答案的概率有多高**。

- 模型预测"N"的概率是 90%？loss 很小，学得不错。
- 模型预测"N"的概率是 1%？loss 很大，还得继续练。

训练过程就是反复调整模型参数，让 loss 一路走低。

在 `train.py` 的训练循环里，前向传播 + 计算 loss 只需要几行：

```python
# train.py — 训练循环核心（前向传播 + 损失计算 + 反向传播）

for batch_idx, batch in enumerate(dataloader):
    input_ids  = batch["input_ids"].to(device)     # [batch, seq_len]
    target_ids = batch["target_ids"].to(device)     # [batch, seq_len]

    # 前向传播：模型预测每个位置的下一个 token
    with torch.amp.autocast("cuda", enabled=use_amp):
        logits = model(input_ids)                   # [batch, seq_len, vocab_size]
        loss = F.cross_entropy(
            logits.view(-1, config.vocab_size),     # 展平为 [batch*seq_len, vocab_size]
            target_ids.view(-1),                    # 展平为 [batch*seq_len]
            ignore_index=-100,                      # 忽略 padding 位置
        )

    # 反向传播 + 梯度裁剪 + 参数更新
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()                   # 反向传播计算梯度
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(                 # 梯度裁剪，防止梯度爆炸
        model.parameters(), max_norm=config.grad_clip
    )
    scaler.step(optimizer)                          # 优化器更新参数
    scaler.update()
```

这段代码就是上面 3.11 节讲的"前向传播 → 算损失 → 反向传播 → 更新参数"的完整实现。注意几个细节：

- `ignore_index=-100`：告诉交叉熵损失函数忽略标签为 -100 的位置。这些位置是 padding（补齐），不是真实内容，算 loss 没意义。
- `torch.amp.autocast`：混合精度训练（AMP），在 CUDA 上自动用 FP16 加速，省显存。
- `clip_grad_norm_`：梯度裁剪，如果梯度的 L2 范数超过 1.0 就按比例缩小。防止某个 batch 产生异常大的梯度导致参数剧烈震荡。

### 4.3 优化器：AdamW

模型知道 loss 要降低，但"该往哪个方向调参数、每次调多少"则由优化器来决定。

Nova 用的是 **AdamW**，深度学习领域目前最主流的优化器。它有三个核心特点：

1. **自适应学习率**：对每个参数独立估算最优步长，而不是所有参数用同一个固定步长
2. **动量**：记住之前几步的更新方向。就像推车——如果连续几步都往右推，说明右边是对的方向，那就推大力一点；如果忽左忽右，说明方向不确定，那就谨慎一点
3. **权重衰减**：每次更新时让参数微微缩小一点，防止参数长到太大（正则化，防过拟合）

创建优化器就一行代码：

```python
# train.py — 创建 AdamW 优化器

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate,       # 学习率（预训练 5e-4，微调 3e-4）
    weight_decay=config.weight_decay,  # 权重衰减系数 0.01
)
```

### 4.4 学习率调度：先热身、再冲刺、最后收尾

学习率（Learning Rate）控制的是"每一步迈多大"。太大容易跑过头（loss 震荡甚至发散），太小又走得太慢。

Nova 的策略是 **Warmup + Cosine Decay**：

```
学习率
  ↑
  │     ╱‾‾‾‾╲
  │    ╱       ╲
  │   ╱         ╲
  │  ╱           ╲
  │ ╱             ╲
  │╱               ╲___
  └────────────────────→ 训练步数
  热身期     余弦衰减期
```

- **Warmup（热身）**：头 100 步，学习率从 0 线性增长到最大值。刚开始训练时模型参数是随机的，步子太大容易"走偏"。
- **Cosine Decay（余弦衰减）**：之后学习率按余弦曲线平滑下降。训练越往后，模型越接近最优解，需要的步子就越小——就像停车，远处可以猛踩油门，快到车位了得轻踩刹车慢慢挪。

对应的 `get_lr` 函数实现非常简洁：

```python
# train.py — 学习率调度器

def get_lr(step, warmup_steps, max_steps, max_lr, min_lr=1e-6):
    # 阶段 1: Warmup — 线性增长
    if step < warmup_steps:
        return max_lr * step / warmup_steps

    # 阶段 2: Cosine Decay — 余弦衰减
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)
```

每个 epoch 开始前，调用一次 `get_lr` 算出当前应该用多大的学习率，然后手动写入优化器：

```python
# train.py — 在训练循环中更新学习率

lr = get_lr(epoch, config.warmup_steps, config.epochs, config.learning_rate)
for param_group in optimizer.param_groups:
    param_group["lr"] = lr
```

### 4.5 两阶段训练：预训练 + 微调

Nova 的训练分两步走，和大模型的训练管线如出一辙：

**阶段 1：预训练（学语言）**

投喂大量纯文本（百科、新闻、文章等），让模型学会"语言本身"——词与词之间的搭配规律、语法结构、常识知识。这一步不管对话能力，只管"学会说人话"。

```bash
python train.py --mode pretrain --data data/pretrain/
```

**阶段 2：微调（学对话）**

在预训练模型的基础上，投喂问答对数据，让模型学会"看到什么问题该怎么回答"。模型已经会说人话了，现在只需要教它"在什么场景下说什么话"。

```bash
python train.py --mode finetune --resume checkpoints/best_model.pt --data data/sft/
```

两个阶段的初始化逻辑分别在 `setup_pretrain` 和 `setup_finetune` 中。核心区别在于数据源和是否加载已有权重：

```python
# train.py — 预训练 vs 微调的初始化差异

# 预训练：从零开始，训练 BPE 分词器 + 随机初始化模型
def setup_pretrain(data_path="data/pretrain", resume_path=None):
    texts = load_pretrain_data(data_path)          # 加载纯文本
    tokenizer.train_from_texts(texts)              # 训练 BPE 分词器
    dataset = PretrainDataset(texts, tokenizer, config.max_seq_len)
    model = NovaModel(config).to(device)           # 模型参数随机初始化
    # ...

# 微调：加载预训练权重，只换数据集
def setup_finetune(data_path="data/sft/", resume_path=None):
    qa_pairs = load_qa_pairs(data_path)            # 加载问答对
    tokenizer.load("data/tokenizer.json")          # 复用预训练的分词器
    dataset = NovaDataset(qa_pairs, tokenizer, config.max_seq_len)
    model = NovaModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])  # 加载预训练权重
    # ...
```

微调时有一个细节值得注意：位置编码表的长度可能需要调整（预训练和微调的 `max_seq_len` 可能不同），`setup_finetune` 中有专门的处理逻辑来截断或扩展位置编码。

---

## 五、数据格式与对话协议

### 5.1 训练数据格式

Nova 支持两种训练数据格式，分别用于预训练和微调：

**预训练数据**（纯文本，JSONL 格式）：

```json
{"text": "计算机是20世纪最先进的科学技术发明之一。"}
{"text": "由硬件系统和软件系统组成..."}
```

拼接后：`<s>计算机是20世纪最先进的科学技术发明之一。<e>`

**微调数据**（问答对，JSONL 格式）：

```json
{"question": "你叫什么名字？", "answer": "我是名为Nova的微型LLM。"}
```

拼接后：`<s>你叫什么名字？<sep>我是名为Nova的微型LLM。<e>`

三个特殊标记的作用：
- `<s>` —— 告诉模型"新对话开始了"
- `<sep>` —— 划分问题和回答的边界。训练时模型看到完整的问答对，推理时模型看到问题 + `<sep>` 就知道"该我回答了"
- `<e>` —— 回答结束的信号

### 5.2 从原始文本到模型输入的完整变换

一条数据从 JSONL 变成模型能吃的 tensor，中间要经过好几步。我们拿微调数据（`NovaDataset`）来看完整过程：

```python
# dataset.py — NovaDataset.__init__（微调数据编码核心逻辑）

def __init__(self, qa_pairs, tokenizer, max_seq_len):
    bos_id = tokenizer.char_to_id[BOS_TOKEN]
    sep_id = tokenizer.char_to_id[SEP_TOKEN]
    eos_id = tokenizer.char_to_id[EOS_TOKEN]

    # 预分配 numpy 数组，避免逐条 append 的内存碎片
    n = len(qa_pairs)
    self.input_ids  = np.zeros((n, max_seq_len), dtype=np.int16)
    self.target_ids = np.full((n, max_seq_len), -100, dtype=np.int16)

    # 批量编码（利用 tokenizers 库的 Rust 加速）
    q_encs = tokenizer._tokenizer.encode_batch(chunk_q)
    a_encs = tokenizer._tokenizer.encode_batch(chunk_a)

    for j, (q, a) in enumerate(zip(q_encs, a_encs)):
        # 拼接: [BOS] + question_ids + [SEP] + answer_ids + [EOS]
        pos = 0
        self.input_ids[i, pos] = bos_id;  pos += 1
        self.input_ids[i, pos:pos+qlen] = q.ids[:qlen];  pos += qlen
        self.input_ids[i, pos] = sep_id;  pos += 1
        self.input_ids[i, pos:pos+alen] = a.ids[:alen];  pos += alen
        self.input_ids[i, pos] = eos_id;  pos += 1

        # 构造 labels: input_ids 左移一位
        self.target_ids[i, :pos-1] = self.input_ids[i, 1:pos]
```

`target_ids` 的构造是关键——它就是 `input_ids` 往左移一位。举个具体例子：

```
input_ids:  [<s>,  你,  好,  <sep>,  我,  是,  <e>,  <pad>, <pad>]
target_ids: [ 你,  好,  <sep>, 我,  是,  <e>,  -100, -100,  -100]
```

每个位置的训练目标就是"下一个 token 是什么"。`-100` 是 PyTorch 交叉熵的"忽略标记"——padding 位置不计入 loss。

### 5.3 推理流程

```
用户输入: "你叫什么名字？"

模型接收: <s>你叫什么名字？<sep>
                              ↑ 从这里开始逐字生成

→ 我 → 是 → 名 → 为 → N → o → v → a → 的
→ 微 → 型 → L → L → M → 。 → <e>
                                  ↑ 遇到结束标记，停止
```

---

## 六、模型规格一览

| 参数 | 值 | 说明 |
|------|-----|------|
| 架构 | Decoder-Only Transformer | 与 GPT/LLaMA 同源 |
| 嵌入维度 (d_model) | 384 | 每个 token 的向量维度 |
| 注意力头数 (n_heads) | 6 | 并行的注意力通道 |
| 每头维度 (head_dim) | 64 | d_model / n_heads |
| Decoder 层数 (n_layers) | 4 | Transformer Block 的堆叠层数 |
| FFN 隐藏维度 (d_ff) | 1536 | 前馈网络的中间扩展维度 |
| 最大序列长度 | 1024 | 单次输入上限 |
| 激活函数 | SwiGLU | LLaMA / PaLM 同款 |
| 归一化 | Pre-RMSNorm | 训练更稳定 |
| 位置编码 | 可学习嵌入 | GPT 系列方案 |
| 参数量 | ~22M | 微型规模 |
| 词表大小 | ~8000-16000 | BPE 子词 |
| 训练策略 | 预训练 + 微调 | 两阶段 |
| 优化器 | AdamW | 自适应学习率 + 权重衰减 |
| 学习率策略 | Warmup + Cosine Decay | 先热身再衰减 |

---

## 七、和主流 LLM 的技术栈对照

| 技术点 | Nova | GPT-2 | LLaMA | 说明 |
|--------|------|-------|-------|------|
| 架构 | Decoder-Only | Decoder-Only | Decoder-Only | 一脉相承 |
| 归一化 | Pre-RMSNorm | Pre-LayerNorm | Pre-RMSNorm | Nova 跟 LLaMA 对齐 |
| 激活函数 | SwiGLU | GELU | SwiGLU | Nova 跟 LLaMA 对齐 |
| 位置编码 | 可学习嵌入 | 可学习嵌入 | RoPE | RoPE 更新但微型模型不需要 |
| 注意力 | MHA | MHA | GQA | GQA 是推理优化，微型模型用不上 |
| 分词器 | BPE | BPE | BPE (SentencePiece) | 同一技术路线 |
| 优化器 | AdamW | Adam | AdamW | 标准选择 |
| 训练策略 | 预训练 + 微调 | 预训练 | 预训练 + SFT + RLHF | 路径一致，RLHF 微型模型不需要 |

简单来说：Nova 在每个技术选型上，要么和 GPT-2 一致，要么和 LLaMA 一致——都是经过工业界大规模验证的方案。区别只在于规模：它们是巨轮，Nova 是快艇，但发动机的原理是一样的。

---

## 八、关键设计决策 FAQ

**为什么选 Decoder-Only 而不是 Encoder-Decoder？**

Encoder-Decoder 适合"输入一段、输出另一段"的任务（比如翻译）。但对于对话场景，Decoder-Only 更自然——它把输入和输出拼成一个序列，统一处理。GPT、LLaMA、Mistral、Qwen 都是这个路子。没有理由在 2024 年做一个对话模型还用 Encoder-Decoder。

**为什么用 BPE 分词器？**

BPE（Byte Pair Encoding）是当前所有主流 LLM 使用的分词方案（GPT 用的 tiktoken、LLaMA 用的 SentencePiece 都是 BPE 的变种）。它的优势在于：同样长度的 token 序列能装下更多语义信息——"人工智能"4 个字可能只需要 1-2 个 token，而字符级分词需要 4 个。这意味着模型的有效上下文窗口更长，不需要浪费容量去学"字怎么组成词"。

**22M 参数能干什么？**

别指望它写论文、做数学题或者聊人生哲理。但对于预设的问答场景（"你叫什么名字""今天天气怎么样"），22M 参数可以做到流畅自然的回答。模型不需要"理解世界"，只需要记住"看到这类问题就输出那类回答"的模式。经过预训练的模型还具备一定的语言通用能力。

**为什么用 RMSNorm 而不是 LayerNorm？**

RMSNorm 去掉了 LayerNorm 中的均值偏移计算，参数更少、计算更快，效果几乎一样。LLaMA 论文实测验证了这一点。既然业界已经帮我们趟过路了，没理由不跟。

**为什么 Pre-Norm 而不是 Post-Norm？**

Pre-Norm 训练稳定性远好于 Post-Norm，尤其在深层网络中。这一点从 GPT-2 开始就已经是共识了。

**为什么要两阶段训练？**

和人学习一样——你得先学会认字、说话（预训练），才能学怎么对话、怎么回答问题（微调）。如果直接用几百条问答对从零训练，模型连基本的语言结构都没学会，怎么可能学好对话？预训练让模型在海量文本上学会语言的底层规律，微调只需要在这个基础上"教它说话的方式"，事半功倍。

---

## 九、源码文件导览

| 文件 | 职责 | 关键类/函数 |
|------|------|-----------|
| `config.py` | 所有超参数的集中管理 | `NovaConfig` dataclass |
| `tokenizer.py` | BPE 分词器：文字 ↔ 数字 ID | `NovaTokenizer`（train_from_texts / encode / decode） |
| `dataset.py` | 数据加载与预处理 | `PretrainDataset` / `NovaDataset` / `create_dataloader` |
| `model.py` | 完整的 Decoder-Only Transformer | `RMSNorm` / `SwiGLUFFN` / `MultiHeadAttention` / `TransformerBlock` / `NovaModel` |
| `train.py` | 两阶段训练流程 | `setup_pretrain` / `setup_finetune` / `train` / `get_lr` |
| `chat.py` | 推理与交互式对话 | `generate` / `chat_loop` / `load_model_for_inference` |
