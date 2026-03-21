# Nova 微型 LLM — 实施计划清单

> 本文档是 Nova 项目的完整实施路线图。每个阶段包含具体的执行步骤、验收标准和预期产出。按顺序执行，每完成一个阶段就在对应的检查框打勾。

---

## 阶段零：项目初始化

**目标：** 搭建开发环境和项目骨架。

### 执行步骤

- [ ] **0.1** 使用 Python 3.11.12 创建虚拟环境（python3.11.12已经安装）
  ```bash
  python3.11 -m venv .venv
  source .venv/bin/activate
  ```
- [ ] **0.2** 创建 `requirements.txt`，包含以下依赖：
  - `torch>=2.0.0`（PyTorch，核心深度学习框架）
  - `numpy>=1.24.0`（数值计算）
- [ ] **0.3** 安装依赖
  ```bash
  pip install -r requirements.txt
  ```
- [ ] **0.4** 创建项目目录结构
  ```
  novaaware-llm/
  ├── requirements.txt
  ├── README.md
  ├── config.py
  ├── tokenizer.py
  ├── dataset.py
  ├── model.py
  ├── train.py
  ├── chat.py
  ├── data/
  │   └── qa_pairs.json
  ├── checkpoints/       ← 训练时自动创建
  └── docs/
      ├── ARCHITECTURE.md
      └── IMPLEMENTATION_PLAN.md
  ```
- [ ] **0.5** 创建 `README.md`，包含项目简介、环境要求、快速上手指南

### 验收标准

- 虚拟环境可正常激活
- `python -c "import torch; print(torch.__version__)"` 输出正常
- 目录结构完整

---

## 阶段一：模型配置

**目标：** 集中管理所有超参数，方便后续调优。

### 执行步骤

- [ ] **1.1** 创建 `config.py`，使用 Python `dataclass` 定义配置类：

  ```python
  @dataclass
  class NovaConfig:
      # 模型架构
      d_model: int = 128          # 嵌入维度
      n_heads: int = 4            # 注意力头数
      n_layers: int = 4           # Transformer 层数
      d_ff: int = 512             # FFN 隐藏维度
      max_seq_len: int = 128      # 最大序列长度
      dropout: float = 0.1        # Dropout 概率
      vocab_size: int = 0         # 由分词器决定，初始化时设置

      # 训练配置
      batch_size: int = 16
      epochs: int = 500
      learning_rate: float = 3e-4
      weight_decay: float = 0.01
      warmup_steps: int = 100
      grad_clip: float = 1.0      # 梯度裁剪

      # 生成配置
      temperature: float = 0.8
      top_k: int = 20
  ```

### 验收标准

- `from config import NovaConfig; cfg = NovaConfig()` 可正常运行
- 所有参数都有合理的默认值

---

## 阶段二：训练数据准备

**目标：** 构建高质量的中文问答数据集，覆盖多种问法和回答。

### 执行步骤

- [ ] **2.1** 创建 `data/qa_pairs.json`，JSON 格式如下：
  ```json
  [
    {"question": "你叫什么名字？", "answer": "我是名为Nova的微型LLM。"},
    {"question": "你的名字是什么？", "answer": "我是名为Nova的微型LLM。"},
    ...
  ]
  ```

- [ ] **2.2** 编写核心问答对（4 组核心主题）：

  | 主题 | 问题变体数 | 回答变体数 |
  |------|-----------|-----------|
  | 名字/身份 | 8-10 种问法 | 3-4 种答法 |
  | 问候/状态 | 8-10 种问法 | 3-4 种答法 |
  | 无法回答的问题 | 8-10 种问法 | 3-4 种答法 |
  | 开发者信息 | 8-10 种问法 | 3-4 种答法 |

- [ ] **2.3** 补充通用对话数据：

  | 主题 | 示例 | 条数 |
  |------|------|------|
  | 打招呼 | "你好" → "你好！有什么可以帮你的吗？" | 8-10 |
  | 感谢 | "谢谢" → "不客气，很高兴能帮到你。" | 5-8 |
  | 告别 | "再见" → "再见！期待下次和你聊天。" | 5-8 |
  | 能力说明 | "你能做什么？" → "我可以回答一些简单的问题..." | 5-8 |
  | 自我认知 | "你有感情吗？" → "我是一个AI程序..." | 5-8 |
  | 简单常识 | "1+1等于几？" → "1+1等于2。" | 8-10 |
  | 中文理解 | "什么是人工智能？" → "人工智能是..." | 5-8 |

- [ ] **2.4** 验证数据：确保总计 **80-120 条**问答对，格式一致、无错别字

### 验收标准

- JSON 文件可被正确解析
- 数据量达到 80-120 条
- 每个核心主题至少有 8 种不同问法
- 所有问答对人工审核通过

---

## 阶段三：字符级分词器

**目标：** 实现一个能双向转换中文字符和数字 ID 的分词器。

### 执行步骤

- [ ] **3.1** 创建 `tokenizer.py`，实现 `NovaTokenizer` 类

- [ ] **3.2** 实现词表构建方法 `build_vocab(texts)`：
  - 扫描所有训练文本，收集所有出现过的字符
  - 预留 5 个特殊 token：`<pad>`(0), `<s>`(1), `<e>`(2), `<sep>`(3), `<unk>`(4)
  - 按字符出现频率排序，依次分配 ID（从 5 开始）
  - 构建 `char_to_id` 和 `id_to_char` 双向映射

- [ ] **3.3** 实现编码方法 `encode(text) -> List[int]`：
  - 遍历文本每个字符，查表得到 ID
  - 不在词表中的字符映射为 `<unk>`

- [ ] **3.4** 实现解码方法 `decode(ids) -> str`：
  - 遍历 ID 列表，查表得到字符
  - 过滤掉特殊 token（`<pad>`, `<s>`, `<e>`, `<sep>`）
  - 拼接成字符串返回

- [ ] **3.5** 实现保存和加载方法：
  - `save(path)`：将词表保存为 JSON 文件
  - `load(path)`：从 JSON 文件加载词表

- [ ] **3.6** 编写单元测试(检查哪些没有)：
  - 测试编码→解码的一致性
  - 测试特殊 token 处理
  - 测试未知字符处理

### 验收标准

- `encode("你好") → [id1, id2]`，`decode([id1, id2]) → "你好"`
- 特殊 token ID 固定且正确
- 词表可保存和加载

---

## 阶段四：数据集与数据加载

**目标：** 实现 PyTorch Dataset，将原始问答对转换为模型可消费的 tensor。

### 执行步骤

- [ ] **4.1** 创建 `dataset.py`，实现 `NovaDataset(torch.utils.data.Dataset)`

- [ ] **4.2** 实现数据编码逻辑：
  - 将每条问答对拼接为 `<s>问题<sep>回答<e>` 的格式
  - 使用分词器编码为 ID 序列
  - 截断超过 `max_seq_len` 的序列
  - 填充（padding）不足 `max_seq_len` 的序列

- [ ] **4.3** 实现 `__getitem__` 方法：
  - 返回 `input_ids`（输入序列）和 `labels`（目标序列）
  - `labels` = `input_ids` 右移一位（即预测下一个 token）
  - padding 位置的 label 设为 -100（PyTorch 的 ignore_index）

- [ ] **4.4** 实现 `__len__` 方法

- [ ] **4.5** 创建 `DataLoader` 工厂函数：
  - 设置 batch_size、shuffle=True
  - 使用 `collate_fn` 统一处理 padding

### 验收标准

- 可以正确迭代 DataLoader，获取 batch
- batch 中的 input_ids 和 labels 维度正确
- padding 位置的 label 为 -100

---

## 阶段五：Transformer 模型实现

**目标：** 从零实现完整的 Decoder-Only Transformer，遵循现代 LLM 标准。

### 执行步骤

- [ ] **5.1** 实现 `RMSNorm` 类（替代标准 LayerNorm）：
  - RMSNorm 是 LLaMA 使用的归一化方式，比 LayerNorm 更简洁高效
  - 参数：可学习的缩放参数 gamma

- [ ] **5.2** 实现 `SwiGLU` 前馈网络类：
  ```python
  class SwiGLUFFN(nn.Module):
      def __init__(self, d_model, d_ff):
          self.w1 = nn.Linear(d_model, d_ff, bias=False)
          self.w2 = nn.Linear(d_ff, d_model, bias=False)
          self.w3 = nn.Linear(d_model, d_ff, bias=False)

      def forward(self, x):
          return self.w2(F.silu(self.w1(x)) * self.w3(x))
  ```

- [ ] **5.3** 实现 `MultiHeadAttention` 类：
  - Q、K、V 线性投影（d_model → d_model）
  - 拆分为多头（reshape 为 [batch, n_heads, seq_len, head_dim]）
  - 缩放点积注意力 + 因果掩码
  - 输出投影（d_model → d_model）
  - Dropout

- [ ] **5.4** 实现 `TransformerBlock` 类（一层 Decoder）：
  ```
  Pre-LN 结构:
  x → RMSNorm → MultiHeadAttention → + x → RMSNorm → SwiGLU FFN → + x
  ```

- [ ] **5.5** 实现 `NovaModel` 类（完整模型）：
  - Token Embedding（vocab_size → d_model）
  - Positional Embedding（max_seq_len → d_model）
  - Dropout
  - N 层 TransformerBlock 堆叠
  - 最终 RMSNorm
  - 输出线性层（d_model → vocab_size）

- [ ] **5.6** 实现权重初始化：
  - 嵌入层：正态分布 N(0, 0.02)
  - 线性层：Xavier 均匀分布
  - LayerNorm/RMSNorm：gamma 初始化为 1

- [ ] **5.7** 添加参数量统计方法：
  - 打印总参数量和各层参数量

- [ ] **5.8** 单元测试：
  - 构造随机输入，验证前向传播输出维度正确
  - 验证因果掩码工作正常
  - 验证参数量在预期范围内（1-2M）

### 验收标准

- 模型可正常前向传播，输出 shape 为 `[batch_size, seq_len, vocab_size]`
- 因果掩码正确：位置 i 只能注意到位置 0~i
- 参数量在 1-2M 范围内
- 无 NaN/Inf 输出

---

## 阶段六：训练流程

**目标：** 实现完整的训练循环，支持断点续训和最优模型保存。

### 执行步骤

- [ ] **6.1** 创建 `train.py`，实现主训练函数

- [ ] **6.2** 实现学习率调度器（Warmup + Cosine Decay）：
  ```python
  def get_lr(step, warmup_steps, max_steps, max_lr, min_lr=1e-6):
      if step < warmup_steps:
          return max_lr * step / warmup_steps
      decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
      coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
      return min_lr + coeff * (max_lr - min_lr)
  ```

- [ ] **6.3** 实现训练循环（核心——所有参数的更新都发生在这里）：
  - 遍历 epochs
  - 遍历 DataLoader 的每个 batch
  - 每个 batch 执行：前向传播 → 计算 loss → 反向传播 → 梯度裁剪 → 参数更新
  - 记录 loss 和学习率
  ```python
  for epoch in range(num_epochs):
      model.train()
      total_loss = 0.0

      # 更新学习率
      lr = get_lr(epoch, warmup_steps, num_epochs, max_lr)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr

      for batch in dataloader:
          input_ids = batch['input_ids'].to(device)    # [batch_size, seq_len]
          target_ids = batch['target_ids'].to(device)   # [batch_size, seq_len]

          # ① 前向传播：输入走完整条流水线（查字义表 → 加位置表 → 4层Transformer Block → 输出层）
          logits = model(input_ids)                     # [batch_size, seq_len, vocab_size]

          # ② 计算损失：预测结果和正确答案差多少
          loss = F.cross_entropy(
              logits.view(-1, vocab_size),
              target_ids.view(-1),
              ignore_index=-100    # 忽略 <pad> 位置，不让模型学填充内容
          )

          # ③ 反向传播：从 loss 出发，沿着 输出层→FFN→注意力→QKV→工作向量→字义表
          #    这条链路倒着走一遍，算出每个参数的梯度（该往哪个方向微调、调多少）
          optimizer.zero_grad()    # 清掉上一轮的梯度，防止累加
          loss.backward()          # PyTorch 自动完成整条反向传播链

          # ④ 梯度裁剪：防止梯度爆炸（某些参数的梯度太大，一步迈太远）
          torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

          # ⑤ 参数更新：AdamW 根据梯度，把所有参数（字义表、位置表、QKV矩阵、FFN权重……）
          #    各自微调一点点。这一步执行完，字义表里的数字才真正发生变化
          optimizer.step()

          total_loss += loss.item()

      avg_loss = total_loss / len(dataloader)
  ```
  > **注意**：`loss.backward()` 这一行就是整个反向传播的触发点。PyTorch 会自动沿着前向传播的路径倒着走一遍，算出所有参数的梯度。`optimizer.step()` 则根据梯度执行实际的参数更新。字义表、位置表、QKV 变换矩阵、FFN 权重等所有参数，都在 `optimizer.step()` 这一刻被统一微调。

- [ ] **6.4** 实现训练日志：
  - 每 10 个 epoch 打印：epoch、loss、学习率、耗时
  - 格式示例：`[Epoch 010/500] loss=3.2456 lr=2.8e-04 time=1.2s`

- [ ] **6.5** 实现 checkpoint 保存：
  - 创建 `checkpoints/` 目录
  - 每 50 个 epoch 保存一次
  - 始终保存 loss 最低的模型为 `best_model.pt`
  - checkpoint 内容：model_state_dict、optimizer_state_dict、epoch、loss、config

- [ ] **6.6** 实现断点续训功能：
  - 命令行参数 `--resume` 指定 checkpoint 路径
  - 加载模型参数、优化器状态、epoch 断点

- [ ] **6.7** 训练结束后输出汇总：
  - 最终 loss
  - 最佳 loss 及对应 epoch
  - 总训练时间
  - 模型保存路径

### 验收标准

- loss 随训练逐步下降
- checkpoint 文件正确保存且可加载
- 断点续训后 loss 能继续下降
- 训练 500 个 epoch 后，loss 降到 0.1 以下（数据量小，应能过拟合）

---

## 阶段七：推理与交互式对话

**目标：** 实现命令行对话界面，用户可以和训练好的 Nova 对话。

### 执行步骤

- [ ] **7.1** 创建 `chat.py`，实现 `generate()` 函数：
  - 输入：问题文本
  - 编码为 `<s>问题<sep>` 的 token 序列
  - 自回归生成：循环调用模型，每次取最后一个位置的 logits
  - 应用 temperature 缩放
  - Top-k 采样：只保留概率最高的 k 个 token，其余设为 -inf
  - 采样得到下一个 token，追加到序列中
  - 遇到 `<e>` 或达到最大长度时停止
  - 解码生成的 token 序列为文本

- [ ] **7.2** 实现命令行交互界面：
  ```
  ╔══════════════════════════════════════╗
  ║     Nova 微型 LLM - 交互式对话       ║
  ╚══════════════════════════════════════╝

  Nova> 你好！我是 Nova，一个微型 LLM。输入 "quit" 退出。

  你: 你叫什么名字？
  Nova: 我是名为Nova的微型LLM。

  你: quit
  Nova> 再见！
  ```

- [ ] **7.3** 命令行参数支持：
  - `--checkpoint`：指定模型 checkpoint 路径（默认 `checkpoints/best_model.pt`）
  - `--temperature`：生成温度（默认 0.8）
  - `--top_k`：Top-k 采样的 k 值（默认 20）

- [ ] **7.4** 错误处理：
  - checkpoint 文件不存在时给出友好提示
  - 空输入跳过
  - Ctrl+C 优雅退出

### 验收标准

- 输入核心问题，能得到正确回答
- 响应时间 < 1 秒（微型模型，CPU 即可）
- 界面友好、无崩溃

---

## 阶段八：端到端测试与调优

**目标：** 验证完整流程，优化模型效果。

### 执行步骤

- [ ] **8.1** 端到端冒烟测试：
  - 从数据准备到训练到推理，走完整个流程
  - 记录发现的问题

- [ ] **8.2** 效果验证（核心问答）：

  | 测试输入 | 期望输出 | 通过？ |
  |---------|---------|-------|
  | 你叫什么名字？ | 我是名为Nova的微型LLM。 | |
  | 你好吗？ | 我很好，谢谢关心。 | |
  | 今天天气怎么样？ | 这是一个需要实时数据的问题，我无法回答。 | |
  | 你是谁开发的？ | 我是由高翔龙开发的微型LLM。 | |
  | 你好 | （某种打招呼的回复） | |
  | 谢谢 | （某种感谢的回复） | |

- [ ] **8.3** 效果验证（泛化能力）：
  - 用训练集中**没有出现过的问法**提问，观察回答质量
  - 例如："请介绍一下你自己"（训练集中可能没有这个原句）

- [ ] **8.4** 参数调优（如果效果不理想）：

  | 问题 | 调优方向 |
  |------|---------|
  | loss 不下降 | 增大学习率、检查数据格式 |
  | loss 下降但回答不对 | 增加训练轮数、增加数据量 |
  | 回答重复/死循环 | 降低 temperature、调整 top_k |
  | 回答乱码 | 检查分词器、检查 padding 处理 |

- [ ] **8.5** 完成 `README.md`：
  - 项目介绍
  - 快速开始指南（环境搭建 → 训练 → 对话）
  - 模型架构说明
  - 参数调优指南

### 验收标准

- 4 个核心问答全部通过
- 至少 2 个泛化问题能得到合理回答
- README 文档完整可用

---

## 执行时间线（预估）

| 阶段 | 任务 | 预估耗时 |
|------|------|---------|
| 阶段零 | 项目初始化 | 10 分钟 |
| 阶段一 | 模型配置 | 10 分钟 |
| 阶段二 | 训练数据准备 | 30 分钟 |
| 阶段三 | 字符级分词器 | 20 分钟 |
| 阶段四 | 数据集与数据加载 | 20 分钟 |
| 阶段五 | Transformer 模型实现 | 60 分钟 |
| 阶段六 | 训练流程 | 30 分钟 |
| 阶段七 | 推理与对话 | 30 分钟 |
| 阶段八 | 端到端测试与调优 | 30 分钟 |
| **合计** | | **约 4 小时** |

---

## 依赖关系图

```
阶段零(项目初始化)
    │
    ├──→ 阶段一(模型配置)
    │        │
    │        ├──→ 阶段五(模型实现) ──→ 阶段六(训练流程) ──→ 阶段八(测试调优)
    │        │                                    ↑
    │        └──→ 阶段七(推理对话) ────────────────┘
    │
    └──→ 阶段二(训练数据)
             │
             ├──→ 阶段三(分词器)
             │        │
             └──→ 阶段四(数据集) ←──┘
```

说明：
- 阶段零完成后，阶段一和阶段二可以**并行**进行
- 阶段三依赖阶段二的数据来构建词表
- 阶段四依赖阶段二的数据和阶段三的分词器
- 阶段五只依赖阶段一的配置
- 阶段六依赖阶段四和阶段五
- 阶段七依赖阶段一和阶段五（可以和阶段六并行开发）
- 阶段八是最终集成测试，依赖前面所有阶段
