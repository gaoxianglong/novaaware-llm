"""基于HuggingFace tokenizers构件的Unicode character-level BPE分词算法实现

因为Transformer不认识文字，只认识数字，所以必须先由分词器把字符和高频词组切分并映射为对应的tokenid，训练或推理时再由分词器进行"文字 <-> tokenid"之间的双向转换

最早的时候我是自己手写的基于字符的分词器，但是模型训练时发现效果不如BPE分词，因为：
  1、虽然他们的初始单元一样，都是Unicode字符，但是BPE分词会针对高频词组做合并动作，那就意味着基于BPE分词器生成的词表文件中包含“字符+词组”的组合；
  2、1个token可以表示的语义更加丰富，因为基于BPE分词后的token可以表示一个由n个字符构成的词组；
  3、token的上下文信息能够承载更多的内容；
  4、模型在预训练阶段整个训练负载会下降，因为BPE训练阶段就已经把高频组词这个训练完成了；

  一条完整对话的 token 序列示例:
    问题: "你好吗？"  回答: "我很好"
    → [1, 42, 15, 88, 7, 3, 56, 23, 91, 2]
       ↑                 ↑               ↑
      <s>              <sep>            <e>
      序列开始        问答分隔         序列结束
"""

from __future__ import annotations

from typing import List

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


# 5 个特殊 token 的名称和固定 ID
# 填充符，训练时把短序列补齐到统一长度
PAD_TOKEN, PAD_ID = "<pad>", 0
# 序列开始标记
BOS_TOKEN, BOS_ID = "<s>", 1
# 序列结束标记（模型生成到这个 token 就停下来）
EOS_TOKEN, EOS_ID = "<e>", 2
# 问题和回答之间的分隔符
SEP_TOKEN, SEP_ID = "<sep>", 3
# 未知字符（分词器无法处理的输入用这个代替）
UNK_TOKEN, UNK_ID = "<unk>", 4

SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, SEP_TOKEN, UNK_TOKEN]
NUM_SPECIAL = len(SPECIAL_TOKENS)  # 普通子词从 ID=5 开始编号


class NovaTokenizer:
    def __init__(self) -> None:
        self._tokenizer: Tokenizer | None = None
        self.vocab_size: int = 0

    # ------------------------------------------------------------------
    # BPE分词训练
    # 创建 BPE 模型 -> 配置预分词器 -> 配置解码器 -> 创建 BPE 训练器 -> 执行 BPE 训练 -> 保存内部引用并更新 vocab_size
    # ------------------------------------------------------------------
    def train_from_texts(
        self,
        # 语料文本列表
        # 预训练: ["一段长文本...", "另一段文本...", ...]
        # 微调:   ["问题1回答1", "问题2回答2", ...]
        texts: List[str],
        # 词表大小的上限，超过这个值后，训练器会停止合并子词
        # 小于则以最终合并的子词为准
        vocab_size: int = 8000,
    ) -> None:
        # 1、创建 BPE 模型
        # 设置后续推理时遇见未知Unicode字符就用 token <unk>代替
        tokenizer = Tokenizer(models.BPE(unk_token=UNK_TOKEN))

        # 2、配置预分词器
        # 粗切，把一段语料文本切成若干段，后续进行token化，如果不做粗切直接在整段文本中找高频token对，会出现一些无意义的词组合并
        # 按 Unicode 脚本边界（不同语言交接的地方） + 数字 + 标点符号进行初步切分，这样中文字符、英文单词、数字、标点会被分到不同的组
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                # UnicodeScripts默认会按照书写系统进行Unicode字符切分，汉字、英文、数字、标点会被分到不同的组
                # 比如：输入: "我今天很happy，你happy吗"
                # UnicodeScripts 切分:
                #   "我今天很" | "happy" | "，" | "你" | "happy" | "吗"
                #    Han        Latin     Common  Han   Latin     Han
                # BPE 的高频合并严格限制在预分词切好的每一段内部进行，不会跨段合并，避免一些无意义的合并
                # 比如："我今天很" 内部：可以合并 "今"+"天" → "今天" ✓， "很"+"h" 跨段了 → 禁止 ✗
                pre_tokenizers.UnicodeScripts(),
                # 除了按脚本边界、数字、标点符号切分外，还要指定预分词器组合空格切分，这主要是给英文这种基于空格分词的语言使用的
                # 如果使用空格分词的语言不加上空格切分，会出现一些跨单词的无意义的组合，比如love you，会组合成e y。
                # Whitespace 切分（在每段内部再按空格切）:
                #   这个例子里没有空格，所以结果不变:
                #   ["我今天很", "happy", "，", "你", "happy", "吗"]
                pre_tokenizers.Whitespace(),
            ]
        )

        # 3、配置解码器
        # 这里用的是Fuse解码器，会把所有相邻的token拼接成一个字符串，比如：[156, 2847] → ["Nova", "可爱"] → “Nova可爱”
        # Fuse解码器在拼接字符串的时候，采用的是直接拼接，不会在token之间加任何的分隔符，比如英文的空格
        tokenizer.decoder = decoders.Fuse()

        # 4、创建 BPE 训练器
        # 这一步主要是从语料中学习子词和词组的合并规则
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            # 指定特殊token，这些token在训练过程中不会被合并
            special_tokens=SPECIAL_TOKENS,
            # 设置BPE分词训练的进度条显示
            show_progress=True,
            # min_frequency=2: 至少出现 2 次的字符对才会被考虑合并
            min_frequency=2,
        )

        # 5、执行 BPE 训练
        # BPE 训练会进行 N 轮，每一轮全量扫描语料中的所有内容，找到频率最高的一组 token对 作为一条 merge 规则存下来，直至满足任意一个条件时停止：
        # 1、所有候选 token对 的频率 < min_frequency
        # 2、词表大小 = vocab_size 上限
        tokenizer.train_from_iterator(texts, trainer=trainer)

        # 6、保存内部引用并更新 vocab_size
        self._tokenizer = tokenizer
        # 更新实际词表大小
        self.vocab_size = self._tokenizer.get_vocab_size()

    # ------------------------------------------------------------------
    # 文本序列 -> token_ids（仅推理使用）
    # ------------------------------------------------------------------

    # 推理时，编码器先把文本拆成单字符序列，然后按 merges 的顺序（从第 1 条到最后一条）逐条检查并合并，合并完成后，再查词表把每个 字符 转成对应的 token ID。
    # 第 1 步: 文本 → 拆成单字符
    #   "什么是" → ["什", "么", "是"]
    # 第 2 步: 按 merges 顺序逐条合并
    #   ["什", "么", "是"]
    #   → 第1条 merge ["什","么"] 命中 → ["什么", "是"]
    #   → 第2条 merge ["什么","是"] 命中 → ["什么是"]
    # 第 3 步: 合并完成后，查词表把每个 token 转成 ID
    #   "什么是" → 5678
    def encode(self, text: str) -> List[int]:
        if not text:
            return []
        if self._tokenizer is None:
            raise RuntimeError("分词器未初始化，请先调用 train_from_texts() 或 load()")
        # 将输入文本序列编码映射为一组tokenids序列
        return self._tokenizer.encode(text).ids

    # ------------------------------------------------------------------
    # token_ids -> 文本序列（仅推理使用）
    # ------------------------------------------------------------------
    def decode(self, ids: List[int]) -> str:
        if not ids:
            return ""
        if self._tokenizer is None:
            raise RuntimeError("分词器未初始化，请先调用 train_from_texts() 或 load()")

        # 过滤特殊标记（PAD/BOS/EOS/SEP）
        skip_ids = {PAD_ID, BOS_ID, EOS_ID, SEP_ID}
        filtered = [tid for tid in ids if tid not in skip_ids]

        if not filtered:
            return ""

        # 执行解码动作，把一组tokenid解码为文本序列
        # 解码动作依赖于训练时候BPE分词器的初始参数tokenizer.decoder = decoders.Fuse()，解码拼接文本的时候不会加任何分隔符
        return self._tokenizer.decode(filtered)

    # ------------------------------------------------------------------
    # 将分词器持久化到磁盘，输出data/tokenizer.json
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        if self._tokenizer is None:
            raise RuntimeError("分词器未初始化，无法保存")
        # tokenizers默认的转储格式就是json
        self._tokenizer.save(path)

    # ------------------------------------------------------------------
    # 推理时从磁盘加载已有分词器
    # ------------------------------------------------------------------
    def load(self, path: str) -> None:
        # 加载已经训练好的BPE分词器
        self._tokenizer = Tokenizer.from_file(path)
        # 获取实际的分词大小
        self.vocab_size = self._tokenizer.get_vocab_size()
