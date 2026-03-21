"""Nova 模型组件单元测试

按模型组件拆分测试类，每个组件对应 IMPLEMENTATION_PLAN 中的一个步骤:
  TestRMSNorm  — RMSNorm 归一化（步骤 5.1）

运行方式:
  .venv/bin/python -m unittest tests.test_model -v
"""

import unittest
import math

import torch
import torch.nn as nn

from config import NovaConfig
from model import RMSNorm, SwiGLUFFN, MultiHeadAttention, TransformerBlock, NovaModel


# ======================================================================
# RMSNorm 测试（步骤 5.1）
#
# RMSNorm 的核心职责:
#   1. 对输入向量的最后一个维度做均方根归一化
#   2. 乘以可学习参数 gamma
#   3. 保持输入输出形状一致
#
# RMSNorm 在 Transformer 中的调用位置:
#   TransformerBlock.forward:
#     x → attn_norm(x) → 自注意力 → +x → ffn_norm(x) → FFN → +x
#   NovaModel.forward:
#     所有 Block 之后 → final_norm(x) → 输出层
# ======================================================================
class TestRMSNorm(unittest.TestCase):
    """测试 RMSNorm 的数值计算和属性。"""

    def setUp(self) -> None:
        self.dim = 128
        self.norm = RMSNorm(self.dim)

    # ---- 形状测试 ----

    def test_output_shape_matches_input(self) -> None:
        """输出形状必须与输入完全一致。"""
        x = torch.randn(2, 10, self.dim)  # [batch=2, seq_len=10, dim=128]
        out = self.norm(x)
        self.assertEqual(out.shape, x.shape)

    def test_works_with_different_batch_sizes(self) -> None:
        """支持不同 batch_size（1, 16, 32 等）。"""
        for batch_size in [1, 4, 16]:
            x = torch.randn(batch_size, 8, self.dim)
            out = self.norm(x)
            self.assertEqual(out.shape, (batch_size, 8, self.dim))

    def test_works_with_different_seq_lengths(self) -> None:
        """支持不同序列长度。"""
        for seq_len in [1, 64, 128]:
            x = torch.randn(2, seq_len, self.dim)
            out = self.norm(x)
            self.assertEqual(out.shape, (2, seq_len, self.dim))

    # ---- 数值正确性测试 ----

    def test_output_rms_is_approximately_one(self) -> None:
        """归一化后，每个向量的 RMS 应约等于 1（gamma 初始全为 1 时）。

        RMS(output) = sqrt(mean(output²)) ≈ 1.0
        """
        x = torch.randn(4, 10, self.dim)
        out = self.norm(x)
        # 计算每个向量的 RMS
        rms_values = torch.sqrt(out.float().pow(2).mean(dim=-1))
        # gamma 初始全为 1 时，归一化后 RMS 应为 1
        torch.testing.assert_close(
            rms_values,
            torch.ones_like(rms_values),
            atol=1e-5, rtol=1e-5,
        )

    def test_manual_computation(self) -> None:
        """手动计算一个简单例子，验证公式正确。

        x = [0.8, -1.2, 0.4, 2.0]
        RMS = sqrt(mean([0.64, 1.44, 0.16, 4.00]) + 1e-6)
            = sqrt(1.56 + 1e-6) ≈ 1.24900
        x_norm = x / RMS
        output = x_norm * gamma (gamma 全为 1)
        """
        norm_4d = RMSNorm(4)
        x = torch.tensor([[[0.8, -1.2, 0.4, 2.0]]])  # [1, 1, 4]

        out = norm_4d(x)

        rms = math.sqrt((0.64 + 1.44 + 0.16 + 4.00) / 4 + 1e-6)
        expected = torch.tensor([[[0.8 / rms, -1.2 / rms, 0.4 / rms, 2.0 / rms]]])
        torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)

    def test_no_nan_or_inf(self) -> None:
        """输出不应包含 NaN 或 Inf。"""
        x = torch.randn(4, 10, self.dim)
        out = self.norm(x)
        self.assertFalse(torch.isnan(out).any(), "输出包含 NaN")
        self.assertFalse(torch.isinf(out).any(), "输出包含 Inf")

    def test_no_nan_with_zero_input(self) -> None:
        """全零输入不应产生 NaN（eps 防除零）。"""
        x = torch.zeros(2, 5, self.dim)
        out = self.norm(x)
        self.assertFalse(torch.isnan(out).any(), "全零输入产生了 NaN")

    # ---- 参数测试 ----

    def test_gamma_is_learnable_parameter(self) -> None:
        """gamma 必须是 nn.Parameter（可学习参数，参与反向传播）。"""
        self.assertIsInstance(self.norm.gamma, nn.Parameter)

    def test_gamma_shape_equals_dim(self) -> None:
        """gamma 的形状必须是 [dim]。"""
        self.assertEqual(self.norm.gamma.shape, (self.dim,))

    def test_gamma_initialized_to_ones(self) -> None:
        """gamma 初始值必须全为 1（不做缩放，等训练来调整）。"""
        torch.testing.assert_close(
            self.norm.gamma.data,
            torch.ones(self.dim),
        )

    def test_gamma_affects_output(self) -> None:
        """修改 gamma 后，输出应相应变化。"""
        x = torch.randn(2, 5, self.dim)
        out_before = self.norm(x).clone()

        # 把 gamma 全部设为 2.0
        with torch.no_grad():
            self.norm.gamma.fill_(2.0)
        out_after = self.norm(x)

        # 输出应该变为原来的 2 倍
        torch.testing.assert_close(out_after, out_before * 2.0, atol=1e-5, rtol=1e-5)

    def test_parameter_count(self) -> None:
        """RMSNorm 只有 gamma 一组参数，参数量 = dim。"""
        total_params = sum(p.numel() for p in self.norm.parameters())
        self.assertEqual(total_params, self.dim)

    # ---- 梯度测试 ----

    def test_gradient_flows_through(self) -> None:
        """梯度能正常流过 RMSNorm（反向传播不会中断）。"""
        x = torch.randn(2, 5, self.dim, requires_grad=True)
        out = self.norm(x)
        loss = out.sum()
        loss.backward()
        # 输入的梯度不应为 None
        self.assertIsNotNone(x.grad)
        # gamma 的梯度也不应为 None
        self.assertIsNotNone(self.norm.gamma.grad)


# ======================================================================
# SwiGLUFFN 测试（步骤 5.2）
#
# SwiGLUFFN 的核心职责:
#   1. 把 d_model 维向量扩展到 d_ff 维（两条通路: W1+SiLU 和 W3）
#   2. 两条通路逐元素相乘（门控机制）
#   3. 压缩回 d_model 维（W2）
#   4. 对每个位置独立处理，位置之间不交互
#
# 在 Transformer 中的调用位置:
#   TransformerBlock.forward:
#     normed = self.ffn_norm(x)
#     ffn_out = self.ffn(normed)     ← 这里调用 SwiGLUFFN
#     x = x + ffn_out               ← 残差连接
# ======================================================================
class TestSwiGLUFFN(unittest.TestCase):
    """测试 SwiGLU 前馈网络的形状、参数和数值行为。"""

    def setUp(self) -> None:
        self.d_model = 128
        self.d_ff = 512
        self.ffn = SwiGLUFFN(self.d_model, self.d_ff)

    # ---- 形状测试 ----

    def test_output_shape_matches_input(self) -> None:
        """输出形状 = 输入形状 [batch, seq_len, d_model]。"""
        x = torch.randn(2, 10, self.d_model)
        out = self.ffn(x)
        self.assertEqual(out.shape, x.shape)

    def test_works_with_different_batch_sizes(self) -> None:
        """支持不同 batch_size。"""
        for bs in [1, 4, 16]:
            x = torch.randn(bs, 8, self.d_model)
            out = self.ffn(x)
            self.assertEqual(out.shape, (bs, 8, self.d_model))

    def test_works_with_different_seq_lengths(self) -> None:
        """支持不同序列长度。"""
        for sl in [1, 64, 128]:
            x = torch.randn(2, sl, self.d_model)
            out = self.ffn(x)
            self.assertEqual(out.shape, (2, sl, self.d_model))

    # ---- 参数测试 ----

    def test_has_three_linear_layers(self) -> None:
        """必须包含 w1, w2, w3 三个线性层。"""
        self.assertIsInstance(self.ffn.w1, nn.Linear)
        self.assertIsInstance(self.ffn.w2, nn.Linear)
        self.assertIsInstance(self.ffn.w3, nn.Linear)

    def test_linear_layers_have_no_bias(self) -> None:
        """三个线性层都不带偏置（bias=False）。"""
        self.assertIsNone(self.ffn.w1.bias)
        self.assertIsNone(self.ffn.w2.bias)
        self.assertIsNone(self.ffn.w3.bias)

    def test_w1_shape(self) -> None:
        """W1 的权重形状: [d_ff, d_model] = [512, 128]。"""
        self.assertEqual(self.ffn.w1.weight.shape, (self.d_ff, self.d_model))

    def test_w2_shape(self) -> None:
        """W2 的权重形状: [d_model, d_ff] = [128, 512]。"""
        self.assertEqual(self.ffn.w2.weight.shape, (self.d_model, self.d_ff))

    def test_w3_shape(self) -> None:
        """W3 的权重形状: [d_ff, d_model] = [512, 128]。"""
        self.assertEqual(self.ffn.w3.weight.shape, (self.d_ff, self.d_model))

    def test_parameter_count(self) -> None:
        """参数量 = 3 × d_model × d_ff（无偏置）。"""
        expected = 3 * self.d_model * self.d_ff
        actual = sum(p.numel() for p in self.ffn.parameters())
        self.assertEqual(actual, expected)

    # ---- 数值行为测试 ----

    def test_no_nan_or_inf(self) -> None:
        """输出不包含 NaN 或 Inf。"""
        x = torch.randn(4, 10, self.d_model)
        out = self.ffn(x)
        self.assertFalse(torch.isnan(out).any(), "输出包含 NaN")
        self.assertFalse(torch.isinf(out).any(), "输出包含 Inf")

    def test_position_independence(self) -> None:
        """各位置独立处理：修改位置 0 不影响位置 1 的输出。

        FFN 对每个 token 独立计算，token 之间无交互。
        """
        x = torch.randn(1, 5, self.d_model)
        out_original = self.ffn(x).clone()

        # 修改位置 0 的输入
        x_modified = x.clone()
        x_modified[0, 0, :] = torch.randn(self.d_model)
        out_modified = self.ffn(x_modified)

        # 位置 1-4 的输出应该完全不变
        torch.testing.assert_close(out_original[0, 1:], out_modified[0, 1:])

    # ---- 梯度测试 ----

    def test_gradient_flows_through(self) -> None:
        """梯度能正常流过（反向传播不中断）。"""
        x = torch.randn(2, 5, self.d_model, requires_grad=True)
        out = self.ffn(x)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        for name, param in self.ffn.named_parameters():
            self.assertIsNotNone(param.grad, f"{name} 的梯度为 None")


# ======================================================================
# MultiHeadAttention 测试（步骤 5.3）
#
# MultiHeadAttention 的核心职责:
#   1. 通过 Q、K、V 投影让 token 之间互相"对话"
#   2. 多头并行，每个头关注不同角度的特征
#   3. 因果掩码保证每个 token 只能看到自己和前面的 token
#   4. 缩放点积防止梯度消失
#   5. 输出投影混合各头信息
#
# 在 Transformer 中的调用位置:
#   TransformerBlock.forward:
#     normed = self.attn_norm(x)       ← RMSNorm
#     attn_out = self.attn(normed)     ← 这里调用 MultiHeadAttention
#     x = x + attn_out                ← 残差连接
# ======================================================================
class TestMultiHeadAttention(unittest.TestCase):
    """测试多头自注意力的形状、因果掩码、参数和数值行为。"""

    def setUp(self) -> None:
        self.d_model = 128
        self.n_heads = 4
        self.head_dim = self.d_model // self.n_heads  # 32
        self.attn = MultiHeadAttention(self.d_model, self.n_heads, dropout=0.0)
        self.attn.eval()

    # ---- 形状测试 ----

    def test_output_shape_matches_input(self) -> None:
        """输出形状 = 输入形状 [batch, seq_len, d_model]。"""
        x = torch.randn(2, 10, self.d_model)
        out = self.attn(x)
        self.assertEqual(out.shape, x.shape)

    def test_works_with_different_batch_sizes(self) -> None:
        """支持不同 batch_size。"""
        for bs in [1, 4, 16]:
            x = torch.randn(bs, 8, self.d_model)
            out = self.attn(x)
            self.assertEqual(out.shape, (bs, 8, self.d_model))

    def test_works_with_different_seq_lengths(self) -> None:
        """支持不同序列长度。"""
        for sl in [1, 32, 128]:
            x = torch.randn(2, sl, self.d_model)
            out = self.attn(x)
            self.assertEqual(out.shape, (2, sl, self.d_model))

    def test_single_token_sequence(self) -> None:
        """seq_len=1 时也能正常工作（只能看自己）。"""
        x = torch.randn(1, 1, self.d_model)
        out = self.attn(x)
        self.assertEqual(out.shape, (1, 1, self.d_model))

    # ---- 参数测试 ----

    def test_has_four_linear_layers(self) -> None:
        """必须包含 w_q, w_k, w_v, w_o 四个线性层。"""
        self.assertIsInstance(self.attn.w_q, nn.Linear)
        self.assertIsInstance(self.attn.w_k, nn.Linear)
        self.assertIsInstance(self.attn.w_v, nn.Linear)
        self.assertIsInstance(self.attn.w_o, nn.Linear)

    def test_all_linear_layers_have_no_bias(self) -> None:
        """四个线性层都不带偏置（bias=False）。"""
        self.assertIsNone(self.attn.w_q.bias)
        self.assertIsNone(self.attn.w_k.bias)
        self.assertIsNone(self.attn.w_v.bias)
        self.assertIsNone(self.attn.w_o.bias)

    def test_projection_weight_shapes(self) -> None:
        """四个投影矩阵的权重形状都是 [d_model, d_model]。"""
        expected = (self.d_model, self.d_model)
        self.assertEqual(self.attn.w_q.weight.shape, expected)
        self.assertEqual(self.attn.w_k.weight.shape, expected)
        self.assertEqual(self.attn.w_v.weight.shape, expected)
        self.assertEqual(self.attn.w_o.weight.shape, expected)

    def test_parameter_count(self) -> None:
        """参数量 = 4 × d_model²（四个投影矩阵，无偏置）。"""
        expected = 4 * self.d_model * self.d_model
        actual = sum(p.numel() for p in self.attn.parameters())
        self.assertEqual(actual, expected)

    def test_head_dim_and_scale(self) -> None:
        """head_dim = d_model / n_heads，scale = 1/√head_dim。"""
        self.assertEqual(self.attn.head_dim, self.head_dim)
        expected_scale = self.head_dim ** -0.5
        self.assertAlmostEqual(self.attn.scale, expected_scale)

    def test_d_model_must_be_divisible_by_n_heads(self) -> None:
        """d_model 不能被 n_heads 整除时应报错。"""
        with self.assertRaises(AssertionError):
            MultiHeadAttention(d_model=128, n_heads=3)

    # ---- 因果掩码测试 ----

    def test_causal_masking_future_tokens_invisible(self) -> None:
        """因果掩码：修改未来 token 不应影响当前 token 的输出。

        核心验证逻辑:
          1. 准备一个 5-token 序列，计算位置 2 的输出
          2. 修改位置 3、4（未来位置）的输入值
          3. 再次计算位置 2 的输出，应该完全相同
        """
        torch.manual_seed(42)
        x = torch.randn(1, 5, self.d_model)
        out1 = self.attn(x)

        x_modified = x.clone()
        x_modified[0, 3:, :] = torch.randn(2, self.d_model)
        out2 = self.attn(x_modified)

        # 位置 0、1、2 只依赖位置 0~2，修改位置 3、4 不应影响它们
        torch.testing.assert_close(out1[0, :3, :], out2[0, :3, :])

    def test_causal_masking_past_tokens_visible(self) -> None:
        """因果掩码：修改过去 token 应当影响当前 token 的输出。

        验证逻辑: 修改位置 0 的输入后，位置 2 的输出应该变化（因为位置 2 能看到位置 0）。
        """
        torch.manual_seed(42)
        x = torch.randn(1, 5, self.d_model)
        out1 = self.attn(x)

        x_modified = x.clone()
        x_modified[0, 0, :] = torch.randn(self.d_model)
        out2 = self.attn(x_modified)

        # 位置 2 能看到位置 0，修改位置 0 后位置 2 的输出应该变化
        self.assertFalse(
            torch.allclose(out1[0, 2, :], out2[0, 2, :], atol=1e-6),
            "修改过去 token 后，当前 token 的输出没有变化——因果掩码或注意力计算有误",
        )

    # ---- 数值行为测试 ----

    def test_no_nan_or_inf(self) -> None:
        """输出不包含 NaN 或 Inf。"""
        x = torch.randn(4, 10, self.d_model)
        out = self.attn(x)
        self.assertFalse(torch.isnan(out).any(), "输出包含 NaN")
        self.assertFalse(torch.isinf(out).any(), "输出包含 Inf")

    def test_deterministic_in_eval_mode(self) -> None:
        """eval 模式下（Dropout 关闭），相同输入应产生相同输出。"""
        self.attn.eval()
        x = torch.randn(2, 8, self.d_model)
        out1 = self.attn(x)
        out2 = self.attn(x)
        torch.testing.assert_close(out1, out2)

    def test_batch_independence(self) -> None:
        """batch 内各样本独立处理：修改样本 0 不影响样本 1 的输出。"""
        x = torch.randn(2, 6, self.d_model)
        out_original = self.attn(x).clone()

        x_modified = x.clone()
        x_modified[0] = torch.randn(6, self.d_model)
        out_modified = self.attn(x_modified)

        torch.testing.assert_close(out_original[1], out_modified[1])

    # ---- 梯度测试 ----

    def test_gradient_flows_through(self) -> None:
        """梯度能正常流过所有参数（反向传播不中断）。"""
        self.attn.train()
        x = torch.randn(2, 5, self.d_model, requires_grad=True)
        out = self.attn(x)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        for name, param in self.attn.named_parameters():
            self.assertIsNotNone(param.grad, f"{name} 的梯度为 None")


# ======================================================================
# TransformerBlock 测试（步骤 5.4）
#
# TransformerBlock 的核心职责:
#   1. Pre-LN 结构: 先归一化再计算（RMSNorm → Attention → 残差）
#   2. 两个阶段: 自注意力(token间交互) + FFN(独立消化)
#   3. 两个残差连接: 保证信息不丢失、梯度不消失
#   4. 输入输出形状不变: [batch, seq_len, d_model]
#
# 在模型中的调用位置:
#   NovaModel.forward:
#     x = token_emb + pos_emb
#     x = dropout(x)
#     for block in self.blocks:     ← 4 层循环
#         x = block(x)             ← 每层调用一次
#     x = final_norm(x)
#     logits = output_layer(x)
# ======================================================================
class TestTransformerBlock(unittest.TestCase):
    """测试 Transformer Decoder Block 的结构、形状和行为。"""

    def setUp(self) -> None:
        self.d_model = 128
        self.n_heads = 4
        self.d_ff = 512
        self.dropout = 0.0
        self.block = TransformerBlock(
            self.d_model, self.n_heads, self.d_ff, self.dropout
        )
        self.block.eval()

    # ---- 形状测试 ----

    def test_output_shape_matches_input(self) -> None:
        """输出形状 = 输入形状 [batch, seq_len, d_model]。"""
        x = torch.randn(2, 10, self.d_model)
        out = self.block(x)
        self.assertEqual(out.shape, x.shape)

    def test_works_with_different_batch_sizes(self) -> None:
        """支持不同 batch_size。"""
        for bs in [1, 4, 16]:
            x = torch.randn(bs, 8, self.d_model)
            out = self.block(x)
            self.assertEqual(out.shape, (bs, 8, self.d_model))

    def test_works_with_different_seq_lengths(self) -> None:
        """支持不同序列长度。"""
        for sl in [1, 32, 128]:
            x = torch.randn(2, sl, self.d_model)
            out = self.block(x)
            self.assertEqual(out.shape, (2, sl, self.d_model))

    # ---- 子模块结构测试 ----

    def test_has_attn_norm(self) -> None:
        """必须包含 attn_norm（自注意力前的 RMSNorm）。"""
        self.assertIsInstance(self.block.attn_norm, RMSNorm)

    def test_has_attention(self) -> None:
        """必须包含 attn（多头自注意力）。"""
        self.assertIsInstance(self.block.attn, MultiHeadAttention)

    def test_has_ffn_norm(self) -> None:
        """必须包含 ffn_norm（FFN 前的 RMSNorm）。"""
        self.assertIsInstance(self.block.ffn_norm, RMSNorm)

    def test_has_ffn(self) -> None:
        """必须包含 ffn（SwiGLU 前馈网络）。"""
        self.assertIsInstance(self.block.ffn, SwiGLUFFN)

    def test_parameter_count(self) -> None:
        """参数量 = 2×RMSNorm + Attention + FFN。

        2 × 128 (RMSNorm gamma) + 4 × 128² (Attention) + 3 × 128 × 512 (FFN)
        = 256 + 65536 + 196608 = 262400
        """
        expected = (
            2 * self.d_model                              # 2 个 RMSNorm
            + 4 * self.d_model * self.d_model             # Attention: W_Q, W_K, W_V, W_O
            + 3 * self.d_model * self.d_ff                # FFN: W1, W2, W3
        )
        actual = sum(p.numel() for p in self.block.parameters())
        self.assertEqual(actual, expected)

    # ---- 残差连接测试 ----

    def test_residual_connection_preserves_input(self) -> None:
        """残差连接确保输出 ≠ 0：即使子模块输出趋零，原始输入仍被保留。

        验证思路: 构造全零权重的注意力和 FFN，此时子模块输出为零，
        残差连接应让 output ≈ input。
        """
        block = TransformerBlock(self.d_model, self.n_heads, self.d_ff, 0.0)
        block.eval()

        with torch.no_grad():
            for param in block.attn.parameters():
                param.zero_()
            for param in block.ffn.parameters():
                param.zero_()

        x = torch.randn(1, 5, self.d_model)
        out = block(x)
        # Attention 和 FFN 输出全零 → 残差后 output ≈ input
        torch.testing.assert_close(out, x, atol=1e-5, rtol=1e-5)

    # ---- 因果掩码测试（继承自 Attention） ----

    def test_causal_masking_through_block(self) -> None:
        """因果掩码在 Block 层面仍然有效：修改未来 token 不影响当前 token。"""
        torch.manual_seed(42)
        x = torch.randn(1, 6, self.d_model)
        out1 = self.block(x)

        x_modified = x.clone()
        x_modified[0, 4:, :] = torch.randn(2, self.d_model)
        out2 = self.block(x_modified)

        # 位置 0-3 只依赖位置 0-3，修改位置 4-5 不应影响
        torch.testing.assert_close(out1[0, :4, :], out2[0, :4, :])

    # ---- 数值行为测试 ----

    def test_no_nan_or_inf(self) -> None:
        """输出不包含 NaN 或 Inf。"""
        x = torch.randn(4, 10, self.d_model)
        out = self.block(x)
        self.assertFalse(torch.isnan(out).any(), "输出包含 NaN")
        self.assertFalse(torch.isinf(out).any(), "输出包含 Inf")

    def test_deterministic_in_eval_mode(self) -> None:
        """eval 模式下相同输入产生相同输出。"""
        self.block.eval()
        x = torch.randn(2, 8, self.d_model)
        out1 = self.block(x)
        out2 = self.block(x)
        torch.testing.assert_close(out1, out2)

    def test_batch_independence(self) -> None:
        """batch 内各样本独立：修改样本 0 不影响样本 1。"""
        x = torch.randn(2, 6, self.d_model)
        out_original = self.block(x).clone()

        x_modified = x.clone()
        x_modified[0] = torch.randn(6, self.d_model)
        out_modified = self.block(x_modified)

        torch.testing.assert_close(out_original[1], out_modified[1])

    # ---- 梯度测试 ----

    def test_gradient_flows_through(self) -> None:
        """梯度能流过整个 Block（包括 Norm → Attention → FFN 全链路）。"""
        self.block.train()
        x = torch.randn(2, 5, self.d_model, requires_grad=True)
        out = self.block(x)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        for name, param in self.block.named_parameters():
            self.assertIsNotNone(param.grad, f"{name} 的梯度为 None")

    # ---- 多层堆叠测试 ----

    def test_stackable(self) -> None:
        """多层 Block 可以堆叠：第一层的输出可以直接作为第二层的输入。"""
        block1 = TransformerBlock(self.d_model, self.n_heads, self.d_ff, 0.0)
        block2 = TransformerBlock(self.d_model, self.n_heads, self.d_ff, 0.0)
        block1.eval()
        block2.eval()

        x = torch.randn(2, 8, self.d_model)
        out1 = block1(x)
        out2 = block2(out1)
        self.assertEqual(out2.shape, x.shape)
        self.assertFalse(torch.isnan(out2).any())


# ======================================================================
# NovaModel 测试（步骤 5.5）
#
# NovaModel 是完整的 Decoder-Only Transformer，串联所有子组件:
#   input_ids [batch, seq_len]
#       │
#   ① Token Embedding: ID → d_model 维向量（查字义表）
#   ② Position Embedding: 位置编号 → d_model 维向量（查位置表）
#   ③ 相加 + Dropout
#   ④ TransformerBlock × n_layers
#   ⑤ Final RMSNorm
#   ⑥ Output Linear: d_model → vocab_size
#       │
#   logits [batch, seq_len, vocab_size]
#
# 调用方:
#   训练: train.py  →  logits = model(input_ids)
#   推理: chat.py   →  logits = model(input_ids)
# ======================================================================
class TestNovaModel(unittest.TestCase):
    """测试完整的 Nova 模型的结构、形状和行为。"""

    def setUp(self) -> None:
        self.config = NovaConfig(
            d_model=64,
            n_heads=4,
            n_layers=2,
            d_ff=256,
            max_seq_len=32,
            dropout=0.0,
            vocab_size=100,
        )
        self.model = NovaModel(self.config)
        self.model.eval()

    # ---- 输出形状测试 ----

    def test_output_shape(self) -> None:
        """输出形状: [batch, seq_len, vocab_size]。"""
        ids = torch.randint(0, self.config.vocab_size, (2, 10))
        logits = self.model(ids)
        self.assertEqual(logits.shape, (2, 10, self.config.vocab_size))

    def test_works_with_different_batch_sizes(self) -> None:
        """支持不同 batch_size。"""
        for bs in [1, 4, 8]:
            ids = torch.randint(0, self.config.vocab_size, (bs, 10))
            logits = self.model(ids)
            self.assertEqual(logits.shape, (bs, 10, self.config.vocab_size))

    def test_works_with_different_seq_lengths(self) -> None:
        """支持不同序列长度（不超过 max_seq_len）。"""
        for sl in [1, 16, self.config.max_seq_len]:
            ids = torch.randint(0, self.config.vocab_size, (2, sl))
            logits = self.model(ids)
            self.assertEqual(logits.shape, (2, sl, self.config.vocab_size))

    # ---- 子模块结构测试 ----

    def test_has_token_embedding(self) -> None:
        """必须包含 token_emb（字义表）。"""
        self.assertIsInstance(self.model.token_emb, nn.Embedding)
        self.assertEqual(
            self.model.token_emb.num_embeddings, self.config.vocab_size
        )
        self.assertEqual(
            self.model.token_emb.embedding_dim, self.config.d_model
        )

    def test_has_position_embedding(self) -> None:
        """必须包含 pos_emb（位置表）。"""
        self.assertIsInstance(self.model.pos_emb, nn.Embedding)
        self.assertEqual(
            self.model.pos_emb.num_embeddings, self.config.max_seq_len
        )
        self.assertEqual(
            self.model.pos_emb.embedding_dim, self.config.d_model
        )

    def test_has_correct_number_of_blocks(self) -> None:
        """TransformerBlock 数量 = config.n_layers。"""
        self.assertEqual(len(self.model.blocks), self.config.n_layers)
        for block in self.model.blocks:
            self.assertIsInstance(block, TransformerBlock)

    def test_has_final_norm(self) -> None:
        """必须包含 final_norm（最终 RMSNorm）。"""
        self.assertIsInstance(self.model.final_norm, RMSNorm)

    def test_has_output_layer(self) -> None:
        """必须包含 output（线性输出层 d_model → vocab_size）。"""
        self.assertIsInstance(self.model.output, nn.Linear)
        self.assertEqual(
            self.model.output.in_features, self.config.d_model
        )
        self.assertEqual(
            self.model.output.out_features, self.config.vocab_size
        )

    def test_output_layer_has_no_bias(self) -> None:
        """输出层不带偏置（bias=False）。"""
        self.assertIsNone(self.model.output.bias)

    # ---- 参数量测试 ----

    def test_parameter_count_reasonable(self) -> None:
        """参数量应在合理范围内。

        对于 d_model=64, n_layers=2, vocab_size=100 的小模型:
          Token Emb:    100 × 64         = 6,400
          Pos Emb:      32 × 64          = 2,048
          Blocks × 2:   2 × (2×64 + 4×64² + 3×64×256) = 2 × (128 + 16384 + 49152) = 131,328
          Final Norm:   64               = 64
          Output:       64 × 100         = 6,400
          ────────────────────────────────────────
          总计:         146,240
        """
        expected = (
            self.config.vocab_size * self.config.d_model          # token_emb
            + self.config.max_seq_len * self.config.d_model       # pos_emb
            + self.config.n_layers * (
                2 * self.config.d_model                           # 2 个 RMSNorm
                + 4 * self.config.d_model ** 2                    # Attention
                + 3 * self.config.d_model * self.config.d_ff      # FFN
            )
            + self.config.d_model                                 # final_norm
            + self.config.d_model * self.config.vocab_size        # output
        )
        actual = self.model.count_parameters()
        self.assertEqual(actual, expected)

    def test_count_parameters_method(self) -> None:
        """count_parameters() 返回值 = sum(p.numel())。"""
        expected = sum(p.numel() for p in self.model.parameters())
        self.assertEqual(self.model.count_parameters(), expected)

    # ---- 因果掩码测试 ----

    def test_causal_masking_end_to_end(self) -> None:
        """端到端因果掩码: 修改未来 token 不影响当前位置的 logits。"""
        torch.manual_seed(42)
        ids = torch.randint(0, self.config.vocab_size, (1, 8))
        logits1 = self.model(ids)

        ids_modified = ids.clone()
        ids_modified[0, 5:] = torch.randint(0, self.config.vocab_size, (3,))
        logits2 = self.model(ids_modified)

        # 位置 0-4 的 logits 应该完全相同
        torch.testing.assert_close(logits1[0, :5, :], logits2[0, :5, :])

    # ---- 数值行为测试 ----

    def test_no_nan_or_inf(self) -> None:
        """输出不包含 NaN 或 Inf。"""
        ids = torch.randint(0, self.config.vocab_size, (4, 16))
        logits = self.model(ids)
        self.assertFalse(torch.isnan(logits).any(), "输出包含 NaN")
        self.assertFalse(torch.isinf(logits).any(), "输出包含 Inf")

    def test_deterministic_in_eval_mode(self) -> None:
        """eval 模式下相同输入产生相同输出。"""
        self.model.eval()
        ids = torch.randint(0, self.config.vocab_size, (2, 10))
        logits1 = self.model(ids)
        logits2 = self.model(ids)
        torch.testing.assert_close(logits1, logits2)

    def test_different_input_ids_produce_different_output(self) -> None:
        """不同的 input_ids 应产生不同的 logits。"""
        ids1 = torch.zeros(1, 5, dtype=torch.long)
        ids2 = torch.ones(1, 5, dtype=torch.long)
        logits1 = self.model(ids1)
        logits2 = self.model(ids2)
        self.assertFalse(torch.allclose(logits1, logits2))

    def test_batch_independence(self) -> None:
        """batch 内各样本独立。"""
        ids = torch.randint(0, self.config.vocab_size, (2, 8))
        logits_orig = self.model(ids).clone()

        ids_mod = ids.clone()
        ids_mod[0] = torch.randint(0, self.config.vocab_size, (8,))
        logits_mod = self.model(ids_mod)

        torch.testing.assert_close(logits_orig[1], logits_mod[1])

    # ---- 梯度测试 ----

    def test_gradient_flows_end_to_end(self) -> None:
        """梯度能从 loss 流回所有可学习参数。"""
        self.model.train()
        ids = torch.randint(0, self.config.vocab_size, (2, 8))
        logits = self.model(ids)
        loss = logits.sum()
        loss.backward()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(
                    param.grad, f"{name} 的梯度为 None"
                )

    # ---- 与 NovaConfig 联动测试 ----

    def test_respects_config_n_layers(self) -> None:
        """n_layers 不同时 Block 数量相应变化。"""
        for n in [1, 3, 6]:
            cfg = NovaConfig(
                d_model=64, n_heads=4, n_layers=n,
                d_ff=256, max_seq_len=32, dropout=0.0, vocab_size=50,
            )
            m = NovaModel(cfg)
            self.assertEqual(len(m.blocks), n)

    def test_respects_config_vocab_size(self) -> None:
        """vocab_size 不同时输出维度相应变化。"""
        for v in [50, 200, 500]:
            cfg = NovaConfig(
                d_model=64, n_heads=4, n_layers=1,
                d_ff=256, max_seq_len=32, dropout=0.0, vocab_size=v,
            )
            m = NovaModel(cfg)
            m.eval()
            ids = torch.randint(0, v, (1, 5))
            logits = m(ids)
            self.assertEqual(logits.shape[-1], v)


# ======================================================================
# 权重初始化测试（步骤 5.6）
#
# 初始化策略:
#   nn.Embedding  → N(0, 0.02)
#   nn.Linear     → Xavier 均匀分布
#   RMSNorm gamma → 全 1（在 RMSNorm.__init__ 中完成）
#
# 调用时机:
#   NovaModel.__init__ 的最后一步调用 self._init_weights()
# ======================================================================
class TestWeightInitialization(unittest.TestCase):
    """测试模型权重初始化是否符合预期策略。"""

    def setUp(self) -> None:
        self.config = NovaConfig(
            d_model=128,
            n_heads=4,
            n_layers=2,
            d_ff=512,
            max_seq_len=128,
            dropout=0.0,
            vocab_size=500,
        )
        self.model = NovaModel(self.config)

    # ---- Embedding 层: N(0, 0.02) ----

    def test_token_embedding_mean_near_zero(self) -> None:
        """Token Embedding 的均值应接近 0。"""
        mean = self.model.token_emb.weight.data.mean().item()
        self.assertAlmostEqual(mean, 0.0, delta=0.01)

    def test_token_embedding_std_near_002(self) -> None:
        """Token Embedding 的标准差应接近 0.02。"""
        std = self.model.token_emb.weight.data.std().item()
        self.assertAlmostEqual(std, 0.02, delta=0.005)

    def test_position_embedding_mean_near_zero(self) -> None:
        """Position Embedding 的均值应接近 0。"""
        mean = self.model.pos_emb.weight.data.mean().item()
        self.assertAlmostEqual(mean, 0.0, delta=0.01)

    def test_position_embedding_std_near_002(self) -> None:
        """Position Embedding 的标准差应接近 0.02。"""
        std = self.model.pos_emb.weight.data.std().item()
        self.assertAlmostEqual(std, 0.02, delta=0.005)

    # ---- Linear 层: Xavier 均匀分布 ----

    def test_linear_layers_xavier_range(self) -> None:
        """线性层权重应在 Xavier 均匀分布的范围内。

        Xavier 均匀: U(-√(6/(fan_in+fan_out)), √(6/(fan_in+fan_out)))
        对于 [128, 128] 的矩阵: bound = √(6/256) ≈ 0.153
        """
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                fan_in = module.in_features
                fan_out = module.out_features
                bound = (6.0 / (fan_in + fan_out)) ** 0.5
                w = module.weight.data
                self.assertTrue(
                    w.min().item() >= -(bound + 0.01),
                    f"{name}: 最小值 {w.min().item():.4f} 超出 Xavier 范围 [-{bound:.4f}]",
                )
                self.assertTrue(
                    w.max().item() <= (bound + 0.01),
                    f"{name}: 最大值 {w.max().item():.4f} 超出 Xavier 范围 [{bound:.4f}]",
                )

    def test_linear_layers_mean_near_zero(self) -> None:
        """线性层权重的均值应接近 0（均匀分布是对称的）。"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                mean = module.weight.data.mean().item()
                self.assertAlmostEqual(
                    mean, 0.0, delta=0.02,
                    msg=f"{name}: 均值 {mean:.4f} 偏离 0 过多",
                )

    def test_linear_layers_no_bias(self) -> None:
        """所有线性层都不带偏置（本项目全部 bias=False）。"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                self.assertIsNone(
                    module.bias, f"{name}: 存在偏置但预期为 None"
                )

    # ---- RMSNorm gamma: 全 1 ----

    def test_rmsnorm_gamma_all_ones(self) -> None:
        """所有 RMSNorm 的 gamma 初始值应为全 1。"""
        for name, module in self.model.named_modules():
            if isinstance(module, RMSNorm):
                torch.testing.assert_close(
                    module.gamma.data,
                    torch.ones_like(module.gamma.data),
                    msg=f"{name}: gamma 不全为 1",
                )

    # ---- 初始化后模型仍可正常前向传播 ----

    def test_forward_works_after_init(self) -> None:
        """初始化后前向传播无 NaN/Inf。"""
        self.model.eval()
        ids = torch.randint(0, self.config.vocab_size, (2, 16))
        logits = self.model(ids)
        self.assertFalse(torch.isnan(logits).any())
        self.assertFalse(torch.isinf(logits).any())


# ======================================================================
# 参数量统计测试（步骤 5.7）
#
# count_parameters()        → 返回总参数量（int）
# print_parameter_summary() → 打印各层参数量详细报表
#
# 调用时机:
#   训练开始前:
#     model = NovaModel(config)
#     model.print_parameter_summary()
# ======================================================================
class TestParameterSummary(unittest.TestCase):
    """测试参数量统计功能。"""

    def setUp(self) -> None:
        self.config = NovaConfig(
            d_model=64,
            n_heads=4,
            n_layers=2,
            d_ff=256,
            max_seq_len=32,
            dropout=0.0,
            vocab_size=100,
        )
        self.model = NovaModel(self.config)

    def test_count_parameters_returns_int(self) -> None:
        """count_parameters() 返回 int 类型。"""
        self.assertIsInstance(self.model.count_parameters(), int)

    def test_count_parameters_matches_manual(self) -> None:
        """count_parameters() 与手动计算一致。"""
        expected = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.assertEqual(self.model.count_parameters(), expected)

    def test_count_parameters_formula(self) -> None:
        """count_parameters() 与公式计算一致。"""
        c = self.config
        expected = (
            c.vocab_size * c.d_model
            + c.max_seq_len * c.d_model
            + c.n_layers * (2 * c.d_model + 4 * c.d_model ** 2 + 3 * c.d_model * c.d_ff)
            + c.d_model
            + c.d_model * c.vocab_size
        )
        self.assertEqual(self.model.count_parameters(), expected)

    def test_print_parameter_summary_runs(self) -> None:
        """print_parameter_summary() 不报错（冒烟测试）。"""
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.model.print_parameter_summary()
        output = buf.getvalue()
        self.assertIn("Nova", output)
        self.assertIn("Token Embedding", output)
        self.assertIn("Position Embedding", output)
        self.assertIn("Block 0", output)
        self.assertIn("Final RMSNorm", output)
        self.assertIn("Output Linear", output)
        self.assertIn("总计", output)

    def test_print_summary_shows_all_blocks(self) -> None:
        """print_parameter_summary() 打印每一层 Block 的信息。"""
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.model.print_parameter_summary()
        output = buf.getvalue()
        for i in range(self.config.n_layers):
            self.assertIn(f"Block {i}", output)


if __name__ == "__main__":
    unittest.main()
