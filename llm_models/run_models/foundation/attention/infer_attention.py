#!/usr/bin/env python
"""
Attention 的四个性质, 每个都用断言验证

1) 为什么除以 √d_k   2) 权重每行和为 1, 被 mask 的列为 0   3) 因果 mask 下过去不受未来影响
4) 没有位置编码的 self-attention 是置换等变的 (打乱输入 = 打乱输出) —— 所以 Transformer 必须另加位置信息
"""

import math

import torch

from llm_models.layers.core.attention import MultiHeadAttention, ScaledDotProductAttention
from llm_models.utils.masks import build_causal_mask


def main():
    torch.manual_seed(42)
    B, T, D, H = 2, 16, 128, 4

    # 1) q·k 的方差 ≈ d_k ⇒ 不缩放时 softmax 饱和
    d_k = 64
    Q, K = torch.randn(1000, d_k), torch.randn(1000, d_k)
    raw = (Q * K).sum(-1)
    p_raw = torch.softmax(torch.randn(T, d_k) @ torch.randn(T, d_k).T, dim=-1).max(-1).values.mean().item()
    p_scaled = torch.softmax(torch.randn(T, d_k) @ torch.randn(T, d_k).T / math.sqrt(d_k), dim=-1).max(-1).values.mean().item()
    print(f"1) var(q·k) = {raw.var().item():.1f} (d_k={d_k}), 缩放后 {(raw / math.sqrt(d_k)).var().item():.2f}; "
          f"softmax 最大权重: 不缩放 {p_raw:.2f} vs 缩放 {p_scaled:.2f}")
    assert abs(raw.var().item() / d_k - 1) < 0.2 and p_raw > 2 * p_scaled

    # 2) 权重是概率分布; 被 mask 的列权重为 0
    x = torch.randn(B, T, D)
    causal = build_causal_mask(T, x.device)                            # [1, T, T]
    _, w = ScaledDotProductAttention()(x, x, x, mask=causal)           # w: [B, T, T]
    assert torch.allclose(w.sum(-1), torch.ones(B, T), atol=1e-5)
    assert (w.masked_select(~causal.bool().expand_as(w)) == 0).all()
    print(f"2) 权重行和 = 1; 因果 mask 下第 0 行只看自己: w[0,0,:3] = {[round(v, 2) for v in w[0, 0, :3].tolist()]}")

    mha = MultiHeadAttention(D, H).eval()
    with torch.inference_mode():
        y = mha(x, mask=causal)
        assert y.shape == x.shape                                      # 形状不变 ⇒ 可以接残差

        # 3) 因果: 改最后一个 token, 前 T-1 个输出不变
        x2 = x.clone(); x2[:, -1] += 1.0
        d_past = (mha(x2, mask=causal)[:, :-1] - y[:, :-1]).abs().max().item()
        assert d_past < 1e-6

        # 4) 置换等变 (无 mask、无位置编码)
        perm = torch.randperm(T)
        d_perm = (mha(x[:, perm]) - mha(x)[:, perm]).abs().max().item()
        assert d_perm < 1e-5
    print(f"3) 改未来 → 过去输出变化 {d_past:.1e}   4) attn(打乱 x) 与 打乱 attn(x) 的差 {d_perm:.1e}")


if __name__ == "__main__":
    main()
