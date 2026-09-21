#!/usr/bin/env python
"""
LLaMA 推理示例 — 四个可以用 assert 证伪的性质

    1. KV cache: 有/无 cache 的贪心生成逐 token 相同; GQA 让 cache 比 MHA 小 H/Hkv 倍
    2. 左 padding: pad 行在 "因果 ∩ padding" 后一个 key 都不剩 → 曾经 softmax 出 NaN;
       现在无 NaN, 且真实位置的 logits 与不 padding 时一致 (RoPE 只看相对位置)
    3. QK-Norm: 把 W_q, W_k 放大 10 倍, 无 QK-Norm 时 logit 放大 100 倍, 有 QK-Norm 时有界 ≤ sqrt(Dh)
    4. attention sink: sink logit → -∞ 时退化为普通 GQA; = 0 时每行注意力之和 < 1; 与 cache 兼容
"""

import math

import torch

from llm_models.layers.core.attention import GroupedQueryAttention
from llm_models.models.language_models.llama import LLaMA
from llm_models.utils.generation import benchmark_kv_cache


def max_attn_logit(attn: GroupedQueryAttention, x: torch.Tensor) -> float:
    """复现 GQA 内部的 QK^T·scale (不加 RoPE: 旋转不改范数, 不影响上界)。"""
    B, T, _ = x.shape
    Q = attn.w_q(x).view(B, T, attn.num_heads, attn.head_dim).transpose(1, 2)      # [B, H, T, Dh]
    K = attn.w_k(x).view(B, T, attn.num_kv_heads, attn.head_dim).transpose(1, 2)   # [B, Hkv, T, Dh]
    if attn.q_norm is not None:
        Q, K = attn.q_norm(Q), attn.k_norm(K)
    K = K.repeat_interleave(attn.num_groups, dim=1)
    return float((Q @ K.transpose(-2, -1) * attn.scale).abs().max())


@torch.inference_mode()
def main():
    torch.manual_seed(42)
    vocab_size = 1000
    cfg = dict(vocab_size=vocab_size, d_model=256, n_heads=8, num_kv_heads=2,
               num_layers=4, max_len=256, dropout=0.0)
    model = LLaMA(**cfg).eval()
    print(f"LLaMA Mini | 参数量: {sum(p.numel() for p in model.parameters()):,} (8 Q heads / 2 KV heads)")

    # ---- 1) KV cache ----
    idx = torch.randint(1, vocab_size, (2, 16))
    assert model(idx).shape == (2, 16, vocab_size)
    speedup = benchmark_kv_cache(model, idx[:, :8], max_new_tokens=200)
    a = model.layers[0].attn
    print(f"[1] KV cache: 生成 200 token 输出完全一致, 加速 {speedup:.1f}x; "
          f"每 token 每层缓存 2·Hkv·Dh = {2 * a.num_kv_heads * a.head_dim} 个数 "
          f"(MHA 要 {2 * a.num_heads * a.head_dim})")

    # ---- 2) 左 padding ----
    P = 5
    real = idx[:1]                                                       # [1, 16]
    padded = torch.cat([torch.zeros(1, P, dtype=torch.long), real], dim=1)
    attn_mask = torch.cat([torch.zeros(1, P), torch.ones(1, 16)], dim=1)
    out_pad = model(padded, attention_mask=attn_mask)
    assert not out_pad.isnan().any(), "左 padding 产生 NaN: 全屏蔽行没有被兜底"
    diff = float((out_pad[:, P:] - model(real)).abs().max())
    assert diff < 1e-4, f"padding 改变了真实位置的输出: {diff}"
    print(f"[2] 左 padding {P} 位: 无 NaN, 真实位置 logits 最大偏差 {diff:.1e}")

    # ---- 3) QK-Norm ----
    x = torch.randn(2, 16, 256)
    for qk_norm in (False, True):
        attn = GroupedQueryAttention(256, 8, 2, qk_norm=qk_norm)
        before = max_attn_logit(attn, x)
        attn.w_q.weight.mul_(10)
        attn.w_k.weight.mul_(10)
        after = max_attn_logit(attn, x)
        print(f"[3] qk_norm={qk_norm!s:5}: max|logit| {before:8.2f} → 权重×10 → {after:8.2f}")
        if qk_norm:
            bound = math.sqrt(attn.head_dim)       # |q̂|=|k̂|=sqrt(Dh) → |q̂·k̂|/sqrt(Dh) ≤ sqrt(Dh)
            assert after <= bound + 1e-3 and abs(after - before) < 1e-3
        else:
            assert after > 50 * before             # 100× 放大 → softmax 饱和成 one-hot, 梯度消失

    # ---- 4) attention sink ----
    plain = GroupedQueryAttention(256, 8, 2)
    sink = GroupedQueryAttention(256, 8, 2, use_sink=True)
    sink.load_state_dict(plain.state_dict(), strict=False)
    causal = torch.ones(16, 16).tril().bool()
    out_sink0 = sink(x, mask=causal)
    assert not torch.allclose(out_sink0, plain(x, mask=causal), atol=1e-4)   # sink=0 分走了概率质量
    sink.sink.fill_(-1e4)                                                     # sink → -∞: 退化为普通 GQA
    assert torch.allclose(sink(x, mask=causal), plain(x, mask=causal), atol=1e-5)
    sink.sink.zero_()
    cache, steps = {}, []
    for t in range(16):                                                       # 逐 token 带 cache 解码
        steps.append(sink(x[:, t:t + 1], mask=causal[t:t + 1, : t + 1], cache=cache))
    assert torch.allclose(torch.cat(steps, dim=1), out_sink0, atol=1e-5)
    print("[4] attention sink: sink→-∞ 等价普通 GQA; sink=0 改变输出; 逐 token cache 解码与整段前向一致")


if __name__ == "__main__":
    main()
