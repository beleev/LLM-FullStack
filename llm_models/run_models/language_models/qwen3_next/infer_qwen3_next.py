#!/usr/bin/env python
"""
Qwen3-Next 推理示例 — 混合线性注意力 (Gated DeltaNet 3 : 1 全注意力)

    1. 层排布 [Δ, Δ, Δ, A, ...]
    2. 因果性: DeltaNet 没有 mask, 但递推天然因果 (前缀 logits 不受未来 token 影响)
    3. 缓存: attn 层的 k/v 随 T 增长, delta 层的 state 大小恒定 —— 直接量 cache 里的张量
    4. 有/无 cache 输出逐 token 相同 (无 cache 时 DeltaNet 每步要从头递推, 所以加速比比纯注意力模型大)
"""

import torch

from llm_models.models.language_models.qwen3_next import Qwen3Next
from llm_models.utils.generation import KVCache, benchmark_kv_cache


@torch.inference_mode()
def main():
    torch.manual_seed(42)
    vocab_size, T = 1000, 32
    model = Qwen3Next(
        vocab_size=vocab_size, d_model=128, n_heads=4, num_kv_heads=2,
        num_layers=8, max_len=128, linear_ratio=3, dropout=0.0,
    ).eval()
    pattern = " ".join("Δ" if t == "delta" else "A" for t in model.layer_types)
    print(f"Qwen3-Next Mini | 参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"[1] 层排布: [{pattern}]")
    assert model.layer_types.count("delta") == 6 and model.layer_types.count("attn") == 2

    idx = torch.randint(1, vocab_size, (2, T))
    logits = model(idx)
    assert logits.shape == (2, T, vocab_size)

    diff = float((logits[:, :8] - model(idx[:, :8])).abs().max())
    assert diff < 1e-4, "DeltaNet 递推必须天然因果"
    print(f"[2] 因果性: |logits(全长)[:8] - logits(截断到 8)| = {diff:.1e}")

    sizes = {}
    for n in (8, 32):
        cache = KVCache(len(model.layers))
        model(idx[:, :n], cache=cache)
        sizes[n] = {
            kind: sum(t.numel() for c, k in zip(cache.layers, model.layer_types) if k == kind
                      for t in c.values())
            for kind in ("attn", "delta")
        }
    assert sizes[32]["attn"] == 4 * sizes[8]["attn"]      # O(T)
    assert sizes[32]["delta"] == sizes[8]["delta"]        # O(1)
    print(f"[3] 缓存元素数 T=8 → T=32:  attn 层 {sizes[8]['attn']} → {sizes[32]['attn']} (×4),  "
          f"delta 层 {sizes[8]['delta']} → {sizes[32]['delta']} (不变)")

    speedup = benchmark_kv_cache(model, idx[:, :8], max_new_tokens=60)
    print(f"[4] 生成 60 token: 有/无 cache 输出一致, 加速 {speedup:.1f}x")


if __name__ == "__main__":
    main()
