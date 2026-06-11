#!/usr/bin/env python
"""
Qwen3-Next 前向示例 — 混合线性注意力 (Gated DeltaNet 3 : 1 全注意力)

重点观察三件事:
    1. 层排布: [Δ, Δ, Δ, A, ...] — 75% 的层用 O(1) 状态替代 KV cache
    2. 因果性验证: DeltaNet 没有 mask, 但递推天然因果 (前缀输出不受未来影响)
    3. "缓存"对比: 全注意力层 O(T) vs DeltaNet 恒定状态矩阵
"""

import torch

from llm_models.models.language_models.qwen3_next import Qwen3Next


def main():
    torch.manual_seed(42)

    vocab_size, T = 1000, 32
    model = Qwen3Next(
        vocab_size=vocab_size, d_model=128, n_heads=4, num_kv_heads=2,
        num_layers=8, max_len=128, linear_ratio=3, dropout=0.0,
    ).eval()

    n_params = sum(p.numel() for p in model.parameters())
    pattern = " ".join("Δ" if t == "delta" else "A" for t in model.layer_types)
    print(f"Qwen3-Next Mini | 参数量: {n_params:,}")
    print(f"层排布 (Δ=DeltaNet, A=全注意力): [{pattern}]")

    idx = torch.randint(0, vocab_size, (2, T))
    with torch.inference_mode():
        logits = model(idx)
    print(f"\n输入: {tuple(idx.shape)}  输出: {tuple(logits.shape)}")

    # ---- 因果性验证: 截断输入, 前缀 logits 必须逐位一致 ----
    with torch.inference_mode():
        logits_prefix = model(idx[:, :8])
    diff = (logits[:, :8] - logits_prefix).abs().max()
    print(f"\n因果性: |前 8 位 logits(全长) - logits(截断)| = {diff:.2e}")
    assert diff < 1e-4, "DeltaNet 递推必须天然因果, 前缀不受未来 token 影响"

    # ---- "缓存"规模对比 ----
    print("\n推理缓存对比 (本配置 8 层 = 6 Δ + 2 A):")
    for t in (1024, 32768, 131072):
        info = model.kv_cache_entries(t)
        print(f"  T={t:>7}: 全注意力层各存 {info['attn_cache_per_layer']:>7} 个位置"
              f" × {info['attn_layers']} 层;  DeltaNet 层各存 1 个状态矩阵"
              f" × {info['delta_layers']} 层 (与 T 无关)")

    gen = model.generate(idx[:1, :4], max_new_tokens=8, temperature=1.0, top_k=10)
    assert gen.shape == (1, 12)
    print("\n✅ Qwen3-Next 前向 + 因果性 + generate 通过"
          " (真实版: 80B 总参 3B 激活, 75% 层线性化)")


if __name__ == "__main__":
    main()
