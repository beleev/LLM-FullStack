#!/usr/bin/env python
"""
Mistral 前向示例 — 滑动窗口注意力 (SWA)

重点观察三件事:
    1. Mistral 与 LLaMA 参数量完全相同 (SWA 只是换 mask, 不加参数)
    2. mask 可见格子数: 全因果 O(T^2/2) vs 带状 O(T·W)
    3. 推理 KV cache 上限: LLaMA O(T) vs Mistral O(W) (rolling buffer)
"""

import torch

from llm_models.models.language_models.llama import LLaMA
from llm_models.models.language_models.mistral import Mistral
from llm_models.utils.masks import build_causal_mask, build_sliding_window_mask


def main():
    torch.manual_seed(42)

    vocab_size, T, W = 1000, 64, 8
    common = dict(
        vocab_size=vocab_size, d_model=256, n_heads=8, num_kv_heads=2,
        num_layers=4, max_len=128, dropout=0.0,
    )
    llama = LLaMA(**common).eval()
    mistral = Mistral(**common, window_size=W).eval()

    n_llama = sum(p.numel() for p in llama.parameters())
    n_mistral = sum(p.numel() for p in mistral.parameters())
    print(f"LLaMA   参数量: {n_llama:,}")
    print(f"Mistral 参数量: {n_mistral:,} (窗口 W={W})")
    assert n_llama == n_mistral, "SWA 只换 mask, 参数量必须一字不差"

    # ---- mask 对比: 可见格子数就是注意力计算量 ----
    full = build_causal_mask(T, torch.device("cpu"))
    band = build_sliding_window_mask(T, W, torch.device("cpu"))
    print(f"\nT={T} 时 mask 可见格子数:")
    print(f"  全因果 (LLaMA):   {int(full.sum()):>5}  (= T(T+1)/2)")
    print(f"  带状   (Mistral): {int(band.sum()):>5}  (≈ T·W)")

    idx = torch.randint(0, vocab_size, (2, T))
    with torch.inference_mode():
        logits = mistral(idx)
    print(f"\n输入: {tuple(idx.shape)}  输出: {tuple(logits.shape)}")

    # ---- 感受野与 KV cache 上限 ----
    print(f"\n理论感受野: {mistral.num_layers} 层 × W={W} ≈ {mistral.receptive_field()} token")
    print("KV cache 需要保留的位置数 (rolling buffer):")
    for t in (16, 64, 4096, 131072):
        print(f"  T={t:>7}:  LLaMA 存 {t:>7} 个  vs  Mistral 封顶 {mistral.kv_cache_entries(t)} 个")

    gen = mistral.generate(idx[:1, :4], max_new_tokens=8, temperature=1.0, top_k=10)
    assert gen.shape == (1, 12)
    print("\n✅ Mistral 前向 + generate 通过 (Mistral-7B: W=4096, 32 层感受野 ≈ 131K)")


if __name__ == "__main__":
    main()
