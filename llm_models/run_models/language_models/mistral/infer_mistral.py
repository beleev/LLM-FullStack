#!/usr/bin/env python
"""
Mistral 推理示例 — 滑动窗口注意力 (SWA)

    1. 与 LLaMA 参数量逐个相同 (SWA 只换 mask)
    2. mask 可见格子: 全因果 T(T+1)/2  vs  带状 ≈ T·W
    3. 窗口真的生效: 单层模型里改位置 0, 只有位置 0..W-1 的输出变 (多层会跨层接力, 所以用 1 层验证)
    4. rolling KV cache: 每层 cache 长度封顶 W, 且有/无 cache 输出逐 token 相同
"""

import torch

from llm_models.models.language_models.llama import LLaMA
from llm_models.models.language_models.mistral import Mistral
from llm_models.utils.generation import KVCache, benchmark_kv_cache
from llm_models.utils.masks import build_causal_mask, build_sliding_window_mask


@torch.inference_mode()
def main():
    torch.manual_seed(42)
    vocab_size, T, W = 1000, 64, 8
    common = dict(vocab_size=vocab_size, d_model=256, n_heads=8, num_kv_heads=2,
                  num_layers=4, max_len=256, dropout=0.0)
    llama, mistral = LLaMA(**common).eval(), Mistral(**common, window_size=W).eval()

    # ---- 1) 参数量 ----
    n_llama = sum(p.numel() for p in llama.parameters())
    n_mistral = sum(p.numel() for p in mistral.parameters())
    assert n_llama == n_mistral, "SWA 只换 mask, 参数量必须一字不差"
    print(f"[1] 参数量 LLaMA = Mistral = {n_mistral:,}")

    # ---- 2) 计算量 ----
    full = int(build_causal_mask(T, torch.device("cpu")).sum())
    band = int(build_sliding_window_mask(T, W, torch.device("cpu")).sum())
    assert full == T * (T + 1) // 2 and band == W * (W + 1) // 2 + (T - W) * W
    print(f"[2] T={T} 可见格子: 全因果 {full}  vs  带状 {band} (W={W})")

    # ---- 3) 窗口生效 (单层: 没有跨层接力) ----
    one = Mistral(**{**common, "num_layers": 1}, window_size=W).eval()
    idx = torch.randint(1, vocab_size, (1, T))
    changed = idx.clone()
    changed[:, 0] = (idx[:, 0] % (vocab_size - 1)) + 1            # 只改位置 0
    same = (one(idx) - one(changed)).abs().amax(dim=(0, 2)) == 0  # [T] 每个位置是否完全不变
    assert not same[:W].any() and same[W:].all(), "位置 0 只应影响 [0, W) 内的 query"
    print(f"[3] 单层: 改位置 0 只影响位置 0..{W - 1}; "
          f"{mistral.num_layers} 层时感受野 ≈ {mistral.receptive_field()}")

    # ---- 4) rolling KV cache ----
    cache = KVCache(len(mistral.layers))
    mistral(idx[:, :20], cache=cache)                             # prefill 20 个 token
    mistral(idx[:, 20:21], cache=cache)                           # 再 decode 1 个
    lens = {c["k"].size(2) for c in cache.layers}
    assert lens == {W} and cache.pos == 21
    speedup = benchmark_kv_cache(mistral, idx[:, :8], max_new_tokens=200)
    print(f"[4] 已读 {cache.pos} 个 token, 每层 cache 只有 {W} 个; "
          f"生成 200 token 有/无 cache 输出一致, 加速 {speedup:.1f}x")


if __name__ == "__main__":
    main()
