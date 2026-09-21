#!/usr/bin/env python
"""
GPT-3 推理示例 — 因果性 + KV cache

验证 (全部是 assert):
    1. 因果 mask: 改动未来的 token, 过去位置的 logits 一位都不变
    2. KV cache: 有/无 cache 的贪心生成逐 token 相同, 打印加速比
       (Sin-PE 与 RoPE 两种位置编码都测: 前者要平移 offset, 后者要平移 position_ids)
"""

import torch

from llm_models.models.language_models.gpt3 import GPT3
from llm_models.utils.generation import benchmark_kv_cache


def main():
    torch.manual_seed(42)
    vocab_size, T = 1000, 16

    for use_rope in (False, True):
        model = GPT3(
            vocab_size=vocab_size, d_model=256, n_heads=8, num_layers=4,
            max_len=256, dropout=0.1, use_rope=use_rope,
        ).eval()
        name = "RoPE" if use_rope else "Sin-PE"
        print(f"GPT-3 Mini ({name}) | 参数量: {sum(p.numel() for p in model.parameters()):,}")

        idx = torch.randint(1, vocab_size, (2, T))
        with torch.inference_mode():
            logits = model(idx)                                   # [B, T, V]
            future_changed = idx.clone()
            future_changed[:, T // 2:] = torch.randint(1, vocab_size, (2, T - T // 2))
            logits2 = model(future_changed)
        assert logits.shape == (2, T, vocab_size)
        # 位置 < T/2 看不到被改动的后半段 → logits 必须完全不变
        assert torch.equal(logits[:, : T // 2], logits2[:, : T // 2]), "因果 mask 失效: 过去看到了未来"
        assert not torch.allclose(logits[:, T // 2:], logits2[:, T // 2:])

        speedup = benchmark_kv_cache(model, idx[:, :8], max_new_tokens=200)
        print(f"  因果性通过 | 生成 200 token: 有/无 cache 输出完全一致, 加速 {speedup:.1f}x")


if __name__ == "__main__":
    main()
