#!/usr/bin/env python
"""
DeepSeek-V3 推理示例: MoE 路由 + MLA 的 latent KV cache

三件事, 每件都有 assert:
  1. 路由: 每 token 选 K 个不同专家, 权重和为 1; 激活参数 ≪ 总参数
  2. 带 cache 的 prefill+decode 与一次性 forward 的 logits 一致; 贪心 generate 输出逐 token 相同
  3. 每 token 每层的 cache 浮点数 (从真实 cache 张量里数出来, 再与公式对照):
        MHA 2·H·Dh   vs   GQA 2·Hkv·Dh   vs   MLA r + rope
模型是随机初始化的, 生成内容无意义; 这里验证的是机制。
"""

import torch

from llm_models.layers.core.attention import GroupedQueryAttention
from llm_models.models.moe.deepseekV3 import DeepSeekV3
from llm_models.utils.generation import KVCache

D_MODEL, N_HEADS, N_KV_HEADS, R, ROPE = 512, 8, 2, 64, 32


def cached_logits(model, idx: torch.Tensor, prefill: int) -> torch.Tensor:
    """prefill 前 `prefill` 个 token, 其余逐个 decode, 拼回 [B, T, V] —— 应与一次性 forward 相同。"""
    cache = KVCache(len(model.layers))
    chunks = [idx[:, :prefill]] + list(idx[:, prefill:].split(1, dim=1))
    with torch.inference_mode():
        return torch.cat([model(c, cache=cache)[0] for c in chunks], dim=1)


def floats_per_token(cache: dict, seq_len: int) -> int:
    return sum(t.numel() for t in cache.values()) // seq_len  # batch = 1


def main():
    torch.manual_seed(42)
    vocab_size, top_k, num_experts = 1000, 2, 8
    model = DeepSeekV3(
        vocab_size=vocab_size, d_model=D_MODEL, n_heads=N_HEADS, num_layers=2, max_len=128,
        num_routed_experts=num_experts, num_shared_experts=1, top_k=top_k,
        latent_dim=R, qk_rope_head_dim=ROPE,
    ).eval()
    idx = torch.randint(0, vocab_size, (1, 10))

    # ---- 1) 路由 ----
    with torch.inference_mode():
        logits, all_routing_info = model(idx)
    assert logits.shape == (1, 10, vocab_size)
    for i, info in enumerate(all_routing_info):
        sel, w = info["selected_experts"], info["routing_weights"]  # [N, K]
        assert sel.shape == (10, top_k) and (sel[:, 0] != sel[:, 1]).all() and sel.max() < num_experts
        assert torch.allclose(w.sum(-1), torch.ones(10), atol=1e-5)
        print(f"Layer {i}: 前 3 个 token 选中专家 {sel[:3].tolist()}, 权重 {[[round(v, 3) for v in r] for r in w[:3].tolist()]}")

    p = model.get_num_active_params()
    assert p["total_params"] == sum(t.numel() for t in model.parameters())
    assert p["active_params"] < p["total_params"]
    print(f"总参数 {p['total_params']:,} | 每 token 激活 {p['active_params']:,} "
          f"({p['active_params'] / p['total_params']:.1%})")

    # ---- 2) KV cache 不改变结果 ----
    full = torch.randint(0, vocab_size, (2, 30))
    with torch.inference_mode():
        diff = (model(full)[0] - cached_logits(model, full, prefill=10)).abs().max().item()
    print(f"prefill 10 + 逐 token decode 20 步 vs 一次性 forward: logits 最大差 {diff:.2e}")
    assert diff < 1e-4, "cache 路径与无 cache 路径的 logits 不一致"
    with_cache = model.generate(idx, max_new_tokens=20, temperature=0, use_cache=True)
    no_cache = model.generate(idx, max_new_tokens=20, temperature=0, use_cache=False)
    assert torch.equal(with_cache, no_cache), "cache 与无 cache 的贪心输出必须完全相同"
    print(f"贪心生成 20 token, cache 与无 cache 完全一致: {with_cache[0, 10:].tolist()}")

    # ---- 3) cache 体积: 直接数真实 cache 里的浮点数 ----
    S = idx.size(1)
    cache = KVCache(len(model.layers))
    with torch.inference_mode():
        model(idx, cache=cache)
        assert set(cache.layers[0]) == {"c_kv", "k_rope"}, "MLA 只该缓存 latent 和共享 k_rope"
        assert cache.layers[0]["c_kv"].shape == (1, S, R) and cache.layers[0]["k_rope"].shape == (1, 1, S, ROPE)
        mla = floats_per_token(cache.layers[0], S)

        x, measured = torch.randn(1, S, D_MODEL), {}
        for name, hkv in [("MHA", N_HEADS), ("GQA", N_KV_HEADS)]:
            c = {}
            GroupedQueryAttention(D_MODEL, N_HEADS, num_kv_heads=hkv)(x, cache=c)
            measured[name] = floats_per_token(c, S)

    d_head = D_MODEL // N_HEADS
    assert mla == R + ROPE
    assert measured["MHA"] == 2 * N_HEADS * d_head and measured["GQA"] == 2 * N_KV_HEADS * d_head
    print(f"\n每 token 每层 cache 浮点数 (d_model={D_MODEL}, H={N_HEADS}, Dh={d_head}):")
    print(f"  MHA  2·H·Dh   = {measured['MHA']}")
    print(f"  GQA  2·Hkv·Dh = {measured['GQA']}  (Hkv={N_KV_HEADS}, MHA 的 {measured['GQA'] / measured['MHA']:.1%})")
    print(f"  MLA  r + rope = {mla}  (r={R}, rope={ROPE}, MHA 的 {mla / measured['MHA']:.1%}, GQA 的 {mla / measured['GQA']:.1%})")
    assert mla < measured["GQA"] < measured["MHA"]


if __name__ == "__main__":
    main()
