#!/usr/bin/env python
"""
Mixtral 推理示例: softmax top-2 路由 + GQA KV cache

assert 验证: 每 token 选 2 个不同专家且权重和为 1; 激活参数 < 总参数;
generate 带 cache 与不带 cache 的贪心输出逐 token 相同。模型随机初始化, 生成内容无意义。
"""

import torch

from llm_models.models.moe.mixtral import Mixtral
from llm_models.utils.generation import KVCache


def cached_logits(model, idx: torch.Tensor, prefill: int) -> torch.Tensor:
    """prefill 前 `prefill` 个 token, 其余逐个 decode, 拼回 [B, T, V] —— 应与一次性 forward 相同。"""
    cache = KVCache(len(model.layers))
    chunks = [idx[:, :prefill]] + list(idx[:, prefill:].split(1, dim=1))
    with torch.inference_mode():
        return torch.cat([model(c, cache=cache)[0] for c in chunks], dim=1)


def main():
    torch.manual_seed(42)

    vocab_size, num_experts, top_k = 1000, 4, 2
    model = Mixtral(
        vocab_size=vocab_size, d_model=256, n_heads=4, num_kv_heads=2,
        num_layers=2, num_experts=num_experts, top_k=top_k, max_len=128,
    ).eval()

    count = lambda m: sum(p.numel() for p in m.parameters())
    total = count(model)
    experts = sum(count(layer.moe.experts) for layer in model.layers)
    active = total - experts + experts * top_k // num_experts
    print(f"Mixtral Mini | 总参数 {total:,} | 每 token 激活 {active:,} ({active / total:.1%})")
    assert active < total

    idx = torch.randint(0, vocab_size, (1, 10))
    with torch.inference_mode():
        logits, all_routing = model(idx)
    assert logits.shape == (1, 10, vocab_size)

    for i, info in enumerate(all_routing):
        sel, w, probs = info["selected_experts"], info["routing_weights"], info["routing_probs"]
        assert (sel[:, 0] != sel[:, 1]).all() and sel.max() < num_experts
        assert torch.allclose(w.sum(-1), torch.ones(10), atol=1e-5)
        assert torch.allclose(probs.sum(-1), torch.ones(10), atol=1e-5)  # softmax 路由: 行和为 1
        print(f"  Layer {i}: 前 3 个 token 的专家 {sel[:3].tolist()}  权重 {[[round(v, 3) for v in r] for r in w[:3].tolist()]}")

    full = torch.randint(0, vocab_size, (2, 30))
    with torch.inference_mode():
        diff = (model(full)[0] - cached_logits(model, full, prefill=10)).abs().max().item()
    print(f"prefill 10 + 逐 token decode 20 步 vs 一次性 forward: logits 最大差 {diff:.2e}")
    assert diff < 1e-4, "cache 路径与无 cache 路径的 logits 不一致"
    with_cache = model.generate(idx, max_new_tokens=20, temperature=0, use_cache=True)
    no_cache = model.generate(idx, max_new_tokens=20, temperature=0, use_cache=False)
    assert torch.equal(with_cache, no_cache), "cache 与无 cache 的贪心输出必须完全相同"
    print(f"贪心生成 20 token, cache 与无 cache 完全一致: {with_cache[0, 10:].tolist()}")


if __name__ == "__main__":
    main()
