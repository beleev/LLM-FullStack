#!/usr/bin/env python
"""
Mixtral (sparse MoE) 前向示例

展示:
- 每层 FFN 被替换为 8 专家 top-2 MoE (softmax + renormalize)
- forward 返回 (logits, routing_info) 与 DeepSeekV3 一致
- 路由信息可用于外部算 Switch-Transformer 风格 aux loss
"""

import torch
from llm_models.models.moe.mixtral import Mixtral


def main():
    torch.manual_seed(42)

    vocab_size = 1000
    model = Mixtral(
        vocab_size=vocab_size, d_model=256, n_heads=4, num_kv_heads=2,
        num_layers=2, num_experts=4, top_k=2, max_len=128,
    ).eval()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Mixtral Mini | 总参数: {num_params:,} (4 专家 top-2)")

    idx = torch.randint(0, vocab_size, (1, 10))
    with torch.inference_mode():
        logits, all_routing = model(idx)
    print(f"输出 logits: {tuple(logits.shape)}")

    print("\n每层 routing (前 3 个 token 的 top-2 专家):")
    for i, info in enumerate(all_routing):
        selected = info["selected_experts"][:3].tolist()
        weights = info["routing_weights"][:3].round(decimals=3).tolist()
        print(f"  Layer {i}: experts {selected}  weights {weights}")

    print("\n✅ Mixtral 前向通过")


if __name__ == "__main__":
    main()
