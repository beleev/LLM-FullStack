#!/usr/bin/env python
"""
LLaMA 前向示例

展示 GQA + SwiGLU + RMSNorm + RoPE 的组合, 与 GPT-3 的组件对比 (MHA + GELU + LN + SinPE)。
输出同为 [B, T, V], 但 KV cache 因 GQA 减半。
"""

import torch
from llm_models.models.language_models.llama import LLaMA


def main():
    torch.manual_seed(42)

    vocab_size = 1000
    model = LLaMA(
        vocab_size=vocab_size, d_model=256, n_heads=8, num_kv_heads=2,
        num_layers=2, max_len=128, dropout=0.0,
    ).eval()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"LLaMA Mini | 参数量: {num_params:,} (GQA: 8 Q heads / 2 KV heads)")

    idx = torch.randint(0, vocab_size, (2, 16))
    with torch.inference_mode():
        logits = model(idx)
    print(f"输入: {tuple(idx.shape)}  输出: {tuple(logits.shape)}")

    # 测试 generate
    gen = model.generate(idx[:1, :4], max_new_tokens=8, temperature=1.0, top_k=10)
    print(f"generate 输出: {tuple(gen.shape)} (4 prompt + 8 new = 12)")
    assert gen.shape == (1, 12)
    print("✅ LLaMA 前向 + generate 通过")


if __name__ == "__main__":
    main()
