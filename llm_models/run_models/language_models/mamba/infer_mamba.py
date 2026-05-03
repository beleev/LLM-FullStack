#!/usr/bin/env python
"""
Mamba (Selective SSM) 前向示例

展示:
- 非注意力架构: 没有 QKV, 没有因果 mask, 只有 SSM + 1D conv + gate
- 线性时间复杂度 O(T)
"""

import torch
from llm_models.models.language_models.mamba import Mamba


def main():
    torch.manual_seed(42)

    vocab_size = 1000
    model = Mamba(
        vocab_size=vocab_size, d_model=128, num_layers=4, d_state=16, d_conv=4,
    ).eval()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Mamba Mini | 参数量: {num_params:,}")

    idx = torch.randint(0, vocab_size, (1, 20))
    with torch.inference_mode():
        logits = model(idx)
    print(f"输入: {tuple(idx.shape)}  输出: {tuple(logits.shape)}")

    # do_sample=False → 纯 argmax, 避免未训练权重的数值噪声导致采样失败
    gen = model.generate(idx[:, :5], max_new_tokens=8, do_sample=False)
    print(f"generate 输出: {tuple(gen.shape)}")
    assert gen.shape == (1, 13)
    print("✅ Mamba 前向 + generate 通过")


if __name__ == "__main__":
    main()
