#!/usr/bin/env python
"""
VAR (Visual AutoRegressive) 前向 + 采样示例

展示:
- 图像 → VQ tokenizer → 离散 token 序列 → GPT 自回归建模
- sample(): 从 [BOS] 自回归生成整张图像, 复用 GPT3.generate 的路径
"""

import torch
from llm_models.models.generative.var import ImageTokenizer, VARModel


def main():
    torch.manual_seed(42)

    tokenizer = ImageTokenizer(
        image_size=32, codebook_size=256, latent_dim=32,
        base_channels=32, levels=2,
    ).eval()
    model = VARModel(
        tokenizer=tokenizer,
        gpt_d_model=128, gpt_n_heads=4, gpt_num_layers=2,
    ).eval()
    print(f"VAR Mini | 参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"序列长度 (grid^2): {model.num_tokens}, codebook: {model.codebook_size}")

    # 前向: teacher forcing
    imgs = torch.randn(2, 3, 32, 32)
    with torch.inference_mode():
        out = model(imgs)
    print(f"logits: {tuple(out['logits'].shape)}  labels: {tuple(out['labels'].shape)}")

    # 采样: 从 BOS 生成整张图
    with torch.inference_mode():
        gen = model.sample(batch_size=1, temperature=1.0, top_k=10)
    print(f"生成图像: {tuple(gen.shape)}")
    print("✅ VAR 前向 + 采样通过")


if __name__ == "__main__":
    main()
