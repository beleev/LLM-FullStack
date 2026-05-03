#!/usr/bin/env python
"""
Image VAE 前向示例 — Latent Diffusion 的前置压缩器
"""

import torch
from llm_models.models.generative.vae import ImageVAE


def main():
    torch.manual_seed(42)

    model = ImageVAE(
        image_channels=3, base_channels=32, latent_dim=4, levels=2,
    ).eval()
    print(f"ImageVAE | 参数量: {sum(p.numel() for p in model.parameters()):,}")

    x = torch.randn(2, 3, 64, 64)
    with torch.inference_mode():
        out = model(x)

    print(f"输入:   {tuple(x.shape)}")
    print(f"重建:   {tuple(out['recon'].shape)}")
    print(f"latent: {tuple(out['z'].shape)}  (空间 ÷ 4)")
    assert out["recon"].shape == x.shape
    assert out["z"].shape[-1] == 64 // 4
    print("✅ VAE 前向通过")


if __name__ == "__main__":
    main()
