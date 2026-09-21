#!/usr/bin/env python
"""
Image VAE 结构自检: 压缩倍数、重参数化的随机性、KL 闭式解
"""

import torch

from llm_models.models.generative.vae import ImageVAE
from llm_models.training.loss import VAELoss


def main():
    torch.manual_seed(42)

    model = ImageVAE(image_channels=3, base_channels=32, latent_dim=4, levels=2).eval()
    print(f"ImageVAE | 参数量: {sum(p.numel() for p in model.parameters()):,}")

    x = torch.rand(2, 3, 64, 64) * 2 - 1
    with torch.inference_mode():
        out1, out2 = model(x), model(x)

    # 1) levels=2 → 空间 ÷4; 输出经 tanh ∈ [-1, 1]
    assert out1["recon"].shape == x.shape and out1["z"].shape == (2, 4, 16, 16)
    assert out1["recon"].abs().max() <= 1.0
    print(f"x {tuple(x.shape)} → z {tuple(out1['z'].shape)}: {x[0].numel()} → {out1['z'][0].numel()} 维 "
          f"({x[0].numel() / out1['z'][0].numel():.0f}× 压缩)")

    # 2) 重参数化: μ 是确定的, z = μ + σ·ε 每次不同; σ→0 时 z 退化为 μ (普通 AE)
    assert torch.equal(out1["mean"], out2["mean"]) and not torch.equal(out1["z"], out2["z"])
    mean = out1["mean"]
    z_det = ImageVAE.reparameterize(mean, torch.full_like(mean, -30.0))
    assert torch.allclose(z_det, mean, atol=1e-5)

    # 3) KL 闭式解: q = N(0, I) 时为 0; μ=1, σ=1 时每维 0.5
    kl = lambda m, lv: VAELoss().compute(
        {"recon": x, "mean": m, "logvar": lv}, x)["kl_loss"].item()
    zeros = torch.zeros(2, 4, 16, 16)
    assert kl(zeros, zeros) == 0.0
    assert abs(kl(zeros + 1, zeros) - 0.5 * 4 * 16 * 16) < 1e-3
    print(f"KL(N(0,I)||N(0,I)) = 0 | KL(N(1,I)||N(0,I)) = {kl(zeros + 1, zeros):.1f} = 0.5 × 1024 维 "
          f"| 未训练 encoder 的 KL = {kl(out1['mean'], out1['logvar']):.2f}")


if __name__ == "__main__":
    main()
