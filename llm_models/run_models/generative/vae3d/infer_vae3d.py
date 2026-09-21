#!/usr/bin/env python
"""
Causal 3D VAE 自检: 时空压缩倍数 + "改未来帧, 过去的输出一个数都不能变" + 短训练
"""

import torch
import torch.nn.functional as F

from llm_models.models.generative.vae3d import CausalVideoVAE
from llm_models.training.loss import VAELoss


def main():
    torch.manual_seed(42)
    model = CausalVideoVAE(base_channels=8, latent_dim=4, spatial_levels=2, time_levels=1).eval()
    print(f"CausalVideoVAE | 参数量: {sum(p.numel() for p in model.parameters()):,}")

    x = torch.rand(1, 3, 8, 16, 16) * 2 - 1                              # [B, 3, T, H, W]
    with torch.no_grad():
        mean = model.encode(x)["mean"]
        assert mean.shape == (1, 4, 4, 4, 4)                              # T÷2, H/W÷4
        print(f"x {tuple(x.shape)} → z {tuple(mean.shape)}: token 数 {8 * 16 * 16} → {4 * 4 * 4}")

        # encoder 因果性: 改第 6、7 帧 → 只有 latent 第 3 帧 (覆盖原始帧 4..6) 可以变
        x2 = x.clone()
        x2[:, :, 6:] += 1.0
        d_enc = (model.encode(x2)["mean"] - mean).abs().amax(dim=(0, 1, 3, 4))
        print("改 x 的 6-7 帧 → latent 各帧变化:", [f"{v:.3f}" for v in d_enc.tolist()])
        assert d_enc[:3].max() == 0 and d_enc[3] > 0

        # decoder 因果性: 改 latent 第 3 帧 → 只有输出第 6、7 帧可以变
        z = torch.randn(1, 4, 4, 4, 4)
        z2 = z.clone()
        z2[:, :, 3:] += 1.0
        d_dec = (model.decode(z2) - model.decode(z)).abs().amax(dim=(0, 1, 3, 4))
        print("改 z 的第 3 帧 → 输出各帧变化:", [f"{v:.3f}" for v in d_dec.tolist()])
        assert d_dec[:6].max() == 0 and d_dec[6:].min() > 0

    # 短训练: 背下一段低频 "视频" (固定 batch), 只验证梯度通路
    video = F.interpolate(torch.randn(1, 3, 2, 4, 4), size=(8, 16, 16), mode="trilinear").tanh()
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    loss_fn = VAELoss(kl_weight=1e-4)
    history = []
    for _ in range(40):
        losses = loss_fn.compute(model(video), video)
        opt.zero_grad()
        losses["total_loss"].backward()
        opt.step()
        history.append(losses["recon_loss"].item())
    print(f"recon: {history[0]:.4f} → {history[-1]:.4f}")
    assert history[-1] < 0.5 * history[0]


if __name__ == "__main__":
    main()
