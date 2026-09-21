#!/usr/bin/env python
"""
Image VAE 训练示例 — loss = MSE(x̂, x) + kl_weight · KL(q(z|x) || N(0, I))

固定的合成 batch (4 张低频色块图): recon 下降 = 背下这 4 张图, 只验证通路。
"""

import torch

from llm_models.models.generative.vae import ImageVAE
from llm_models.training.config import TrainingConfig
from llm_models.training.data import ImageDataGenerator
from llm_models.training.loss import VAELoss
from llm_models.training.trainer import Trainer


def main():
    cfg = TrainingConfig(
        learning_rate=2e-3, batch_size=4, num_steps=80,
        warmup_steps=5, log_interval=20, seed=42,
    )
    torch.manual_seed(cfg.seed)

    model = ImageVAE(image_channels=3, base_channels=16, latent_dim=4, levels=2)
    print(f"VAE | 参数量: {sum(p.numel() for p in model.parameters()):,} "
          f"| 压缩: 3×32×32={3 * 32 * 32} → 4×8×8={4 * 8 * 8} 维")

    data_gen = ImageDataGenerator(batch_size=cfg.batch_size, image_size=32)
    x = data_gen.generate_batch()["x"]
    assert x.abs().max() <= 1.0, "图像值域应与 decoder 的 tanh 输出一致"

    metrics = Trainer(model, cfg, data_gen, VAELoss(recon_weight=1.0, kl_weight=1e-4)).train()
    first, last = metrics[0], metrics[-1]
    print(f"recon: {first['recon_loss']:.4f} → {last['recon_loss']:.4f} | "
          f"KL: {first['kl_loss']:.2f} → {last['kl_loss']:.2f}")

    assert last["recon_loss"] < 0.3 * first["recon_loss"], "重建 loss 应大幅下降"
    # kl_weight 只有 1e-4: 模型会用更 "尖" 的后验换重建质量, KL 不降反升是预期行为
    assert last["kl_loss"] > 0


if __name__ == "__main__":
    main()
