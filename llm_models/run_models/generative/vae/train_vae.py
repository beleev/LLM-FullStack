#!/usr/bin/env python
"""
Image VAE 训练示例 — 重建 + KL loss
"""

import torch
from llm_models.models.generative.vae import ImageVAE
from llm_models.training import (
    Trainer, TrainingConfig, VAELoss, ImageDataGenerator,
)


def main():
    cfg = TrainingConfig(
        learning_rate=1e-3, batch_size=4, num_steps=30,
        warmup_steps=3, log_interval=5, seed=42,
    )
    torch.manual_seed(cfg.seed)

    model = ImageVAE(
        image_channels=3, base_channels=16, latent_dim=4, levels=2,
    )
    print(f"VAE | 参数量: {sum(p.numel() for p in model.parameters()):,}")

    data_gen = ImageDataGenerator(batch_size=cfg.batch_size, image_size=32)
    loss_fn = VAELoss(recon_weight=1.0, kl_weight=1e-4)

    trainer = Trainer(model, cfg, data_gen, loss_fn)
    metrics = trainer.train()

    print(f"初始 recon: {metrics[0]['recon_loss']:.4f} | 终态 recon: {metrics[-1]['recon_loss']:.4f}")
    print("VAE 训练链路验证通过!")


if __name__ == "__main__":
    main()
