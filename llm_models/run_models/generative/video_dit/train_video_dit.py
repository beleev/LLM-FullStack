#!/usr/bin/env python
"""
Video DiT 训练示例
"""

import torch
from llm_models.models.generative.video_dit import VideoDiT
from llm_models.training import (
    Trainer, TrainingConfig, DiffusionLoss,
    DDPMScheduler, VideoDiffusionDataGenerator,
)


def main():
    cfg = TrainingConfig(
        learning_rate=1e-4, batch_size=2, num_steps=20,
        warmup_steps=2, log_interval=5, seed=42,
    )
    torch.manual_seed(cfg.seed)

    num_classes = 4
    model = VideoDiT(
        latent_channels=4, video_latent_size=(4, 8, 8),
        patch_size_t=2, patch_size_hw=2,
        d_model=96, n_heads=4, num_layers=2, num_classes=num_classes,
    )
    print(f"VideoDiT Mini | 参数量: {sum(p.numel() for p in model.parameters()):,}")

    scheduler = DDPMScheduler(num_train_timesteps=1000)
    data_gen = VideoDiffusionDataGenerator(
        scheduler=scheduler, batch_size=cfg.batch_size,
        latent_channels=4, latent_size=(4, 8, 8), num_classes=num_classes,
    )
    loss_fn = DiffusionLoss()

    trainer = Trainer(model, cfg, data_gen, loss_fn)
    metrics = trainer.train()

    print(f"初始 loss: {metrics[0]['total_loss']:.4f} | 终态: {metrics[-1]['total_loss']:.4f}")
    print("VideoDiT 训练链路验证通过!")


if __name__ == "__main__":
    main()
