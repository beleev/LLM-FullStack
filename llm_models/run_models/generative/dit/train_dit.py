#!/usr/bin/env python
"""
DiT 训练示例 — DDPM ε-prediction

训练目标: 模型预测当前 x_t 的噪声 ε, 与 scheduler 采出来的噪声做 MSE。
合成数据: 把随机张量当 latent, 仅验证训练链路可收敛。
"""

import torch
from llm_models.models.generative.dit import DiT
from llm_models.training import (
    Trainer, TrainingConfig, DiffusionLoss,
    DDPMScheduler, DiffusionDataGenerator,
)


def main():
    cfg = TrainingConfig(
        learning_rate=1e-4, batch_size=4, num_steps=30,
        warmup_steps=3, log_interval=5, seed=42,
    )
    torch.manual_seed(cfg.seed)

    num_classes = 8
    model = DiT(
        latent_channels=4, image_size=8, patch_size=2,
        d_model=96, n_heads=4, num_layers=2,
        num_classes=num_classes, class_dropout=0.1,
    )
    print(f"DiT Mini | 参数量: {sum(p.numel() for p in model.parameters()):,}")

    scheduler = DDPMScheduler(num_train_timesteps=1000)
    data_gen = DiffusionDataGenerator(
        scheduler=scheduler, batch_size=cfg.batch_size,
        latent_channels=4, latent_size=8, num_classes=num_classes,
    )
    loss_fn = DiffusionLoss()

    trainer = Trainer(model, cfg, data_gen, loss_fn)
    metrics = trainer.train()

    print(f"初始 loss: {metrics[0]['total_loss']:.4f} | 终态: {metrics[-1]['total_loss']:.4f}")
    print("DiT 训练链路验证通过!")


if __name__ == "__main__":
    main()
