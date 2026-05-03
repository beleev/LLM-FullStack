#!/usr/bin/env python
"""
MM-DiT 训练示例 — Rectified Flow (velocity prediction)
"""

import torch
from llm_models.models.generative.mmdit import MMDiT
from llm_models.training import (
    Trainer, TrainingConfig, DiffusionLoss,
    FlowMatchingScheduler, DiffusionDataGenerator,
)


def main():
    cfg = TrainingConfig(
        learning_rate=1e-4, batch_size=2, num_steps=20,
        warmup_steps=2, log_interval=5, seed=42,
    )
    torch.manual_seed(cfg.seed)

    model = MMDiT(
        latent_channels=4, image_size=8, patch_size=2,
        d_model=96, n_heads=4, num_layers=2,
        text_seq_len=16, text_dim=64,
    )
    print(f"MM-DiT Mini | 参数量: {sum(p.numel() for p in model.parameters()):,}")

    scheduler = FlowMatchingScheduler(num_train_timesteps=1000)
    data_gen = DiffusionDataGenerator(
        scheduler=scheduler, batch_size=cfg.batch_size,
        latent_channels=4, latent_size=8,
        text_seq_len=16, text_dim=64,     # 开启 MM-DiT 的文本流
    )
    loss_fn = DiffusionLoss()

    trainer = Trainer(model, cfg, data_gen, loss_fn)
    metrics = trainer.train()

    print(f"初始 loss: {metrics[0]['total_loss']:.4f} | 终态: {metrics[-1]['total_loss']:.4f}")
    print("MM-DiT 训练链路验证通过!")


if __name__ == "__main__":
    main()
