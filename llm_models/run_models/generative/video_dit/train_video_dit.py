#!/usr/bin/env python
"""
Video DiT 训练示例 — DDPM ε-prediction, 输入是 5D 视频 latent [B, C, T, H, W]

固定的合成 batch: loss 下降 = 背下这 2 段 "视频" 的 ε, 只验证通路。
"""

import torch

from llm_models.models.generative.video_dit import VideoDiT
from llm_models.training.config import TrainingConfig
from llm_models.training.data import VideoDiffusionDataGenerator
from llm_models.training.diffusion import DDPMScheduler, DiffusionLoss
from llm_models.training.trainer import Trainer


def main():
    cfg = TrainingConfig(
        learning_rate=1e-3, batch_size=2, num_steps=60,
        warmup_steps=5, log_interval=10, seed=42,
    )
    torch.manual_seed(cfg.seed)

    num_classes = 4
    model = VideoDiT(
        latent_channels=4, video_latent_size=(4, 8, 8),
        patch_size_t=2, patch_size_hw=2,
        d_model=96, n_heads=4, num_layers=2, num_classes=num_classes,
    )
    print(f"VideoDiT Mini | 参数量: {sum(p.numel() for p in model.parameters()):,} "
          f"| token 数 = (4/2)·(8/2)·(8/2) = {model.num_patches}")

    data_gen = VideoDiffusionDataGenerator(
        scheduler=DDPMScheduler(num_train_timesteps=1000), batch_size=cfg.batch_size,
        latent_channels=4, latent_size=(4, 8, 8), num_classes=num_classes,
    )

    metrics = Trainer(model, cfg, data_gen, DiffusionLoss()).train()
    first, last = metrics[0]["total_loss"], metrics[-1]["total_loss"]
    print(f"初始 loss: {first:.4f} (理论 E[ε²]=1) | 终态: {last:.4f}")
    assert abs(first - 1.0) < 0.15, "零初始化 → pred=0 → 初始 loss 应 ≈ 1.0"
    assert last < 0.2 * first, "固定 batch 上 loss 应大幅下降"


if __name__ == "__main__":
    main()
