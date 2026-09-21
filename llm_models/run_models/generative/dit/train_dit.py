#!/usr/bin/env python
"""
DiT 训练示例 — DDPM ε-prediction, loss = MSE(pred, ε)

数据是一个固定的合成 batch (x_t, t, ε 都被缓存): loss 下降 = 模型背下了这 4 个样本的 ε,
只证明 前向 / 反向 / adaLN 条件通路 是通的, 不代表学会了去噪。
"""

import torch

from llm_models.models.generative.dit import DiT
from llm_models.training.config import TrainingConfig
from llm_models.training.data import DiffusionDataGenerator
from llm_models.training.diffusion import DDPMScheduler, DiffusionLoss
from llm_models.training.trainer import Trainer


def main():
    cfg = TrainingConfig(
        learning_rate=1e-3, batch_size=4, num_steps=60,
        warmup_steps=5, log_interval=10, seed=42,
    )
    torch.manual_seed(cfg.seed)

    num_classes = 8
    model = DiT(
        latent_channels=4, image_size=8, patch_size=2,
        d_model=96, n_heads=4, num_layers=2,
        num_classes=num_classes, class_dropout=0.1,
    )
    print(f"DiT Mini | 参数量: {sum(p.numel() for p in model.parameters()):,}")

    data_gen = DiffusionDataGenerator(
        scheduler=DDPMScheduler(num_train_timesteps=1000), batch_size=cfg.batch_size,
        latent_channels=4, latent_size=8, num_classes=num_classes,
    )
    loss_fn = DiffusionLoss()

    # loss_mask: 只在 mask=1 的位置算 MSE, 按有效元素数平均
    pred, target = torch.zeros(1, 1, 2, 2), torch.tensor([[[[1.0, 1.0], [3.0, 3.0]]]])
    top_row = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    assert loss_fn.compute(pred, target)["total_loss"].item() == 5.0              # (1+1+9+9)/4
    assert loss_fn.compute(pred, target, loss_mask=top_row)["total_loss"].item() == 1.0

    metrics = Trainer(model, cfg, data_gen, loss_fn).train()
    first, last = metrics[0]["total_loss"], metrics[-1]["total_loss"]
    print(f"初始 loss: {first:.4f} (理论 E[ε²]=1) | 终态: {last:.4f}")

    # FinalLayer 零初始化 → 初始 pred=0 → loss = mean(ε²) ≈ 1
    assert abs(first - 1.0) < 0.15, "初始 loss 应 ≈ 1.0"
    assert last < 0.2 * first, "固定 batch 上 loss 应大幅下降"


if __name__ == "__main__":
    main()
