#!/usr/bin/env python
"""
MM-DiT 训练示例 — Rectified Flow, loss = MSE(pred, v), v = ε - x_0

固定的合成 batch (x_t, t, v, 文本都被缓存): loss 下降 = 背下这 2 个样本, 只验证通路。
"""

import torch

from llm_models.models.generative.mmdit import MMDiT
from llm_models.training.config import TrainingConfig
from llm_models.training.data import DiffusionDataGenerator
from llm_models.training.diffusion import DiffusionLoss, FlowMatchingScheduler
from llm_models.training.trainer import Trainer


def main():
    cfg = TrainingConfig(
        learning_rate=1e-3, batch_size=2, num_steps=60,
        warmup_steps=5, log_interval=10, seed=42,
    )
    torch.manual_seed(cfg.seed)

    model = MMDiT(
        latent_channels=4, image_size=8, patch_size=2,
        d_model=96, n_heads=4, num_layers=2,
        text_seq_len=16, text_dim=64,
    )
    print(f"MM-DiT Mini | 参数量: {sum(p.numel() for p in model.parameters()):,}")

    data_gen = DiffusionDataGenerator(
        scheduler=FlowMatchingScheduler(num_train_timesteps=1000), batch_size=cfg.batch_size,
        latent_channels=4, latent_size=8,
        text_seq_len=16, text_dim=64,     # 开启文本流
    )
    t = data_gen.generate_batch()["t"]
    print(f"喂给模型的 t: {t.tolist()}  (Flow Matching 的 t∈[0,1] 已 ×1000)")
    assert t.max() > 1.0, "t 没有被缩放到 [0, 1000) 量纲"

    metrics = Trainer(model, cfg, data_gen, DiffusionLoss()).train()
    first, last = metrics[0]["total_loss"], metrics[-1]["total_loss"]
    print(f"初始 loss: {first:.4f} (理论 Var(ε)+Var(x_0)=2) | 终态: {last:.4f}")

    # 零初始化 → pred=0 → loss = mean(v²) ≈ 2; 只有 512 个元素, 估计的标准差约 0.125
    assert abs(first - 2.0) < 0.5, "初始 loss 应 ≈ 2.0"
    assert last < 0.2 * first, "固定 batch 上 loss 应大幅下降"


if __name__ == "__main__":
    main()
