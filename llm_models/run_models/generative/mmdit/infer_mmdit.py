#!/usr/bin/env python
"""
MM-DiT (SD3 风格) 前向示例 — Rectified Flow 训练目标

展示:
- 文本/图像双流同层 attention, 参数各自独立
- Flow Matching 调度: 线性路径 + velocity-prediction
"""

import torch
from llm_models.models.generative.mmdit import MMDiT
from llm_models.training import FlowMatchingScheduler, EulerFlowSampler


def main():
    torch.manual_seed(42)

    model = MMDiT(
        latent_channels=4, image_size=8, patch_size=2,
        d_model=128, n_heads=4, num_layers=2,
        text_seq_len=16, text_dim=64,
    ).eval()
    print(f"MM-DiT Mini | 参数量: {sum(p.numel() for p in model.parameters()):,}")

    scheduler = FlowMatchingScheduler(num_train_timesteps=1000)
    x0 = torch.randn(2, 4, 8, 8)
    t = scheduler.sample_timesteps(2, x0.device)
    noised = scheduler.add_noise(x0, t)

    text_embeds = torch.randn(2, 16, 64)
    text_pooled = torch.randn(2, 64)

    with torch.inference_mode():
        pred_v = model(noised.noisy, noised.t_norm, text_embeds, text_pooled)
    print(f"训练前向: x_t {tuple(noised.noisy.shape)} → velocity {tuple(pred_v.shape)}")
    print("✅ MM-DiT 前向通过")


if __name__ == "__main__":
    main()
