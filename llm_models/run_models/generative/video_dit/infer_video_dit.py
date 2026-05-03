#!/usr/bin/env python
"""
Video DiT (Sora-lite) 前向 + 采样示例
"""

import torch
from llm_models.models.generative.video_dit import VideoDiT
from llm_models.training import DDPMScheduler, DDIMSampler


def main():
    torch.manual_seed(42)

    model = VideoDiT(
        latent_channels=4, video_latent_size=(4, 8, 8),
        patch_size_t=2, patch_size_hw=2,
        d_model=128, n_heads=4, num_layers=2,
        num_classes=4,
    ).eval()
    print(f"VideoDiT Mini | 参数量: {sum(p.numel() for p in model.parameters()):,}")

    scheduler = DDPMScheduler(num_train_timesteps=1000)
    x0 = torch.randn(1, 4, 4, 8, 8)
    t = scheduler.sample_timesteps(1, x0.device)
    noised = scheduler.add_noise(x0, t)

    with torch.inference_mode():
        pred = model(noised.noisy, noised.t_norm, y=torch.tensor([1]))
    print(f"训练前向: {tuple(noised.noisy.shape)} → {tuple(pred.shape)}")

    sampler = DDIMSampler(scheduler, num_inference_steps=10)
    with torch.inference_mode():
        x_gen = sampler.sample(
            model, shape=(1, 4, 4, 8, 8), device=torch.device("cpu"),
            class_labels=torch.tensor([0]), guidance_scale=2.0,
            null_class_id=model.null_class_idx,
        )
    print(f"DDIM 视频采样输出: {tuple(x_gen.shape)}")
    print("✅ VideoDiT 前向 + 采样通过")


if __name__ == "__main__":
    main()
