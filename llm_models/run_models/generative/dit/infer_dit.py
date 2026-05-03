#!/usr/bin/env python
"""
DiT (Diffusion Transformer) 前向 + 采样示例

展示:
- adaLN-Zero 条件注入机制
- DDIM 采样器从纯噪声生成 latent (推理 20 步)
- 简化配置, CPU 可跑
"""

import torch
from llm_models.models.generative.dit import DiT
from llm_models.training import DDPMScheduler, DDIMSampler


def main():
    torch.manual_seed(42)

    # 小 DiT: 只处理 latent 8×8×4, num_layers=4
    model = DiT(
        latent_channels=4, image_size=8, patch_size=2,
        d_model=128, n_heads=4, num_layers=4,
        num_classes=10, class_dropout=0.1,
    ).eval()
    print(f"DiT Mini | 参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练前向一次: 模拟 scheduler 加噪后调用
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    x0 = torch.randn(2, 4, 8, 8)
    t = scheduler.sample_timesteps(2, x0.device)
    noised = scheduler.add_noise(x0, t)
    y = torch.randint(0, 10, (2,))

    with torch.inference_mode():
        pred = model(noised.noisy, noised.t_norm, y)
    print(f"训练前向: x_t {tuple(noised.noisy.shape)} → pred ε {tuple(pred.shape)}")

    # 采样: 从噪声生成
    sampler = DDIMSampler(scheduler, num_inference_steps=20)
    with torch.inference_mode():
        x_gen = sampler.sample(
            model, shape=(1, 4, 8, 8), device=torch.device("cpu"),
            class_labels=torch.tensor([3]), guidance_scale=4.0,
            null_class_id=model.null_class_idx,
        )
    print(f"DDIM 采样输出: {tuple(x_gen.shape)}")
    print("✅ DiT 前向 + 采样通过")


if __name__ == "__main__":
    main()
