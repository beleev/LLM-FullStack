#!/usr/bin/env python
"""
MM-DiT 结构自检: 时间嵌入的量纲、双流联合注意力、Euler 采样
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_models.models.generative.mmdit import MMDiT
from llm_models.training.diffusion import EulerFlowSampler, FlowMatchingScheduler


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
    t = torch.tensor([0.1, 0.9])
    noised = scheduler.add_noise(x0, t)

    # 1) 时间嵌入量纲: 裸 t∈[0,1] 的 sinusoidal 特征几乎不可分, scheduler ×1000 后才可分
    emb = model.t_embed._sin_embed
    cos_raw = F.cosine_similarity(*emb(t), dim=0).item()
    cos_scaled = F.cosine_similarity(*emb(noised.t_norm), dim=0).item()
    print(f"cos(emb(t=0.1), emb(t=0.9)): 裸 t = {cos_raw:.3f} | scheduler 输出的 t_norm = {cos_scaled:.3f}")
    assert cos_raw > 0.95, "不缩放的症状: 两个时间几乎同一个嵌入"
    assert cos_scaled < 0.9 and torch.allclose(noised.t_norm, t * 1000)

    # 2) x_t 与 velocity 的定义
    assert torch.allclose(noised.noisy, (1 - t.view(2, 1, 1, 1)) * x0 + t.view(2, 1, 1, 1) * noised.noise)
    assert torch.allclose(noised.target, noised.noise - x0)

    text_embeds, text_pooled = torch.randn(2, 16, 64), torch.randn(2, 64)
    with torch.inference_mode():
        pred_v = model(noised.noisy, noised.t_norm, text_embeds, text_pooled)
        assert pred_v.shape == x0.shape and pred_v.abs().max() == 0       # 零初始化起点

        # 3) 模拟训练后 (随机化零初始化层): 文本 token 经联合注意力影响图像输出
        for m in model.modules():
            if isinstance(m, nn.Linear) and m.weight.abs().sum() == 0:
                nn.init.normal_(m.weight, std=0.02)
        base = model(noised.noisy, noised.t_norm, text_embeds, text_pooled)
        d_txt = (model(noised.noisy, noised.t_norm, text_embeds.flip(0), text_pooled) - base).abs().max().item()
        d_t = (model(noised.noisy, noised.t_norm.flip(0), text_embeds, text_pooled) - base).abs().max().item()
        print(f"换文本 token 输出变化 {d_txt:.3e} | 换 t 输出变化 {d_t:.3e}")
        assert d_txt > 1e-5 and d_t > 1e-5

    # 4) Euler 采样: 积分用 t∈[0,1], 喂模型的 t 与训练同量纲 (1000 → 50)
    seen_t = []

    def denoiser(x, t_model, _labels):
        seen_t.append(t_model[0].item())
        return model(x, t_model, text_embeds[:1], text_pooled[:1])

    x_gen = EulerFlowSampler(num_inference_steps=20).sample(
        denoiser, shape=(1, 4, 8, 8), device=torch.device("cpu"))
    assert abs(seen_t[0] - 1000.0) < 1e-3 and abs(seen_t[-1] - 50.0) < 1e-3 and len(seen_t) == 20
    assert torch.isfinite(x_gen).all()
    print(f"Euler 20 步, 模型看到的 t: {seen_t[0]:.0f} → {seen_t[-1]:.0f}")


if __name__ == "__main__":
    main()
