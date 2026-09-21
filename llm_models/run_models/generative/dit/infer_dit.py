#!/usr/bin/env python
"""
DiT 结构自检: adaLN-Zero 起点、patchify/unpatchify 互逆、条件通路、DDIM + CFG 采样
"""

import torch
import torch.nn as nn

from llm_models.models.generative.dit import DiT
from llm_models.training.diffusion import DDIMSampler, DDPMScheduler


def main():
    torch.manual_seed(42)

    model = DiT(
        latent_channels=4, image_size=8, patch_size=2,
        d_model=128, n_heads=4, num_layers=4,
        num_classes=10, class_dropout=0.1,
    ).eval()
    print(f"DiT Mini | 参数量: {sum(p.numel() for p in model.parameters()):,}")

    scheduler = DDPMScheduler(num_train_timesteps=1000)
    x0 = torch.randn(2, 4, 8, 8)
    noised = scheduler.add_noise(x0, torch.tensor([100, 900]))
    y = torch.tensor([3, 7])

    with torch.inference_mode():
        # 1) adaLN-Zero + FinalLayer 零初始化: 未训练的 DiT 输出恒为 0
        pred = model(noised.noisy, noised.t_norm, y)
        assert pred.shape == x0.shape and pred.abs().max() == 0
        print("未训练 DiT 的输出全 0 (adaLN-Zero 起点) ✔")

        # 2) unpatchify 是 "切 patch" 的逆: [B,C,H,W] → [B,N,p²C] → [B,C,H,W]
        B, C, g, p = 2, 4, model.grid_size, model.patch_size
        tokens = x0.view(B, C, g, p, g, p).permute(0, 2, 4, 3, 5, 1).reshape(B, g * g, p * p * C)
        assert torch.equal(model.unpatchify(tokens), x0)

        # 3) 把零初始化的层随机化 (模拟训练后), 条件 t / y 才会真正影响输出
        for m in model.modules():
            if isinstance(m, nn.Linear) and m.weight.abs().sum() == 0:
                nn.init.normal_(m.weight, std=0.02)
        base = model(noised.noisy, noised.t_norm, y)
        d_t = (model(noised.noisy, noised.t_norm.flip(0), y) - base).abs().max().item()
        d_y = (model(noised.noisy, noised.t_norm, y.flip(0)) - base).abs().max().item()
        print(f"换 t 输出变化 {d_t:.3e} | 换 y 输出变化 {d_y:.3e}")
        assert d_t > 1e-4 and d_y > 1e-4

    # 4) DDIM + CFG: 记录采样器喂给模型的 t, 必须是 [0, 999] 的原始步数 (没有被再 ×1000)
    seen_t = []

    def spy(x, t, labels):
        seen_t.append(t[0].item())
        return model(x, t, labels)

    # clip_x0: ᾱ_999 ≈ 2.4e-9, 不截断的话第一步就把数值放大 ~2 万倍
    sampler = DDIMSampler(scheduler, num_inference_steps=20, clip_x0=3.0)
    raw = DDIMSampler(scheduler, num_inference_steps=20).sample(
        model, shape=(1, 4, 8, 8), device=torch.device("cpu"))
    kwargs = dict(shape=(1, 4, 8, 8), device=torch.device("cpu"),
                  class_labels=torch.tensor([3]), null_class_id=model.null_class_idx)
    torch.manual_seed(0)
    x_cfg = sampler.sample(spy, guidance_scale=4.0, **kwargs)
    torch.manual_seed(0)
    x_plain = sampler.sample(model, guidance_scale=1.0, **kwargs)
    assert max(seen_t) == 999 and min(seen_t) == 0 and len(seen_t) == 2 * 20   # CFG: 每步 2 次前向
    assert x_cfg.abs().max() <= 3.0 and not torch.allclose(x_cfg, x_plain)
    print(f"|x|max: 不截断 {raw.abs().max():.1f} | clip_x0=3 → {x_cfg.abs().max():.2f}")
    print(f"DDIM 20 步, 模型前向 {len(seen_t)} 次 (CFG ×2), t: {max(seen_t):.0f} → {min(seen_t):.0f}")


if __name__ == "__main__":
    main()
