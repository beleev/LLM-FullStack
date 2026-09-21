#!/usr/bin/env python
"""
Video DiT 结构自检: spacetime patch 数、3D unpatchify 互逆、时空位置嵌入、DDIM 采样
"""

import torch
import torch.nn as nn

from llm_models.models.generative.video_dit import VideoDiT
from llm_models.training.diffusion import DDIMSampler, DDPMScheduler


def main():
    torch.manual_seed(42)

    model = VideoDiT(
        latent_channels=4, video_latent_size=(4, 8, 8),
        patch_size_t=2, patch_size_hw=2,
        d_model=128, n_heads=4, num_layers=2, num_classes=4,
    ).eval()
    print(f"VideoDiT Mini | 参数量: {sum(p.numel() for p in model.parameters()):,}")

    scheduler = DDPMScheduler(num_train_timesteps=1000)
    x0 = torch.randn(1, 4, 4, 8, 8)                                       # [B, C, T, H, W]
    noised = scheduler.add_noise(x0, torch.tensor([500]))

    with torch.inference_mode():
        # 1) token 数 = (T/p_t)·(H/p)·(W/p)
        tokens, grid = model.patchify(noised.noisy)
        assert grid == (2, 4, 4) and tokens.shape == (1, 32, 128) and model.num_patches == 32

        # 2) 3D unpatchify 是 "切 tubelet" 的逆
        C, pt, p = 4, 2, 2
        T, H, W = grid
        tub = x0.view(1, C, T, pt, H, p, W, p).permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(1, T * H * W, -1)
        assert torch.equal(model.unpatchify(tub, grid), x0)

        # 3) 位置嵌入 = time_pos + space_pos (广播相加): 参数量 T'+H'W' 而不是 T'·H'W'
        n_pos = model.time_pos.numel() + model.space_pos.numel()
        assert n_pos == (2 + 16) * 128
        print(f"位置嵌入参数: {n_pos} (分解式) vs {32 * 128} (每个时空位置一个)")

        # 4) 零初始化起点 → 输出恒 0
        y = torch.tensor([1])
        assert model(noised.noisy, noised.t_norm, y=y).abs().max() == 0
        for m in model.modules():                                         # 模拟训练后
            if isinstance(m, nn.Linear) and m.weight.abs().sum() == 0:
                nn.init.normal_(m.weight, std=0.02)

        # 5) 交换两个 tubelet (帧 [0,1] ↔ [2,3]): 注意力本身对 token 置换等变,
        #    能区分先后全靠 time_pos —— 把它清零, 输出就只是跟着换了个位置
        swap = [2, 3, 0, 1]
        def swap_gap():
            out = model(noised.noisy, noised.t_norm, y=y)
            return (model(noised.noisy[:, :, swap], noised.t_norm, y=y)[:, :, swap] - out).abs().max().item()
        with_pos = swap_gap()
        model.time_pos.zero_()
        without_pos = swap_gap()
        print(f"交换 tubelet 后输出差: 有 time_pos {with_pos:.3e} | time_pos 清零 {without_pos:.3e}")
        assert with_pos > 1e-4 and without_pos < 1e-5

    sampler = DDIMSampler(scheduler, num_inference_steps=10, clip_x0=3.0)
    x_gen = sampler.sample(
        model, shape=(1, 4, 4, 8, 8), device=torch.device("cpu"),
        class_labels=torch.tensor([0]), guidance_scale=2.0,
        null_class_id=model.null_class_idx,
    )
    assert x_gen.shape == (1, 4, 4, 8, 8) and x_gen.abs().max() <= 3.0
    print(f"DDIM 10 步采样输出: {tuple(x_gen.shape)}, |x|max = {x_gen.abs().max():.3f} (clip_x0=3)")


if __name__ == "__main__":
    main()
