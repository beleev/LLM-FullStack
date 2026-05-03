"""
Causal 3D VAE — 视频生成的时空压缩器

背景:
    Sora / HunyuanVideo / CogVideoX / Wan 等 SOTA 视频 DiT 的共同前置:
    一个把 [B, 3, T, H, W] 视频压到 [B, C_latent, T', H', W'] 的 3D VAE。

为什么视频需要 **Causal** 3D VAE?
    1) 时间维下采样 (典型 4×): 让 DiT 的序列长度下降, 算力下降 O(T²) 倍
    2) 因果时间卷积: 每个时间步只看过去帧, 保证推理时"边生成边 decode",
       无需等待整段视频, 支持流式
    3) 空间 2D + 时间 1D 解耦卷积: 比 full 3D conv 便宜, 但捕捉时空结构足够

本文件提供最小教学版:
    - 空间 2× 下采样 (方向数可配) + 时间 2× 下采样 (因果)
    - Encoder/Decoder 对称
    - 采用 "Conv3d + GroupNorm + SiLU" 基本块; 教学优先保留可读性
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _causal_pad(x: torch.Tensor, pad_t: int) -> torch.Tensor:
    """
    时间维因果 padding: 只在 "过去" 侧补零, 不看未来帧。
    x: [B, C, T, H, W];  pad_t 个 0 补在 T 维左侧。
    """
    return F.pad(x, (0, 0, 0, 0, pad_t, 0))  # pad order: (W_left, W_right, H_..., T_left, T_right)


class CausalConv3dBlock(nn.Module):
    """
    因果 3D 卷积块:
        causal_pad(time) -> Conv3d (无时间 padding) -> GroupNorm -> SiLU

    kernel_time=3 → 卷积窗口 [t-2, t-1, t], 天然因果。
    kernel_space=3, padding=1 让空间维保持长度 (下采样独立用 stride)。
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        spatial_stride: int = 1,
        time_stride: int = 1,
        kernel_time: int = 3,
    ):
        super().__init__()

        self.kernel_time = kernel_time
        self.conv = nn.Conv3d(
            in_ch, out_ch,
            kernel_size=(kernel_time, 3, 3),
            stride=(time_stride, spatial_stride, spatial_stride),
            padding=(0, 1, 1),          # 时间维 padding 外部手动做, 保证因果
        )
        self.norm = nn.GroupNorm(num_groups=min(32, out_ch), num_channels=out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _causal_pad(x, self.kernel_time - 1)
        return F.silu(self.norm(self.conv(x)))


class CausalVAE3DEncoder(nn.Module):
    """
    Encoder: [B, 3, T, H, W] → (μ, logσ²), 每个 shape [B, latent_dim, T', H', W']

    各级做 spatial 2× 与 time 2× 的独立下采样, 交叉出现避免一次性压太多导致训练不稳。
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        latent_dim: int = 4,
        spatial_levels: int = 2,      # 空间下采样次数 (H/W ÷ 2^n)
        time_levels: int = 2,         # 时间下采样次数 (T ÷ 2^n)
    ):
        super().__init__()

        layers = [CausalConv3dBlock(in_channels, base_channels)]

        ch = base_channels
        # 交替做空间 / 时间下采样, 直到两个预算都花光
        s = spatial_levels
        t = time_levels
        while s > 0 or t > 0:
            if s > 0:
                layers.append(CausalConv3dBlock(ch, ch * 2, spatial_stride=2))
                ch *= 2
                s -= 1
            if t > 0:
                layers.append(CausalConv3dBlock(ch, ch * 2, time_stride=2))
                ch *= 2
                t -= 1

        self.trunk = nn.Sequential(*layers)
        self.mean_head = nn.Conv3d(ch, latent_dim, kernel_size=1)
        self.logvar_head = nn.Conv3d(ch, latent_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.mean_head(h), self.logvar_head(h)


class CausalVAE3DDecoder(nn.Module):
    """
    Decoder: [B, latent_dim, T', H', W'] → [B, 3, T, H, W]
    用 Upsample + Causal Conv 还原, 避免 ConvTranspose 的 checkerboard。
    """

    def __init__(
        self,
        out_channels: int = 3,
        base_channels: int = 32,
        latent_dim: int = 4,
        spatial_levels: int = 2,
        time_levels: int = 2,
    ):
        super().__init__()

        ch = base_channels * (2 ** (spatial_levels + time_levels))
        layers = [CausalConv3dBlock(latent_dim, ch)]

        s, t = spatial_levels, time_levels
        while s > 0 or t > 0:
            if t > 0:
                layers.append(nn.Upsample(scale_factor=(2, 1, 1), mode="nearest"))
                layers.append(CausalConv3dBlock(ch, ch // 2))
                ch //= 2
                t -= 1
            if s > 0:
                layers.append(nn.Upsample(scale_factor=(1, 2, 2), mode="nearest"))
                layers.append(CausalConv3dBlock(ch, ch // 2))
                ch //= 2
                s -= 1

        # 出口: 回到图像通道, tanh 限幅到 [-1, 1]
        layers.append(nn.Conv3d(ch, out_channels, kernel_size=3, padding=1))
        self.trunk = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.trunk(z))


class CausalVideoVAE(nn.Module):
    """
    Causal 3D VAE (教学版)

    Args:
        image_channels: 像素通道 (3)
        base_channels:  首层通道
        latent_dim:     潜空间通道数
        spatial_levels: 空间下采样次数
        time_levels:    时间下采样次数
    """

    def __init__(
        self,
        image_channels: int = 3,
        base_channels: int = 32,
        latent_dim: int = 4,
        spatial_levels: int = 2,
        time_levels: int = 2,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.encoder = CausalVAE3DEncoder(
            in_channels=image_channels, base_channels=base_channels,
            latent_dim=latent_dim,
            spatial_levels=spatial_levels, time_levels=time_levels,
        )
        self.decoder = CausalVAE3DDecoder(
            out_channels=image_channels, base_channels=base_channels,
            latent_dim=latent_dim,
            spatial_levels=spatial_levels, time_levels=time_levels,
        )

    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        mean, logvar = self.encoder(x)
        logvar = logvar.clamp(-30.0, 20.0)
        return {"mean": mean, "logvar": logvar}

    @staticmethod
    def reparameterize(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        enc = self.encode(x)
        z = self.reparameterize(enc["mean"], enc["logvar"])
        return {"recon": self.decode(z), "z": z, "mean": enc["mean"], "logvar": enc["logvar"]}
