"""
Causal 3D VAE — 视频 DiT 的时空压缩器 (Sora / HunyuanVideo / CogVideoX / Wan 的共同前置)

解决的问题: 视频 token 数 = T·H·W, 直接喂 DiT 注意力 O(N²) 吃不消;
3D VAE 把 [B, 3, T, H, W] 压到 [B, C, T/2^a, H/2^b, W/2^b], token 数降 2^(a+2b) 倍。

"Causal" = 时间维只向过去 padding: 卷积核 k_t=3 时第 t 帧输出只依赖 [t-2, t-1, t]。
好处: 第一帧可单独当图像编码 (图像/视频联合训练), 长视频可分块流式编解码。
要整条链因果, 三处都不能偷看未来: 卷积 (左 padding)、归一化 (逐帧 GroupNorm)、
上采样 (nearest, 第 i 帧 → 第 2i, 2i+1 帧)。

读代码时盯住: _causal_pad 的 pad 顺序, 以及 CausalConv3dBlock 里 norm 前后的 reshape。
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _causal_pad(x: torch.Tensor, pad_t: int) -> torch.Tensor:
    """x: [B, C, T, H, W], 只在时间维左侧 (过去) 补 pad_t 帧 0。"""
    return F.pad(x, (0, 0, 0, 0, pad_t, 0))  # F.pad 从最后一维往前数: (W左, W右, H左, H右, T左, T右)


class CausalConv3dBlock(nn.Module):
    """
    causal_pad(time) → Conv3d (时间维不再 padding) → 逐帧 GroupNorm → SiLU

    kernel_time=3 → 窗口 [t-2, t-1, t]。GroupNorm 若直接作用在 5D 张量上, 均值/方差会跨
    全部 T 帧统计, 未来帧就经统计量泄漏到过去; 所以把 T 并进 batch 维逐帧归一化。
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
        x = self.conv(_causal_pad(x, self.kernel_time - 1))           # [B, C', T', H', W']
        B, C, T, H, W = x.shape
        x = self.norm(x.transpose(1, 2).reshape(B * T, C, H, W))      # 逐帧统计, 不跨时间
        x = x.view(B, T, C, H, W).transpose(1, 2)                     # [B, C', T', H', W']
        return F.silu(x)


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

        self.trunk = nn.Sequential(*layers)
        # 出口卷积同样要因果: 时间维 padding 交给 _causal_pad (对称 padding=1 会偷看 t+1)
        self.out_conv = nn.Conv3d(ch, out_channels, kernel_size=3, padding=(0, 1, 1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.out_conv(_causal_pad(self.trunk(z), 2)))  # tanh → [-1, 1]


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
