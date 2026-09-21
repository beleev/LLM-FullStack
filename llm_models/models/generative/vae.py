"""
VAE — Latent Diffusion 的前置压缩器 (Kingma & Welling 2013; Rombach et al. 2022)

解决的问题: 在 512×512×3 像素上做扩散太贵; 先压到 64×64×4 的 latent (元素数 ÷48) 再扩散。
为什么不是普通 AE: AE 的 latent 分布任意, 扩散模型无从假设; VAE 用 KL 把它拉向 N(0, I)。

    μ, logσ² = Encoder(x);   z = μ + σ·ε, ε~N(0,I)      # 重参数化: 随机性挪到 ε, 梯度能穿过采样
    loss = MSE(Decoder(z), x) + kl_weight · KL,   KL = -0.5·Σ(1 + logσ² - μ² - σ²)

关键数字: SD 的 kl_weight ~1e-6 —— 几乎就是 AE, 只要 latent 别离 N(0,I) 太远。
读代码时盯住: ImageVAE.reparameterize, 以及 encoder 的两个 1×1 头 (mean_head / logvar_head)。
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    """一个"Conv → GroupNorm → SiLU"块, GN 在小 batch 上比 BN 稳定。"""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
        nn.GroupNorm(num_groups=min(32, out_ch), num_channels=out_ch),
        nn.SiLU(),
    )


class ImageVAEEncoder(nn.Module):
    """
    VAE 图像 encoder

    Args:
        in_channels:   输入图像通道 (3 for RGB)
        base_channels: 最内层通道数, 每次下采样翻倍 (128 → 256 → 512)
        latent_dim:    潜空间通道数 (SD 1.5 用 4; 教学默认 4)
        downsample_levels: 下采样次数 (3 → 空间 ÷8; 教学默认 2 → ÷4)
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        latent_dim: int = 4,
        downsample_levels: int = 2,
    ):
        super().__init__()

        # 首层卷积: 进入 base_channels
        layers = [_conv_block(in_channels, base_channels)]

        ch = base_channels
        for _ in range(downsample_levels):
            # 每级: 一次 stride-2 下采样 + 一次普通 conv
            layers.append(_conv_block(ch, ch * 2, stride=2))
            layers.append(_conv_block(ch * 2, ch * 2))
            ch *= 2

        self.trunk = nn.Sequential(*layers)
        # 瓶颈用 1×1 卷积产出 μ 和 log σ² 两组张量
        self.mean_head = nn.Conv2d(ch, latent_dim, kernel_size=1)
        self.logvar_head = nn.Conv2d(ch, latent_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)                                   # [B, ch, H/2^L, W/2^L]
        return self.mean_head(h), self.logvar_head(h)       # 2 × [B, latent_dim, H/2^L, W/2^L]


class ImageVAEDecoder(nn.Module):
    """
    VAE 图像 decoder (与 encoder 镜像)
    """

    def __init__(
        self,
        out_channels: int = 3,
        base_channels: int = 64,
        latent_dim: int = 4,
        upsample_levels: int = 2,
    ):
        super().__init__()

        # 最深通道 = base_channels * 2^upsample_levels
        ch = base_channels * (2 ** upsample_levels)
        layers = [_conv_block(latent_dim, ch)]

        for _ in range(upsample_levels):
            # 每级: upsample (bilinear) + conv, 比 ConvTranspose 不易出 checkerboard
            layers.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
            layers.append(_conv_block(ch, ch // 2))
            layers.append(_conv_block(ch // 2, ch // 2))
            ch //= 2

        # 输出层: 回到 pixel 通道 (forward 里再过 tanh → [-1, 1])
        layers.append(nn.Conv2d(ch, out_channels, kernel_size=3, padding=1))

        self.trunk = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.trunk(z))


class ImageVAE(nn.Module):
    """
    完整 VAE (encoder + decoder + reparameterization)

    forward 流程:
        μ, logσ² = encoder(x)
        ε ~ N(0, I)
        z = μ + σ · ε                   # reparameterization trick (让梯度可回传)
        x̂ = decoder(z)

    返回重建 + 分布参数, 供外部计算 loss:
        recon_loss = ||x - x̂||²
        kl_loss    = -0.5 · Σ (1 + logσ² - μ² - σ²)

    Args:
        image_channels / base_channels / latent_dim / levels 见 encoder/decoder
    """

    def __init__(
        self,
        image_channels: int = 3,
        base_channels: int = 64,
        latent_dim: int = 4,
        levels: int = 2,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.encoder = ImageVAEEncoder(
            in_channels=image_channels, base_channels=base_channels,
            latent_dim=latent_dim, downsample_levels=levels,
        )
        self.decoder = ImageVAEDecoder(
            out_channels=image_channels, base_channels=base_channels,
            latent_dim=latent_dim, upsample_levels=levels,
        )

    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        mean, logvar = self.encoder(x)
        # 数值安全: clamp logvar 避免 exp 爆炸 (SD 常见做法)
        logvar = logvar.clamp(-30.0, 20.0)
        return {"mean": mean, "logvar": logvar}

    @staticmethod
    def reparameterize(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)                       # logσ² → σ
        eps = torch.randn_like(std)
        return mean + std * eps                             # [B, latent_dim, h, w]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        enc = self.encode(x)
        z = self.reparameterize(enc["mean"], enc["logvar"])
        recon = self.decode(z)
        return {"recon": recon, "z": z, "mean": enc["mean"], "logvar": enc["logvar"]}
