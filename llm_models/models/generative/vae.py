"""
Variational AutoEncoder (VAE) — Latent Diffusion 的前置压缩器

论文出处:
    "Auto-Encoding Variational Bayes" (Kingma & Welling, 2013)
    "High-Resolution Image Synthesis with Latent Diffusion Models"
    (Rombach et al., CVPR 2022) 把 VAE 用作 SD 的潜空间压缩

为什么扩散模型需要 VAE?
    直接在像素空间 (512×512×3) 做扩散, 每步都要在 ~10⁵ 维张量上去噪, 算力爆炸。
    Latent Diffusion 先用 VAE 把图像压到 (64×64×4) 的潜空间, 面积 ÷64,
    扩散只在潜空间里做, 最后 decoder 还原为像素, 算力省 64×, 质量几乎无损。

与普通 AE 的差别 (教学重点):
    - AE 的 latent 可能散布在潜空间任意位置, 不好被扩散模型对齐 (分布未知)
    - VAE 用 reparameterization: 让 encoder 输出 (μ, logσ²) 并从该高斯采样,
      配合 KL(q || N(0, I)) 把潜分布拉近标准正态, 下游扩散就能默认 latent 服从
      "接近正态"的分布, 训练更稳

本文件提供 2D 图像 VAE, 结构参考 SD 1.5 的简化版:
    - Encoder: 3 次 stride-2 卷积, 把 H×W 压到 H/8 × W/8 (教学用 4×, 可调)
    - Decoder: 对应的反卷积 (ConvTranspose / Upsample+Conv) 还原
    - 瓶颈: mean_head / logvar_head 两个 1×1 卷积产出参数
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
        h = self.trunk(x)
        return self.mean_head(h), self.logvar_head(h)


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

        # 输出层: 回到 pixel 通道, 用 tanh 把 [−1, 1] 作为标准像素范围
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
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        enc = self.encode(x)
        z = self.reparameterize(enc["mean"], enc["logvar"])
        recon = self.decode(z)
        return {"recon": recon, "z": z, "mean": enc["mean"], "logvar": enc["logvar"]}
