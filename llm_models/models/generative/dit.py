"""
Diffusion Transformer (DiT) — 图像扩散生成的 SOTA 骨架

论文出处:
    "Scalable Diffusion Models with Transformers" (Peebles & Xie, ICCV 2023)

在本库中的位置:
    生成模型主线. 是 Stable Diffusion 3 / FLUX / Sora / Hunyuan Video / Wan 2.2
    等 SOTA 生图生视频模型的共同骨架:
        UNet 时代 (SD 1.5) → DiT 时代 (SD3, FLUX, Sora)

核心设计三件套:
    1) Patchify: 把 latent 张量 [B, C, H, W] 切成 patch token [B, N, D]
       (与 ViT 完全相同, 但作用在 VAE 潜空间)
    2) 条件注入 adaLN-Zero (见 layers/adaln.py):
       时间步 t + 可选类别/文本 c → 6 段 (γ, β, α), 每个 block 做 FiLM 调制
    3) FinalLayer: adaLN + Linear 到 patch_size² * C, 再 unpatchify 还原

训练:
    输入: 含噪 latent x_t + 时间步 t + 条件 c
    输出: 预测的噪声 ε 或 velocity v (由 scheduler 决定)
    loss: MSE(pred, target)

本库对扩散采样的完整链:
    layers/adaln.py       — 条件注入机制
    models/dit.py         — 这里: 网络骨架
    training/diffusion.py — scheduler + sampler + loss + CFG
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from llm_models.layers.diffusion.adaln import AdaLNZeroBlock, FinalLayer, TimestepEmbedding
from llm_models.layers.core.attention import MultiHeadAttention
from llm_models.layers.core.feedforward import GeLUFeedForward


class PatchifyConv(nn.Module):
    """
    图像 → patch token 的 Conv2d 实现 (与 PatchEmbed2D 同机制, 但不做 shape 校验,
    便于 DiT 在训练 / 推理时接受不同分辨率 latent)。

    [B, C, H, W] → Conv2d(kernel=stride=patch_size) → [B, D, H/p, W/p]
                → flatten(2).transpose → [B, N, D]
    """

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int):
        super().__init__()

        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                       # [B, D, gH, gW]
        return x.flatten(2).transpose(1, 2)    # [B, N, D]


class DiT(nn.Module):
    """
    DiT 图像扩散 Transformer

    架构:
        x_t  -> PatchifyConv -> tokens
        t    -> TimestepEmbedding ── +
        y    -> Embedding(num_classes) ──┘── c (条件向量)
        tokens + pos_embed -> N x AdaLNZeroBlock(attn + FFN, 受 c 调制)
                          -> FinalLayer(c) -> [B, N, p²·C]
                          -> unpatchify -> [B, C, H, W]  (预测噪声 / velocity)

    默认配置近似 DiT-B/2 (DiT-Base, patch_size=2), 教学可按需缩小。

    Args:
        latent_channels: VAE 潜空间通道 (SD 1.5 为 4)
        image_size:      输入 latent 的空间大小 (注意是 latent 的 H, 不是原图 H)
        patch_size:      patchify 的 patch 边长
        d_model:         DiT 骨架隐藏维度
        n_heads:         注意力头数
        num_layers:      DiT block 数量
        c_dim:           条件嵌入维度 (默认等于 d_model)
        num_classes:     类别条件数; 0 表示无类别条件 (仅 timestep)
        class_dropout:   Classifier-free guidance 训练用的 class drop 概率
                         (训练时以此概率把类别换成 "null class", 推理时通过两套输出融合)
    """

    def __init__(
        self,
        latent_channels: int = 4,
        image_size: int = 32,
        patch_size: int = 2,
        d_model: int = 384,
        n_heads: int = 6,
        num_layers: int = 12,
        num_classes: int = 0,
        class_dropout: float = 0.1,
        c_dim: Optional[int] = None,
    ):
        super().__init__()

        if image_size % patch_size != 0:
            raise ValueError(f"image_size ({image_size}) 必须能被 patch_size ({patch_size}) 整除")

        self.latent_channels = latent_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size ** 2

        if c_dim is None:
            c_dim = d_model
        self.c_dim = c_dim

        # 1) patchify
        self.patchify = PatchifyConv(latent_channels, d_model, patch_size)
        # 2) 空间位置 (2D ViT 同款, 可学习)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # 3) 条件嵌入: timestep + 可选 class
        self.t_embed = TimestepEmbedding(c_dim)
        self.num_classes = num_classes
        self.class_dropout = class_dropout
        if num_classes > 0:
            # +1 多一个 "null" class 给 CFG 用
            self.class_embed = nn.Embedding(num_classes + 1, c_dim)
            self.null_class_idx = num_classes
        else:
            self.class_embed = None
            self.null_class_idx = None

        # 4) N 个 adaLN-Zero block (attn + FFN 都受 c 调制)
        d_ff = 4 * d_model
        self.blocks = nn.ModuleList(
            [
                AdaLNZeroBlock(
                    d_model=d_model,
                    c_dim=c_dim,
                    attn=MultiHeadAttention(d_model, n_heads),
                    ffn=GeLUFeedForward(d_model, d_ff),
                )
                for _ in range(num_layers)
            ]
        )

        # 5) 最终层: [B, N, patch_size² · C]
        self.final = FinalLayer(
            d_model=d_model, c_dim=c_dim,
            patch_out_dim=patch_size * patch_size * latent_channels,
        )

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        [B, N, p²·C] → [B, C, H, W]

        N = (H/p) * (W/p); 按行优先 reshape, 与 PatchifyConv 的 flatten 顺序一致。
        """
        B, N, _ = x.shape
        C = self.latent_channels
        p = self.patch_size
        H_grid = self.grid_size

        # [B, N, p²·C] → [B, H/p, W/p, p, p, C] → [B, C, H, W]
        x = x.view(B, H_grid, H_grid, p, p, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()      # [B, C, H/p, p, W/p, p]
        return x.view(B, C, H_grid * p, H_grid * p)

    def _make_condition(
        self, t: torch.Tensor, y: Optional[torch.Tensor], training: bool
    ) -> torch.Tensor:
        """组合 timestep + class embedding → 单个 c 向量。"""
        c = self.t_embed(t)                                               # [B, c_dim]
        if self.class_embed is not None and y is not None:
            if training and self.class_dropout > 0:
                # Classifier-free guidance 训练: 随机把类别换成 "null"
                drop = torch.rand(y.shape[0], device=y.device) < self.class_dropout
                y = torch.where(drop, torch.full_like(y, self.null_class_idx), y)
            c = c + self.class_embed(y)
        return c

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] 含噪 latent (来自 VAE 潜空间)
            t: [B] 时间步 (可连续可离散, 由 scheduler 传入)
            y: [B] 类别 id (可选)
        Returns:
            [B, C, H, W] 预测的噪声 / velocity (由训练目标决定语义)
        """
        c = self._make_condition(t, y, training=self.training)

        tokens = self.patchify(x) + self.pos_embed                       # [B, N, D]
        for block in self.blocks:
            tokens = block(tokens, c=c)

        out = self.final(tokens, c)                                      # [B, N, p²·C]
        return self.unpatchify(out)
