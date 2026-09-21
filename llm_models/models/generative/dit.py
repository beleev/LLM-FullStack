"""
DiT — 用 Transformer 取代 UNet 做扩散去噪骨架 (Peebles & Xie, 2023); SD3 / FLUX / Sora 的共同祖先

解决的问题: UNet 的卷积归纳偏置难以按 scaling law 放大; DiT 把 latent 当 token 序列, 直接吃 LLM 的扩展经验。

    x_t [B,C,H,W] ─patchify→ [B,N,D] (+pos) ─N × AdaLNZeroBlock(c)→ FinalLayer(c) ─unpatchify→ [B,C,H,W]
    c = TimestepEmbedding(t) + ClassEmbedding(y)          # 全局条件, 经 adaLN 调制每一层
    loss = MSE(pred, target), target 是 ε 还是 velocity 由 scheduler 决定 (training/diffusion.py)

关键数字: N = (H/p)², patch_size 减半 → token ×4 → 注意力算力 ×16 (DiT-XL/2 优于 /4 /8 的代价)。
CFG: 训练时以 class_dropout 概率把 y 换成 null 类, 推理时 pred = uncond + s·(cond - uncond)。
读代码时盯住: c 怎么来 (_make_condition), 以及 unpatchify 的 permute 顺序。
"""

from typing import Optional

import torch
import torch.nn as nn

from llm_models.layers.diffusion.adaln import AdaLNZeroBlock, FinalLayer, TimestepEmbedding
from llm_models.layers.core.attention import MultiHeadAttention
from llm_models.layers.core.feedforward import GeLUFeedForward


class PatchifyConv(nn.Module):
    """[B, C, H, W] → Conv2d(kernel=stride=p) → [B, D, H/p, W/p] → [B, N, D] (与 ViT 的 patch embed 同机制)。"""

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int):
        super().__init__()

        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                       # [B, D, gH, gW]
        return x.flatten(2).transpose(1, 2)    # [B, N, D]


class DiT(nn.Module):
    """
    DiT 图像扩散 Transformer (默认配置近似 DiT-B/2 的缩小版)。

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
        """[B, N, p²·C] → [B, C, H, W]; token 按行优先排列, 与 PatchifyConv 的 flatten 一致。"""
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
            t: [B] 时间步, [0, 1000) 量纲 (用 AddNoiseResult.t_norm, 不要直接传 t∈[0,1])
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
