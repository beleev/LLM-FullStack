"""
Video DiT — Sora-lite 教学版

论文出处:
    "Video generation models as world simulators" (OpenAI Sora, 2024)
    "HunyuanVideo" / "CogVideoX" / "Wan 2.2" 等开源视频 DiT

核心设计 (与 Image DiT 的差异):
    1) 输入张量是 [B, C, T, H, W] 的 5D 视频 latent (来自 Causal 3D VAE)
    2) **Spacetime Patches**: 用 3D patchify (同 PatchEmbed3D), 把时空体积切块:
         tubelet = (p_t, p_h, p_w), token 数 = (T/p_t) * (H/p_h) * (W/p_w)
    3) 位置嵌入需要覆盖 3D 网格 — 本实现用 (temporal × spatial) 可学习嵌入的外积和
    4) 其余 (adaLN-Zero, 条件注入, FinalLayer) 完全复用 Image DiT 的机制

本教学版简化:
    - 用自注意力统一处理所有时空 token (O(N²)); Sora 实际会做 spatial/temporal 分离
    - 不含文本条件, 仅支持 timestep + 可选类别 (概念等价)
    - 训练目标与 Image DiT 一致 (ε-pred 或 velocity-pred)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from llm_models.layers.diffusion.adaln import AdaLNZeroBlock, FinalLayer, TimestepEmbedding
from llm_models.layers.core.attention import MultiHeadAttention
from llm_models.layers.core.feedforward import GeLUFeedForward


class Patchify3D(nn.Module):
    """
    [B, C, T, H, W] -> [B, N, D] via Conv3d(kernel=stride=(p_t, p_h, p_w))

    与 PatchEmbed3D 几乎相同, 但 DiT 里不做形状校验, 允许任意 T/H/W 输入。
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size_t: int,
        patch_size_hw: int,
    ):
        super().__init__()
        self.patch_size_t = patch_size_t
        self.patch_size_hw = patch_size_hw
        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=(patch_size_t, patch_size_hw, patch_size_hw),
            stride=(patch_size_t, patch_size_hw, patch_size_hw),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        x = self.proj(x)                               # [B, D, T', H', W']
        _, _, t, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)               # [B, N, D]
        return x, (t, h, w)


class VideoDiT(nn.Module):
    """
    Video DiT (Sora-lite)

    Args:
        latent_channels:  3D VAE 潜空间通道 (默认 4)
        video_latent_size: (T', H', W')  预期的 latent 时空尺寸
        patch_size_t / patch_size_hw: tubelet 尺寸
        d_model / n_heads / num_layers: DiT 骨架参数
        num_classes: 类别条件数, 0 表示仅 timestep 条件
        class_dropout: CFG 训练用的 class drop 概率
    """

    def __init__(
        self,
        latent_channels: int = 4,
        video_latent_size: Tuple[int, int, int] = (8, 32, 32),
        patch_size_t: int = 1,
        patch_size_hw: int = 2,
        d_model: int = 384,
        n_heads: int = 6,
        num_layers: int = 12,
        num_classes: int = 0,
        class_dropout: float = 0.1,
        c_dim: Optional[int] = None,
    ):
        super().__init__()

        T, H, W = video_latent_size
        if T % patch_size_t != 0 or H % patch_size_hw != 0 or W % patch_size_hw != 0:
            raise ValueError(
                f"video_latent_size {video_latent_size} 必须能被 "
                f"(patch_size_t={patch_size_t}, patch_size_hw={patch_size_hw}) 整除"
            )

        self.latent_channels = latent_channels
        self.video_latent_size = video_latent_size
        self.patch_size_t = patch_size_t
        self.patch_size_hw = patch_size_hw

        self.t_grid = T // patch_size_t
        self.hw_grid = H // patch_size_hw
        self.num_patches = self.t_grid * self.hw_grid * self.hw_grid

        if c_dim is None:
            c_dim = d_model

        # Patchify 3D
        self.patchify = Patchify3D(latent_channels, d_model, patch_size_t, patch_size_hw)

        # 3D 位置嵌入: 简化为"时间嵌入 + 空间嵌入"的可学习外积和, 参数省 T·H·W 的乘积
        self.time_pos = nn.Parameter(torch.zeros(1, self.t_grid, 1, d_model))
        self.space_pos = nn.Parameter(torch.zeros(1, 1, self.hw_grid * self.hw_grid, d_model))
        nn.init.trunc_normal_(self.time_pos, std=0.02)
        nn.init.trunc_normal_(self.space_pos, std=0.02)

        # 条件
        self.t_embed = TimestepEmbedding(c_dim)
        self.num_classes = num_classes
        self.class_dropout = class_dropout
        if num_classes > 0:
            self.class_embed = nn.Embedding(num_classes + 1, c_dim)
            self.null_class_idx = num_classes
        else:
            self.class_embed = None
            self.null_class_idx = None

        # DiT block 栈
        d_ff = 4 * d_model
        self.blocks = nn.ModuleList(
            [
                AdaLNZeroBlock(
                    d_model=d_model, c_dim=c_dim,
                    attn=MultiHeadAttention(d_model, n_heads),
                    ffn=GeLUFeedForward(d_model, d_ff),
                )
                for _ in range(num_layers)
            ]
        )

        # FinalLayer 输出每 tubelet 的像素数 = p_t * p_hw² * C
        patch_out_dim = patch_size_t * patch_size_hw * patch_size_hw * latent_channels
        self.final = FinalLayer(d_model=d_model, c_dim=c_dim, patch_out_dim=patch_out_dim)

    def unpatchify(
        self, x: torch.Tensor, grid: Tuple[int, int, int]
    ) -> torch.Tensor:
        """
        [B, N, p_t·p_hw²·C] → [B, C, T, H, W]
        grid: (T', H', W') — 从 patchify 得到的 latent 网格
        """
        B, N, _ = x.shape
        T_grid, H_grid, W_grid = grid
        C = self.latent_channels
        p_t = self.patch_size_t
        p_hw = self.patch_size_hw

        x = x.view(B, T_grid, H_grid, W_grid, p_t, p_hw, p_hw, C)
        # 按 [B, C, T, H, W] 的顺序重排
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        return x.view(B, C, T_grid * p_t, H_grid * p_hw, W_grid * p_hw)

    def _make_condition(
        self, t: torch.Tensor, y: Optional[torch.Tensor], training: bool
    ) -> torch.Tensor:
        c = self.t_embed(t)
        if self.class_embed is not None and y is not None:
            if training and self.class_dropout > 0:
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
            x: [B, C, T, H, W] 含噪视频 latent
            t: [B] timestep
            y: [B] 类别 (可选)
        Returns:
            [B, C, T, H, W] 预测的 ε 或 velocity
        """
        c = self._make_condition(t, y, training=self.training)

        tokens, grid = self.patchify(x)                                    # [B, N, D]

        # 把 time / space 位置加到对应 token 上:
        # token 排列是 (t_idx 外, 空间 idx 内), 与 flatten(2) 的 (T', H', W') 行主序一致
        T_grid, H_grid, W_grid = grid
        spatial_n = H_grid * W_grid
        pos = (self.time_pos + self.space_pos).expand(-1, T_grid, spatial_n, -1)  # [1, T, HW, D]
        pos = pos.reshape(1, T_grid * spatial_n, -1)
        tokens = tokens + pos

        for block in self.blocks:
            tokens = block(tokens, c=c)

        out = self.final(tokens, c)                                        # [B, N, patch_out]
        return self.unpatchify(out, grid)
