"""
多模态通用构建块

设计背景:
    现代多模态大模型 (LLaVA / Flamingo / Qwen-VL / Qwen2.5-Omni 等) 通常共享
    "模态编码器 -> (重采样) -> 投影 -> LLM" 的三段式范式。本文件把这些跨模型
    可复用的底层组件抽离，避免在各 VLM / Omni 文件中重复实现。

组件一览:
    - PatchEmbed2D / PatchEmbed3D: ViT (Dosovitskiy, 2020) 与 ViViT (2021) 风格的
      图像 / 视频切分，把连续像素映射为离散 patch token 序列
    - PatchTransformerEncoder: 标准 ViT 骨架，用于视觉 / 声谱图 / 视频编码器
    - PerceiverResamplerBlock / PerceiverResampler:
      源自 Perceiver IO (2021) 与 Flamingo (DeepMind, 2022)，用少量可学习 latent
      通过 cross-attention 把任意长度源 tokens 压缩为定长，避免视觉 token 数
      随分辨率爆炸式膨胀，显著减轻 LLM 侧上下文压力
    - ModalityProjector: 把模态特征映射到 LLM 嵌入空间
      (LLaVA 的关键实现细节 — 共享 LLM 词嵌入维度以迁移文本预训练知识)
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from llm_models.layers.core.attention import MultiHeadAttention
from llm_models.layers.core.blocks import PreLNBlock
from llm_models.layers.core.feedforward import FeedForward


Size2D = Union[Tuple[int, int], int]


def _as_pair(x: Size2D) -> Tuple[int, int]:
    return (x, x) if isinstance(x, int) else x


class PatchEmbed2D(nn.Module):
    """
    2D Patch Embedding — ViT 风格的图像/声谱图切块

    把 [B, C, H, W] 按 patch 切块投影为 [B, N, D]，N = (H/ph) * (W/pw)。

    工程要点:
        - 用一次 stride=patch_size 的 Conv2d 等价实现"切块 + 线性投影"两步，
          比手工 unfold + matmul 更高效，且权重布局对硬件更友好
        - 共享给视觉 (RGB 3 通道) 与音频 mel 声谱图 (1 通道) 复用 —
          声谱图的 (frequency, time) 二维结构本质上也是 2D 张量
    """

    def __init__(
        self,
        input_size: Size2D,
        patch_size: Size2D,
        in_channels: int,
        embed_dim: int,
    ):
        super().__init__()

        self.input_size = _as_pair(input_size)
        self.patch_size = _as_pair(patch_size)

        if self.input_size[0] % self.patch_size[0] != 0 or \
                self.input_size[1] % self.patch_size[1] != 0:
            raise ValueError(
                f"input_size {self.input_size} 必须能被 patch_size {self.patch_size} 整除"
            )

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        # kernel_size == stride == patch_size: 各 patch 之间无重叠，等价于"先切块再线性投影"
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=self.patch_size, stride=self.patch_size, bias=False,
        )

        grid_h = self.input_size[0] // self.patch_size[0]
        grid_w = self.input_size[1] // self.patch_size[1]
        self.grid_size = (grid_h, grid_w)
        self.num_patches = grid_h * grid_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"输入必须是 4D Tensor [B, C, H, W]，当前 {tuple(x.shape)}")
        _, _, h, w = x.shape
        if (h, w) != self.input_size:
            raise ValueError(
                f"输入尺寸 {(h, w)} 与初始化 input_size {self.input_size} 不一致"
            )
        x = self.proj(x)                    # [B, D, gH, gW] — Conv2d 输出空间网格
        # flatten(2) 把 (gH, gW) 拍成一维序列，transpose 到 [B, N, D]
        # 之后由 Transformer 序列建模来重新捕捉空间关系 (位置编码补回结构信息)
        return x.flatten(2).transpose(1, 2)  # [B, N, D]


class PatchEmbed3D(nn.Module):
    """
    3D Patch Embedding — ViViT / VideoMAE 风格的 "tubelet" 切块

    把 [B, C, T, H, W] 用 (tubelet, ph, pw) 三维卷积切成 [B, N, D]，
    N = (T/tubelet) * (H/ph) * (W/pw)。

    为什么用 tubelet 而不是逐帧 2D patch?
        - 逐帧 patch 数 = T * (H/ph) * (W/pw)，长视频会爆炸式增长
        - tubelet 在时间维上做下采样 (典型 tubelet=2)，token 数减半
        - 同时让每个 token 自带短时运动信息，省去"先空间编码再做时间聚合"的两阶段
    """

    def __init__(
        self,
        video_size: Tuple[int, int, int],
        tubelet_size: int,
        patch_size: Size2D,
        in_channels: int,
        embed_dim: int,
    ):
        super().__init__()

        patch_size = _as_pair(patch_size)
        t, h, w = video_size
        if t % tubelet_size != 0:
            raise ValueError(f"video T ({t}) 必须能被 tubelet_size ({tubelet_size}) 整除")
        if h % patch_size[0] != 0 or w % patch_size[1] != 0:
            raise ValueError(f"video H/W {(h, w)} 必须能被 patch_size {patch_size} 整除")

        self.video_size = video_size
        self.tubelet_size = tubelet_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel = (tubelet_size, patch_size[0], patch_size[1])
        self.proj = nn.Conv3d(
            in_channels, embed_dim, kernel_size=kernel, stride=kernel, bias=False,
        )

        grid_t = t // tubelet_size
        grid_h = h // patch_size[0]
        grid_w = w // patch_size[1]
        self.grid_size = (grid_t, grid_h, grid_w)
        self.num_patches = grid_t * grid_h * grid_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"输入必须是 5D Tensor [B, C, T, H, W]，当前 {tuple(x.shape)}")
        _, _, t, h, w = x.shape
        if (t, h, w) != self.video_size:
            raise ValueError(
                f"输入尺寸 {(t, h, w)} 与初始化 video_size {self.video_size} 不一致"
            )
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class PatchTransformerEncoder(nn.Module):
    """
    通用 ViT 风格编码器: PatchEmbed -> 可学习 PosEmbed -> N x PreLNBlock -> LayerNorm

    设计理念:
        视觉 / 音频声谱图 / 视频虽然来自不同模态，但经过 patch 化后都是
        token 序列，因此可以共享同一个 Transformer 骨架，差异只体现在
        前置的 PatchEmbed 与超参数 (维度、层数)。

    为什么用可学习位置嵌入而不是 RoPE?
        - 视觉/音频编码器是非自回归 (双向注意力)，序列长度固定，
          可学习绝对位置嵌入更直观且与 ViT 原始论文对齐
        - LLM 解码端才需要 RoPE 处理变长外推与因果注意力
    """

    def __init__(
        self,
        patch_embed: nn.Module,
        d_model: int,
        n_heads: int,
        num_layers: int,
        dropout: float = 0.1,
        d_ff: Optional[int] = None,
    ):
        super().__init__()

        self.patch_embed = patch_embed
        # 可学习位置嵌入 [1, N, D]，broadcast 到 batch
        # 截断正态初始化 (std=0.02) 是 ViT/BERT 系列的标准做法，
        # 避免初始权重过大导致前几层激活饱和
        self.pos_embed = nn.Parameter(torch.zeros(1, patch_embed.num_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        if d_ff is None:
            d_ff = 4 * d_model

        # 视觉分支继续沿用 ReLU FFN + LayerNorm (与原始 ViT 对齐)
        self.layers = nn.ModuleList(
            [
                PreLNBlock(
                    d_model=d_model,
                    attn=MultiHeadAttention(d_model, n_heads),
                    ffn=FeedForward(d_model, d_ff),
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.dropout(x + self.pos_embed)
        for block in self.layers:
            x = block(x)
        return self.norm(x)


class PerceiverResamplerBlock(nn.Module):
    """
    Perceiver Resampler Block — 单层 cross-attention + FFN

    数据流:
        latents -> LN -> CrossAttn(Q=latents, K/V=source) -> Add
                -> LN -> FFN -> Add

    源自 Flamingo (NeurIPS 2022)。核心思想是 Q 来自固定数量的 latent，
    K/V 来自变长源 token，因此输出长度恒等于 latent 数量，与源长度解耦。
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, latents: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        residual = latents
        h = self.norm1(latents)
        h = self.cross_attn(q=h, k=source, v=source)
        latents = residual + self.dropout(h)

        residual = latents
        h = self.norm2(latents)
        h = self.ffn(h)
        return residual + self.dropout(h)


class PerceiverResampler(nn.Module):
    """
    Perceiver Resampler — 把变长视觉/音频 tokens 压缩到固定长度

    用 num_latents 个可学习 latent tokens 去 cross-attend 源 tokens，
    输出形状 [B, num_latents, d_model]，与源 tokens 数量完全解耦。

    为什么需要它?
        - ViT-Large 在 336x336 输入下产生 576 个 patch，多张图就上千 tokens
        - 直接拼到 LLM 上下文会快速吃光预算 (尤其是长对话场景)
        - Resampler 通常压到 32~256 个 token，几乎不损失下游性能
    """

    def __init__(
        self,
        num_latents: int,
        d_model: int,
        n_heads: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        if num_latents <= 0:
            raise ValueError(f"num_latents 必须为正数，当前 {num_latents}")

        self.num_latents = num_latents
        self.d_model = d_model

        self.latents = nn.Parameter(torch.randn(num_latents, d_model))

        d_ff = 4 * d_model
        self.layers = nn.ModuleList(
            [
                PerceiverResamplerBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        B = source.size(0)
        # 同一组 latent 在 batch 内共享 (expand 不复制内存，节省显存)
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)
        for block in self.layers:
            latents = block(latents, source)
        return latents


class ModalityProjector(nn.Module):
    """
    模态投影器 — 把视觉/音频/视频特征对齐到 LLM 嵌入空间

    - hidden_dim is None: 单层线性 (LLaVA-1 风格，轻量)
    - 否则: Linear -> GELU -> Linear (LLaVA-1.5 / MLP 风格，表达力更强)
    最后接 LayerNorm 稳定输出分布，便于与 LLM 预训练 token embedding 混合。

    为什么要"投影"而不是直接拼接?
        - 视觉编码器 (ViT) 维度常与 LLM 隐维度不同 (e.g. 1024 vs 4096)
        - 视觉特征分布与文本 embedding 差异大，直接塞入会破坏 LLM 内部分布
        - 投影层只需少量训练数据即可对齐两模态的表示空间 — 这是 LLaVA
          "视觉指令微调"能以极低代价取得好效果的关键
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()

        if hidden_dim is None:
            self.projector: nn.Module = nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.projector = nn.Sequential(
                nn.Linear(in_dim, hidden_dim, bias=False),
                nn.GELU(),
                nn.Linear(hidden_dim, out_dim, bias=False),
            )

        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.projector(x))
