"""
多模态通用积木: "模态编码器 → (重采样) → 投影 → LLM" 三段式里的前三段

是什么: 把图像 / 声谱图 / 视频变成 LLM 能直接拼进上下文的 token 序列 [B, N, D_llm]。
解决了什么: LLM 只吃 token 序列; 像素是稠密网格, 且 token 数随分辨率平方增长。
    PatchEmbed2D/3D        切块 + 线性投影 (一次 stride=patch 的卷积): [B,C,H,W] → [B, N, D], N = (H/p)·(W/p)
    PatchTransformerEncoder ViT 骨架: patch + 可学习位置 → N × 双向 block
    PerceiverResampler      K 个可学习 latent 作 Q 去 cross-attend N 个 patch ⇒ 输出恒为 K 个 token (Flamingo)
    ModalityProjector       Linear / MLP 把编码器维度对齐到 LLM 维度 (LLaVA)
关键数字: 224² 图, patch 14 → 256 token; 336² → 576 token; Resampler 通常压到 32~256。
读代码时盯住: 序列长度 N 在每一步怎么变 (N_patch → num_latents), 这就是 LLM 要付的上下文成本。
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
    """[B, C, H, W] → [B, N, D], N = (H/ph)·(W/pw)。图像 (C=3) 与 mel 声谱图 (C=1, 频率×时间) 共用。"""

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
        # kernel = stride = patch: 无重叠卷积 ≡ "切块 → 展平 → Linear"
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
        x = self.proj(x)                     # [B, D, gH, gW]
        return x.flatten(2).transpose(1, 2)  # [B, N, D] 按行优先展平; 2-D 结构靠位置编码补回


class PatchEmbed3D(nn.Module):
    """
    [B, C, T, H, W] → [B, N, D], N = (T/tubelet)·(H/ph)·(W/pw)  (ViViT 的 tubelet 切块)

    相比逐帧 2-D patch: 时间维也下采样 (token 数 ÷ tubelet), 且每个 token 自带短时运动信息。
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
        x = self.proj(x)                     # [B, D, gT, gH, gW]
        return x.flatten(2).transpose(1, 2)  # [B, N, D]


class PatchTransformerEncoder(nn.Module):
    """
    ViT 骨架: PatchEmbed → + 可学习位置 → N × PreLNBlock (双向, 无 mask) → LayerNorm。

    图像 / 声谱图 / 视频 patch 化之后都是 token 序列, 共用这一个骨架, 差别只在传入的 patch_embed。
    编码器输入长度固定且非自回归, 用可学习绝对位置即可; 变长外推是 LLM 侧 RoPE 的事。
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
        self.pos_embed = nn.Parameter(torch.zeros(1, patch_embed.num_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)                # [1, N, D]

        if d_ff is None:
            d_ff = 4 * d_model

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
        x = self.patch_embed(x)                                        # [B, N, D]
        x = self.dropout(x + self.pos_embed)
        for block in self.layers:
            x = block(x)
        return self.norm(x)                                            # [B, N, D]


class PerceiverResamplerBlock(nn.Module):
    """latents += CrossAttn(Q=LN(latents), K=V=source);  latents += FFN(LN(latents))"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, latents: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        # latents [B, K, D], source [B, N, D]; 注意力矩阵 [B, K, N]: 每个 latent 从 N 个 patch 里"挑"信息
        h = self.cross_attn(q=self.norm1(latents), k=source, v=source)
        latents = latents + self.dropout(h)
        return latents + self.dropout(self.ffn(self.norm2(latents)))   # [B, K, D]


class PerceiverResampler(nn.Module):
    """
    [B, N, D] → [B, num_latents, D]: 输出长度只由可学习 latent 的个数决定, 与输入 token 数 N 无关 (Flamingo)。

    代价: latent 不再对应具体的空间位置, 细粒度定位 / OCR 类任务会受损 —— 所以 Qwen2-VL 改用相邻 patch 合并。
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
        latents = self.latents.unsqueeze(0).expand(source.size(0), -1, -1)   # [B, K, D] 同一组 latent 全 batch 共享
        for block in self.layers:
            latents = block(latents, source)
        return latents


class ModalityProjector(nn.Module):
    """
    [B, N, in_dim] → [B, N, out_dim]: 单层 Linear (LLaVA-1) 或 Linear-GELU-Linear (LLaVA-1.5), 末尾 LayerNorm。

    LLaVA 的关键发现: 冻结视觉编码器和 LLM, 只训这一个小投影层就能把两个空间对齐。
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
