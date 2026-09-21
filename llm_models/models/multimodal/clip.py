"""
CLIP — 图文对比学习双塔 (Radford et al., OpenAI, 2021)

是什么: 图像塔 (ViT) 和文本塔 (Transformer) 各自把输入压成一个向量, 训练目标是让配对的图文向量靠近、不配对的远离。
解决了什么: 传统视觉模型只能在固定的 N 个类别上做监督分类; CLIP 用 4 亿图文对的自然语言做监督,
           得到可零样本迁移的视觉表征 —— 后来 LLaVA 等 VLM 的"眼睛"就是它。
关键公式: logits = exp(t) · norm(img) @ norm(txt)ᵀ   [B, B]
         loss = ½ [CE(logits, diag) + CE(logitsᵀ, diag)];  初始 1/τ = exp(t) = 1/0.07 ≈ 14.3, 上限 100
         未训练时各向量几乎无区分度 ⇒ 初始 loss ≈ ln B。
读代码时盯住: [B, B] 相似度矩阵的对角线 —— batch 内其余 B−1 个样本就是免费的负样本。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from llm_models.layers.core.attention import MultiHeadAttention
from llm_models.layers.core.blocks import PreLNBlock
from llm_models.layers.core.feedforward import GeLUFeedForward
from llm_models.layers.multimodal import PatchEmbed2D
from llm_models.utils.init import init_weights
from llm_models.utils.masks import build_causal_mask


class CLIPTextEncoder(nn.Module):
    """
    文本塔: GPT 式因果 Transformer, 取 [EOS] 位置的 hidden 作句向量, 再线性投影到共享空间。

    为什么是 EOS 而不是 [CLS]: 因果 mask 下只有最后一个 token 看得到全句。
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        max_len: int,
        embed_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.d_model = d_model
        self.max_len = max_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)

        d_ff = 4 * d_model
        self.layers = nn.ModuleList(
            [
                PreLNBlock(
                    d_model=d_model,
                    attn=MultiHeadAttention(d_model, n_heads),
                    ffn=GeLUFeedForward(d_model, d_ff),
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.text_projection = nn.Linear(d_model, embed_dim, bias=False)

        self.register_buffer("causal_mask", build_causal_mask(max_len, torch.device("cpu")), persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """input_ids [B, T] -> [B, embed_dim] (未归一化)。eos_token_id=None 时取每行最后一个位置。"""
        B, T = input_ids.shape
        position_ids = torch.arange(T, device=input_ids.device)

        x = self.token_embedding(input_ids) + self.position_embedding(position_ids)   # [B, T, D]
        causal = self.causal_mask[:, :T, :T]                           # [1, T, T]

        for layer in self.layers:
            x = layer(x, mask=causal)
        x = self.ln_f(x)

        if eos_token_id is not None:
            # argmax 取第一个 EOS: 它之后都是 padding, 因果 mask 保证 padding 影响不到它。(行内没有 EOS 会落到位置 0)
            eos_pos = (input_ids == eos_token_id).long().argmax(dim=-1)  # [B]
        else:
            eos_pos = torch.full((B,), T - 1, device=input_ids.device, dtype=torch.long)

        pooled = x[torch.arange(B, device=x.device), eos_pos]          # [B, D]
        return self.text_projection(pooled)                            # [B, embed_dim]


class CLIPVisionEncoder(nn.Module):
    """视觉塔: ViT。patch 序列前拼一个可学习 [CLS], 双向 attention 后取 [CLS] 位置, 线性投影到共享空间。"""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        embed_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.patch_embed = PatchEmbed2D(
            input_size=image_size, patch_size=patch_size,
            in_channels=3, embed_dim=d_model,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches + 1, d_model) * 0.02)

        d_ff = 4 * d_model
        self.layers = nn.ModuleList(
            [
                PreLNBlock(
                    d_model=d_model,
                    attn=MultiHeadAttention(d_model, n_heads),
                    ffn=GeLUFeedForward(d_model, d_ff),
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.visual_projection = nn.Linear(d_model, embed_dim, bias=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images [B, 3, H, W] -> [B, embed_dim] (未归一化)"""
        B = images.size(0)
        x = self.patch_embed(images)                                   # [B, N, D]
        cls = self.cls_token.expand(B, -1, -1)                         # [B, 1, D]
        x = torch.cat([cls, x], dim=1)                                 # [B, N+1, D]
        x = x + self.position_embedding

        for layer in self.layers:
            x = layer(x)                                               # 双向, 无 mask
        return self.visual_projection(self.ln_f(x)[:, 0])              # 取 [CLS] -> [B, embed_dim]


class CLIPModel(nn.Module):
    """
    双塔 + 可学习温度。forward 只返回归一化特征和 logit_scale, 对比 loss 在 training/loss.py::ContrastiveLoss 里算
    (特征单独暴露, 方便直接拿去做检索 / 零样本分类)。

    ViT-B/32 原版: embed_dim=512, 文本 12 层 d=512 max_len=77, 视觉 12 层 d=768 patch=32。
    """

    def __init__(
        self,
        embed_dim: int = 512,
        vocab_size: int = 49408,
        text_d_model: int = 512,
        text_n_heads: int = 8,
        text_num_layers: int = 12,
        text_max_len: int = 77,
        image_size: int = 224,
        patch_size: int = 32,
        vision_d_model: int = 768,
        vision_n_heads: int = 12,
        vision_num_layers: int = 12,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.text_encoder = CLIPTextEncoder(
            vocab_size=vocab_size, d_model=text_d_model, n_heads=text_n_heads,
            num_layers=text_num_layers, max_len=text_max_len,
            embed_dim=embed_dim, dropout=dropout,
        )
        self.vision_encoder = CLIPVisionEncoder(
            image_size=image_size, patch_size=patch_size,
            d_model=vision_d_model, n_heads=vision_n_heads, num_layers=vision_num_layers,
            embed_dim=embed_dim, dropout=dropout,
        )

        # 学的是 log(1/τ): 保证温度恒正, 且梯度尺度与 τ 无关
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / 0.07)))

        init_weights(self)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        x = self.vision_encoder(images)
        return F.normalize(x, dim=-1)

    def encode_text(
        self, input_ids: torch.Tensor, eos_token_id: Optional[int] = None
    ) -> torch.Tensor:
        x = self.text_encoder(input_ids, eos_token_id=eos_token_id)
        return F.normalize(x, dim=-1)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        eos_token_id: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """images [B, 3, H, W], input_ids [B, T] -> {image_features [B, E], text_features [B, E] (均 L2 归一化), logit_scale 标量}"""
        return {
            "image_features": self.encode_image(images),
            "text_features": self.encode_text(input_ids, eos_token_id=eos_token_id),
            # 上限 100 (τ ≥ 0.01): 否则模型会靠无限放大 logits 来压低 loss
            "logit_scale": self.logit_scale.exp().clamp(max=100.0),
        }
