"""
CLIP 模型模块

论文出处:
    "Learning Transferable Visual Models From Natural Language Supervision"
    (Radford et al., OpenAI, 2021)

在本库中的位置:
    与 Qwen2-VL (早融合 prefix-token 生成式多模态) 并列的 **另一条多模态主线**:
    对比学习双塔 + 共享 latent 空间。

核心范式:
    - 两个独立 encoder: 图像塔 (ViT) + 文本塔 (Transformer), **不共享权重**
    - 各自提取特征 → L2 归一化 → 余弦相似度 → 对角线为正样本的对比 loss
    - 训练完成后, 两塔就在同一"语义空间"对齐, 可做零样本分类 / 检索 / VLM 基座

与 LLaVA / Qwen2-VL 的关系:
    CLIP 本身 **不是生成模型**, 但它的 vision encoder 是现代 VLM 的默认"眼睛"。
    LLaVA 系列就是在 CLIP 视觉编码器之上再接 projector + LLM 做指令微调。

教学重点:
    - 为什么要 logit_scale 温度参数? 对比 loss 对温度极敏感, 可学习温度让
      模型自动找到合适的 "softmax 锐度", 比手调的 τ=0.07 更稳健
    - 为什么 L2 normalize? 让相似度退化为纯角度, 与长度解耦;
      数值稳定, 也是对比学习的事实标准
    - 双向对比: image→text 与 text→image 两个方向都算 loss, 对称性更强
"""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_models.layers.core.attention import MultiHeadAttention
from llm_models.layers.core.blocks import PreLNBlock
from llm_models.layers.core.feedforward import GeLUFeedForward
from llm_models.layers.multimodal import PatchEmbed2D, PatchTransformerEncoder


class CLIPTextEncoder(nn.Module):
    """
    CLIP 文本塔: 标准 Transformer encoder + [EOS] 池化

    设计细节:
        - 与 GPT 几乎同构的 decoder-only 风格 (因果 mask),
          但 **用 [EOS] 位置的 hidden 而非 [CLS]** 做句子表征
          (OpenAI CLIP 官方做法: 文本以 EOS 结尾, EOS 之前的 causal attn 已经看到全句)
        - 最终再过一个 Linear projection 到共享 embed_dim
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
        # CLIP 文本塔用 causal mask (和 GPT 一致, 只是用 EOS pool 而非生成)
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

        # 因果 mask 预构建 (文本塔仍需因果, 保证 EOS 是"看完全句后"的表示)
        from llm_models.utils.masks import build_causal_mask
        causal = build_causal_mask(max_len, torch.device("cpu"))
        self.register_buffer("causal_mask", causal, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [B, T] 文本 token, 最后一个有效位置通常是 [EOS]
            eos_token_id: 若提供, 按该 id 找 pooler 位置; 否则默认取每行最后一个 token

        Returns:
            [B, embed_dim] 文本嵌入 (未 L2 归一化)
        """
        B, T = input_ids.shape
        position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)

        x = self.token_embedding(input_ids) + self.position_embedding(position_ids)
        causal = self.causal_mask[:, :T, :T]

        for layer in self.layers:
            x = layer(x, mask=causal)
        x = self.ln_f(x)

        # Pool: 取每行 EOS 位置的 hidden
        if eos_token_id is not None:
            # argmax 会取"首次出现"; 约定 EOS 在句末, 所以反转后再找
            eos_pos = (input_ids == eos_token_id).long().argmax(dim=-1)  # [B]
        else:
            # 默认: 最后一个 token (假设 padding 在 batch 外部处理)
            eos_pos = torch.full((B,), T - 1, device=input_ids.device, dtype=torch.long)

        # gather: x[b, eos_pos[b], :]
        pooled = x[torch.arange(B, device=x.device), eos_pos]
        return self.text_projection(pooled)


class CLIPVisionEncoder(nn.Module):
    """
    CLIP 视觉塔: ViT + [CLS] pooler + Linear projection

    - 在 patch 序列前拼一个可学习 [CLS] token, 最终用 [CLS] 位置的 hidden 做句子级表征
    - 等价实现: 直接复用 PatchTransformerEncoder, 然后在外部拼接 CLS
      (PatchTransformerEncoder 本身不自带 CLS, 这里手动加一个)
    """

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

        # 可学习 [CLS] token; 初始值标准正态即可, ViT 默认做法
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        # 位置嵌入: num_patches + 1 (+1 为 CLS)
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
        """
        Args:
            images: [B, 3, H, W]
        Returns:
            [B, embed_dim] 图像嵌入 (未 L2 归一化)
        """
        B = images.size(0)
        x = self.patch_embed(images)                                   # [B, N, D]
        cls = self.cls_token.expand(B, -1, -1)                         # [B, 1, D]
        x = torch.cat([cls, x], dim=1)                                 # [B, N+1, D]
        x = x + self.position_embedding

        for layer in self.layers:
            x = layer(x)                                               # 视觉塔双向 attn, 无 mask
        x = self.ln_f(x)

        # 取 [CLS] 位置
        pooled = x[:, 0]
        return self.visual_projection(pooled)


class CLIPModel(nn.Module):
    """
    CLIP 双塔模型 (教学版)

    forward 返回:
        image_features:  [B, embed_dim] L2-normalized
        text_features:   [B, embed_dim] L2-normalized
        logit_scale:     标量, 可学习温度倒数, 用于缩放 image↔text 相似度

    训练损失由外部 (training/loss.py::ContrastiveLoss) 计算:
        logits_per_image = logit_scale * image_features @ text_features.T
        loss = (CE(logits_per_image, diag) + CE(logits_per_text, diag)) / 2

    为什么把 features 与 logit_scale 分开暴露?
        让 loss 计算与模型前向解耦; 也便于把 features 单独拿去做下游任务
        (零样本分类 / 检索) 而不必经过 loss 模块。

    Args:
        embed_dim:        共享对比空间维度 (CLIP ViT-B/32 原版为 512)
        vocab_size:       文本词表大小 (CLIP 用 BPE, 49408)
        text_*:           文本塔超参
        image_size/patch_size: 图像塔输入
        vision_*:         视觉塔超参
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

        # 可学习温度的对数形式, 初始化 exp(init) ≈ 1/0.07 (CLIP 官方默认)
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / 0.07)))

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
        """
        Args:
            images:    [B, 3, H, W]
            input_ids: [B, T]
            eos_token_id: 文本塔 pooler 位置
        Returns:
            dict:
              image_features: [B, embed_dim] L2 归一化
              text_features:  [B, embed_dim] L2 归一化
              logit_scale:    标量 (scalar tensor)
        """
        return {
            "image_features": self.encode_image(images),
            "text_features": self.encode_text(input_ids, eos_token_id=eos_token_id),
            # 裁剪 logit_scale 避免温度爆炸 (官方做法: 最大 100 ≈ τ ≥ 0.01)
            "logit_scale": self.logit_scale.exp().clamp(max=100.0),
        }
