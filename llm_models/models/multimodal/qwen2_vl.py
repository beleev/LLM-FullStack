"""
Qwen2-VL 架构演示模块

论文出处:
    "Qwen2-VL: Enhancing Vision-Language Model's Perception of the World
     at Any Resolution" (Wang et al., 2024, 阿里通义千问团队)

核心创新 (相比 Qwen-VL / LLaVA 等前辈):
    1) 动态分辨率 (Naive Dynamic Resolution): 不再强制 resize 到 224x224，
       而是按原始长宽比切 patch，保留视觉细节并支持任意分辨率推理
    2) M-RoPE (Multimodal RoPE): 把 RoPE 频率维度切成三段 (temporal/height/width)，
       让位置编码同时表达"时间步、高度坐标、宽度坐标"，统一处理图像/视频/文本
    3) Vision token 与 LLM 共享嵌入空间: 经过投影后直接和文本 token 拼接送入
       同一个 decoder，最大化复用文本预训练知识

架构思路 (本文件采用早融合 / prefix-token 范式):
    1) Vision Encoder (ViT): 把图像切成 patch tokens
    2) Perceiver Resampler (可选): 把大量视觉 tokens 压缩为固定长度
    3) ModalityProjector: 投影到 LLM 维度
    4) Qwen2VLDecoder: decoder-only LLM，可接受 M-RoPE 位置索引区分 text/vision 位置

    注意: Qwen2-VL 选择 prefix-token 而非 cross-attention (Flamingo 路线) 的原因 —
    简单、可复用 KV-cache、便于多图/交错图文输入。

文件职责仅为 "VLM 模型胶水"，视觉/投影组件复用 multimodal.py，
解码器构建块复用 layers/blocks.py。
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from llm_models.layers.core.attention import GroupedQueryAttention
from llm_models.layers.core.blocks import PreLNBlock
from llm_models.layers.core.feedforward import SwiGLUFeedForward
from llm_models.layers.core.normalization import RMSNorm
from llm_models.layers.core.position_encoding import (
    MultimodalRotaryEmbedding,
    RotaryPositionalEncoding,
    SinPositionalEncoding,
)
from llm_models.layers.multimodal import (
    ModalityProjector,
    PatchEmbed2D,
    PatchTransformerEncoder,
    PerceiverResampler,
)
from llm_models.utils.masks import build_causal_mask, combine_causal_and_padding_mask


class Qwen2VLDecoder(nn.Module):
    """
    Qwen2-VL 文本解码器 (decoder-only LLM)

    架构要点 (对齐现代 Qwen2 / LLaMA 系列):
        - GQA (num_kv_heads ≤ num_heads): 推理时 K/V 内存与带宽减半，质量近乎无损
        - SwiGLU FFN: 门控激活，比 ReLU/GeLU 在等参数下更稳更强
        - RMSNorm (Pre-Norm): 比 LayerNorm 少一次均值计算且更稳定
        - RoPE / M-RoPE: 相对位置编码，支持长度外推

    支持两种输入:
        1) input_ids            [B, T]            — 纯文本
        2) inputs_embeds        [B, T, D]         — 跨模态拼接后传入 (vision+text)

    位置编码语义:
        - position_ids 为 None                -> 普通 1D RoPE，按序列下标 0..T-1
        - position_ids 形状 [T] 或 [B, T]      -> 普通 1D RoPE，可指定偏移 (KV-cache 续写)
        - position_ids 形状 [3, B, T]          -> M-RoPE，三通道分别表示 (temporal, h, w)
          视觉 token 的三维坐标可设为该 patch 在视频 (t, h, w) 中的位置，
          文本 token 三个通道全设相同的 1D 步进，从而同 LLM 框架兼容。
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        max_len: int,
        num_kv_heads: Optional[int] = None,
        dropout: float = 0.1,
        use_rope: bool = True,
        use_mrope: bool = False,
        mrope_sections: Optional[tuple] = None,
        d_ff: Optional[int] = None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.use_rope = use_rope
        self.use_mrope = use_mrope

        self.token_embedding = nn.Embedding(vocab_size, d_model)

        d_head = d_model // n_heads
        if use_mrope:
            if not use_rope:
                raise ValueError("use_mrope=True 需要 use_rope=True")
            # M-RoPE: section_dims 决定 d_head 在 (t, h, w) 三方向的频率分配
            # Qwen2-VL 默认 16/24/24 维，文本场景三段使用相同 position 即退化为 1D RoPE
            self.pos_encoder = MultimodalRotaryEmbedding(
                d_head=d_head,
                section_dims=mrope_sections,
                max_len=max_len,
            )
        elif use_rope:
            self.pos_encoder = RotaryPositionalEncoding(d_head, max_len)
        else:
            # 兜底使用经典 Sinusoidal (不推荐于 LLM，仅为对比/调试)
            self.pos_encoder = SinPositionalEncoding(d_model, max_len)

        if d_ff is None:
            # SwiGLU 有 3 个权重矩阵 (gate/up/down)，参数 ~= 1.5x 普通 FFN
            # 因此把 4*d_model 的传统 hidden 按 2/3 缩放，使总参数与 ReLU FFN 持平
            d_ff = int(4 * d_model * 2 / 3)

        self.layers = nn.ModuleList(
            [
                PreLNBlock(
                    d_model=d_model,
                    attn=GroupedQueryAttention(
                        d_model=d_model,
                        num_heads=n_heads,
                        num_kv_heads=num_kv_heads,
                    ),
                    ffn=SwiGLUFeedForward(d_model, d_ff),
                    norm_cls=RMSNorm,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # 输入嵌入与输出投影权重共享 (weight tying)
        # 节省 vocab_size * d_model 个参数，并轻微提升收敛性 — GPT-2 起的标准做法
        self.lm_head.weight = self.token_embedding.weight

        # 预构建 max_len 范围的因果掩码并 register 到 buffer
        # 避免每次 forward 重建；persistent=False 让其不进入 state_dict
        causal = build_causal_mask(max_len, torch.device("cpu"))
        self.register_buffer("causal_mask", causal, persistent=False)

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        # 缩放 sqrt(d_model) 来源于 Vaswani 2017 — 让 embedding 与 PE/激活方差匹配
        return self.token_embedding(input_ids) * math.sqrt(self.d_model)

    def _causal_mask(self, seq_len: int) -> torch.Tensor:
        if seq_len <= self.causal_mask.size(-1):
            return self.causal_mask[:, :seq_len, :seq_len]
        return build_causal_mask(seq_len, self.causal_mask.device)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ):
        # 二选一: 纯文本走 input_ids 节省内存; 跨模态走 inputs_embeds (已在外层拼接好)
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("input_ids 与 inputs_embeds 必须且只能传入一个")

        x = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        _, seq_len, _ = x.shape
        if seq_len > self.max_len:
            raise ValueError(f"序列长度 {seq_len} 超过最大长度 {self.max_len}")

        # 位置编码: RoPE 在每层 attention 内部对 Q/K 做旋转 (传 handler 进去)
        # Sinusoidal 则是直接加到 embedding 上一次性完成
        rope_handler: Optional[nn.Module] = None
        if self.use_rope:
            rope_handler = self.pos_encoder
        else:
            x = self.pos_encoder(x)

        # M-RoPE 的 position_ids 形状校验
        if self.use_mrope:
            if position_ids is None:
                raise ValueError("use_mrope=True 时必须传入 [3, B, T] position_ids")
            if position_ids.dim() != 3 or position_ids.size(0) != 3:
                raise ValueError(
                    f"M-RoPE 要求 position_ids 形状 [3, B, T]，当前 {tuple(position_ids.shape)}"
                )

        # 注意力掩码: 因果 (下三角) ∧ padding mask (来自外部)
        # 注意视觉 prefix 也参与因果注意力 — 后续文本可看到所有视觉 token，但反之不行
        causal = self._causal_mask(seq_len)
        attn_mask = combine_causal_and_padding_mask(causal, attention_mask)

        for layer in self.layers:
            x = layer(x, mask=attn_mask, rope=rope_handler, position_ids=position_ids)

        hidden = self.ln_f(x)
        logits = self.lm_head(hidden)
        if return_hidden:
            return logits, hidden
        return logits


class Qwen2VLModel(nn.Module):
    """
    Qwen2-VL 顶层模型 — 把 Vision Encoder + Resampler + Projector + LLM 串起来

    采用早融合 / prefix-token 策略:
        [vision_tokens; text_tokens] -> Decoder -> logits
    相比 Flamingo 的 cross-attention 路线，prefix 范式更简单、推理时 KV-cache
    管理也更直观 (视觉 KV 一次算完缓存即可)。

    Args:
        vocab_size: 词表大小
        text_*: Decoder 配置
        vision_*: Vision 编码器配置
        vision_num_latents: Resampler 输出 token 数; <=0 表示关闭 Resampler
        projector_hidden_dim: Projector 中间维度 (None 走单层 Linear)
        use_modality_embedding: 是否用一张 Embedding(2, D) 标记 vision/text 两个段
            类似 BERT 的 segment embedding，给模型一个显式信号区分模态来源
    """

    # 模态标识 — 用作 modality_embedding 的查表索引
    MODALITY_VISION = 0
    MODALITY_TEXT = 1

    def __init__(
        self,
        vocab_size: int,
        text_d_model: int = 1024,
        text_n_heads: int = 16,
        text_num_kv_heads: Optional[int] = None,
        text_num_layers: int = 24,
        max_len: int = 2048,
        vision_image_size: int = 224,
        vision_patch_size: int = 14,
        vision_d_model: int = 1024,
        vision_n_heads: int = 16,
        vision_num_layers: int = 24,
        vision_num_latents: int = 64,
        vision_num_latent_layers: int = 2,
        projector_hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_rope: bool = True,
        use_mrope: bool = False,
        use_modality_embedding: bool = True,
    ):
        super().__init__()

        # --- Vision encoder: 标准 ViT 把图像编为 (H/p)*(W/p) 个 patch token ---
        self.vision_encoder = PatchTransformerEncoder(
            patch_embed=PatchEmbed2D(
                input_size=vision_image_size,
                patch_size=vision_patch_size,
                in_channels=3,
                embed_dim=vision_d_model,
            ),
            d_model=vision_d_model,
            n_heads=vision_n_heads,
            num_layers=vision_num_layers,
            dropout=dropout,
        )

        # --- Resampler (可选): 把 256+ 视觉 token 压成 64 个，节省 LLM 上下文预算 ---
        self.vision_resampler: Optional[PerceiverResampler]
        if vision_num_latents and vision_num_latents > 0:
            self.vision_resampler = PerceiverResampler(
                num_latents=vision_num_latents,
                d_model=vision_d_model,
                n_heads=vision_n_heads,
                num_layers=vision_num_latent_layers,
                dropout=dropout,
            )
        else:
            self.vision_resampler = None

        # --- Projector: 把 vision 维度 (1024) 对齐到 text 维度，桥接两模态空间 ---
        self.vision_projector = ModalityProjector(
            in_dim=vision_d_model,
            out_dim=text_d_model,
            hidden_dim=projector_hidden_dim,
        )

        # --- Text decoder ---
        self.text_decoder = Qwen2VLDecoder(
            vocab_size=vocab_size,
            d_model=text_d_model,
            n_heads=text_n_heads,
            num_kv_heads=text_num_kv_heads,
            num_layers=text_num_layers,
            max_len=max_len,
            dropout=dropout,
            use_rope=use_rope,
            use_mrope=use_mrope,
        )

        # --- Modality Embedding (可选) ---
        self.use_modality_embedding = use_modality_embedding
        if use_modality_embedding:
            self.modality_embedding = nn.Embedding(2, text_d_model)
        else:
            self.modality_embedding = None

    def _build_modality_embeddings(
        self,
        batch_size: int,
        num_visual_tokens: int,
        num_text_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        if self.modality_embedding is None:
            raise ValueError("modality_embedding 未启用")
        vision_ids = torch.full(
            (batch_size, num_visual_tokens), self.MODALITY_VISION,
            dtype=torch.long, device=device,
        )
        text_ids = torch.full(
            (batch_size, num_text_tokens), self.MODALITY_TEXT,
            dtype=torch.long, device=device,
        )
        return self.modality_embedding(torch.cat([vision_ids, text_ids], dim=1))

    def forward(
        self,
        input_ids: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: 文本 token IDs [B, T_text]
            images: 图像输入 [B, C, H, W]，可选 (None 时退化为纯文本 LLM)
            text_attention_mask: 文本 padding mask [B, T_text]
            position_ids: 可选位置索引 (普通 RoPE: [T] or [B, T]; M-RoPE: [3, B, T_total])

        Returns:
            logits: [B, T_total, vocab_size]，T_total = T_vision + T_text
        """
        batch_size = input_ids.size(0)

        text_embeds = self.text_decoder.embed_tokens(input_ids)

        # 视觉分支三阶段: encoder -> (resampler) -> projector
        # 输出 vision_embeds 与 text_embeds 在最后一维同 d_model，可直接拼接
        vision_embeds = None
        if images is not None:
            vision_tokens = self.vision_encoder(images)
            if self.vision_resampler is not None:
                vision_tokens = self.vision_resampler(vision_tokens)
            vision_embeds = self.vision_projector(vision_tokens)

        if vision_embeds is None:
            combined_embeds = text_embeds
            combined_attention_mask = text_attention_mask
            num_visual_tokens = 0
        else:
            # 视觉 token 放前缀，文本放后面 — 让因果注意力中文本可以看到完整图像信息
            combined_embeds = torch.cat([vision_embeds, text_embeds], dim=1)
            num_visual_tokens = vision_embeds.size(1)

            if text_attention_mask is None:
                combined_attention_mask = None
            else:
                # 视觉 prefix 全部有效 (无 padding)，扩展 mask 使其与拼接后序列对齐
                vision_mask = torch.ones(
                    batch_size, num_visual_tokens,
                    device=text_attention_mask.device,
                    dtype=text_attention_mask.dtype,
                )
                combined_attention_mask = torch.cat(
                    [vision_mask, text_attention_mask], dim=1
                )

        # Modality embedding 类似 BERT 的 segment embedding,
        # 给 LLM 一个显式标记区分"这段是视觉"还是"这段是文本"
        if self.use_modality_embedding:
            modality_embeds = self._build_modality_embeddings(
                batch_size,
                num_visual_tokens,
                input_ids.size(1),
                combined_embeds.device,
            )
            combined_embeds = combined_embeds + modality_embeds

        return self.text_decoder(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention_mask,
            position_ids=position_ids,
        )
