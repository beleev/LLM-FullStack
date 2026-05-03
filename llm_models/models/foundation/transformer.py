"""
Transformer 模型模块

标准 Encoder-Decoder 架构 (Vaswani et al., 2017, "Attention Is All You Need")。

历史背景:
    Transformer 诞生于机器翻译任务 (源/目标语言不同),
    因此天然采用 Encoder-Decoder: Encoder 编码源句, Decoder 自回归生成译文,
    并通过 Cross-Attention 让译文每一步都"看到"整个源句。
    这一架构后来分化出两条主线:
        - Encoder-only (BERT 系): 双向理解类任务
        - Decoder-only (GPT 系):   生成类任务, 现已统一几乎所有 NLP

教学重点:
    - Encoder Block 与 GPT Block 本质上相同: Pre-LN(Self-Attn) + FFN
    - Decoder Block 多了一层 Cross-Attention: Pre-LN(Self-Attn) + Cross-Attn + FFN
    - Cross-Attention 不应使用 RoPE
      (RoPE 假设 Q/K 在同一位置空间; 而 Decoder 的 Q 与 Encoder 的 K 来自
       不同序列, 强行旋转会引入伪位置关系)
    - 原论文使用 Post-LN, 本实现采用更稳定的 Pre-LN (避免 warmup 强依赖)
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from llm_models.layers.core.attention import MultiHeadAttention
from llm_models.layers.core.blocks import PreLNBlock, PreLNCrossBlock
from llm_models.layers.core.feedforward import FeedForward
from llm_models.layers.core.position_encoding import (
    RotaryPositionalEncoding,
    SinPositionalEncoding,
)


def _make_encoder_layer(
    d_model: int, n_heads: int, d_ff: int, dropout: float
) -> PreLNBlock:
    """构造一个 Encoder 层: Pre-LN(Self-Attn) + Pre-LN(FFN)。

    使用原始 ReLU FFN 与 LayerNorm, 与论文一致 (GPT 系会换成 GELU/RMSNorm)。
    """
    return PreLNBlock(
        d_model=d_model,
        attn=MultiHeadAttention(d_model, n_heads),
        ffn=FeedForward(d_model, d_ff),
        norm_cls=nn.LayerNorm,
        dropout=dropout,
    )


def _make_decoder_layer(
    d_model: int, n_heads: int, d_ff: int, dropout: float
) -> PreLNCrossBlock:
    """构造一个 Decoder 层: Pre-LN(Self-Attn) + Pre-LN(Cross-Attn) + Pre-LN(FFN)。

    Cross-Attention 的 Q 来自目标端 (decoder hidden), K/V 来自源端 (encoder output),
    这是 seq2seq 任务中"对齐"机制的关键。
    """
    return PreLNCrossBlock(
        d_model=d_model,
        self_attn=MultiHeadAttention(d_model, n_heads),
        cross_attn=MultiHeadAttention(d_model, n_heads),
        ffn=FeedForward(d_model, d_ff),
        norm_cls=nn.LayerNorm,
        dropout=dropout,
    )


# 保留旧符号作为别名 (新代码用 PreLNBlock/PreLNCrossBlock, 旧测试/教程仍可 import)
EncoderLayer = PreLNBlock
DecoderLayer = PreLNCrossBlock


class Transformer(nn.Module):
    """
    Encoder-Decoder Transformer (原始 "Attention Is All You Need" 架构)

    架构流程:
        src_ids -> src_emb * sqrt(d_model) (+Sin-PE) -> EncoderLayer × N -> enc_final_norm
        tgt_ids -> tgt_emb * sqrt(d_model) (+Sin-PE)
                -> DecoderLayer × N (cross-attend enc_output)
                -> dec_final_norm -> fc_out

    设计说明:
        - 源/目标使用独立 Embedding (词表通常不同, 如 EN→DE)。
        - emb * sqrt(d_model): 论文做法, 让 embedding 与 PE 量级匹配。
        - Sin-PE: 绝对位置编码, 训练外推能力弱; 提供 use_rope 选项以支持
          相对位置编码 (RoPE), 但仅用在 Self-Attn (Cross-Attn 不可用)。
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 5000,
        dropout: float = 0.1,
        use_rope: bool = False,
    ):
        super().__init__()

        self.use_rope = use_rope
        self.d_model = d_model

        if use_rope:
            # RoPE 作用在每个注意力头上, 维度为 d_head 而非 d_model
            d_head = d_model // n_heads
            self.pos_encoder = RotaryPositionalEncoding(d_head, max_len)
        else:
            # Sin-PE 直接加到 token embedding 上, 维度为 d_model
            self.pos_encoder = SinPositionalEncoding(d_model, max_len)

        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        self.encoder_layers = nn.ModuleList(
            [_make_encoder_layer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [_make_decoder_layer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.enc_final_norm = nn.LayerNorm(d_model)
        self.dec_final_norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def _embed(
        self, ids: torch.Tensor, embedding: nn.Embedding
    ) -> torch.Tensor:
        """token id -> 向量 + (可选) Sin 位置编码。

        RoPE 模式下不在此处加位置, 而是在每层 attention 内部对 Q/K 旋转。

        Args:
            ids:        [B, T] long
            embedding:  源端或目标端的 nn.Embedding
        Returns:
            [B, T, d_model]
        """
        emb = embedding(ids) * math.sqrt(self.d_model)
        return emb if self.use_rope else self.pos_encoder(emb)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            src:      [B, S] 源端 token ids
            tgt:      [B, T] 目标端 token ids (训练时为 teacher-forcing 输入)
            src_mask: padding mask (Encoder 与 Cross-Attn 的 K 端共用)
            tgt_mask: 因果 + padding mask (Decoder Self-Attn 用)
        Returns:
            [B, T, tgt_vocab_size] logits
        """
        # rope=None 时 attention 内部走 Sin-PE 已加在 emb 上的路径
        rope_handler: Optional[nn.Module] = self.pos_encoder if self.use_rope else None

        src_emb = self._embed(src, self.src_embedding)
        tgt_emb = self._embed(tgt, self.tgt_embedding)

        # Encoder: 双向 self-attn, 对源句做编码
        enc_output = src_emb
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, mask=src_mask, rope=rope_handler)
        enc_output = self.enc_final_norm(enc_output)

        # Decoder: 因果 self-attn + cross-attn(K/V 来自 enc_output)
        # context_mask 用 src_mask 是为了在 cross-attn 中屏蔽源端 padding
        dec_output = tgt_emb
        for layer in self.decoder_layers:
            dec_output = layer(
                dec_output,
                context=enc_output,
                self_mask=tgt_mask,
                context_mask=src_mask,
                rope=rope_handler,
            )
        dec_output = self.dec_final_norm(dec_output)

        return self.fc_out(dec_output)
