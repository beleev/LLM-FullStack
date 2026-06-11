"""
Mistral 模型模块

论文出处:
    "Mistral 7B" (Jiang et al., 2023, arXiv:2310.06825)

历史意义:
    Mistral-7B 用 7B 参数打平/超过 Llama-2 13B, 架构上它对 LLaMA 只做了一处
    "减法" —— **Sliding Window Attention (SWA, 滑动窗口注意力)**:

        - 每个位置只看最近 W 个 token (Mistral-7B: W = 4096)
        - 注意力计算量从 O(T^2) 降到 O(T·W)
        - 推理 KV cache 用 rolling buffer (环形缓冲覆写), 显存从 O(T) 封顶到 O(W)
        - 感受野并没有被掐断: 信息可以跨层接力传播,
          L 层 × 窗口 W 的理论感受野 ≈ L·W (Mistral: 32 × 4096 ≈ 131K token)

    这条"局部注意力"路线被后来者广泛继承:
        Gemma 2/3 (2024-25)  全局层与 SWA 层交替堆叠 (5:1)
        GPT-OSS  (2025)      SWA + 可学习 attention sink, 层间交替
        Character/MiniMax 等 线性注意力 + 全注意力的混合也属于同一思想谱系

在本库的演进地图里的位置:
    LLaMA (2023, GQA + SwiGLU + RMSNorm + RoPE, 因果全注意力)
        ↓  把注意力矩阵从"下三角"裁成"带状"  ← 改 mask, 省计算/显存
    Mistral (2023, = LLaMA + 滑动窗口 mask)
        ↓  另一条压缩 KV 的路线: 改投影而不是改 mask
    DeepSeek-V3 (2024, MLA 低秩压缩 KV)

实现说明 (本文件的教学重点):
    SWA **不需要新的注意力类** —— 它只是换了一张 mask。
    注意力本体 (QKV 投影 + softmax + 加权和) 与 LLaMA 完全相同, 谁能看见谁
    完全由 mask 决定。因此本文件复用 GroupedQueryAttention, 仅把
    build_causal_mask 换成 build_sliding_window_mask, 参数量与 LLaMA 一字不差。
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_models.layers.core.attention import GroupedQueryAttention
from llm_models.layers.core.blocks import PreLNBlock
from llm_models.layers.core.feedforward import SwiGLUFeedForward
from llm_models.layers.core.normalization import RMSNorm
from llm_models.layers.core.position_encoding import RotaryPositionalEncoding
from llm_models.utils.masks import (
    build_sliding_window_mask,
    combine_causal_and_padding_mask,
)


def _make_mistral_block(
    d_model: int,
    n_heads: int,
    num_kv_heads: Optional[int],
    d_ff: int,
    dropout: float,
) -> PreLNBlock:
    """
    组装一个 Mistral Block: GQA + SwiGLU-FFN + RMSNorm (Pre-Norm)。
    与 LLaMA Block 完全一致 —— SWA 的差异不在 Block 里, 在喂给它的 mask 上。
    """
    return PreLNBlock(
        d_model=d_model,
        attn=GroupedQueryAttention(
            d_model=d_model, num_heads=n_heads, num_kv_heads=num_kv_heads,
        ),
        ffn=SwiGLUFeedForward(d_model, d_ff),
        norm_cls=RMSNorm,
        dropout=dropout,
    )


# 社区常称 MistralDecoderLayer; 结构上与 LlamaBlock 同为 PreLNBlock
MistralBlock = PreLNBlock


class Mistral(nn.Module):
    """
    Mistral decoder-only LLM (教学版)

    架构:
        idx -> TokenEmbed * sqrt(d_model)
            -> N x PreLNBlock(GQA + SwiGLU + RMSNorm) [RoPE 注入 Q/K]
               其中注意力使用 **带状因果 mask** (窗口 W)
            -> RMSNorm
            -> lm_head (与 token_embedding 共享权重)

    与 LLaMA 的一行差异对照:
        LLaMA:   mask = 下三角        (位置 t 看 [0, t])
        Mistral: mask = 带状下三角    (位置 t 看 (t-W, t])

    Args:
        vocab_size:      词表大小 (Mistral 原版 32000)
        d_model:         隐藏维度
        n_heads:         Q head 数
        num_kv_heads:    K/V head 数 (Mistral-7B: 32 Q / 8 KV, GQA 4x 压缩)
        num_layers:      Transformer 层数
        max_len:         最大上下文 (预构建 mask 所需)
        window_size:     滑动窗口大小 W (Mistral-7B: 4096); 教学默认 8 便于观察
        d_ff:            SwiGLU 隐藏维度, None 时同 LLaMA 取 ~(8/3)·d_model
        dropout:         Dropout (官方训练为 0)
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 4096,
        n_heads: int = 32,
        num_kv_heads: Optional[int] = None,
        num_layers: int = 32,
        max_len: int = 4096,
        window_size: int = 8,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.window_size = window_size
        self.num_layers = num_layers

        self.token_embedding = nn.Embedding(vocab_size, d_model)

        d_head = d_model // n_heads
        self.rope = RotaryPositionalEncoding(d_head, max_len)

        if d_ff is None:
            d_ff = int(8 / 3 * d_model)
            d_ff = ((d_ff + 63) // 64) * 64

        self.layers = nn.ModuleList(
            [
                _make_mistral_block(d_model, n_heads, num_kv_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )

        self.ln_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        # 预构建带状因果 mask — 与 LLaMA 唯一的结构性差异
        banded = build_sliding_window_mask(
            max_len, window_size, torch.device("cpu")
        )
        self.register_buffer("window_mask", banded, persistent=False)

    def _window_mask(self, seq_len: int) -> torch.Tensor:
        if seq_len <= self.window_mask.size(-1):
            return self.window_mask[:, :seq_len, :seq_len]
        return build_sliding_window_mask(
            seq_len, self.window_size, self.window_mask.device
        )

    def receptive_field(self) -> int:
        """
        理论感受野 ≈ num_layers × window_size。
        第 1 层的位置 t 聚合了 (t-W, t]; 第 2 层在此之上再向左延伸 W, 以此类推。
        这是 SWA "局部注意力不等于局部信息" 的关键: 深度替代了宽度。
        """
        return self.num_layers * self.window_size

    def kv_cache_entries(self, seq_len: int) -> int:
        """推理时 KV cache 实际需要保留的位置数: min(T, W) — rolling buffer 上限。"""
        return min(seq_len, self.window_size)

    def forward(
        self,
        idx: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            idx:            [B, T] token IDs
            attention_mask: [B, T] padding mask, 1=有效 0=pad
        Returns:
            logits: [B, T, vocab_size]
        """
        B, T = idx.shape
        if T > self.max_len:
            raise ValueError(f"序列长度 {T} 超过 max_len={self.max_len}")

        x = self.token_embedding(idx) * math.sqrt(self.d_model)

        # 带状因果 mask ∩ padding mask
        banded = self._window_mask(T)
        mask = combine_causal_and_padding_mask(banded, attention_mask)

        for layer in self.layers:
            x = layer(x, mask=mask, rope=self.rope)

        x = self.ln_f(x)
        return self.lm_head(x)

    @torch.inference_mode()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """朴素自回归生成 (教学实现, 同 LLaMA.generate)。"""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.max_len else idx[:, -self.max_len :]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx
