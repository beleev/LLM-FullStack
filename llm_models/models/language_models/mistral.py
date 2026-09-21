"""
Mistral-7B (Jiang et al., 2023) — LLaMA + 滑动窗口注意力 (SWA)

是什么: 与 LLaMA 逐参数相同, 只把因果 mask 从 "下三角" 裁成 "带状": 位置 t 只看 (t-W, t]。
解决什么: 全注意力计算 O(T²)、KV cache O(T); SWA 降到 O(T·W) 与 O(W) (rolling buffer)。
关键数字: 感受野没被掐断 —— 信息跨层接力, L 层 × 窗口 W ≈ L·W (Mistral: 32×4096 ≈ 131K)。
后继: Gemma 2/3 (全局层 : SWA 层交替)、GPT-OSS (SWA + attention sink)。
读代码时盯住: `window_mask` 与 forward 里的 `kept` —— SWA 不需要新的注意力类, 只是换了一张 mask;
             有 KV cache 时 cache 被滚动裁到 W, mask 的列要跟着只取最后 kept+T 列。
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from llm_models.layers.core.attention import GroupedQueryAttention
from llm_models.layers.core.blocks import PreLNBlock
from llm_models.layers.core.feedforward import SwiGLUFeedForward
from llm_models.layers.core.normalization import RMSNorm
from llm_models.layers.core.position_encoding import RotaryPositionalEncoding
from llm_models.utils.generation import GenerationMixin, KVCache
from llm_models.utils.init import init_weights
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


class Mistral(GenerationMixin, nn.Module):
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

        init_weights(self)   # weight tying 之后; 初始 CE ≈ ln V

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
        idx: torch.Tensor,                              # [B, T]
        attention_mask: Optional[torch.Tensor] = None,  # [B, past+T], 1=有效 0=pad
        cache: Optional[KVCache] = None,
    ) -> torch.Tensor:                                  # [B, T, V]
        B, T = idx.shape
        past = cache.pos if cache is not None else 0
        if past + T > self.max_len:
            raise ValueError(f"序列长度 {past + T} 超过 max_len={self.max_len}")

        x = self.token_embedding(idx) * math.sqrt(self.d_model)          # [B, T, D]
        position_ids = torch.arange(past, past + T, device=idx.device)

        # 带状 mask 的行 past:past+T (新 query) × 列 :past+T (全部历史), 再 ∩ padding
        banded = self._window_mask(past + T)[:, past:]                   # [1, T, past+T]
        mask = combine_causal_and_padding_mask(banded, attention_mask)
        # rolling buffer: cache 里只留了最近 kept 个 key, mask 的列要对齐到这 kept+T 个
        kept = min(past, self.window_size)
        mask = mask[..., -(kept + T):]                                   # [*, T, kept+T]

        for i, layer in enumerate(self.layers):
            layer_cache = cache.layers[i] if cache is not None else None
            x = layer(x, mask=mask, rope=self.rope, position_ids=position_ids, cache=layer_cache)
            if layer_cache is not None:
                # 窗口外的 K/V 永远不会再被看到 → 丢掉。K 已带 RoPE (绝对位置), 裁剪不影响正确性
                layer_cache["k"] = layer_cache["k"][:, :, -self.window_size:]
                layer_cache["v"] = layer_cache["v"][:, :, -self.window_size:]
                assert layer_cache["k"].size(2) <= self.window_size
        if cache is not None:
            cache.pos += T

        return self.lm_head(self.ln_f(x))
