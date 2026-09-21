"""
LLaMA (Touvron et al., 2023) — 现代开源 LLM 的事实模板

是什么: decoder-only LM, 把 GPT-3 的四个零件全部换成现代版:
    MHA → GQA (KV cache ÷ 组数) | GELU-FFN → SwiGLU | LayerNorm → RMSNorm | Sin-PE → RoPE
    外加: 无 bias、无 dropout、lm_head 与 embedding 共享权重。
关键数字: d_ff ≈ 8/3·d_model (SwiGLU 有 3 个矩阵, 8/3 让参数量与 4·d 的两矩阵 FFN 持平);
          初始 CE 必须 ≈ ln V (init_weights 保证; 默认 N(0,1) embedding + weight tying 会给出 ~250)。
演进: GPT-3 → LLaMA → Mistral (换 mask) / Mixtral (FFN 换 MoE) → DeepSeek-V3 (GQA 换 MLA)。
读代码时盯住: forward 里的 `past` —— 无 cache 时为 0, 有 cache 时它同时平移 RoPE 位置和 mask 行。
本文件几乎是纯组装, 零件都在 layers/core/。
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
from llm_models.utils.masks import build_causal_mask, combine_causal_and_padding_mask


def _make_llama_block(
    d_model: int,
    n_heads: int,
    num_kv_heads: Optional[int],
    d_ff: int,
    dropout: float,
    qk_norm: bool = False,
) -> PreLNBlock:
    """一个 LLaMA Block = PreLNBlock(GQA + SwiGLU + RMSNorm)。"""
    return PreLNBlock(
        d_model=d_model,
        attn=GroupedQueryAttention(
            d_model=d_model, num_heads=n_heads, num_kv_heads=num_kv_heads, qk_norm=qk_norm,
        ),
        ffn=SwiGLUFeedForward(d_model, d_ff),
        norm_cls=RMSNorm,
        dropout=dropout,
    )


# 社区常称 LlamaDecoderLayer / LlamaBlock
LlamaBlock = PreLNBlock


class LLaMA(GenerationMixin, nn.Module):
    """
    idx -> Embed·sqrt(D) -> N × PreLNBlock(GQA+SwiGLU+RMSNorm, RoPE 注入 Q/K) -> RMSNorm -> lm_head (tied)

    Args:
        vocab_size / d_model / n_heads / num_layers: 常规
        num_kv_heads: K/V head 数; None = n_heads (MHA)。LLaMA-2 70B 用 8。
        max_len:      最大上下文 (因果 mask 与 RoPE 表的大小)
        d_ff:         None 时取 8/3·d_model 并向上对齐到 64
        dropout:      官方训练为 0
        qk_norm:      Q/K 过 RMSNorm (Qwen3 / OLMo-2 风格), 默认关
        rope_scaling / rope_factor / rope_original_max_len:
                      长上下文外推 (None | "ntk" | "yarn"), 见 position_encoding.py
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 4096,
        n_heads: int = 32,
        num_kv_heads: Optional[int] = None,
        num_layers: int = 32,
        max_len: int = 4096,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        qk_norm: bool = False,
        rope_scaling: Optional[str] = None,
        rope_factor: float = 1.0,
        rope_original_max_len: Optional[int] = None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # RoPE 作用在 per-head 维度; 无参数, 所有层共享一份 cos/sin 表
        self.rope = RotaryPositionalEncoding(
            d_model // n_heads, max_len,
            scaling=rope_scaling, factor=rope_factor, original_max_len=rope_original_max_len,
        )

        if d_ff is None:
            d_ff = ((int(8 / 3 * d_model) + 63) // 64) * 64   # 对齐 64: matmul 对硬件友好

        self.layers = nn.ModuleList(
            [
                _make_llama_block(d_model, n_heads, num_kv_heads, d_ff, dropout, qk_norm)
                for _ in range(num_layers)
            ]
        )

        self.ln_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight     # weight tying

        self.register_buffer(
            "causal_mask", build_causal_mask(max_len, torch.device("cpu")), persistent=False
        )

        # 必须在 weight tying 之后: N(0, 0.02²) 让初始 logits ≈ 0 → CE ≈ ln V
        init_weights(self)

    def _causal_mask(self, seq_len: int) -> torch.Tensor:
        if seq_len <= self.causal_mask.size(-1):
            return self.causal_mask[:, :seq_len, :seq_len]
        return build_causal_mask(seq_len, self.causal_mask.device)

    def forward(
        self,
        idx: torch.Tensor,                              # [B, T]
        attention_mask: Optional[torch.Tensor] = None,  # [B, past+T], 1=有效 0=pad
        cache: Optional[KVCache] = None,
    ) -> torch.Tensor:                                  # [B, T, V]
        B, T = idx.shape
        past = cache.pos if cache is not None else 0    # 已缓存的 token 数
        if past + T > self.max_len:
            raise ValueError(f"序列长度 {past + T} 超过 max_len={self.max_len}")

        # ·sqrt(D): 抵消 0.02 的小初始化, 让 embedding 与残差分支同量级
        x = self.token_embedding(idx) * math.sqrt(self.d_model)          # [B, T, D]

        position_ids = torch.arange(past, past + T, device=idx.device)   # 新 token 的绝对位置
        # 新 token 是 query (行 past:past+T), 能看到全部历史 (列 :past+T)
        causal = self._causal_mask(past + T)[:, past:]                   # [1, T, past+T]
        mask = combine_causal_and_padding_mask(causal, attention_mask)

        for i, layer in enumerate(self.layers):
            x = layer(
                x, mask=mask, rope=self.rope, position_ids=position_ids,
                cache=cache.layers[i] if cache is not None else None,
            )
        if cache is not None:
            cache.pos += T

        return self.lm_head(self.ln_f(x))
