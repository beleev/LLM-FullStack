"""
Qwen3-Next (Alibaba, 2025) — 混合线性注意力架构 (教学版)

是什么: 75% 的层用 Gated DeltaNet (线性注意力, 状态 O(1)), 25% 用全注意力 (GQA, cache O(T))。
        层排布 (linear_ratio=3): [Δ, Δ, Δ, A, Δ, Δ, Δ, A, ...]
解决什么: 全层注意力的长上下文成本 O(T²)/O(T); 纯线性模型 (Mamba) 又召回弱。
          混合 = 线性层管 "流畅的局部建模", 少量全注意力层兜底 "精准的长程检索"。
          同路线: Jamba (Mamba+Attn), MiniMax-Text (Lightning Attention 7:1)。
关键数字: 推理缓存 = n_attn × T × 2·Hkv·Dh  +  n_delta × H·Dh² (后一项与 T 无关)。
简化: 真实模型还有超稀疏 MoE (见 DeepSeekMoE)、MTP (见 mtp.py)、zero-centered RMSNorm, 此处只留 "混合层"。
读代码时盯住: `layer_types` 和每层 cache dict 里存的东西 —— attn 层是 k/v (随 T 增长), delta 层是 state (恒定)。
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn

from llm_models.layers.core.attention import GroupedQueryAttention
from llm_models.layers.core.blocks import PreLNBlock
from llm_models.layers.core.feedforward import SwiGLUFeedForward
from llm_models.layers.core.normalization import RMSNorm
from llm_models.layers.core.position_encoding import RotaryPositionalEncoding
from llm_models.layers.sparse.linear_attention import GatedDeltaNet
from llm_models.utils.generation import GenerationMixin, KVCache
from llm_models.utils.init import init_weights
from llm_models.utils.masks import build_causal_mask, combine_causal_and_padding_mask


class Qwen3Next(GenerationMixin, nn.Module):
    """
    混合架构 decoder-only LM: Gated DeltaNet 与全注意力按比例交替。

    层排布 (linear_ratio=3 时): [Δ, Δ, Δ, A, Δ, Δ, Δ, A, ...]
        Δ = PreLNBlock(GatedDeltaNet + SwiGLU)   状态 O(1)
        A = PreLNBlock(GQA + SwiGLU)             KV cache O(T)

    Args:
        vocab_size / d_model / n_heads / num_kv_heads / num_layers /
        max_len / d_ff / dropout: 同 LLaMA。
        linear_ratio: 每 (linear_ratio + 1) 层里放 linear_ratio 个 DeltaNet 层
                      (Qwen3-Next 取 3, 即 75% 线性层)。
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 4096,
        n_heads: int = 32,
        num_kv_heads: Optional[int] = None,
        num_layers: int = 32,
        max_len: int = 4096,
        linear_ratio: int = 3,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if linear_ratio < 1:
            raise ValueError(f"linear_ratio 至少为 1, 当前 {linear_ratio}")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        d_head = d_model // n_heads
        self.rope = RotaryPositionalEncoding(d_head, max_len)

        if d_ff is None:
            d_ff = int(8 / 3 * d_model)
            d_ff = ((d_ff + 63) // 64) * 64

        # 周期排布: 每个周期的最后一层是全注意力, 其余是 DeltaNet
        period = linear_ratio + 1
        self.layer_types: List[str] = [
            "attn" if (i + 1) % period == 0 else "delta" for i in range(num_layers)
        ]
        self.layers = nn.ModuleList(
            [
                PreLNBlock(
                    d_model=d_model,
                    attn=(
                        GroupedQueryAttention(
                            d_model=d_model, num_heads=n_heads, num_kv_heads=num_kv_heads,
                        )
                        if kind == "attn"
                        else GatedDeltaNet(d_model=d_model, num_heads=n_heads)
                    ),
                    ffn=SwiGLUFeedForward(d_model, d_ff),
                    norm_cls=RMSNorm,
                    dropout=dropout,
                )
                for kind in self.layer_types
            ]
        )

        self.ln_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        causal = build_causal_mask(max_len, torch.device("cpu"))
        self.register_buffer("causal_mask", causal, persistent=False)

        init_weights(self)   # weight tying 之后; 初始 CE ≈ ln V
        # init_weights 会把所有 Linear bias 清零, 而 α 门需要 bias=+2 (sigmoid≈0.88, 初期偏向 "记住")
        for m in self.modules():
            if isinstance(m, GatedDeltaNet):
                nn.init.constant_(m.gate_alpha.bias, 2.0)

    def _causal_mask(self, seq_len: int) -> torch.Tensor:
        if seq_len <= self.causal_mask.size(-1):
            return self.causal_mask[:, :seq_len, :seq_len]
        return build_causal_mask(seq_len, self.causal_mask.device)

    def kv_cache_entries(self, seq_len: int) -> dict:
        """推理时每类层的"缓存"规模 (条目数): 全注意力 O(T), DeltaNet O(1)。"""
        n_attn = self.layer_types.count("attn")
        n_delta = self.layer_types.count("delta")
        return {
            "attn_layers": n_attn,
            "delta_layers": n_delta,
            "attn_cache_per_layer": seq_len,           # 随 T 增长
            "delta_state_per_layer": 1,                # 恒定 (一个状态矩阵)
        }

    def forward(
        self,
        idx: torch.Tensor,                              # [B, T]
        attention_mask: Optional[torch.Tensor] = None,  # [B, past+T]; 只作用于 attn 层
        cache: Optional[KVCache] = None,
    ) -> torch.Tensor:                                  # [B, T, V]
        B, T = idx.shape
        past = cache.pos if cache is not None else 0
        if past + T > self.max_len:
            raise ValueError(f"序列长度 {past + T} 超过 max_len={self.max_len}")

        x = self.token_embedding(idx) * math.sqrt(self.d_model)          # [B, T, D]
        position_ids = torch.arange(past, past + T, device=idx.device)
        causal = self._causal_mask(past + T)[:, past:]                   # [1, T, past+T]
        mask = combine_causal_and_padding_mask(causal, attention_mask)

        # mask / rope 对 DeltaNet 层是 no-op (递推天然因果, 衰减门隐式编码位置);
        # 同一个 cache dict 协议: attn 层往里放 k/v, delta 层往里放 state —— 主干循环不区分层类型
        for i, layer in enumerate(self.layers):
            x = layer(
                x, mask=mask, rope=self.rope, position_ids=position_ids,
                cache=cache.layers[i] if cache is not None else None,
            )
        if cache is not None:
            cache.pos += T

        return self.lm_head(self.ln_f(x))
