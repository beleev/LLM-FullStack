"""
Qwen3-Next 模型模块 (混合线性注意力架构, 教学版)

论文/技术报告出处:
    Qwen3-Next (Alibaba, 2025) — 80B-A3B, 混合架构 + 超稀疏 MoE
    Gated DeltaNet: Yang et al., 2024 (NeurIPS)

历史意义:
    2025 年的共识雏形: **不是所有层都需要完整注意力**。
        - Gated DeltaNet 层 (75%): O(1) 状态, O(T) 计算, 管"流畅的局部建模"
        - 全注意力层 (25%):        O(T) cache, O(T^2) 计算, 管"精准的长程检索"
    MiniMax-Text (Lightning Attention 7:1)、Jamba (Mamba+Attn) 走的同一条路。
    长上下文的成本被砍掉大半, 而召回精度由少量全注意力层兜底。

在本库的演进地图里的位置:
    Mamba (2023, 纯 SSM, O(T))            —— 线性序列建模的极端
        ↓  纯线性召回弱, 混一点全注意力
    Qwen3-Next (2025, DeltaNet 3 : 1 Attn) —— 混合架构
        ↑  另一个极端: LLaMA/Mistral 全层注意力

实现说明 (教学简化):
    - 真实 Qwen3-Next 还有超稀疏 MoE (本库见 DeepSeekMoE)、MTP (见 mtp.py)、
      zero-centered RMSNorm 等; 此处只保留"混合层"这一核心创新
    - 全注意力层用库内 GQA + RoPE; DeltaNet 层不需要 mask 和 RoPE
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_models.layers.core.attention import GroupedQueryAttention
from llm_models.layers.core.blocks import PreLNBlock
from llm_models.layers.core.feedforward import SwiGLUFeedForward
from llm_models.layers.core.normalization import RMSNorm
from llm_models.layers.core.position_encoding import RotaryPositionalEncoding
from llm_models.layers.sparse.linear_attention import GatedDeltaNet
from llm_models.utils.masks import build_causal_mask, combine_causal_and_padding_mask


class Qwen3Next(nn.Module):
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
        idx: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            idx: [B, T] token IDs
        Returns:
            logits: [B, T, vocab_size]
        """
        B, T = idx.shape
        if T > self.max_len:
            raise ValueError(f"序列长度 {T} 超过 max_len={self.max_len}")

        x = self.token_embedding(idx) * math.sqrt(self.d_model)

        causal = self._causal_mask(T)
        mask = combine_causal_and_padding_mask(causal, attention_mask)

        # mask/rope 对 DeltaNet 层是 no-op (递推天然因果, 衰减门隐式编码位置),
        # PreLNBlock 统一转发, 层类型对主干循环完全透明
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
