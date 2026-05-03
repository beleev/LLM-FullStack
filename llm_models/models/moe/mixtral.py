"""
Mixtral 模型模块

论文出处:
    "Mixtral of Experts" (Jiang et al., Mistral AI, 2024)
    8x7B: 总参 47B, 每 token 激活 ~13B, 首个主流开源稀疏 MoE。

在本库中的位置:
    - LLaMA (dense) 的稀疏化版本: 把每层 SwiGLU FFN 换成 MixtralMoE (8 专家, top-2)
    - 与 DeepSeek-V3 的对照教学:
        Mixtral: softmax + top-k, 无共享专家, 无 aux-loss-free bias
        DeepSeekV3: sigmoid + top-k + renormalize, 有共享专家, aux-free bias
      二者都用 LLaMA-风格骨架 (GQA + SwiGLU 专家 + RMSNorm + RoPE)

教学重点:
    - MoE 并未改变 attention 或 norm 结构, 纯粹是 FFN 稀疏化
    - routing_info 的返回形态与 DeepSeekMoE 保持一致, 训练端可以复用同一个
      MoELMLoss (Switch-Transformer 风格 aux loss)
    - Mixtral 的 attention 是 GQA (LLaMA 同款); 本教学版不加 SWA (sliding window),
      保持简单; 真实 Mixtral 还有 window=4096 的 SWA
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from llm_models.layers.core.attention import GroupedQueryAttention
from llm_models.layers.core.feedforward import SwiGLUFeedForward
from llm_models.layers.sparse.moe import MixtralMoE
from llm_models.layers.core.normalization import RMSNorm
from llm_models.layers.core.position_encoding import RotaryPositionalEncoding
from llm_models.utils.masks import build_causal_mask, combine_causal_and_padding_mask


class MixtralBlock(nn.Module):
    """
    Mixtral Block: GQA + MixtralMoE + RMSNorm (Pre-Norm)

    数据流:
        x -> RMSNorm -> GQA           -> Add
          -> RMSNorm -> MixtralMoE    -> Add  (同时返回 routing_info)

    由于 MoE 层返回 (out, routing_info) 而非单张量, 本 Block 不能直接复用 PreLNBlock;
    逻辑与 DeepSeekBlock 完全一致, 只是把 MLA→GQA, DeepSeekMoE→MixtralMoE。
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_kv_heads: Optional[int],
        d_ff: int,
        num_experts: int,
        top_k: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.attn = GroupedQueryAttention(
            d_model=d_model, num_heads=n_heads, num_kv_heads=num_kv_heads,
        )
        self.moe = MixtralMoE(
            d_model=d_model, d_ff=d_ff,
            num_experts=num_experts, top_k=top_k, dropout=dropout,
        )
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        rope: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # 子层 1: GQA self-attention
        h = self.norm1(x)
        h = self.attn(q=h, k=h, v=h, mask=mask, rope=rope)
        x = x + self.dropout(h)

        # 子层 2: 稀疏 MoE
        h = self.norm2(x)
        h, routing_info = self.moe(h)
        x = x + self.dropout(h)
        return x, routing_info


class Mixtral(nn.Module):
    """
    Mixtral 教学版 (sparse MoE decoder-only)

    架构:
        idx -> TokenEmbed * sqrt(d_model)
            -> N x MixtralBlock(GQA + MixtralMoE + RMSNorm)   [RoPE 注入 Q/K]
            -> RMSNorm
            -> lm_head  (与 token_embedding 共享)

    forward 返回 (logits, all_routing_info), 与 DeepSeekV3 对齐, 可复用 MoELMLoss。

    Args:
        vocab_size:    词表大小
        d_model:       隐藏维度 (Mixtral 8x7B 为 4096)
        n_heads:       Q head 数 (Mixtral 8x7B 为 32)
        num_kv_heads:  K/V head 数 (Mixtral 8x7B 为 8)
        num_layers:    Transformer 层数 (Mixtral 8x7B 为 32)
        num_experts:   每层专家数 (Mixtral 8x7B 为 8)
        top_k:         每 token 激活专家数 (Mixtral 8x7B 为 2)
        max_len:       最大上下文
        d_ff:          专家 SwiGLU 隐藏维度, 默认 int(8/3 * d_model)
        dropout:       Dropout
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 4096,
        n_heads: int = 32,
        num_kv_heads: Optional[int] = 8,
        num_layers: int = 32,
        num_experts: int = 8,
        top_k: int = 2,
        max_len: int = 4096,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.d_model = d_model
        self.max_len = max_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)

        d_head = d_model // n_heads
        self.rope = RotaryPositionalEncoding(d_head, max_len)

        if d_ff is None:
            d_ff = int(8 / 3 * d_model)
            d_ff = ((d_ff + 63) // 64) * 64  # 对齐到 64

        self.layers = nn.ModuleList(
            [
                MixtralBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    num_kv_heads=num_kv_heads,
                    d_ff=d_ff,
                    num_experts=num_experts,
                    top_k=top_k,
                    dropout=dropout,
                )
                for _ in range(num_layers)
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

    def forward(
        self,
        idx: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        """
        Args:
            idx:            [B, T] token IDs
            attention_mask: [B, T] padding mask (可选)
        Returns:
            logits:           [B, T, vocab_size]
            all_routing_info: 每层一份 routing_info 的列表, 供外部算 Switch aux loss
        """
        B, T = idx.shape
        if T > self.max_len:
            raise ValueError(f"序列长度 {T} 超过 max_len={self.max_len}")

        x = self.token_embedding(idx) * math.sqrt(self.d_model)

        causal = self._causal_mask(T)
        mask = combine_causal_and_padding_mask(causal, attention_mask)

        all_routing_info: List[Dict[str, torch.Tensor]] = []
        for layer in self.layers:
            x, routing_info = layer(x, mask=mask, rope=self.rope)
            all_routing_info.append(routing_info)

        x = self.ln_f(x)
        return self.lm_head(x), all_routing_info
