"""
Mixtral (Mistral AI, 2024) — LLaMA 骨架 + 每层 FFN 换成稀疏 MoE

解决什么: dense LLaMA 想加容量只能整体加宽加深, 算力同步上涨。Mixtral 只把 FFN 换成
    8 个专家、每 token 选 2 个: 8x7B 总参 47B, 每 token 激活 ~13B。attention/norm/RoPE 一律不变。
路由: softmax → top-2 → 选中权重再归一 (layers/sparse/moe.py::MixtralMoE)。
负载均衡: 没有 DeepSeek 那样的 bias, 全靠 aux loss (training/loss.py::MoELMLoss, 均衡时 = K)。
forward 返回 (logits, all_routing_info), 与 DeepSeekV3 同接口。
读代码时盯住: MixtralBlock.forward 里 moe 返回的 routing_info —— 它一路传到 loss。
教学省略: 真实 Mixtral 的 sliding-window attention (window=4096)。
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from llm_models.layers.core.attention import GroupedQueryAttention
from llm_models.layers.sparse.moe import MixtralMoE
from llm_models.layers.core.normalization import RMSNorm
from llm_models.layers.core.position_encoding import RotaryPositionalEncoding
from llm_models.utils.generation import GenerationMixin, KVCache
from llm_models.utils.init import init_weights
from llm_models.utils.masks import build_causal_mask, combine_causal_and_padding_mask


class MixtralBlock(nn.Module):
    """Pre-RMSNorm Block:  x → norm → GQA → +  → norm → MixtralMoE → +  (顺带返回 routing_info)"""

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
            num_experts=num_experts, top_k=top_k,
        )
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)  # 残差 dropout, 每个子层恰好一次 (MoE 内部不再做)

    def forward(
        self,
        x: torch.Tensor,                       # [B, T, D]
        mask: Optional[torch.Tensor] = None,
        rope: Optional[nn.Module] = None,
        position_ids: Optional[torch.Tensor] = None,
        cache: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # 子层 1: GQA self-attention
        h = self.norm1(x)
        h = self.attn(q=h, mask=mask, rope=rope, position_ids=position_ids, cache=cache)
        x = x + self.dropout(h)

        # 子层 2: 稀疏 MoE
        h = self.norm2(x)
        h, routing_info = self.moe(h)
        x = x + self.dropout(h)
        return x, routing_info


class Mixtral(GenerationMixin, nn.Module):
    """
    idx → Embed·sqrt(D) → N × MixtralBlock → RMSNorm → lm_head (与 embedding 共享权重)

    默认参数即 Mixtral 8x7B: d_model=4096, 32 头 / 8 KV 头, 32 层, 8 专家 top-2。
    d_ff 默认 int(8/3·D) 向上对齐到 64。generate() 来自 GenerationMixin (GQA KV cache)。
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

        causal = build_causal_mask(max_len, torch.device("cpu"))  # [1, L, L]
        self.register_buffer("causal_mask", causal, persistent=False)

        # 默认 N(0,1) embedding + tying + ·sqrt(D) 会让初始 CE ≈ 127; N(0, 0.02²) 后 ≈ ln V
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
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        """返回 (logits [B, T, V], 每层一份 routing_info)。"""
        B, T = idx.shape
        past = cache.pos if cache is not None else 0  # 已缓存 token 数: 同时平移 RoPE 位置和 mask 行
        if past + T > self.max_len:
            raise ValueError(f"序列长度 {past + T} 超过 max_len={self.max_len}")

        x = self.token_embedding(idx) * math.sqrt(self.d_model)  # [B, T, D]

        position_ids = torch.arange(past, past + T, device=idx.device)
        causal = self._causal_mask(past + T)[:, past:]  # [1, T, past+T]
        mask = combine_causal_and_padding_mask(causal, attention_mask)

        all_routing_info: List[Dict[str, torch.Tensor]] = []
        for i, layer in enumerate(self.layers):
            x, routing_info = layer(
                x, mask=mask, rope=self.rope, position_ids=position_ids,
                cache=cache.layers[i] if cache is not None else None,
            )
            all_routing_info.append(routing_info)
        if cache is not None:
            cache.pos += T

        return self.lm_head(self.ln_f(x)), all_routing_info
