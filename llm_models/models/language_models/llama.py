"""
LLaMA 模型模块

论文出处:
    "LLaMA: Open and Efficient Foundation Language Models" (Touvron et al., 2023)
    "Llama 2" / "Llama 3" (Meta, 2023/2024)

历史意义:
    LLaMA 系列是 **现代开源 LLM 的事实模板**:
        - GQA (Grouped Query Attention): 减小 KV cache, LLaMA-2 70B / Llama-3 全尺寸启用
        - SwiGLU FFN: 门控激活, 比 GELU 在同参数预算下更强
        - RMSNorm (Pre-Norm): 比 LayerNorm 少一次均值, 数值更稳
        - RoPE: 相对位置, 长度外推友好
        - 无 bias + 无 dropout (大模型训练惯例)
        - Weight Tying: lm_head 与 token embedding 共享权重

在本库的演进地图里的位置:
    GPT-3 (2020, MHA + GELU + LayerNorm + Sin-PE)
        ↓  把零件全部换成现代版
    LLaMA (2023, GQA + SwiGLU + RMSNorm + RoPE)
        ↓  继续稀疏化
    Mixtral (2023, LLaMA + sparse MoE 替换 FFN)
        ↓  进一步压缩 KV cache
    DeepSeek-V3 (2024, MLA + 细粒度 MoE + 共享专家)
        ↓  稀疏长上下文
    DeepSeek-V3.2 (2025, MLA + DSA + MoE)

因此本文件几乎是纯组装: 所有零件都已在 layers/ 中, 这里只是按 LLaMA 的方式组合。
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
from llm_models.utils.masks import build_causal_mask, combine_causal_and_padding_mask


def _make_llama_block(
    d_model: int,
    n_heads: int,
    num_kv_heads: Optional[int],
    d_ff: int,
    dropout: float,
) -> PreLNBlock:
    """
    组装一个 LLaMA Block:  GQA + SwiGLU-FFN + RMSNorm (Pre-Norm)
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


# 对外保留别名: LLaMA 社区常称其为 LlamaDecoderLayer / LlamaBlock
LlamaBlock = PreLNBlock


class LLaMA(nn.Module):
    """
    LLaMA decoder-only LLM (教学版)

    架构:
        idx -> TokenEmbed * sqrt(d_model)
            -> N x PreLNBlock(GQA + SwiGLU + RMSNorm) [RoPE 注入 Q/K]
            -> RMSNorm
            -> lm_head  (与 token_embedding 共享权重)

    与 GPT3 的一行差异对照:
        GPT3:  MHA   + GELU-FFN + LayerNorm + Sin-PE (or RoPE)
        LLaMA: GQA   + SwiGLU   + RMSNorm   + RoPE

    默认参数对齐 LLaMA-2 7B (d_model=4096, n_heads=32, num_kv_heads=32, layers=32);
    教学默认缩小到 GPT-Medium 尺寸, 方便 CPU 实验。

    Args:
        vocab_size:      词表大小 (LLaMA 原版 32000)
        d_model:         隐藏维度
        n_heads:         Q head 数
        num_kv_heads:    K/V head 数, None 时等于 n_heads (退化为 MHA);
                         LLaMA-2 70B 使用 8 (GQA 压缩 4×)
        num_layers:      Transformer 层数
        max_len:         最大上下文 (预构建因果 mask 所需)
        d_ff:            SwiGLU 隐藏维度, None 时用 int(8/3 * d_model) 对齐 LLaMA
        dropout:         Dropout (LLaMA 官方训练设 0, 教学保留参数)
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
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # RoPE 作用在 per-head 维度, 所有层共享同一个 encoder (无参数开销)
        d_head = d_model // n_heads
        self.rope = RotaryPositionalEncoding(d_head, max_len)

        # d_ff 默认: LLaMA 官方用 (8/3) * d_model, 圆整到 256 的倍数
        if d_ff is None:
            d_ff = int(8 / 3 * d_model)
            # 向上对齐到 64 的整数倍, 让 matmul 对硬件友好
            d_ff = ((d_ff + 63) // 64) * 64

        self.layers = nn.ModuleList(
            [
                _make_llama_block(d_model, n_heads, num_kv_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )

        self.ln_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying: lm_head 与 token_embedding 共享
        self.lm_head.weight = self.token_embedding.weight

        # 预构建 max_len × max_len 因果 mask
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

        # * sqrt(d_model) 与 GPT3 一致 (让 embedding 与激活方差匹配)
        x = self.token_embedding(idx) * math.sqrt(self.d_model)

        # 因果 mask ∩ padding mask
        causal = self._causal_mask(T)
        mask = combine_causal_and_padding_mask(causal, attention_mask)

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
        """
        朴素自回归生成 (每步重算, 无 KV cache) — 同 GPT3.generate 的教学实现。
        生产用 KV cache + 连续 batching 能把 O(T^2) 降到 O(T)。
        """
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
