"""
GPT-3 (Brown et al., 2020) — decoder-only Transformer, "一切任务 = 文本续写"

是什么: N × PreLNBlock(MHA + GELU-FFN + LayerNorm), 目标只有一个: 预测下一个 token。
解决什么: 相比原始 Transformer / GPT-1 的 Post-LN, Pre-LN `x + f(LN(x))` 让深层训练稳定、不再强依赖 warmup;
          175B 参数首次系统验证 Scaling Law 与 in-context learning。
关键数字: d_ff = 4·d_model;  初始 CE ≈ ln V (N(0,0.02²) 初始化; 默认 N(0,1) embedding + weight tying 会给 ~250)。
三个实现细节: causal mask 用 register_buffer 缓存;  lm_head 与 embedding 共享权重 (Press & Wolf 2017);
             generate() 来自 GenerationMixin, 带 KV cache (每步只算 1 个新 token)。
读代码时盯住: forward 里的 `past` —— 它平移 Sin-PE 的 offset (或 RoPE 的 position_ids) 和 mask 的行。
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from llm_models.layers.core.attention import GroupedQueryAttention
from llm_models.layers.core.blocks import PreLNBlock
from llm_models.layers.core.feedforward import GeLUFeedForward
from llm_models.layers.core.position_encoding import (
    RotaryPositionalEncoding,
    SinPositionalEncoding,
)
from llm_models.utils.generation import GenerationMixin, KVCache
from llm_models.utils.init import init_weights
from llm_models.utils.masks import build_causal_mask


def _make_gpt_block(d_model: int, n_heads: int, d_ff: int, dropout: float) -> PreLNBlock:
    """GPT-3 Block = PreLNBlock(MHA + GELU-FFN + LayerNorm)。

    MHA 用 GQA(num_kv_heads = num_heads, bias=True) 实现 —— 数学上就是 MHA, 且自带 KV cache;
    逐头循环的教学版见 layers/core/attention.py::MultiHeadAttention。
    """
    return PreLNBlock(
        d_model=d_model,
        attn=GroupedQueryAttention(d_model, n_heads, bias=True),
        ffn=GeLUFeedForward(d_model, d_ff),
        norm_cls=nn.LayerNorm,
        dropout=dropout,
    )


GPTBlock = PreLNBlock   # 旧符号


class GPT3(GenerationMixin, nn.Module):
    """
    idx -> Embed·sqrt(D) -> (Sin-PE 加到 x | RoPE 注入每层 Q/K)
        -> N × PreLNBlock(MHA + GELU-FFN + LN) -> LayerNorm -> lm_head (tied)

    末尾 ln_f 是 Pre-LN 必需的 "出口规范化": 残差主路从不过 norm, 范数随层数累积。
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 768,
        n_heads: int = 12,
        num_layers: int = 12,
        max_len: int = 2048,
        dropout: float = 0.1,
        use_rope: bool = False,
    ):
        super().__init__()

        self.d_model = d_model
        self.use_rope = use_rope
        self.max_len = max_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)

        if use_rope:
            self.pos_encoder = RotaryPositionalEncoding(d_model // n_heads, max_len)  # 作用于 head 内 Q/K
        else:
            self.pos_encoder = SinPositionalEncoding(d_model, max_len)                # 加在 embedding 上

        d_ff = 4 * d_model
        self.layers = nn.ModuleList(
            [_make_gpt_block(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight    # weight tying: 参数减半 + 隐式正则

        self.register_buffer(
            "causal_mask", build_causal_mask(max_len, torch.device("cpu")), persistent=False
        )

        init_weights(self)   # weight tying 之后; 初始 CE ≈ ln V

    def _causal_mask(self, seq_len: int) -> torch.Tensor:
        if seq_len <= self.causal_mask.size(-1):
            return self.causal_mask[:, :seq_len, :seq_len]
        return build_causal_mask(seq_len, self.causal_mask.device)

    def forward(
        self,
        idx: torch.Tensor,                   # [B, T]
        cache: Optional[KVCache] = None,
    ) -> torch.Tensor:                       # [B, T, V]
        _, T = idx.size()
        past = cache.pos if cache is not None else 0
        if past + T > self.max_len:
            raise ValueError(f"序列长度 {past + T} 超过最大上下文窗口 {self.max_len}")

        x = self.token_embedding(idx) * math.sqrt(self.d_model)          # [B, T, D]

        # 两种位置编码二选一: RoPE 不动主干, 交给每层去旋转 Q/K; Sin-PE 直接加到 x 上
        rope = self.pos_encoder if self.use_rope else None
        if not self.use_rope:
            x = self.pos_encoder(x, offset=past)

        position_ids = torch.arange(past, past + T, device=idx.device)
        mask = self._causal_mask(past + T)[:, past:]                     # [1, T, past+T]
        for i, layer in enumerate(self.layers):
            x = layer(
                x, mask=mask, rope=rope, position_ids=position_ids,
                cache=cache.layers[i] if cache is not None else None,
            )
        if cache is not None:
            cache.pos += T

        return self.lm_head(self.ln_f(x))
