"""
GPT-OSS (OpenAI, 2025) — Mixtral 骨架 + 交替 SWA/全注意力 + attention sink

是什么: 稀疏 MoE decoder。偶数层用滑动窗口 (只看最近 W 个), 奇数层用全因果注意力;
        每个 attention head 多一个可学的 sink logit。
解决什么: 全 SWA (Mistral) 远程信息只能跨层接力、越传越糊; 全 full 的 KV cache 又是 O(T)·L。
        交替: 一半的层 cache 封顶 W, 另一半保留全局直达通路。
        sink: softmax 行和必须为 1, head "不想看任何人" 时只能把概率倒给首 token (StreamingLLM);
        给它一个专用的空列: attn = softmax([scores, sink])[..., :-1], 行和 < 1。
关键数字: gpt-oss-20b: 24 层, W=128, 64 Q / 8 KV 头, 32 专家 top-4 (总 21B, 激活 3.6B), YaRN 4K→131K。
路由: 官方是 "先 top-k 再对 k 个 logit 做 softmax"; 它与 MixtralMoE 的 "softmax → top-k → 重归一"
      逐项相等 (e^{l_i}/Σ_topk e^{l_j}, 全局分母约掉), 所以直接复用 MixtralMoE。
读代码时盯住: forward 里的 `is_swa` —— 每层拿哪张 mask、cache 裁不裁, 全由它决定。
教学省略: attention/专家的 bias, 带 clamp 的 SwiGLU, MXFP4 量化。
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from llm_models.layers.core.attention import GroupedQueryAttention
from llm_models.layers.core.normalization import RMSNorm
from llm_models.layers.core.position_encoding import RotaryPositionalEncoding
from llm_models.models.moe.mixtral import MixtralBlock
from llm_models.utils.generation import GenerationMixin, KVCache
from llm_models.utils.init import init_weights
from llm_models.utils.masks import (
    build_causal_mask,
    build_sliding_window_mask,
    combine_causal_and_padding_mask,
)


class GPTOSSBlock(MixtralBlock):
    """与 MixtralBlock 唯一的差异: GQA 打开 use_sink (每 head 一个可学 logit)。"""

    def __init__(self, d_model: int, n_heads: int, num_kv_heads: Optional[int], **kwargs):
        super().__init__(d_model=d_model, n_heads=n_heads, num_kv_heads=num_kv_heads, **kwargs)
        self.attn = GroupedQueryAttention(
            d_model=d_model, num_heads=n_heads, num_kv_heads=num_kv_heads, use_sink=True,
        )


class GPTOSSMini(GenerationMixin, nn.Module):
    """
    idx → Embed·sqrt(D) → N × GPTOSSBlock (偶数层 SWA / 奇数层 full) → RMSNorm → lm_head (tied)

    Args:
        window_size:  SWA 层的窗口 W (gpt-oss: 128)
        rope_kwargs:  透传给 RotaryPositionalEncoding, 如
                      dict(base=150000.0, scaling="yarn", factor=32.0, original_max_len=4096)
        其余同 Mixtral。forward 返回 (logits, all_routing_info), 与 Mixtral / DeepSeekV3 同接口。
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 128,
        n_heads: int = 4,
        num_kv_heads: Optional[int] = 2,
        num_layers: int = 4,
        num_experts: int = 4,
        top_k: int = 2,
        max_len: int = 256,
        window_size: int = 8,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        rope_kwargs: Optional[dict] = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.max_len = max_len
        self.window_size = window_size

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.rope = RotaryPositionalEncoding(d_model // n_heads, max_len, **(rope_kwargs or {}))

        if d_ff is None:
            d_ff = ((int(8 / 3 * d_model) + 63) // 64) * 64

        self.layers = nn.ModuleList(
            [
                GPTOSSBlock(
                    d_model=d_model, n_heads=n_heads, num_kv_heads=num_kv_heads, d_ff=d_ff,
                    num_experts=num_experts, top_k=top_k, dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        cpu = torch.device("cpu")
        self.register_buffer("causal_mask", build_causal_mask(max_len, cpu), persistent=False)
        self.register_buffer(
            "window_mask", build_sliding_window_mask(max_len, window_size, cpu), persistent=False
        )

        init_weights(self)   # weight tying 之后; sink 是裸 Parameter, 不受影响, 保持 0

    @staticmethod
    def is_swa(layer_idx: int) -> bool:
        """偶数层滑动窗口, 奇数层全注意力 (与官方 layer_types 的交替顺序一致)。"""
        return layer_idx % 2 == 0

    def forward(
        self,
        idx: torch.Tensor,                              # [B, T]
        attention_mask: Optional[torch.Tensor] = None,  # [B, past+T], 1=有效 0=pad
        cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        B, T = idx.shape
        past = cache.pos if cache is not None else 0
        total = past + T
        if total > self.max_len:
            raise ValueError(f"序列长度 {total} 超过 max_len={self.max_len}")

        x = self.token_embedding(idx) * math.sqrt(self.d_model)          # [B, T, D]
        position_ids = torch.arange(past, total, device=idx.device)

        # 两张 mask, 都取行 past: (新 query) × 列 :total (全部历史), 再 ∩ padding
        full_mask = combine_causal_and_padding_mask(
            self.causal_mask[:, past:total, :total], attention_mask)     # [*, T, past+T]
        swa_mask = combine_causal_and_padding_mask(
            self.window_mask[:, past:total, :total], attention_mask)
        # SWA 层的 cache 被滚动裁到 W, 只剩最近 kept 个 key → mask 的列对齐到最后 kept+T 列
        kept = min(past, self.window_size)
        swa_mask = swa_mask[..., -(kept + T):]                           # [*, T, kept+T]

        all_routing_info: List[Dict[str, torch.Tensor]] = []
        for i, layer in enumerate(self.layers):
            layer_cache = cache.layers[i] if cache is not None else None
            x, routing_info = layer(
                x, mask=swa_mask if self.is_swa(i) else full_mask,
                rope=self.rope, position_ids=position_ids, cache=layer_cache,
            )
            all_routing_info.append(routing_info)
            if layer_cache is not None and self.is_swa(i):
                # 窗口外的 K/V 再也不会被看到 → 丢掉 (K 已带 RoPE 绝对位置, 裁剪不会错位)。
                # full 层不裁: 它们就是为 "直达任意远处" 而留的。
                layer_cache["k"] = layer_cache["k"][:, :, -self.window_size:]
                layer_cache["v"] = layer_cache["v"][:, :, -self.window_size:]
        if cache is not None:
            cache.pos += T

        return self.lm_head(self.ln_f(x)), all_routing_info
