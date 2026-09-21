"""
sink_cache.py — StreamingLLM 的滚动 KV cache: [开头 n_sink 个 token] + [最近 window 个 token]

是什么: 无限流式输入下显存有界的 KV cache, 逐出中间 token, 永远保留开头几个 "attention sink"。
瓶颈  : 显存 — 完整 cache 是 O(T) 且 T 迟早超过 max_seq_len (RoPE 表 / 训练长度) 直接跑不了。
关键  : cache 条目恒 ≤ n_sink + window; 位置**按 cache 槽位重新编号** 0..L-1, 永远不超过训练长度。
        为了能重编号, K 必须存 **未旋转 (pre-RoPE)** 的版本, 每步按当前槽位现转 —— 与 core 里
        "K 存 post-RoPE" 的普通 cache 正好相反。
盯住  : `SinkCache.k[li]` (L,D) 未旋转; `stream_step` 里的 `positions = arange(L)`。
对应  : StreamingLLM (Xiao et al. 2023) 的 StartRecentKVCache + pos_shift 补丁;
        HF transformers 的 SinkCache; TensorRT-LLM 的 sink_token_length; GPT-OSS 的可学习 sink logit。
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from llm_infer.core import TinyLM, dense_attention, rms_norm
from llm_infer.core.tiny_model import apply_rope, mlp_forward


class SinkCache:
    """每层一份 (K_raw, V), 形状 (L, D), L ≤ n_sink + window。

    n_sink=0 → 纯滑动窗口; window=很大 → 完整 cache (此时槽位 == 绝对位置, 与 TinyLM.forward 等价)。
    """

    def __init__(self, n_layer: int, d_model: int, n_sink: int, window: int):
        self.n_sink, self.window = n_sink, window
        self.k = [np.zeros((0, d_model), dtype=np.float32) for _ in range(n_layer)]   # 未旋转的 K
        self.v = [np.zeros((0, d_model), dtype=np.float32) for _ in range(n_layer)]

    def __len__(self) -> int:
        return self.k[0].shape[0]

    def append(self, li: int, k_raw: np.ndarray, v: np.ndarray) -> None:
        """追加 1 个 token 的 (k_raw, v) (1,D); 超预算就逐出窗口里最老的 (槽位 n_sink), sink 不动。"""
        K = np.concatenate([self.k[li], k_raw])                      # (L+1, D)
        V = np.concatenate([self.v[li], v])
        if K.shape[0] > self.n_sink + self.window:
            keep = np.r_[0:self.n_sink, self.n_sink + 1:K.shape[0]]  # 去掉槽位 n_sink
            K, V = K[keep], V[keep]
        self.k[li], self.v[li] = K, V


def stream_step(lm: TinyLM, cache: SinkCache, tok: int,
                weights_out: Optional[List[np.ndarray]] = None) -> np.ndarray:
    """喂 1 个 token → logits (V,)。自己的逐层循环 (core 的 attn_forward 把 K 转完才存, 没法重编号)。

    weights_out 非 None 时, 追加每层的注意力权重 (L,) 供观测 sink 现象。
    """
    x = lm.w.tok_emb[[tok]]                                          # (1, D)
    for li, layer in enumerate(lm.w.layers):
        h = rms_norm(x, layer.norm1_g)
        q, k_raw, v = h @ layer.wq, h @ layer.wk, h @ layer.wv       # 各 (1, D); k 不旋转就入 cache
        cache.append(li, k_raw, v)
        L = len(cache.k[li])
        pos = np.arange(L)                                           # 位置 = cache 槽位, 而非流里的绝对位置
        K = apply_rope(cache.k[li], lm.cos, lm.sin, positions=pos)   # (L, D) 每步整体重转, O(L·D)
        q = apply_rope(q, lm.cos, lm.sin, positions=pos[-1:])        # query 在最后一个槽位
        x = x + dense_attention(q, K, cache.v[li]) @ layer.wo        # Tq=1 → 默认因果 mask 即全可见
        if weights_out is not None:
            weights_out.append(dense_attention(q, K, np.eye(L, dtype=np.float32))[0])  # V=I → 输出即权重
        x = x + mlp_forward(rms_norm(x, layer.norm2_g), layer)
    return (rms_norm(x, lm.w.norm_f_g) @ lm.w.lm_head)[0]            # (V,)
