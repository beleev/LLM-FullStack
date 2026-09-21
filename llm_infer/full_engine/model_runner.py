"""
model_runner.py — 在**物理分页 KV pool** 上跑 TinyLM 前向

是什么: 引擎里唯一碰张量的地方。KV 不挂在序列上, 而是写进全局 pool[layer] (num_blocks, block_size, D),
        attention 经 block_table 读回 (m02.paged_attention) —— 所以 prefix cache 命中的 block 真的不用再算。
盯住: run() 的 start_pos: 只算 ids[start_pos:] 的 Q/K/V, 前 start_pos 个 token 的 KV 直接从 pool 读
      (它们来自: 前缀命中 / 上一个 prefill chunk / 之前的 decode 步)。
与真实系统的差距: 这里逐序列循环; vLLM `GPUModelRunner` 把整个 batch 的 token 摊平成一个 (ΣT, D) 张量,
      靠 slot_mapping + block_tables + FlashAttention varlen kernel 一次算完, 再套 CUDA Graph (m12)。
"""
from __future__ import annotations
from typing import Sequence
import numpy as np

from llm_infer.core import TinyLM, rms_norm
from llm_infer.core.tiny_model import apply_rope, mlp_forward
from llm_infer.m02_paged_attention.paged_attention import write_kv, paged_attention


class ModelRunner:
    def __init__(self, lm: TinyLM, num_blocks: int, block_size: int):
        self.lm = lm
        shape = (lm.cfg.n_layer, num_blocks, block_size, lm.cfg.d_model)
        dtype = lm.w.tok_emb.dtype                         # pool 精度必须与模型一致, 否则写入时悄悄截断
        self.k_pool = np.zeros(shape, dtype=dtype)         # (L, num_blocks, bs, D), 存 RoPE 之后的 K
        self.v_pool = np.zeros(shape, dtype=dtype)
        self.tokens_computed = 0                           # 真正做了前向的 token 数 (诚实的省算力统计)

    def run(self, ids: Sequence[int], block_table: Sequence[int], start_pos: int) -> np.ndarray:
        """算 ids[start_pos:] 这 n 个 token, KV 写入 pool → 最后一个 token 的 logits (V,)。"""
        lm = self.lm
        new = np.asarray(ids[start_pos:], dtype=np.int64)
        pos = np.arange(start_pos, len(ids))               # (n,) 绝对位置: RoPE 与 slot 换算都用它
        x = lm.w.tok_emb[new]                              # (n, D)
        for li, layer in enumerate(lm.w.layers):
            h = rms_norm(x, layer.norm1_g)
            q = apply_rope(h @ layer.wq, lm.cos, lm.sin, positions=pos)   # (n, D)
            k = apply_rope(h @ layer.wk, lm.cos, lm.sin, positions=pos)
            write_kv(self.k_pool[li], self.v_pool[li], block_table, pos, k, h @ layer.wv)
            # q 只有 n 行, K/V 是经页表读回的全部 len(ids) 行 (含别的请求算好的共享前缀)
            attn = paged_attention(q, self.k_pool[li], self.v_pool[li], block_table, len(ids))
            x = x + attn @ layer.wo
            x = x + mlp_forward(rms_norm(x, layer.norm2_g), layer)
        self.tokens_computed += len(new)
        return rms_norm(x[-1], lm.w.norm_f_g) @ lm.w.lm_head               # (V,)
