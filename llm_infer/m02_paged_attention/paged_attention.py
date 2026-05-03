"""
paged_attention.py — 在分页 KV pool 上做 attention

与 m01 的"连续 KV"区别:
    m01: K, V 是 (T, D) 的连续数组, 直接 q @ K.T
    m02: KV 散落在 pool 中, 需要按 block_table 把它们"串"起来再算

简化:
    - 只演示 query 是 1 个 token (decode 场景), prefill 类似但更繁
    - 先 gather 出连续的 K, V, 再按经典公式算; 真实 vLLM 是融合 kernel
"""
from __future__ import annotations
import numpy as np

from llm_infer.core.utils import softmax, causal_mask


def gather_kv(
    k_pool: np.ndarray,       # (num_blocks, block_size, D)
    v_pool: np.ndarray,
    block_table: list,        # 该序列的物理 block ids
    ctx_len: int,             # 当前序列总 token 数
) -> tuple:
    """从分页 pool 中按 block_table 把该序列的 K, V 拼成连续数组。

    教学用; 真实实现会让 attention kernel 直接"按 block 跳着读", 避免拷贝。
    """
    block_size = k_pool.shape[1]
    pieces_k, pieces_v = [], []
    remaining = ctx_len
    for blk in block_table:
        take = min(block_size, remaining)
        pieces_k.append(k_pool[blk, :take])
        pieces_v.append(v_pool[blk, :take])
        remaining -= take
        if remaining == 0:
            break
    K = np.concatenate(pieces_k, axis=0)   # (ctx_len, D)
    V = np.concatenate(pieces_v, axis=0)
    return K, V


def write_kv(
    k_pool: np.ndarray, v_pool: np.ndarray,
    block_table: list,
    pos: int,                 # token 在序列中的下标
    k_new: np.ndarray,        # (D,)
    v_new: np.ndarray,
) -> None:
    """把新算出来的 K_new, V_new 写到正确的 (block, slot)。"""
    block_size = k_pool.shape[1]
    block_idx_in_seq = pos // block_size
    slot_in_block = pos % block_size
    physical_blk = block_table[block_idx_in_seq]
    k_pool[physical_blk, slot_in_block] = k_new
    v_pool[physical_blk, slot_in_block] = v_new


def paged_attention(
    q: np.ndarray,            # (T_q, D)
    k_pool: np.ndarray,
    v_pool: np.ndarray,
    block_table: list,
    ctx_len: int,
) -> np.ndarray:
    """对一个序列做 attention, KV 来自分页 pool。"""
    K, V = gather_kv(k_pool, v_pool, block_table, ctx_len)
    d = q.shape[-1]
    scores = (q @ K.T) / np.sqrt(d)
    scores = scores + causal_mask(q.shape[0], K.shape[0])
    attn = softmax(scores, axis=-1)
    return attn @ V
