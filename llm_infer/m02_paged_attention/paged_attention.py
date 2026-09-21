"""
paged_attention.py — 在分页 KV pool 上读写 KV 并做 attention

是什么: KV 不再是每序列一块连续 (T, D) 数组, 而是散落在全局 pool (num_blocks, block_size, D) 里,
        token 位置 pos 的 KV 存在 pool[block_table[pos // bs], pos % bs]。
解决什么: 让 BlockManager 的"页表"真正落到张量上 —— 多序列共享同一物理 block 时, KV 只存一份。
盯住: write_kv / gather_kv 里 pos → (block, slot) 的换算; 其余与 m01 的连续 KV 完全一样。
与真实系统的差别: 这里先 gather 成连续数组再调 dense_attention (多一次拷贝);
        vLLM 的 PagedAttention / FlashInfer kernel 直接按 block_table 跳着读, 不拷贝。
"""
from __future__ import annotations
from typing import Sequence, Tuple
import numpy as np

from llm_infer.core.utils import dense_attention


def write_kv(k_pool: np.ndarray, v_pool: np.ndarray,      # (num_blocks, bs, D)
             block_table: Sequence[int],
             pos,                                          # int 或 (n,) 序列内 token 位置
             k_new: np.ndarray, v_new: np.ndarray) -> None:  # (D,) 或 (n, D)
    """把位置 pos 的新 K/V 写进各自的 (物理 block, slot) —— 即 vLLM 的 slot_mapping。"""
    bs = k_pool.shape[1]
    pos = np.atleast_1d(pos)
    blk = np.asarray(block_table)[pos // bs]               # (n,) 逻辑页号 → 物理 block
    k_pool[blk, pos % bs] = k_new
    v_pool[blk, pos % bs] = v_new


def gather_kv(k_pool: np.ndarray, v_pool: np.ndarray,
              block_table: Sequence[int], ctx_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """按页表把前 ctx_len 个 token 的 K, V 拼回连续数组 (ctx_len, D)。"""
    bs, D = k_pool.shape[1], k_pool.shape[2]
    blks = list(block_table[:-(-ctx_len // bs)])           # 只取用得到的页
    K = k_pool[blks].reshape(-1, D)[:ctx_len]              # (n_blk, bs, D) → (n_blk·bs, D) → 截掉末页空槽
    V = v_pool[blks].reshape(-1, D)[:ctx_len]
    return K, V


def paged_attention(q: np.ndarray,                         # (T_q, D), 对齐序列尾部
                    k_pool: np.ndarray, v_pool: np.ndarray,
                    block_table: Sequence[int], ctx_len: int) -> np.ndarray:
    """T_q=1 是 decode; T_q>1 是 (chunked) prefill —— 因果 mask 由 dense_attention 按尾部对齐生成。"""
    K, V = gather_kv(k_pool, v_pool, block_table, ctx_len)
    return dense_attention(q, K, V)                        # (T_q, D)
