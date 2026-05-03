"""
m02 demo — Paged Attention 与 BlockManager

观察三件事:
    1) 多序列在同一 pool 下并发分配/释放, 显存被复用
    2) gather + attention 在分页 KV 上算出的结果与连续 KV 等价
    3) 池子用满 → can_allocate 返回 False, 触发抢占的契机 (m03 会处理)
"""
from __future__ import annotations
import numpy as np

from llm_infer.core.utils import banner, kv, softmax, causal_mask
from llm_infer.m02_paged_attention.block_manager import BlockManager
from llm_infer.m02_paged_attention.paged_attention import (
    gather_kv, write_kv, paged_attention,
)


def main():
    banner("M02 - Paged Attention: 显存的虚拟内存")

    # --- 1) 多序列并发分配 ---------------------------------------- #
    print("\n[1] 多序列并发分配 (pool=8 blocks, block_size=4)")
    bm = BlockManager(num_blocks=8, block_size=4)
    bm.allocate(seq_id=101, n_tokens=10)  # ceil(10/4)=3 blocks
    bm.allocate(seq_id=102, n_tokens=6)   # ceil(6/4) =2 blocks
    bm.allocate(seq_id=103, n_tokens=3)   # 1 block
    print(f"  seq 101 block_table: {bm.block_table(101)}")
    print(f"  seq 102 block_table: {bm.block_table(102)}")
    print(f"  seq 103 block_table: {bm.block_table(103)}")
    print(f"  pool stats:          {bm.stats()}")

    # --- 2) 序列结束, block 自动归还 ----------------------------- #
    print("\n[2] seq 102 结束, 它的 block 立即被 free_list 吃掉")
    bm.free(102)
    print(f"  pool stats:          {bm.stats()}")
    print(f"  free_list:           {list(bm.free_list)}")

    # --- 3) 序列追加 token, 必要时分配新 block -------------------- #
    print("\n[3] seq 101 增长到 13 token, 触发新 block 分配")
    blk = bm.append(seq_id=101, current_len=12)   # 12→13, 13/4=3 block 满, 需新 block
    print(f"  appended new block: {blk}")
    print(f"  seq 101 block_table: {bm.block_table(101)}")

    # --- 4) 池子用满, 申请失败 ----------------------------------- #
    print("\n[4] 申请超出 pool 容量")
    try:
        bm.allocate(seq_id=999, n_tokens=100)
    except MemoryError as e:
        print(f"  MemoryError: {e}")
    print(f"  → 这就是 m03 scheduler 触发 preempt 的信号")

    # --- 5) 数值正确性: paged vs 连续 KV ------------------------- #
    banner("[5] 数值验证: paged_attention 与连续 KV 计算结果一致")
    rs = np.random.RandomState(0)
    D, T = 8, 11
    Q = rs.randn(1, D).astype(np.float32)        # 一个 query (decode 场景)
    K_full = rs.randn(T, D).astype(np.float32)
    V_full = rs.randn(T, D).astype(np.float32)

    # 把 K_full / V_full 装进 pool
    bm2 = BlockManager(num_blocks=10, block_size=4)
    bm2.allocate(seq_id=1, n_tokens=T)
    block_size = bm2.block_size
    k_pool = np.zeros((bm2.num_blocks, block_size, D), dtype=np.float32)
    v_pool = np.zeros_like(k_pool)
    table = bm2.block_table(1)
    for pos in range(T):
        write_kv(k_pool, v_pool, table, pos, K_full[pos], V_full[pos])

    # paged 路径
    out_paged = paged_attention(Q, k_pool, v_pool, table, ctx_len=T)
    # 连续路径 (m01 风格)
    scores = (Q @ K_full.T) / np.sqrt(D)
    out_dense = softmax(scores + causal_mask(1, T), axis=-1) @ V_full

    diff = np.max(np.abs(out_paged - out_dense))
    kv("max |paged - dense|", f"{diff:.2e}")
    assert diff < 1e-6, "paged_attention 实现错误"
    print("  ✓ 数值一致, 分页对外语义透明")


if __name__ == "__main__":
    main()
