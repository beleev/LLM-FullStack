"""
m04 demo — Prefix Cache 命中演示

场景: 同一段 system prompt 被多个用户共享, 第二个用户应该命中前缀。
"""
from __future__ import annotations
from llm_infer.core.utils import banner, kv
from llm_infer.m02_paged_attention.block_manager import BlockManager
from llm_infer.m04_prefix_cache.prefix_cache import PrefixCache


def main():
    banner("M04 - Prefix Cache (hash 链)")

    # block_size=4 容易看清"完整 block 才命中"
    bm = BlockManager(num_blocks=20, block_size=4)
    cache = PrefixCache(bm)

    SYSTEM = [10, 11, 12, 13, 20, 21, 22, 23]    # 8 token = 整 2 个 block
    USER_A = SYSTEM + [30, 31, 32, 33, 40, 41]    # 14 token, 末 6 token 是 A 私有
    USER_B = SYSTEM + [50, 51, 52, 53, 60, 61]    # 14 token, 末 6 token 是 B 私有
    USER_C = [10, 11, 12, 13, 99, 99, 99, 99]    # 共享前 4 token, 第 1 块就 miss

    # ---- 用户 A 第一次, 完全 miss --------------------------------- #
    print("\n[1] User A 入场 (cold)")
    hits_a, n_hit_a = cache.match_prefix(USER_A)
    kv("命中 block", hits_a)
    kv("命中 token 数", n_hit_a)
    # 模拟分配 + 计算 + 注册
    n_blocks_a = (len(USER_A) + bm.block_size - 1) // bm.block_size
    bm.allocate(seq_id=1, n_tokens=len(USER_A))
    table_a = bm.block_table(1)
    parent = None
    for i in range(len(USER_A) // bm.block_size):  # 只对完整 block 注册
        chunk = USER_A[i * bm.block_size:(i + 1) * bm.block_size]
        h = cache.register_block(parent, chunk, table_a[i])
        parent = h
    print(f"  A block_table: {table_a}")
    print(f"  cache stats:   {cache.stats()}")

    # ---- 用户 B 入场, 应命中前 2 个 block ------------------------- #
    print("\n[2] User B 入场, 共享 SYSTEM 前缀")
    hits_b, n_hit_b = cache.match_prefix(USER_B)
    kv("命中 block (复用)", hits_b)
    kv("命中 token 数", n_hit_b)
    # 把命中部分 share, 未命中部分新分配
    for blk in hits_b:
        bm.share_block(blk)
    n_new = (len(USER_B) - n_hit_b + bm.block_size - 1) // bm.block_size
    new_blocks = []
    for _ in range(n_new):
        new_blocks.append(bm.free_list.popleft())
        bm.ref_count[new_blocks[-1]] = 1
    table_b = hits_b + new_blocks
    bm.block_tables[2] = table_b
    parent = None
    for i in range(len(USER_B) // bm.block_size):
        chunk = USER_B[i * bm.block_size:(i + 1) * bm.block_size]
        h = cache.register_block(parent, chunk, table_b[i])
        parent = h
    print(f"  B block_table: {table_b}")
    print(f"  ref_count[0]={bm.ref_count[hits_b[0]]} (A,B 共享)")
    print(f"  cache stats:   {cache.stats()}")

    # ---- 用户 C 入场, 第 1 个 block 命中, 第 2 个不命中 ----------- #
    print("\n[3] User C 入场, 共享前 4 token, 第 2 块就分叉")
    hits_c, n_hit_c = cache.match_prefix(USER_C)
    kv("命中 block", hits_c)
    kv("命中 token 数", n_hit_c)
    print(f"  → 后半 token 内容不同 (USER_A=20,21..., USER_C=99,99...), miss")

    # ---- 节省统计 -------------------------------------------------- #
    banner("节省 vs naive (无 cache)")
    naive_blocks = sum((len(u) + bm.block_size - 1) // bm.block_size
                       for u in [USER_A, USER_B, USER_C])
    actual_blocks = bm.used_blocks()
    kv("naive 总分配 block", naive_blocks)
    kv("actual 占用 block", actual_blocks)
    kv("节省比例", f"{100 * (naive_blocks - actual_blocks) / naive_blocks:.1f}%")


if __name__ == "__main__":
    main()
