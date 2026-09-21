"""
m04 demo — Prefix Cache: 命中 / 分叉 / 释放后仍可命中 / 被覆盖才失效

运行: python -m llm_infer.m04_prefix_cache.demo
这里只演示索引与 block 生命周期; "命中的 token 真的不再做前向" 在 full_engine demo 里对账。
"""
from __future__ import annotations
from llm_infer.core.utils import banner, kv
from llm_infer.m02_paged_attention.block_manager import BlockManager
from llm_infer.m04_prefix_cache.prefix_cache import PrefixCache


def admit(bm: BlockManager, cache: PrefixCache, seq_id: int, ids):
    """一条请求进场的完整流程: 查前缀 → 命中的 share、其余新分配 → (假装算完 KV) → 登记。"""
    hits, n_hit = cache.match_prefix(ids)
    table = bm.allocate(seq_id, len(ids), shared=hits)
    cache.register(ids, table, num_computed=len(ids))
    return hits, n_hit, table


def main():
    banner("M04 - Prefix Cache (block 链式 hash)")
    bm = BlockManager(num_blocks=10, block_size=4)
    cache = PrefixCache(bm)

    SYSTEM = [10, 11, 12, 13, 20, 21, 22, 23]            # 8 token = 2 个完整 block
    USER_A = SYSTEM + [30, 31, 32, 33, 40, 41]
    USER_B = SYSTEM + [50, 51, 52, 53, 60, 61]
    USER_C = [10, 11, 12, 13, 99, 99, 99, 99, 5]          # 第 2 块就分叉

    print("\n[1] A 冷启动 → 全 miss")
    hits_a, n_a, table_a = admit(bm, cache, 1, USER_A)
    kv("A 命中 token / 页表", f"{n_a} / {table_a}")
    assert n_a == 0

    print("\n[2] B 共享 SYSTEM → 命中前 2 块, 物理 block 与 A 相同")
    hits_b, n_b, table_b = admit(bm, cache, 2, USER_B)
    kv("B 命中 token / 页表", f"{n_b} / {table_b}")
    kv("共享 block 的 ref_count", [bm.ref_count[b] for b in hits_b])
    assert n_b == 8 and table_b[:2] == table_a[:2] and bm.ref_count[table_a[0]] == 2

    print("\n[3] C 只有第 1 块相同 → 命中 4 token (链式 hash: 第 2 块内容不同, 后面全断)")
    hits_c, n_c, _ = admit(bm, cache, 3, USER_C)
    kv("C 命中 token", n_c)
    assert n_c == 4

    naive = sum(bm.blocks_needed(len(u)) for u in (USER_A, USER_B, USER_C))
    kv("占用 block: 无缓存 vs 实际", f"{naive} vs {bm.used_blocks()}")
    assert bm.used_blocks() == naive - 3                  # B 省 2 块, C 省 1 块

    print("\n[4] A、B、C 全部结束 → block 回到 free_list, 但内容和索引还在 → 新请求 D 照样命中")
    for sid in (1, 2, 3):
        bm.free(sid)
    assert bm.used_blocks() == 0
    hits_d, n_d, table_d = admit(bm, cache, 4, USER_A)
    kv("D (= A 的 prompt) 命中 token", n_d)
    assert n_d == 12 and table_d[:3] == table_a[:3] and all(bm.ref_count[b] == 1 for b in table_d)
    bm.free(4)

    print("\n[5] 来一条占满整个 pool 的请求 → 旧 block 被覆盖, 索引同步失效 (不会命中到脏数据)")
    big = list(range(1000, 1000 + 40))                    # 40 token = 10 block, 且 id ≥ 256
    _, n_big, _ = admit(bm, cache, 5, big)
    hits_e, n_e = cache.match_prefix(USER_A)
    kv("pool 被占满后 A 的 prompt 命中", n_e)
    kv("索引里的 block 数", cache.stats()["indexed_blocks"])
    assert n_big == 0 and n_e == 0
    assert all(bm.ref_count[b] > 0 or b in bm.free_list for b in cache.block_to_hash)
    assert set(cache.hash_to_block.values()) == set(cache.block_to_hash)
    print("  ✓ 缓存项寿命 = block 内容寿命: 释放 ≠ 失效, 覆盖才失效 (free_list 顺序即 LRU)")


if __name__ == "__main__":
    main()
