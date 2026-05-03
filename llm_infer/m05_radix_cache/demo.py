"""
m05 demo — Radix Tree 前缀缓存

观察 4 件事:
    1) 不同请求的最长公共前缀, 树自动找出
    2) 部分匹配触发 split (m04 hash 永远做不到这个)
    3) ref_count 防止正在用的节点被驱逐
    4) LRU 驱逐
"""
from __future__ import annotations
from llm_infer.core.utils import banner, kv
from llm_infer.m05_radix_cache.radix_tree import RadixCache


def main():
    banner("M05 - Radix Cache (SGLang)")

    rc = RadixCache()

    # ---- 1) 第一条 ---------------------------------------------- #
    print("\n[1] insert SEQ_A = [1,2,3,4,5,6,7,8]")
    rc.insert([1, 2, 3, 4, 5, 6, 7, 8], kv_handle=100)
    print(rc.pretty())

    # ---- 2) SEQ_B 部分共享 → split ----------------------------- #
    print("\n[2] insert SEQ_B = [1,2,3,4,9,9]  (前 4 共享, 第 5 起分叉 → split)")
    rc.insert([1, 2, 3, 4, 9, 9], kv_handle=200)
    print(rc.pretty())
    print("  注意: 原节点被切成 [1,2,3,4] 和 [5,6,7,8] 两段, [9,9] 作新分支")

    # ---- 3) match 命中长度 ------------------------------------- #
    print("\n[3] match")
    for s in ([1, 2, 3, 4, 5, 6], [1, 2, 3, 9], [1, 2], [42]):
        node, n = rc.match(s)
        print(f"  query={s}  →  matched_len={n}")

    # ---- 4) lock 防止驱逐 + 驱逐 ------------------------------- #
    print("\n[4] lock SEQ_B 路径, 然后驱逐 1 个 → 应只能驱逐 SEQ_A 末段")
    node_b, _ = rc.match([1, 2, 3, 4, 9, 9])
    rc.lock_path(node_b)
    evicted = rc.evict_lru(k=1)
    kv("evicted kv_handles", evicted)
    print(rc.pretty())

    print("\n  → kv=100 ([5,6,7,8]) 被踢, kv=200 ([9,9]) 因为 lock 留下")


if __name__ == "__main__":
    main()
