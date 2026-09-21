"""
M09 — 通信原语 (Collectives)

是什么: 所有并行策略最终都落到 4 个原语上 ——
    all-reduce      每人一份 → 每人都拿到总和          DDP 梯度 / TP 激活
    reduce-scatter  每人一份 → 每人拿到总和的 1/N      ZeRO-2/3 梯度
    all-gather      每人 1/N → 每人都拿到完整          ZeRO-3/FSDP 参数
    all-to-all      每人给每人发一份不同的             MoE 路由 / Ulysses
解决的瓶颈: 通信带宽。ring 算法让 all-reduce 每 rank 只发 2(N-1)/N·S 字节, 与 N 几乎无关;
           朴素的 "汇总到 rank 0 再广播" 则让 rank 0 吞吐 2(N-1)·S, 随 N 线性爆炸。
关键恒等式: all-reduce == reduce-scatter + all-gather   (ZeRO 不比 DDP 多通信的原因)
读代码盯住: core/collectives.py 里 `ring_all_reduce_sum` 的两个阶段, 和 `comm` 计数器。
"""
from __future__ import annotations

import numpy as np

from llm_train.core import (
    all_gather, all_reduce_sum, all_to_all, banner, comm, kv, max_abs_diff,
    reduce_scatter_sum, ring_all_reduce_sum,
)


def main() -> None:
    banner("M09 - Distributed Collectives")

    N = 4
    ranks = [np.arange(8, dtype=np.float32) + 10 * r for r in range(N)]     # 每 rank [8]
    S = ranks[0].nbytes

    comm.reset()
    ar = all_reduce_sum(ranks)                                  # N × [8] -> N × [8]
    rs = reduce_scatter_sum(ranks)                              # N × [8] -> N × [2]
    ag = all_gather(rs)                                         # N × [2] -> N × [8]
    ring = ring_all_reduce_sum(ranks)

    kv("rank r 的输入", "arange(8) + 10r")
    kv("all_reduce", ar[0].tolist())
    kv("reduce_scatter (rank 0..3 各得)", [x.tolist() for x in rs])
    kv("all_gather(reduce_scatter)", ag[0].tolist())
    kv("ring all_reduce (逐步传 chunk)", ring[0].tolist())

    # all-to-all: rank s 给 rank d 发 [s*10+d]; 收到的是发送矩阵的转置
    send = [[np.array([s * 10 + d], dtype=np.float32) for d in range(N)] for s in range(N)]
    recv = all_to_all(send)
    kv("all_to_all: rank 1 发出 / 收到", f"{[int(t[0]) for t in send[1]]} / {[int(t[0]) for t in recv[1]]}")

    print(f"\n  每 rank 发送字节 (S = {S} B, N = {N}):")
    for op in ("all_reduce", "reduce_scatter", "all_gather", "ring_all_reduce", "all_to_all"):
        kv(f"  {op}", f"{comm.bytes[op]:.0f} B")
    naive_root = 2 * (N - 1) * S
    kv("  朴素 reduce→broadcast 的 rank 0", f"{naive_root} B  ({naive_root / comm.bytes['all_reduce']:.0f}x, 且随 N 线性增长)")

    assert all(max_abs_diff(x, ar[0]) == 0 for x in ar)
    assert max_abs_diff(ag[0], ar[0]) == 0, "all-reduce == reduce-scatter + all-gather"
    assert all(max_abs_diff(x, ar[0]) == 0 for x in ring), "ring 实现与直接求和一致"
    assert comm.bytes["ring_all_reduce"] == 2 * (N - 1) / N * S, "真实逐步传输的字节数 == 公式"
    assert comm.bytes["reduce_scatter"] + comm.bytes["all_gather"] == comm.bytes["all_reduce"]
    assert all(recv[d][s][0] == s * 10 + d for s in range(N) for d in range(N))
    print("\n  OK: ring all-reduce 实测字节 == 2(N-1)/N·S == reduce-scatter + all-gather。")


if __name__ == "__main__":
    main()
