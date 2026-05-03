"""
M09 — Communication Collectives

分布式训练里的高层技术大多落到少数通信原语:
    all-reduce      DDP 梯度同步
    reduce-scatter  ZeRO/FSDP 梯度分片
    all-gather      FSDP 参数聚合
    all-to-all      MoE expert parallel token routing
"""
from __future__ import annotations

import numpy as np

from llm_train.core import all_gather, all_reduce_sum, all_to_all, reduce_scatter_sum, banner, kv


def main() -> None:
    banner("M09 - Distributed Collectives")

    ranks = [np.array([r, r + 10, r + 20, r + 30], dtype=np.float32) for r in range(4)]
    ar = all_reduce_sum(ranks)
    rs = reduce_scatter_sum(ranks, axis=0)
    ag = all_gather(rs, axis=0)

    tokens = [
        [np.array([0]), np.array([1])],
        [np.array([2]), np.array([3])],
    ]
    routed = all_to_all(tokens)

    kv("rank inputs", [x.tolist() for x in ranks])
    kv("all_reduce_sum result", ar[0].tolist())
    kv("reduce_scatter_sum shards", [x.tolist() for x in rs])
    kv("all_gather(reduce_scatter)", ag[0].tolist())
    kv("all_to_all 2x2", [[x.tolist() for x in row] for row in routed])
    kv("mental model", "通信原语决定并行策略的主要成本")

    assert all(np.allclose(x, ar[0]) for x in ar)
    assert np.allclose(ag[0], ar[0])
    print("\n  OK: 看懂这四个原语, 就能读懂大多数训练并行论文/框架。")


if __name__ == "__main__":
    main()
