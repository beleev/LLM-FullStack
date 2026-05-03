"""
M02 — Data Parallel / DDP

每张卡保留完整模型副本, 处理不同数据 shard, 反向后 all-reduce 梯度。
真实 PyTorch DDP 还会把梯度按 bucket 分组以重叠通信和反向计算。

说明: 本 demo 在单进程内用 N 个 list 元素模拟 N 个 rank, 通信原语
      (all-reduce) 是纯 numpy 求和。真实分布式场景需 NCCL/Gloo 后端。
"""
from __future__ import annotations

import numpy as np

from llm_train.core import LinearModel, all_reduce_mean, banner, kv, max_abs_diff


def _average_grad_trees(local_grads):
    return {
        name: all_reduce_mean([g[name] for g in local_grads])[0]
        for name in local_grads[0]
    }


def main() -> None:
    banner("M02 - Data Parallel / DDP")

    rs = np.random.RandomState(2)
    world_size = 4
    batch = 16
    model = LinearModel.init(d_in=6, d_out=2, seed=3)
    replicas = [model.copy() for _ in range(world_size)]
    x = rs.randn(batch, 6).astype(np.float32)
    y = rs.randn(batch, 2).astype(np.float32)

    # Single-process baseline on the whole batch.
    dense = model.copy()
    _, dense_grads = dense.loss_and_grads(x, y)
    dense.apply_grads(dense_grads, lr=0.1)

    # DDP: split batch, compute local grads, all-reduce mean, update replicas.
    local_grads = []
    for rank, replica in enumerate(replicas):
        xb = np.array_split(x, world_size, axis=0)[rank]
        yb = np.array_split(y, world_size, axis=0)[rank]
        _, grads = replica.loss_and_grads(xb, yb)
        local_grads.append(grads)

    synced_grads = _average_grad_trees(local_grads)
    for replica in replicas:
        replica.apply_grads(synced_grads, lr=0.1)

    diff = max_abs_diff(dense.W, replicas[0].W)
    grad_bytes = sum(v.nbytes for v in synced_grads.values())
    kv("world_size", world_size)
    kv("per-rank batch", batch // world_size)
    kv("gradient all-reduce payload", f"{grad_bytes} bytes / step")
    kv("max |single - ddp| after step", f"{diff:.2e}")

    assert diff < 1e-6
    print("\n  OK: DDP 扩大吞吐, 代价是每步同步一次全模型梯度。")


if __name__ == "__main__":
    main()

