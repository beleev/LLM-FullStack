"""
M02 — 数据并行 (DDP)

是什么: 每个 rank 一份完整模型, 各吃 batch 的 1/N, 反向后 all-reduce(mean) 梯度, 再各自 step。
解决的瓶颈: 吞吐 (算力线性扩展)。不省显存 —— 参数/梯度/优化器状态每卡一整份 (→ m05 ZeRO)。
关键数字: ring all-reduce 每 rank 每步发送 2(N-1)/N · |grad| 字节, 几乎不随 N 增长。
读代码盯住: `average_grad_trees` —— 它之后所有副本梯度相同, 所以参数永远保持一致。
"""
from __future__ import annotations

import numpy as np

from llm_train.core import LinearModel, average_grad_trees, banner, bytes_of, comm, kv, max_abs_diff


def main() -> None:
    banner("M02 - Data Parallel / DDP")

    rs = np.random.RandomState(2)
    world, batch, lr = 4, 16, 0.1
    model = LinearModel.init(d_in=6, d_out=2, seed=3)
    x = rs.randn(batch, 6).astype(np.float32)
    y = rs.randn(batch, 2).astype(np.float32)

    # 基线: 单卡吃整个 batch
    dense = model.copy()
    _, dense_grads = dense.loss_and_grads(x, y)
    dense.apply_grads(dense_grads, lr)

    # DDP: x [16,6] -> 4 × [4,6]; 每个 rank 只看到自己的那片
    replicas = [model.copy() for _ in range(world)]
    x_shards, y_shards = np.split(x, world), np.split(y, world)
    local_grads = [rep.loss_and_grads(xs, ys)[1] for rep, xs, ys in zip(replicas, x_shards, y_shards)]

    comm.reset()
    synced = average_grad_trees(local_grads)          # 等分 shard 时 mean(local mean) == global mean
    for rep in replicas:
        rep.apply_grads(synced, lr)

    grad_bytes = bytes_of(synced)
    kv("world_size / per-rank batch", f"{world} / {batch // world}")
    kv("local grad 彼此不同", f"{max_abs_diff(local_grads[0], local_grads[1]):.3f}")
    kv("梯度大小", f"{grad_bytes} B")
    kv("通信量", comm.summary())
    kv("max |single - ddp| after step", f"{max_abs_diff(dense.params(), replicas[0].params()):.2e}")

    assert max_abs_diff(dense.params(), replicas[0].params()) < 1e-6
    assert all(max_abs_diff(replicas[0].params(), r.params()) == 0 for r in replicas[1:]), "副本必须逐位一致"
    assert abs(comm.total - 2 * (world - 1) / world * grad_bytes) < 1e-9
    print("\n  OK: DDP == 单卡大 batch; 代价是每步 2(N-1)/N·|grad| 的 all-reduce。")
    print("      真实 DDP 还把梯度装桶 (bucket, 默认 25MB), 边反向边通信来隐藏这部分时间。")


if __name__ == "__main__":
    main()
