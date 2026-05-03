"""
M05 — ZeRO / FSDP

DDP 每张卡都有完整 params / grads / optimizer states。ZeRO 和 FSDP 的核心是:
    - ZeRO-1: optimizer states 分片
    - ZeRO-2: optimizer states + gradients 分片
    - ZeRO-3/FSDP: params + gradients + optimizer states 都分片

本 demo 用一个向量参数演示 reduce-scatter 梯度 + 本地 shard 更新 + all-gather 参数。
"""
from __future__ import annotations

import numpy as np

from llm_train.core import all_gather, reduce_scatter_sum, banner, kv


def memory_units(num_params: int, world: int):
    # Count arrays, not bytes: param + grad + Adam m + Adam v.
    full = num_params
    shard = num_params / world
    return {
        "DDP": full + full + full + full,
        "ZeRO-1": full + full + shard + shard,
        "ZeRO-2": full + shard + shard + shard,
        "ZeRO-3/FSDP": shard + shard + shard + shard,
    }


def main() -> None:
    banner("M05 - ZeRO / FSDP Sharding")

    world = 4
    lr = 0.1
    param = np.arange(16, dtype=np.float32)
    grad_per_rank = [np.ones_like(param) * (r + 1) for r in range(world)]

    dense_grad = sum(grad_per_rank)
    dense_updated = param - lr * dense_grad

    # ZeRO-2/3 gradient path: sum gradients, then scatter one shard per rank.
    grad_shards = reduce_scatter_sum(grad_per_rank, axis=0)
    param_shards = list(np.split(param, world, axis=0))
    updated_shards = [p - lr * g for p, g in zip(param_shards, grad_shards)]

    # FSDP gathers full params before a layer forward; after update it can keep shards.
    gathered = all_gather(updated_shards, axis=0)[0]

    print("Per-rank memory units for 16 params, Adam optimizer:")
    for name, units in memory_units(num_params=16, world=world).items():
        kv(name, f"{units:.1f} floats")

    kv("dense updated param", dense_updated.tolist())
    kv("sharded updated param", gathered.tolist())
    kv("max update diff", f"{np.max(np.abs(dense_updated - gathered)):.2e}")
    kv("key collectives", "reduce-scatter gradients, all-gather params")

    assert np.allclose(dense_updated, gathered)
    print("\n  OK: ZeRO/FSDP 用通信换显存, 让更大的模型放进多卡集群。")


if __name__ == "__main__":
    main()

