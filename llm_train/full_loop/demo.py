"""
Full Loop — Mini Distributed Training Loop

把前面模块组合成一个最小闭环:
    data parallel       batch 按 rank 切分
    gradient accumulation   rank 内按 micro-batch 累积
    mixed precision path    loss scale / unscale
    grad clipping           控制全局范数
    ZeRO-style update       参数 shard 本地更新后 all-gather
    checkpoint              保存最终状态

仍然是单进程 numpy 模拟, 但控制流对应真实训练脚本的主干。
"""
from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import numpy as np

from llm_train.core import LinearModel, ToyDataStream, all_reduce_mean, banner, clip_by_global_norm, kv
from llm_train.core.utils import add_inplace, zeros_like


def mse(model: LinearModel, x: np.ndarray, y: np.ndarray) -> float:
    diff = model.forward(x) - y
    return float(np.mean(diff * diff))


def average_grad_trees(local_grads):
    return {
        name: all_reduce_mean([g[name] for g in local_grads])[0]
        for name in local_grads[0]
    }


def zero_style_sgd_step(model: LinearModel, grads, lr: float, world_size: int) -> None:
    """Update parameter shards locally, then all-gather into a full model copy."""
    new_params = {}
    for name, param in model.params().items():
        param_shards = np.array_split(param, world_size, axis=0)
        grad_shards = np.array_split(grads[name], world_size, axis=0)
        updated = [p - lr * g for p, g in zip(param_shards, grad_shards)]
        new_params[name] = np.concatenate(updated, axis=0)
    model.load_params(new_params)


def distributed_step(
    replicas,
    x: np.ndarray,
    y: np.ndarray,
    lr: float,
    micro_batches_per_rank: int,
    loss_scale: float,
    max_grad_norm: float,
) -> tuple[float, float]:
    world = len(replicas)
    local_grads = []
    local_losses = []

    x_ranks = np.array_split(x, world, axis=0)
    y_ranks = np.array_split(y, world, axis=0)
    for rank, replica in enumerate(replicas):
        accum = zeros_like(replica.params())
        rank_x = x_ranks[rank]
        rank_y = y_ranks[rank]
        x_micros = np.array_split(rank_x, micro_batches_per_rank, axis=0)
        y_micros = np.array_split(rank_y, micro_batches_per_rank, axis=0)

        for xb, yb in zip(x_micros, y_micros):
            loss, grads = replica.loss_and_grads(xb, yb)
            local_losses.append(loss)

            # Simulate AMP: scale gradients during backward, then unscale before sync.
            scaled = {k: v * loss_scale for k, v in grads.items()}
            unscaled = {k: v / loss_scale for k, v in scaled.items()}
            add_inplace(accum, unscaled, scale=len(xb) / len(rank_x))

        local_grads.append(accum)

    synced = average_grad_trees(local_grads)
    clipped, grad_norm, _ = clip_by_global_norm(synced, max_norm=max_grad_norm)

    # One full model is reconstructed after the sharded optimizer step, then broadcast
    # back to all data-parallel replicas.
    zero_style_sgd_step(replicas[0], clipped, lr=lr, world_size=world)
    for replica in replicas[1:]:
        replica.load_params(replicas[0].params())

    return float(np.mean(local_losses)), grad_norm


def save_final_checkpoint(path: Path, model: LinearModel, step: int, data: ToyDataStream) -> None:
    payload = {
        "step": step,
        "model": {k: v.copy() for k, v in model.params().items()},
        "data_cursor": data.cursor,
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)


def main() -> None:
    banner("Full Loop - Mini Distributed Training")

    world = 2
    steps = 30
    lr = 0.12
    data = ToyDataStream(d_in=4, d_out=2, batch_size=16, seed=42)
    replicas = [LinearModel.init(4, 2, seed=0) for _ in range(world)]

    rs = np.random.RandomState(99)
    val_x = rs.randn(32, 4).astype(np.float32)
    val_y = val_x @ data.true_W + data.true_b
    start_loss = mse(replicas[0], val_x, val_y)

    for step in range(1, steps + 1):
        x, y = data.next_batch()
        train_loss, grad_norm = distributed_step(
            replicas,
            x,
            y,
            lr=lr,
            micro_batches_per_rank=2,
            loss_scale=2**10,
            max_grad_norm=1.0,
        )
        if step in {1, 10, 20, 30}:
            kv(f"step {step:02d}", f"train_loss={train_loss:.4f}, grad_norm={grad_norm:.3f}")

    end_loss = mse(replicas[0], val_x, val_y)

    with tempfile.TemporaryDirectory(prefix="llm_train_full_") as tmp:
        path = Path(tmp) / "final.pkl"
        save_final_checkpoint(path, replicas[0], step=steps, data=data)
        kv("checkpoint", str(path))

    kv("validation loss", f"{start_loss:.4f} -> {end_loss:.4f}")
    kv("combined techniques", "DDP + grad accumulation + AMP scale + clipping + ZeRO-style shard")

    assert end_loss < start_loss
    print("\n  OK: 这是一个可运行的分布式训练主循环骨架, 只是把多机多卡压缩成了 numpy 列表。")


if __name__ == "__main__":
    main()

