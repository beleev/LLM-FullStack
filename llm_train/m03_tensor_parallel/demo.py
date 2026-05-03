"""
M03 — Tensor Parallelism

把单层矩阵按维度切给多个 rank。这里演示 Megatron 常用模式:
    W1: column-parallel   X @ [W1_0 | W1_1]
    W2: row-parallel      [H_0, H_1] @ [W2_0; W2_1] -> all-reduce

前向和反向的梯度都与 dense MLP 对齐。
"""
from __future__ import annotations

import numpy as np

from llm_train.core import all_reduce_sum, banner, kv


def relu(x):
    return np.maximum(x, 0.0)


def dense_mlp_grads(x, target, W1, b1, W2, b2):
    z = x @ W1 + b1
    h = relu(z)
    out = h @ W2 + b2
    diff = out - target
    loss = float(np.mean(diff * diff))
    d_out = (2.0 / diff.size) * diff
    grads = {
        "W2": h.T @ d_out,
        "b2": np.sum(d_out, axis=0),
    }
    d_h = d_out @ W2.T
    d_z = d_h * (z > 0)
    grads["W1"] = x.T @ d_z
    grads["b1"] = np.sum(d_z, axis=0)
    return loss, grads, out


def main() -> None:
    banner("M03 - Tensor Parallel MLP")

    rs = np.random.RandomState(4)
    B, D, H, O = 3, 4, 8, 2
    world = 2
    x = rs.randn(B, D).astype(np.float32)
    target = rs.randn(B, O).astype(np.float32)
    W1 = (rs.randn(D, H) * 0.1).astype(np.float32)
    b1 = np.zeros(H, dtype=np.float32)
    W2 = (rs.randn(H, O) * 0.1).astype(np.float32)
    b2 = np.zeros(O, dtype=np.float32)

    dense_loss, dense_grads, dense_out = dense_mlp_grads(x, target, W1, b1, W2, b2)

    W1_shards = np.split(W1, world, axis=1)
    b1_shards = np.split(b1, world, axis=0)
    W2_shards = np.split(W2, world, axis=0)

    # Forward: each rank owns one hidden slice.
    z_shards = [x @ W1_s + b1_s for W1_s, b1_s in zip(W1_shards, b1_shards)]
    h_shards = [relu(z) for z in z_shards]
    partial_outs = [h @ W2_s for h, W2_s in zip(h_shards, W2_shards)]
    tp_out = all_reduce_sum(partial_outs)[0] + b2

    # Backward: output gradient is replicated after row-parallel all-reduce.
    diff = tp_out - target
    tp_loss = float(np.mean(diff * diff))
    d_out = (2.0 / diff.size) * diff
    grad_b2 = np.sum(d_out, axis=0)
    grad_W2_shards = [h.T @ d_out for h in h_shards]
    d_h_shards = [d_out @ W2_s.T for W2_s in W2_shards]
    d_z_shards = [dh * (z > 0) for dh, z in zip(d_h_shards, z_shards)]
    grad_W1_shards = [x.T @ dz for dz in d_z_shards]
    grad_b1_shards = [np.sum(dz, axis=0) for dz in d_z_shards]

    tp_grads = {
        "W1": np.concatenate(grad_W1_shards, axis=1),
        "b1": np.concatenate(grad_b1_shards, axis=0),
        "W2": np.concatenate(grad_W2_shards, axis=0),
        "b2": grad_b2,
    }

    kv("dense loss", f"{dense_loss:.6f}")
    kv("tp loss", f"{tp_loss:.6f}")
    kv("max output diff", f"{np.max(np.abs(dense_out - tp_out)):.2e}")
    kv("max grad W1 diff", f"{np.max(np.abs(dense_grads['W1'] - tp_grads['W1'])):.2e}")
    kv("max grad W2 diff", f"{np.max(np.abs(dense_grads['W2'] - tp_grads['W2'])):.2e}")
    kv("communication", "W2 row-parallel forward: 1 all-reduce; backward类似")

    assert np.allclose(dense_out, tp_out, atol=1e-6)
    assert np.allclose(dense_grads["W1"], tp_grads["W1"], atol=1e-6)
    assert np.allclose(dense_grads["W2"], tp_grads["W2"], atol=1e-6)
    print("\n  OK: 张量并行降低单卡参数/激活宽度, 代价是层内通信。")


if __name__ == "__main__":
    main()

