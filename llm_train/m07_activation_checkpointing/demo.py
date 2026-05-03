"""
M07 — Activation Checkpointing

反向传播需要前向激活。默认每层都存, 显存随层数线性增长。
checkpointing 只存少量边界激活, 反向时重新跑局部前向:
    显存下降, 计算量上升。
"""
from __future__ import annotations

import numpy as np

from llm_train.core import banner, kv


def forward_range(x, weights, start: int, end: int, counter: dict) -> np.ndarray:
    h = x
    for i in range(start, end):
        counter["calls"] += 1
        h = np.tanh(h @ weights[i])
    return h


def main() -> None:
    banner("M07 - Activation Checkpointing")

    rs = np.random.RandomState(7)
    layers = 8
    hidden = 64
    segment = 2
    x = rs.randn(2, hidden).astype(np.float32)
    weights = [(rs.randn(hidden, hidden) * 0.02).astype(np.float32) for _ in range(layers)]

    # Naive forward: store every layer output for backward.
    naive_counter = {"calls": 0}
    activations = [x]
    h = x
    for i in range(layers):
        h = forward_range(h, weights, i, i + 1, naive_counter)
        activations.append(h)
    naive_saved = len(activations)

    # Checkpoint forward: store only segment boundaries.
    ckpt_counter = {"calls": 0}
    checkpoints = {0: x}
    h = x
    for start in range(0, layers, segment):
        h = forward_range(h, weights, start, min(start + segment, layers), ckpt_counter)
        checkpoints[min(start + segment, layers)] = h
    ckpt_saved = len(checkpoints)

    # During backward, recompute each segment's internal activations.
    for start in range(0, layers, segment):
        _ = forward_range(checkpoints[start], weights, start, min(start + segment, layers), ckpt_counter)

    naive_bytes = naive_saved * x.nbytes
    ckpt_bytes = ckpt_saved * x.nbytes

    kv("layers", layers)
    kv("checkpoint segment size", segment)
    kv("naive saved activations", naive_saved)
    kv("checkpoint saved activations", ckpt_saved)
    kv("activation memory", f"{naive_bytes} -> {ckpt_bytes} bytes")
    kv("forward layer calls", f"{naive_counter['calls']} -> {ckpt_counter['calls']}")

    assert np.allclose(activations[-1], checkpoints[layers])
    print("\n  OK: 激活重算用额外前向计算换显存, 是长序列/深层模型的常规手段。")


if __name__ == "__main__":
    main()

