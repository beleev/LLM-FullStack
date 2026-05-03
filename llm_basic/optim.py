"""
optim.py — 手写 Adam 优化器（pure numpy）。

为什么不用 SGD？
  对 Transformer 来说，SGD 收敛极慢；Adam 加上 1st/2nd moment + 偏置修正
  在小学习率下也能稳定收敛。代码就 30 行，写一遍最直观。

公式（Kingma & Ba 2014）：
  t       ← t + 1
  m_t     ← β1 * m_{t-1} + (1-β1) * g
  v_t     ← β2 * v_{t-1} + (1-β2) * g²
  m_hat   ← m_t / (1 - β1^t)
  v_hat   ← v_t / (1 - β2^t)
  W       ← W - lr * m_hat / (√v_hat + eps)
"""
from __future__ import annotations

from typing import Any

import numpy as np


def adam_init(W: dict[str, np.ndarray]) -> dict[str, Any]:
    """为每个参数初始化一阶/二阶矩为 0；步数 t=0。"""
    return {
        "m": {k: np.zeros_like(v) for k, v in W.items()},
        "v": {k: np.zeros_like(v) for k, v in W.items()},
        "t": 0,
    }


def adam_step(
    W: dict[str, np.ndarray],
    grads: dict[str, np.ndarray],
    state: dict[str, Any],
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """返回新的 (W, state)，不就地修改。"""
    t = state["t"] + 1
    new_m: dict[str, np.ndarray] = {}
    new_v: dict[str, np.ndarray] = {}
    new_W: dict[str, np.ndarray] = {}

    bc1 = 1.0 - beta1 ** t   # bias correction 1
    bc2 = 1.0 - beta2 ** t   # bias correction 2

    for k, w in W.items():
        g = grads[k]
        m = beta1 * state["m"][k] + (1.0 - beta1) * g
        v = beta2 * state["v"][k] + (1.0 - beta2) * g * g
        m_hat = m / bc1
        v_hat = v / bc2
        new_m[k] = m
        new_v[k] = v
        new_W[k] = w - lr * m_hat / (np.sqrt(v_hat) + eps)

    return new_W, {"m": new_m, "v": new_v, "t": t}
