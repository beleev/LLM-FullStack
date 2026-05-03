"""
Tiny differentiable model used by multiple demos.

The model is a single linear layer trained with MSE:
    y_hat = x @ W + b

Gradients are derived by hand so distributed-training mechanics are not hidden
behind autograd.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np

from .utils import ArrayDict


@dataclass
class LinearModel:
    W: np.ndarray
    b: np.ndarray

    @classmethod
    def init(cls, d_in: int, d_out: int, seed: int = 0, scale: float = 0.02) -> "LinearModel":
        rs = np.random.RandomState(seed)
        return cls(
            W=(rs.randn(d_in, d_out) * scale).astype(np.float32),
            b=np.zeros((d_out,), dtype=np.float32),
        )

    def copy(self) -> "LinearModel":
        return LinearModel(self.W.copy(), self.b.copy())

    def params(self) -> ArrayDict:
        return {"W": self.W, "b": self.b}

    def load_params(self, params: ArrayDict) -> None:
        self.W[...] = params["W"]
        self.b[...] = params["b"]

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x @ self.W + self.b

    def loss_and_grads(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, ArrayDict]:
        pred = self.forward(x)
        diff = pred - y
        loss = float(np.mean(diff * diff))
        d_pred = (2.0 / diff.size) * diff
        grads = {
            "W": x.T @ d_pred,
            "b": np.sum(d_pred, axis=0),
        }
        return loss, grads

    def apply_grads(self, grads: ArrayDict, lr: float) -> None:
        self.W -= lr * grads["W"]
        self.b -= lr * grads["b"]


class MomentumSGD:
    """Small optimizer with state, useful for checkpoint/resume demos."""

    def __init__(self, params: ArrayDict, lr: float = 0.1, momentum: float = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocity: ArrayDict = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params: ArrayDict, grads: ArrayDict) -> None:
        for k in params:
            self.velocity[k] = self.momentum * self.velocity[k] + grads[k]
            params[k] -= self.lr * self.velocity[k]

    def state_dict(self) -> Dict[str, object]:
        return {
            "lr": self.lr,
            "momentum": self.momentum,
            "velocity": {k: v.copy() for k, v in self.velocity.items()},
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.lr = float(state["lr"])
        self.momentum = float(state["momentum"])
        velocity = state["velocity"]
        self.velocity = {k: v.copy() for k, v in velocity.items()}


class ToyDataStream:
    """Deterministic regression data stream with a resumable cursor."""

    def __init__(self, d_in: int, d_out: int, batch_size: int, seed: int = 123):
        self.d_in = d_in
        self.d_out = d_out
        self.batch_size = batch_size
        self.seed = seed
        self.cursor = 0
        rs = np.random.RandomState(seed)
        self.true_W = (rs.randn(d_in, d_out) * 0.5).astype(np.float32)
        self.true_b = (rs.randn(d_out) * 0.1).astype(np.float32)

    def next_batch(self) -> tuple[np.ndarray, np.ndarray]:
        # Use cursor-derived seeds so restoring cursor restores the exact data order.
        rs = np.random.RandomState(self.seed + 10_000 + self.cursor)
        x = rs.randn(self.batch_size, self.d_in).astype(np.float32)
        y = x @ self.true_W + self.true_b
        self.cursor += 1
        return x, y.astype(np.float32)

    def state_dict(self) -> Dict[str, int]:
        return {"cursor": self.cursor}

    def load_state_dict(self, state: Dict[str, int]) -> None:
        self.cursor = int(state["cursor"])

