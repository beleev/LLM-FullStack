"""
多个 demo 共用的玩具模型 / 优化器 / 数据流。

模型只有一层线性 + MSE:  y_hat = x @ W + b。梯度手推, 不藏在 autograd 后面,
这样 "梯度在哪一步被切分 / 同步 / 缩放" 在代码里一眼可见。
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

    def loss_and_grads(
        self, x: np.ndarray, y: np.ndarray, loss_scale: float = 1.0
    ) -> Tuple[float, ArrayDict]:
        """返回 (未缩放的 loss, 乘了 loss_scale 的梯度)。dtype 跟随输入: 全 fp16 输入就是 fp16 反向。"""
        diff = self.forward(x) - y                                   # [B, d_out]
        loss = float(np.mean(diff.astype(np.float32) ** 2))          # loss 总是用 fp32 算
        # loss scaling 发生在反向的起点: 之后每个 fp16 中间梯度都被同比放大
        d_pred = (diff * np.float32(loss_scale * 2.0 / diff.size)).astype(diff.dtype)
        grads = {
            "W": x.T @ d_pred,
            "b": np.sum(d_pred, axis=0),
        }
        return loss, grads

    def apply_grads(self, grads: ArrayDict, lr: float) -> None:
        self.W -= lr * grads["W"]
        self.b -= lr * grads["b"]


def adam_update(p, g, m, v, t: int, lr: float, b1=0.9, b2=0.999, eps=1e-8, wd=0.0) -> None:
    """原地 Adam(W) 更新。全是逐元素运算 → 对任意切片单独做, 结果与整体做逐位相同,
    这正是 ZeRO 能把优化器状态按 rank 切开的数学前提。"""
    m[...] = b1 * m + (1 - b1) * g
    v[...] = b2 * v + (1 - b2) * g * g
    m_hat = m / (1 - b1**t)                                          # 偏差修正, t 从 1 开始
    v_hat = v / (1 - b2**t)
    p[...] = p - lr * (m_hat / (np.sqrt(v_hat) + eps) + wd * p)      # wd>0 即 AdamW (解耦衰减)


class MomentumSGD:
    """带状态 (velocity) 的最小优化器: 状态不存, resume 就对不上。"""

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
    """确定性回归数据流。可恢复状态 = (seed, cursor): 两者都要进 checkpoint。"""

    def __init__(self, d_in: int, d_out: int, batch_size: int, seed: int = 123):
        self.d_in = d_in
        self.d_out = d_out
        self.batch_size = batch_size
        self.cursor = 0
        self._set_seed(seed)

    def _set_seed(self, seed: int) -> None:
        self.seed = seed
        rs = np.random.RandomState(seed)
        self.true_W = (rs.randn(self.d_in, self.d_out) * 0.5).astype(np.float32)
        self.true_b = (rs.randn(self.d_out) * 0.1).astype(np.float32)

    def next_batch(self) -> tuple[np.ndarray, np.ndarray]:
        # 第 cursor 个 batch 只由 (seed, cursor) 决定 → 恢复这两个数就恢复了数据顺序
        rs = np.random.RandomState(self.seed + 10_000 + self.cursor)
        x = rs.randn(self.batch_size, self.d_in).astype(np.float32)
        y = x @ self.true_W + self.true_b
        self.cursor += 1
        return x, y.astype(np.float32)

    def state_dict(self) -> Dict[str, int]:
        return {"seed": self.seed, "cursor": self.cursor}

    def load_state_dict(self, state: Dict[str, int]) -> None:
        self._set_seed(int(state["seed"]))
        self.cursor = int(state["cursor"])

