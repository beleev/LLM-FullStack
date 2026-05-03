"""
kv_quant.py — Per-token 对称 INT8 KV cache 量化

为什么 per-token?
    K, V 在序列方向变化平缓 (相邻 token 数值范围差异小),
    每个 token 一个 scale 已足够; 不必 per-channel (那样 scale 就太大了)。

存储:
    K_q   : int8     (T, D)
    scale : float32  (T,)

写入:
    把新算的 K_new (D,) quant → 追加到 K_q[t], scale[t] = ...

读取 (做 attention):
    需要 dequant 回 fp 算 softmax (整 INT 路径需特殊 kernel, 这里教学略过)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class QKVCache:
    """量化 K / V cache, 仅一层一头。"""
    d_model: int
    K_q: List[np.ndarray] = field(default_factory=list)   # 每元素 (D,) int8
    V_q: List[np.ndarray] = field(default_factory=list)
    K_scale: List[float] = field(default_factory=list)
    V_scale: List[float] = field(default_factory=list)

    def append(self, k: np.ndarray, v: np.ndarray) -> None:
        self.K_q.append(_quantize_vec(k))
        self.V_q.append(_quantize_vec(v))
        self.K_scale.append(_scale_of(k))
        self.V_scale.append(_scale_of(v))

    def get_full(self) -> tuple:
        """dequant 回 (K, V) ∈ fp32, 用于 attention。"""
        T = len(self.K_q)
        K = np.stack([self.K_q[t].astype(np.float32) * self.K_scale[t]
                      for t in range(T)])
        V = np.stack([self.V_q[t].astype(np.float32) * self.V_scale[t]
                      for t in range(T)])
        return K, V

    def nbytes(self) -> int:
        # int8 每个 D + 4 字节 scale; 双倍 (K, V)
        return 2 * (sum(a.nbytes for a in self.K_q) +
                    len(self.K_scale) * 4)


def _scale_of(x: np.ndarray) -> float:
    a = float(np.max(np.abs(x)))
    return (a / 127.0) if a > 0 else 1e-8


def _quantize_vec(x: np.ndarray) -> np.ndarray:
    s = _scale_of(x)
    q = np.round(x / s).astype(np.int32)
    return np.clip(q, -127, 127).astype(np.int8)
