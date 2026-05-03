"""
int8_weight.py — Per-channel 对称 INT8 权重量化 (RTN)

为什么 per-channel 而非 per-tensor?
    权重 W ∈ R^{D_in × D_out}, 不同输出通道 (列) 的数值范围差异很大;
    每列独立 scale 能显著降低量化误差。

存储:
    W_q   : int8     (D_in, D_out)
    scale : float32  (D_out,)        # 每列一个

解量化:
    W' = W_q * scale[None, :]        # 广播
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class QInt8Tensor:
    """容器: int8 权重 + 列方向 scale。"""
    q: np.ndarray            # int8, (D_in, D_out)
    scale: np.ndarray        # float32, (D_out,)

    def dequantize(self) -> np.ndarray:
        return self.q.astype(np.float32) * self.scale[None, :]

    def nbytes(self) -> int:
        return self.q.nbytes + self.scale.nbytes


def quantize_int8(W: np.ndarray) -> QInt8Tensor:
    """对称 RTN per-channel 量化: scale = max(|col|) / 127"""
    assert W.ndim == 2
    abs_max = np.max(np.abs(W), axis=0)              # (D_out,)
    abs_max = np.where(abs_max == 0, 1e-8, abs_max)  # 防 0
    scale = abs_max / 127.0
    q = np.round(W / scale[None, :]).astype(np.int32)
    q = np.clip(q, -127, 127).astype(np.int8)
    return QInt8Tensor(q=q, scale=scale.astype(np.float32))


def matmul_qint8(x: np.ndarray, qw: QInt8Tensor) -> np.ndarray:
    """y = x @ W; W 用 INT8 存储, 计算时融合 dequant: y = (x @ W_q) * scale。

    数学等价: x @ W = x @ (W_q * scale_diag) = (x @ W_q) * scale[None,:]
    收益: 中间 (x @ W_q) 仍是 int 累加, 写显存 / 读显存按 int8 算, 砍带宽。
    本演示用 numpy float 模拟, 不真省内存, 但接口与真实 INT8 GEMM 一致。
    """
    return (x @ qw.q.astype(np.float32)) * qw.scale[None, :]
