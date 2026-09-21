"""
int8_weight.py — RTN (round-to-nearest) 量化原语: 对称 INT8 per-channel + 通用非对称 n-bit 量化器

是什么: 把 fp 张量映射到 2^bits 个等间距格点, 只存整数 + 每组一个 scale (非对称再加一个 zero/最小值)。
解决的瓶颈: 显存容量, 以及 decode 阶段的显存带宽 (每步都要把全部权重 / KV 读一遍 → 字节少 = 更快)。
关键数字: 误差上界 = scale/2 = (组内 max-min) / (2·(2^bits-1)) → 组内有一个离群值, 同组所有数的精度一起变差。
    所以整个模块只在回答一个问题: **统计 min/max 的"组"怎么划** (`axis` 参数)。
读代码盯住: `quantize_affine` 的 `axis` —— kv_quant.py (per-token / per-channel) 与 int4_awq.py (group-wise)
    都只是用不同的 reshape + axis 调它。
对应真实系统: bitsandbytes / TensorRT-LLM weight-only INT8; vLLM 的 W8A16 / W4A16 kernel (Marlin 等)。
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class QInt8Tensor:
    q: np.ndarray            # int8, (D_in, D_out)
    scale: np.ndarray        # float32, (D_out,) 每个输出通道 (列) 一个

    def dequantize(self) -> np.ndarray:
        return self.q.astype(np.float32) * self.scale[None, :]

    def nbytes(self) -> int:
        return self.q.nbytes + self.scale.nbytes


def quantize_int8(W: np.ndarray) -> QInt8Tensor:
    """对称 per-channel: scale = max|列| / 127。权重近似零均值, 对称量化省掉 zero-point。"""
    abs_max = np.maximum(np.max(np.abs(W), axis=0), 1e-8)                 # (D_out,)
    scale = (abs_max / 127.0).astype(np.float32)
    q = np.clip(np.round(W / scale[None, :]), -127, 127).astype(np.int8)
    return QInt8Tensor(q=q, scale=scale)


def matmul_qint8(x: np.ndarray, qw: QInt8Tensor) -> np.ndarray:
    """y = x @ W ≈ (x @ W_q) * scale: per-channel scale 可以提到 matmul 外面 (每列一个标量)。

    weight-only: 激活仍是浮点, 算力没省; 省的是从显存读权重的字节数 (decode 是 memory-bound)。
    numpy 里 W_q 转回 float 再乘, 只为验证数值。
    """
    return (x @ qw.q.astype(np.float32)) * qw.scale[None, :]              # (B,D_in)@(D_in,D_out) * (1,D_out)


@dataclass
class QTensor:
    """非对称 n-bit: x ≈ q * scale + lo。scale/lo 用 fp16 存 (真实系统的常见选择)。"""
    q: np.ndarray            # uint8, 值域 [0, 2^bits-1]; 未做 bit-packing, nbytes() 按 pack 后计
    scale: np.ndarray        # float16, 形状 = x 在 axis 上 keepdims 归约后的形状
    lo: np.ndarray           # float16, 组内最小值 (等价于 zero-point)
    bits: int

    def dequantize(self) -> np.ndarray:
        return self.q.astype(np.float32) * self.scale.astype(np.float32) + self.lo.astype(np.float32)

    def nbytes(self) -> int:
        return -(-self.q.size * self.bits // 8) + self.scale.nbytes + self.lo.nbytes

    def bits_per_elem(self) -> float:
        """含 scale/lo 开销的等效 bit 数。"""
        return self.nbytes() * 8 / self.q.size


def quantize_affine(x: np.ndarray, bits: int, axis) -> QTensor:
    """在 `axis` 上统计 min/max (这些维上的元素共用一组 scale/lo), 非对称 RTN。axis=None → per-tensor。"""
    lo = x.min(axis=axis, keepdims=True).astype(np.float16)
    hi = x.max(axis=axis, keepdims=True)
    scale = np.maximum((hi - lo.astype(np.float32)) / (2 ** bits - 1), 1e-4).astype(np.float16)
    # 用存下来的 fp16 scale/lo 算 q, 这样 dequantize 与这里严格互逆
    q = np.round((x - lo.astype(np.float32)) / scale.astype(np.float32))
    return QTensor(np.clip(q, 0, 2 ** bits - 1).astype(np.uint8), scale, lo, bits)
