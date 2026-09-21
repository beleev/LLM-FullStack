"""
低精度浮点的 "假量化" (fake quantization): 把高精度数舍入到目标格式的可表示网格上。

numpy 没有 bfloat16 / FP8 / FP4 dtype, 但一个浮点格式只由三个数决定:
    m_bits  尾数位数 → normal 区相对精度 ≈ 2^-(m_bits+1)
    max_val 最大可表示值 → 超出即饱和 (FP8/FP4 习惯饱和而不是变 Inf)
    e_min   最小 normal 指数 → 低于 2^e_min 进入 subnormal (固定步长), 再小就冲成 0
m06 (BF16)、m13 (FP8)、m15 (FP4) 共用这一个函数。
"""
from __future__ import annotations

import numpy as np

FLOAT_FORMATS = {
    #        m_bits, max_val,        e_min
    "bf16": (7, 3.3895e38, -126),    # 指数与 FP32 相同 (8 位), 尾数只剩 7 位
    "e5m2": (2, 57344.0, -14),       # FP8 大范围
    "e4m3": (3, 448.0, -6),          # FP8 高精度 (448 而非 480: 最高编码留给 NaN)
    "e2m1": (1, 6.0, 0),             # FP4: 只有 ±{0, .5, 1, 1.5, 2, 3, 4, 6}
}


def fake_quant_float(x: np.ndarray, fmt: str) -> np.ndarray:
    """逐元素舍入到 fmt 的网格 (round-to-nearest-even, 上界饱和, subnormal, 下溢为 0)。"""
    m_bits, max_val, e_min = FLOAT_FORMATS[fmt]
    x = np.clip(np.asarray(x, dtype=np.float64), -max_val, max_val)
    mant, exp = np.frexp(x)                          # x = mant · 2^exp, mant ∈ [0.5, 1)
    grid = 2.0 ** (m_bits + 1)                       # 含隐含位共 m_bits+1 个有效二进制位
    q_normal = np.ldexp(np.round(mant * grid) / grid, exp)
    quantum = 2.0 ** (e_min - m_bits)                # subnormal 区固定步长
    q_sub = np.round(x / quantum) * quantum
    return np.where(np.abs(x) >= 2.0 ** e_min, q_normal, q_sub)


def quant_blockwise(x: np.ndarray, fmt: str, block: int) -> np.ndarray:
    """每 block 个元素共享一个 scale (= amax / max_val), 缩放到满量程后量化再还原。

    block >= x.size 时退化为 per-tensor scaling。scale 本身保持高精度
    (真实系统存 FP32 或 E8M0, 见 m15)。
    """
    max_val = FLOAT_FORMATS[fmt][1]
    flat = np.asarray(x, dtype=np.float64).reshape(-1)
    pad = (-flat.size) % block
    blocks = np.pad(flat, (0, pad)).reshape(-1, block)               # [n_blocks, block]
    scale = np.abs(blocks).max(axis=1, keepdims=True) / max_val      # [n_blocks, 1]
    scale[scale == 0] = 1.0
    deq = fake_quant_float(blocks / scale, fmt) * scale
    return deq.reshape(-1)[: flat.size].reshape(np.shape(x))
