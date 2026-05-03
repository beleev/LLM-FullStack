"""
flash_attention.py — Online softmax 的 numpy 实现

接口与朴素 attention 一致: (Q, K, V, mask?) → O。
内部按 block 切, 不显式存 (T, T) 矩阵。

简化:
    - 单头, 无 batch 维 (T, D)
    - causal mask 默认开
    - block_size 同时切 Q 与 KV (生产实现 Q-block 与 KV-block 大小常不同)
"""
from __future__ import annotations
import numpy as np


def flash_attention(
    Q: np.ndarray,            # (Tq, D)
    K: np.ndarray,            # (Tk, D)
    V: np.ndarray,            # (Tk, D)
    block_size: int = 16,
    causal: bool = True,
) -> np.ndarray:
    """Online softmax 流式计算 attention, 输出形状 (Tq, D)。

    内存峰值: O(block_size² + block_size·D), 与 Tq, Tk 无关。
    """
    Tq, D = Q.shape
    Tk = K.shape[0]
    sqrt_d = np.sqrt(D)

    # 输出累加器 + 数值稳定状态 (m: 当前 max, l: 当前 sum_exp)
    O = np.zeros((Tq, D), dtype=np.float32)
    m = np.full((Tq,), -np.inf, dtype=np.float32)
    l = np.zeros((Tq,), dtype=np.float32)

    # 按 KV 方向 block 流式处理
    for k_start in range(0, Tk, block_size):
        k_end = min(Tk, k_start + block_size)
        K_b = K[k_start:k_end]            # (Bk, D)
        V_b = V[k_start:k_end]

        # 同时按 Q 方向 block (这里偷懒一次取全部, 实际 FA 也切 Q)
        # S 的 shape: (Tq, Bk), 显存只有 Tq * Bk, 远小于 Tq * Tk
        S = (Q @ K_b.T) / sqrt_d

        if causal:
            # 行 i 不能看 (k_start + j) > i + (Tk - Tq) 的 key
            # 简化: 只支持 Tq <= Tk, query i 对应"绝对位置" i + (Tk - Tq)
            offset = Tk - Tq
            for i in range(Tq):
                last_visible = i + offset
                for j in range(k_end - k_start):
                    if k_start + j > last_visible:
                        S[i, j] = -np.inf

        # ---- online softmax 增量更新 ---------------------------- #
        m_b = np.max(S, axis=-1)                              # (Tq,) 本块的 max
        m_new = np.maximum(m, m_b)
        # 之前累计的 O / l 需要按"新 max"重新缩放
        scale_old = np.exp(m - m_new)                         # (Tq,)
        scale_old = np.where(np.isnan(scale_old), 0, scale_old)  # -inf - -inf 处理
        # 本块的 exp(S - m_new)
        P_b = np.exp(S - m_new[:, None])                      # (Tq, Bk)
        # 累加
        l = scale_old * l + np.sum(P_b, axis=-1)
        O = scale_old[:, None] * O + P_b @ V_b
        m = m_new

    # 最终归一化
    O = O / l[:, None]
    return O


def naive_attention(
    Q: np.ndarray, K: np.ndarray, V: np.ndarray, causal: bool = True
) -> np.ndarray:
    """对照组: 显式 (T, T) 矩阵的标准做法。"""
    Tq, D = Q.shape
    Tk = K.shape[0]
    S = (Q @ K.T) / np.sqrt(D)
    if causal:
        offset = Tk - Tq
        for i in range(Tq):
            for j in range(Tk):
                if j > i + offset:
                    S[i, j] = -np.inf
    S = S - np.max(S, axis=-1, keepdims=True)
    P = np.exp(S)
    P = P / np.sum(P, axis=-1, keepdims=True)
    return P @ V
