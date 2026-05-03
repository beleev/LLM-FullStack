"""
parallel_linear.py — 单进程模拟的 column / row parallel 线性层

约定:
    单卡:    Y = X @ W,  X (B,D_in)  W (D_in,D_out)  Y (B,D_out)
    column:  W → [W_1 | W_2 | ... | W_N] split axis=1
    row:     W → [W_1 ; W_2 ; ... ; W_N] split axis=0
"""
from __future__ import annotations
from typing import List
import numpy as np


# --------------------------------------------------------------------- #
# 切分                                                                  #
# --------------------------------------------------------------------- #

def split_column(W: np.ndarray, N: int) -> List[np.ndarray]:
    """按输出维 (axis=1) 切 N 份。"""
    assert W.shape[1] % N == 0, f"D_out={W.shape[1]} 不能被 N={N} 整除"
    return list(np.split(W, N, axis=1))


def split_row(W: np.ndarray, N: int) -> List[np.ndarray]:
    """按输入维 (axis=0) 切 N 份。"""
    assert W.shape[0] % N == 0, f"D_in={W.shape[0]} 不能被 N={N} 整除"
    return list(np.split(W, N, axis=0))


def split_x_for_row(x: np.ndarray, N: int) -> List[np.ndarray]:
    """row-parallel 的 X 也要按 D_in 切; 真实分布式里这步是上一层 column 的天然产出。"""
    return list(np.split(x, N, axis=-1))


# --------------------------------------------------------------------- #
# 前向                                                                  #
# --------------------------------------------------------------------- #

def column_parallel_linear(x: np.ndarray, W_shards: List[np.ndarray]) -> List[np.ndarray]:
    """每个 rank 持有 W_shards[i], 全员算 x @ W_i, 输出 list 长度 = N。

    通信: 0 次 (输出按 D_out 维分散, 下层若是 row-parallel 直接对接)。
    """
    return [x @ W for W in W_shards]


def row_parallel_linear(
    x_shards: List[np.ndarray], W_shards: List[np.ndarray]
) -> np.ndarray:
    """每个 rank 持有 x_shards[i] 与 W_shards[i] (按 D_in 切), 算部分和后 all-reduce。

    通信: 1 次 all-reduce (本演示用 sum 模拟)。
    """
    partial = [x_i @ W_i for x_i, W_i in zip(x_shards, W_shards)]
    return all_reduce_sum(partial)


# --------------------------------------------------------------------- #
# 通信原语 (单进程模拟)                                                  #
# --------------------------------------------------------------------- #

def all_reduce_sum(tensors: List[np.ndarray]) -> np.ndarray:
    """模拟 NCCL all-reduce: 每个 rank 持有一个张量, 结束后所有 rank 拿到 sum。"""
    out = tensors[0].copy()
    for t in tensors[1:]:
        out = out + t
    return out


def all_gather(tensors: List[np.ndarray], axis: int = -1) -> np.ndarray:
    """模拟 all-gather: 每 rank 出一片, 拼起来。"""
    return np.concatenate(tensors, axis=axis)
