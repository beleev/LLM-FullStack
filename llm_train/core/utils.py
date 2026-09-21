"""
各 demo 共用的小工具。

约定: "参数树" 就是 dict[str, np.ndarray], 梯度/优化器状态用同样的结构。
凡是两个以上模块要用的东西 (softmax、溢出检测、梯度平均、checkpoint 读写)
都放这里, demo 里不再各写一份。
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Union
import numpy as np

from .collectives import all_reduce_mean

ArrayDict = Dict[str, np.ndarray]


def make_rng(seed: int = 0) -> np.random.RandomState:
    """局部随机源。不碰 numpy 的全局种子, demo 之间互不影响。"""
    return np.random.RandomState(seed)


def banner(title: str, width: int = 72) -> None:
    pad = max(0, (width - len(title) - 2) // 2)
    line = "=" * width
    print(f"\n{line}\n{' ' * pad} {title}\n{line}")


def kv(key: str, value, indent: int = 2) -> None:
    print(f"{' ' * indent}{key:<34} = {value}")


# ---- 数学小件 ---------------------------------------------------------- #

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - x.max(axis=axis, keepdims=True)          # 减最大值防 exp 溢出
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


# ---- 参数树 ------------------------------------------------------------ #

def zeros_like(tree: ArrayDict) -> ArrayDict:
    return {k: np.zeros_like(v) for k, v in tree.items()}


def add_inplace(dst: ArrayDict, src: ArrayDict, scale: float = 1.0) -> None:
    for k in dst:
        dst[k] += src[k] * scale


def scaled(tree: ArrayDict, scale: float) -> ArrayDict:
    return {k: v * scale for k, v in tree.items()}


def flatten_tree(tree: ArrayDict) -> np.ndarray:
    """按 key 顺序拍平成一维向量 (ZeRO/FSDP 的 flat parameter)。"""
    return np.concatenate([v.reshape(-1) for v in tree.values()])


def unflatten_like(flat: np.ndarray, like: ArrayDict) -> ArrayDict:
    out, i = {}, 0
    for k, v in like.items():
        out[k] = flat[i : i + v.size].reshape(v.shape)
        i += v.size
    return out


def average_grad_trees(local_grads: List[ArrayDict]) -> ArrayDict:
    """DDP 梯度同步: 对每个参数做一次 all-reduce(mean), 返回 rank 0 的那份。"""
    return {
        name: all_reduce_mean([g[name] for g in local_grads])[0]
        for name in local_grads[0]
    }


def global_norm(tree: ArrayDict) -> float:
    """所有张量拼在一起的 L2 范数 (用 float64 累加, 防止平方和自己溢出)。"""
    return float(np.sqrt(sum(float(np.sum(v.astype(np.float64) ** 2)) for v in tree.values())))


def clip_by_global_norm(tree: ArrayDict, max_norm: float) -> tuple[ArrayDict, float, float]:
    """返回 (裁剪后的梯度, 裁剪前范数, 缩放系数)。方向不变, 只缩长度。"""
    norm = global_norm(tree)
    scale = min(1.0, max_norm / (norm + 1e-12))
    return scaled(tree, scale), norm, scale


def has_overflow(grads: Union[ArrayDict, Iterable[np.ndarray]]) -> bool:
    """任一梯度含 NaN/Inf 即为坏 step。AMP 的 skip-step 和 NaN guard 共用这一个判断。"""
    arrays = grads.values() if isinstance(grads, dict) else grads
    return any(not np.isfinite(g).all() for g in arrays)


def max_abs_diff(a, b) -> float:
    """数组或参数树的最大绝对误差。"""
    if isinstance(a, dict):
        return max(max_abs_diff(a[k], b[k]) for k in a)
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def bytes_of(obj) -> int:
    """数组 / dict / list 里所有 ndarray 的真实字节数 (显存账本用)。"""
    if isinstance(obj, dict):
        return sum(bytes_of(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(bytes_of(v) for v in obj)
    return int(obj.nbytes) if isinstance(obj, np.ndarray) else 0


# ---- checkpoint -------------------------------------------------------- #

def save_checkpoint(path: Path, payload: dict) -> None:
    """先写临时文件再原子 rename: 写到一半被杀也不会留下半个 checkpoint。"""
    tmp = Path(str(path) + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(payload, f)
    os.replace(tmp, path)


def load_checkpoint(path: Path) -> dict:
    # ponytail: pickle 只能读自己写的可信文件; 真实系统用 safetensors / DCP。
    with Path(path).open("rb") as f:
        return pickle.load(f)
