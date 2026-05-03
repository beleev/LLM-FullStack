"""
Small shared helpers.

The demos intentionally avoid framework magic.  A "parameter tree" is just a
dict[str, np.ndarray], and gradients use the same structure.
"""
from __future__ import annotations

from typing import Dict
import numpy as np


ArrayDict = Dict[str, np.ndarray]


def set_seed(seed: int = 0) -> np.random.RandomState:
    """Return a local RandomState so demos stay deterministic."""
    return np.random.RandomState(seed)


def banner(title: str, width: int = 72) -> None:
    pad = max(0, (width - len(title) - 2) // 2)
    line = "=" * width
    print(f"\n{line}\n{' ' * pad} {title}\n{line}")


def kv(key: str, value, indent: int = 2) -> None:
    print(f"{' ' * indent}{key:<34} = {value}")


def zeros_like(tree: ArrayDict) -> ArrayDict:
    return {k: np.zeros_like(v) for k, v in tree.items()}


def add_inplace(dst: ArrayDict, src: ArrayDict, scale: float = 1.0) -> None:
    for k in dst:
        dst[k] += src[k] * scale


def scaled(tree: ArrayDict, scale: float) -> ArrayDict:
    return {k: v * scale for k, v in tree.items()}


def global_norm(tree: ArrayDict) -> float:
    """L2 norm over all tensors in a gradient tree."""
    total = 0.0
    for v in tree.values():
        total += float(np.sum(v.astype(np.float64) ** 2))
    return float(np.sqrt(total))


def clip_by_global_norm(tree: ArrayDict, max_norm: float) -> tuple[ArrayDict, float, float]:
    """Return clipped gradients plus (old_norm, scale)."""
    norm = global_norm(tree)
    scale = min(1.0, max_norm / (norm + 1e-12))
    return scaled(tree, scale), norm, scale


def max_abs_diff(a, b) -> float:
    """Max absolute difference for arrays or dicts of arrays."""
    if isinstance(a, dict):
        return max(float(np.max(np.abs(a[k] - b[k]))) for k in a)
    return float(np.max(np.abs(a - b)))


def bytes_of(tree_or_array) -> int:
    if isinstance(tree_or_array, dict):
        return sum(v.nbytes for v in tree_or_array.values())
    if isinstance(tree_or_array, list):
        return sum(bytes_of(x) for x in tree_or_array)
    return int(tree_or_array.nbytes)

