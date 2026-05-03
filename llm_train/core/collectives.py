"""
Single-process simulations of distributed communication collectives.

In real training these are NCCL/RCCL/MPI calls.  Here every "rank" is simply an
item in a Python list, which keeps the data movement visible.
"""
from __future__ import annotations

from typing import List
import numpy as np


def all_reduce_sum(tensors: List[np.ndarray]) -> List[np.ndarray]:
    """Every rank starts with one tensor; every rank receives the sum."""
    total = np.zeros_like(tensors[0])
    for t in tensors:
        total = total + t
    return [total.copy() for _ in tensors]


def all_reduce_mean(tensors: List[np.ndarray]) -> List[np.ndarray]:
    """DDP gradient synchronization is usually all-reduce(sum) / world_size."""
    summed = all_reduce_sum(tensors)
    return [t / len(tensors) for t in summed]


def all_gather(shards: List[np.ndarray], axis: int = 0) -> List[np.ndarray]:
    """Every rank receives the concatenation of all shards."""
    full = np.concatenate(shards, axis=axis)
    return [full.copy() for _ in shards]


def reduce_scatter_sum(tensors: List[np.ndarray], axis: int = 0) -> List[np.ndarray]:
    """Sum all inputs, then split the result back across ranks."""
    total = all_reduce_sum(tensors)[0]
    assert total.shape[axis] % len(tensors) == 0
    return list(np.split(total, len(tensors), axis=axis))


def all_to_all(shards_by_rank: List[List[np.ndarray]]) -> List[List[np.ndarray]]:
    """Transpose a rank->destination matrix of shards.

    Used by MoE expert parallelism: each rank sends tokens to the rank that owns
    the selected expert, then receives tokens for its local experts.
    """
    world = len(shards_by_rank)
    assert all(len(row) == world for row in shards_by_rank)
    return [[shards_by_rank[src][dst] for src in range(world)] for dst in range(world)]

