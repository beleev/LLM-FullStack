"""
单进程模拟的通信原语: 一个 "rank" 就是 Python list 里的一个元素。

真实训练里它们是 NCCL 调用。这里语义相同 (谁最后拿到什么), 默认实现走捷径
(先求和再切), 但通信量按 **ring 算法每个 rank 实际发送的字节** 记到全局计数器
`comm` 上, 于是每个 demo 都能打印自己的通信成本:

    all-reduce      2(N-1)/N · S      (= reduce-scatter + all-gather)
    reduce-scatter   (N-1)/N · S
    all-gather       (N-1)/N · S      (S = 聚合后的完整张量字节数)
    all-to-all       实际发给别的 rank 的字节 (对角块留在本地, 不上网线)

`ring_all_reduce_sum` 是真的一步步传 chunk 的实现, 用来验证上面的公式。
"""
from __future__ import annotations

from collections import defaultdict
from typing import List
import numpy as np


class CommCounter:
    """每个 rank 平均发送的字节数, 按原语分类累计。"""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.bytes = defaultdict(float)
        self.calls = defaultdict(int)

    def add(self, op: str, nbytes: float) -> None:
        self.bytes[op] += nbytes
        self.calls[op] += 1

    @property
    def total(self) -> float:
        return float(sum(self.bytes.values()))

    def summary(self) -> str:
        if not self.bytes:
            return "0 B (无通信)"
        parts = [f"{op}×{self.calls[op]}={int(b)}B" for op, b in self.bytes.items()]
        return f"{int(self.total)} B/rank  [" + ", ".join(parts) + "]"


comm = CommCounter()


def all_reduce_sum(tensors: List[np.ndarray], ring: bool = False) -> List[np.ndarray]:
    """每个 rank 拿到所有 rank 的和。ring=True 走真实的环形实现 (结果相同)。"""
    if ring:
        return ring_all_reduce_sum(tensors)
    n = len(tensors)
    total = tensors[0].copy()
    for t in tensors[1:]:
        total = total + t
    comm.add("all_reduce", 2 * (n - 1) / n * total.nbytes)
    return [total.copy() for _ in tensors]


def all_reduce_mean(tensors: List[np.ndarray]) -> List[np.ndarray]:
    """DDP 梯度同步 = all-reduce(sum) / world_size。"""
    return [t / len(tensors) for t in all_reduce_sum(tensors)]


def ring_all_reduce_sum(tensors: List[np.ndarray]) -> List[np.ndarray]:
    """Ring all-reduce: 张量切 N 块, 每步每个 rank 只给右邻居发 1 块。

    阶段 1 reduce-scatter (N-1 步): 第 t 步 rank r 发 chunk (r-t)%N, 右邻居累加;
                                    结束时 rank r 持有 chunk (r+1)%N 的全局和。
    阶段 2 all-gather     (N-1 步): 把已归约好的 chunk 再绕环传一圈 (覆盖而非累加)。
    每 rank 共发 2(N-1) 块 = 2(N-1)/N · S 字节, 与 N 几乎无关 → 带宽最优。
    """
    n = len(tensors)
    shape = tensors[0].shape
    # chunks[r][c]: rank r 上的第 c 块, 形状约 [S/N]
    chunks = [np.array_split(t.reshape(-1).copy(), n) for t in tensors]
    sent = 0
    for phase in ("reduce", "gather"):
        for t in range(n - 1):
            shift = 0 if phase == "reduce" else 1
            msgs = [(r, (r + shift - t) % n) for r in range(n)]           # (发送方, chunk id)
            payload = [chunks[r][c].copy() for r, c in msgs]              # 同一步内先全部 "发出"
            for (r, c), buf in zip(msgs, payload):
                dst = (r + 1) % n
                chunks[dst][c] = chunks[dst][c] + buf if phase == "reduce" else buf
                sent += buf.nbytes
    comm.add("ring_all_reduce", sent / n)
    return [np.concatenate(ch).reshape(shape) for ch in chunks]


def all_gather(shards: List[np.ndarray], axis: int = 0) -> List[np.ndarray]:
    """每个 rank 拿到所有 shard 的拼接: N × [S/N, ...] -> [S, ...]。"""
    n = len(shards)
    full = np.concatenate(shards, axis=axis)
    comm.add("all_gather", (n - 1) / n * full.nbytes)
    return [full.copy() for _ in shards]


def reduce_scatter_sum(tensors: List[np.ndarray], axis: int = 0) -> List[np.ndarray]:
    """先求和, 再让 rank r 只留第 r 片: N × [S, ...] -> N × [S/N, ...]。"""
    n = len(tensors)
    total = tensors[0].copy()
    for t in tensors[1:]:
        total = total + t
    assert total.shape[axis] % n == 0, "reduce-scatter 要求该维能被 world 整除 (真实框架靠 padding)"
    comm.add("reduce_scatter", (n - 1) / n * total.nbytes)
    return list(np.split(total, n, axis=axis))


def all_to_all(shards_by_rank: List[List[np.ndarray]]) -> List[List[np.ndarray]]:
    """out[dst][src] = in[src][dst], 即 "发送矩阵" 的转置。MoE / Ulysses 的核心原语。"""
    world = len(shards_by_rank)
    assert all(len(row) == world for row in shards_by_rank)
    off_diag = sum(
        shards_by_rank[s][d].nbytes for s in range(world) for d in range(world) if s != d
    )
    comm.add("all_to_all", off_diag / world)
    return [[shards_by_rank[src][dst] for src in range(world)] for dst in range(world)]


def ring_shift(tensors: List[np.ndarray]) -> List[np.ndarray]:
    """环形 P2P: rank r 把自己的张量发给 r+1, 同时收 r-1 的 (Ring Attention 传 KV 用)。"""
    n = len(tensors)
    comm.add("ring_p2p", sum(t.nbytes for t in tensors) / n)
    return [tensors[(r - 1) % n] for r in range(n)]
