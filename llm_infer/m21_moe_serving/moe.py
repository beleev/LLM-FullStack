"""
moe.py — MoE serving: top-k 路由 + 专家并行 (EP) dispatch/combine + EPLB 冗余专家负载均衡

是什么: 每个 token 只过 top-k 个专家 FFN; 专家分布在多个 rank 上, token 经 all-to-all 发到专家所在
        rank (dispatch), 算完再 all-to-all 发回并按 gate 加权求和 (combine)。
瓶颈:   吞吐 / 延迟。EP 每层都要同步, 一步的耗时 = 最慢 rank 的耗时 → 看 max/mean rank 负载, 不看平均。
关键数字: 路由倾斜时热专家所在 rank 负载可达均值的 2~3 倍; EPLB 把热专家复制到空闲 slot 并拆分其
        token, 可把 max/mean 压回 ≈1.0x, 且模型输出逐位不变 (副本权重相同)。
盯住:   ep_forward 里的 slot_of (每个 (token, k) 分配去哪个 slot) 和 rank_load。
真实系统: DeepSeek-V3 的 DeepEP (dispatch/combine kernel) + EPLB; vLLM / SGLang 的 --enable-eplb。
注意:   all-to-all 用数组分组模拟, 没有真实通信; "负载" = 每 rank 处理的 token 数。
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List

import numpy as np

from llm_infer.core import softmax
from llm_infer.core.utils import silu


@dataclass
class MoELayer:
    w_router: np.ndarray       # (D, E)
    router_bias: np.ndarray    # (E,)  demo 用它制造热专家 (Zipf 偏置)
    w_gate: np.ndarray         # (E, D, H)
    w_up: np.ndarray           # (E, D, H)
    w_down: np.ndarray         # (E, H, D)
    top_k: int

    @property
    def n_expert(self) -> int:
        return self.w_router.shape[1]

    def route(self, x):
        """x (T, D) → (expert ids (T, k), gates (T, k)); gate = 只在被选中的 k 个 logit 上做 softmax。"""
        logits = x @ self.w_router + self.router_bias                     # (T, E)
        idx = np.argsort(-logits, axis=-1, kind="stable")[:, :self.top_k]  # (T, k)
        return idx, softmax(np.take_along_axis(logits, idx, -1))

    def expert(self, e: int, x):
        """第 e 个专家的 SwiGLU FFN: x (n, D) → (n, D)。"""
        return (silu(x @ self.w_gate[e]) * (x @ self.w_up[e])) @ self.w_down[e]


def make_layer(D=32, H=64, E=16, top_k=2, zipf_s=1.0, seed=0) -> MoELayer:
    rs = np.random.RandomState(seed)
    r = lambda *s: (rs.randn(*s) / np.sqrt(s[-2])).astype(np.float32)
    bias = (zipf_s * -np.log(np.arange(1, E + 1))).astype(np.float32)     # 专家 e 的先验 ∝ (e+1)^-s
    return MoELayer(r(D, E), bias, r(E, D, H), r(E, D, H), r(E, H, D), top_k)


def dense_reference(layer: MoELayer, x):
    """朴素基线: 逐 token、逐被选专家循环。"""
    idx, gates = layer.route(x)
    out = np.zeros_like(x)
    for t in range(x.shape[0]):
        for e, g in zip(idx[t], gates[t]):
            out[t] += g * layer.expert(e, x[t:t + 1])[0]
    return out


@dataclass
class Placement:
    slot_expert: np.ndarray    # (S,) 每个物理 slot 装哪个逻辑专家 (S ≥ E, 多出来的是冗余副本)
    slot_rank: np.ndarray      # (S,) 每个 slot 在哪个 rank


def contiguous_placement(n_expert: int, n_rank: int) -> Placement:
    """默认放法: 专家 0..E/R-1 在 rank 0, 依此类推 (无冗余)。"""
    e = np.arange(n_expert)
    return Placement(e, e // (n_expert // n_rank))


def eplb_placement(expert_load: np.ndarray, n_rank: int, n_redundant: int) -> Placement:
    """EPLB 式两步贪心: ① 副本数 — 反复给 "每副本负载" 最大的专家加一个副本;
    ② 放置 — slot 按每副本负载从大到小, 依次放到当前最轻且还有空位的 rank。"""
    E = len(expert_load)
    n_rep = np.ones(E, int)
    for _ in range(n_redundant):
        n_rep[np.argmax(expert_load / n_rep)] += 1
    slot_expert = np.repeat(np.arange(E), n_rep)                          # (S,)
    slot_load = (expert_load / n_rep)[slot_expert]                        # 副本间均分 token
    cap = len(slot_expert) // n_rank
    assert cap * n_rank == len(slot_expert)
    slot_rank = np.zeros(len(slot_expert), int)
    rank_load, rank_free = np.zeros(n_rank), np.full(n_rank, cap)
    for s in np.argsort(-slot_load, kind="stable"):
        r = np.argmin(np.where(rank_free > 0, rank_load, np.inf))
        slot_rank[s] = r
        rank_load[r] += slot_load[s]
        rank_free[r] -= 1
    return Placement(slot_expert, slot_rank)


def ep_forward(layer: MoELayer, x, pl: Placement, n_rank: int):
    """专家并行前向 (all-to-all 用分组模拟) → (out (T, D), rank_load (R,))。"""
    T, k = x.shape[0], layer.top_k
    idx, gates = layer.route(x)
    tok = np.repeat(np.arange(T), k)                                      # (T·k,) 每条分配属于哪个 token
    exp, gate = idx.reshape(-1), gates.reshape(-1)                        # (T·k,)

    # 逻辑专家 → 物理 slot: 有副本的专家把自己的 token 轮流分给各副本
    slot_of = np.empty(T * k, int)
    for e in range(layer.n_expert):
        mine = np.flatnonzero(exp == e)
        replicas = np.flatnonzero(pl.slot_expert == e)
        slot_of[mine] = replicas[np.arange(len(mine)) % len(replicas)]

    out = np.zeros_like(x)
    rank_load = np.zeros(n_rank, int)
    for r in range(n_rank):
        # dispatch: rank r 收到所有目的 slot 在自己身上的分配 (真实系统: all-to-all 的 recv buffer)
        recv = np.flatnonzero(pl.slot_rank[slot_of] == r)
        rank_load[r] = len(recv)
        y = np.empty((len(recv), x.shape[1]), x.dtype)                    # (n_r, D)
        for s in np.flatnonzero(pl.slot_rank == r):                       # 本 rank 的每个 slot 批量算一次 FFN
            sel = slot_of[recv] == s
            y[sel] = layer.expert(pl.slot_expert[s], x[tok[recv[sel]]])
        # combine: 结果发回 token 原位置, 按 gate 加权累加 (同一 token 的 k 份来自不同 rank)
        np.add.at(out, tok[recv], gate[recv, None] * y)
    return out, rank_load
