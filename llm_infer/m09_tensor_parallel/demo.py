"""
m09 demo — Tensor Parallel: 一个完整 block (多头 attention + SwiGLU MLP), tp ∈ {1,2,4}

验证 (全部 assert):
    0) 两个恒等式: 列切 = concat, 行切 = sum
    1) tp_block 输出 vs 不切分的 dense_block: max-abs-diff < 1e-5
    2) 每个 block 恰好 2 次 all-reduce; 每卡权重 ≈ 1/tp
    3) 反例: 不按 head 边界切 (把单头硬切 4 份) 结果就错了
all-reduce 是单进程求和模拟, 只统计次数与字节, 不含真实通信耗时。
"""
from __future__ import annotations
import numpy as np

from llm_infer.core.utils import banner, kv
from llm_infer.m09_tensor_parallel.parallel_linear import (
    Comm, split_column, split_row, mha, dense_block, shard_weights, tp_block,
)

TOL = 1e-5


def main():
    banner("M09 - Tensor Parallel (MHA + SwiGLU, tp=1/2/4)")
    rs = np.random.RandomState(0)
    T, D, n_head, d_mlp = 16, 64, 8, 192
    rand = lambda *s: (rs.randn(*s) * 0.1).astype(np.float32)
    W = {"q": rand(D, D), "k": rand(D, D), "v": rand(D, D), "o": rand(D, D),
         "gate": rand(D, d_mlp), "up": rand(D, d_mlp), "down": rand(d_mlp, D),
         "ln1": 1 + rand(D), "ln2": 1 + rand(D)}
    x = rs.randn(T, D).astype(np.float32)
    kv("配置", f"T={T} D={D} n_head={n_head} (d_head={D // n_head}) d_mlp={d_mlp}")

    print("\n[0] 两个恒等式 (tp=4)")
    col = np.concatenate([x @ Wr for Wr in split_column(W["up"], 4)], axis=-1)       # 4×(T,d_mlp/4) → (T,d_mlp)
    act = rand(T, d_mlp)
    row = sum(a @ Wr for a, Wr in zip(np.split(act, 4, axis=-1), split_row(W["down"], 4)))   # Σ (T,d_mlp/4)@(d_mlp/4,D)
    d_col, d_row = np.max(np.abs(col - x @ W["up"])), np.max(np.abs(row - act @ W["down"]))
    kv("列切: concat_r(X@W_r) vs X@W", f"{d_col:.2e}")
    kv("行切: Σ_r X_r@W_r    vs X@W", f"{d_row:.2e}")
    assert d_col < TOL and d_row < TOL

    print("\n[1][2] tp_block vs dense_block")
    ref = dense_block(x, W, n_head)
    mat_bytes = sum(W[k].nbytes for k in W if W[k].ndim == 2)
    print(f"  {'tp':>3} {'max|Δ|':>10} {'all-reduce/block':>17} {'载荷 bytes':>11} {'ring 每卡发送':>13} {'每卡权重 bytes':>15} {'每卡头数':>8}")
    for tp in (1, 2, 4):
        ranks, comm = shard_weights(W, tp), Comm()
        out = tp_block(x, ranks, n_head, comm)
        diff = np.max(np.abs(out - ref))
        rank_bytes = sum(v.nbytes for v in ranks[0].values())
        ring = comm.payload_bytes * 2 * (tp - 1) // tp    # ring all-reduce: 每卡发送 2(tp-1)/tp × 载荷
        print(f"  {tp:>3} {diff:>10.2e} {comm.n_allreduce:>17} {comm.payload_bytes:>11,} {ring:>13,} {rank_bytes:>15,} {n_head // tp:>8}")
        assert diff < TOL
        assert comm.n_allreduce == (2 if tp > 1 else 0)
        assert comm.payload_bytes == comm.n_allreduce * T * D * 4         # 每次载荷 = 整个 (T,D) 激活
        assert rank_bytes == mat_bytes // tp + 2 * D * 4                  # 矩阵 1/tp, 两个 gamma 复制
    kv("不切分的权重 bytes", f"{mat_bytes + 2 * D * 4:,}")

    print("\n[3] 反例: 列切不落在 head 边界 → 单头被硬切成 4 个 '头', 各自 softmax")
    one_head = mha(x, W["q"], W["k"], W["v"], W["o"], n_head=1)
    cut_in_4 = mha(x, W["q"], W["k"], W["v"], W["o"], n_head=4)          # 等价于 tp=4 每卡对 D/4 维各做 softmax
    d_bad = np.max(np.abs(one_head - cut_in_4))
    kv("max|单头 - 硬切 4 份|", f"{d_bad:.2e}  ← softmax 不能跨 rank 拆开")
    assert d_bad > 100 * TOL

    print(f"\n  ✓ tp=1/2/4 输出与单卡一致 (< {TOL:g}, 差异只来自浮点求和顺序)")
    print("  ✓ 每 block 2 次 all-reduce, 载荷与 tp 无关 (都是 T×D 激活); 每卡权重 ≈ 1/tp")


if __name__ == "__main__":
    main()
