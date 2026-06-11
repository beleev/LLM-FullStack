"""
M12 — Sequence / Context Parallelism (Ring Attention)

长上下文训练的新瓶颈: T=128K 时, 单层注意力的激活 (Q/K/V 与 softmax 中间量)
就能挤爆一张卡 —— 这次模型和 batch 都没问题, 是 **序列本身** 放不下。

序列并行 (Megatron CP / Ring Attention, Liu et al. 2023) 把序列切成 D 段:
    - 每张卡常驻自己那段的 Q_r / K_r / V_r  (显存 O(T/D))
    - 注意力需要"我的 Q × 所有人的 K/V" → K/V 块在 D 张卡之间按环传递
    - 每一步对收到的 KV 块算一次局部注意力, 用 **online softmax** 增量合并
      (与 FlashAttention 的分块技巧完全相同, 只是块来自别的卡)

    step 0:  卡 r 处理 KV_r       (自己的块)
    step 1:  卡 r 处理 KV_{r-1}   (邻居寄来的)
    ...
    step D-1: 每张卡都见过了完整序列, 但**任何时刻**只持有 1/D 的 KV

因果模型还能省一半: 块 j > 块 i 的 KV 对 Q_i 全不可见, 直接跳过。

说明: 本 demo 用 list 模拟 D 张卡, "环传递" 就是数组下标轮转。
"""
from __future__ import annotations

import numpy as np

from llm_train.core import banner, kv, max_abs_diff, set_seed


def full_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """单卡基线: 完整因果 softmax 注意力 (T×T 分数矩阵一次成形)。"""
    t, d = q.shape
    scores = q @ k.T / np.sqrt(d)
    scores[np.triu_indices(t, k=1)] = -np.inf      # 因果: 上三角屏蔽
    scores -= scores.max(axis=1, keepdims=True)
    w = np.exp(scores)
    return (w / w.sum(axis=1, keepdims=True)) @ v


def ring_attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, world: int
) -> tuple[np.ndarray, dict]:
    """
    Ring Attention: 序列均分到 world 张卡, KV 块沿环传递 world-1 次。

    每张卡维护 online-softmax 的三件套 (与 FlashAttention 相同):
        m     — 已见分数的逐行最大值   [t_local]
        denom — 归一化分母 (指数和)    [t_local]
        acc   — 未归一化的加权 V 累计  [t_local, d]
    收到新 KV 块时按公式增量合并, 数值上与一次性 softmax 完全等价。
    """
    t_total, d = q.shape
    t_local = t_total // world

    def blocks(a: np.ndarray) -> list[np.ndarray]:
        return [a[r * t_local : (r + 1) * t_local] for r in range(world)]

    q_blk, k_blk, v_blk = blocks(q), blocks(k), blocks(v)

    m = [np.full(t_local, -np.inf) for _ in range(world)]
    denom = [np.zeros(t_local) for _ in range(world)]
    acc = [np.zeros((t_local, d)) for _ in range(world)]
    comm_floats = 0
    skipped = 0

    for step in range(world):
        for r in range(world):
            src = (r - step) % world          # 本步处理来自 src 卡的 KV 块
            if src > r:
                skipped += 1                  # 因果性: 未来块整块不可见, 不算
                continue
            scores = q_blk[r] @ k_blk[src].T / np.sqrt(d)   # [t_local, t_local]
            if src == r:                      # 对角块: 块内仍要因果屏蔽
                scores[np.triu_indices(t_local, k=1)] = -np.inf

            # ---- online softmax 增量合并 ----
            m_new = np.maximum(m[r], scores.max(axis=1))
            scale = np.exp(m[r] - m_new)                  # 旧累计量的修正系数
            p = np.exp(scores - m_new[:, None])
            denom[r] = denom[r] * scale + p.sum(axis=1)
            acc[r] = acc[r] * scale[:, None] + p @ v_blk[src]
            m[r] = m_new
        if step < world - 1:
            comm_floats += world * t_local * d * 2        # 每步全环传一轮 K+V

    out = np.concatenate([acc[r] / denom[r][:, None] for r in range(world)])
    stats = {
        "comm_floats": comm_floats,
        "skipped_blocks": skipped,
        "peak_kv_per_dev": t_local * d * 2,
        "full_kv": t_total * d * 2,
    }
    return out, stats


def main() -> None:
    banner("M12 - Sequence Parallel / Ring Attention")

    rs = set_seed(11)
    world, t_total, d = 4, 32, 16
    q = rs.randn(t_total, d)
    k = rs.randn(t_total, d)
    v = rs.randn(t_total, d)

    base = full_attention(q, k, v)
    ring, stats = ring_attention(q, k, v, world)

    print("\n[1] 正确性 (online softmax 与一次性 softmax 数值等价)")
    kv("max |full - ring|", f"{max_abs_diff(base, ring):.2e}")
    assert max_abs_diff(base, ring) < 1e-9

    print("\n[2] 显存: 每张卡任何时刻只持有 1/D 的 KV")
    kv("完整 KV (单卡基线)", f"{stats['full_kv']} floats")
    kv(f"每卡常驻 KV (D={world})", f"{stats['peak_kv_per_dev']} floats (1/{world})")

    print("\n[3] 通信与因果跳过")
    kv("环传递轮数", f"{world - 1} (每步只和邻居收发, 可与计算重叠)")
    kv("KV 环传递总量", f"{stats['comm_floats']} floats")
    kv("因果性跳过的块", f"{stats['skipped_blocks']} / {world * world} (上三角整块不算)")

    print("\n  OK: 序列并行解决 '一条序列放不下一张卡' 的问题;")
    print("      online softmax 让分块合并精确无损 —— 同一个技巧, 在单卡内是")
    print("      FlashAttention (llm_infer/m11), 跨卡传块就是 Ring Attention。")


if __name__ == "__main__":
    main()
