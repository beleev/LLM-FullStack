"""
M12 — 上下文并行 (Context Parallel): Ring Attention + zigzag 负载均衡

命名: 目录沿用 sequence_parallel, 但这里做的是 **context parallel** —— 在注意力内部按序列切。
      (Megatron 的 "sequence parallel" 指 TP 组内切 LayerNorm/Dropout 激活, 是另一回事;
       另一条按序列切的路线是 Ulysses, 见 m16。)
是什么: 序列切成 D 段, 每卡常驻自己的 Q/K/V; KV 块沿环传 D-1 次, 每到一块就用 online softmax 并入结果。
解决的瓶颈: 长序列的激活显存 (每卡 O(T/D))。通信是邻居间 P2P, 可与计算重叠。
关键公式: online softmax (同 FlashAttention):  m' = max(m, max s);  den' = den·e^{m-m'} + Σe^{s-m'};  acc 同理
因果负载: 连续切分时 rank r 只有 r+1 个块要算 → 每一轮都有人在算整块, 墙钟时间一点不省;
          zigzag 把序列切 2D 块, rank r 拿第 r 和第 2D-1-r 块 → 每卡每轮工作量完全相同。
读代码盯住: `work[step][rank]` (每轮每卡算了多少个 q·k 对) 和它的 "每轮取 max 再求和"。
"""
from __future__ import annotations

import numpy as np

from llm_train.core import banner, comm, kv, make_rng, max_abs_diff, ring_shift, softmax


def full_attention(q, k, v):
    """单卡基线: 一次成形的 [T, T] 因果注意力。"""
    t, d = q.shape
    scores = q @ k.T / np.sqrt(d)
    scores[np.triu_indices(t, k=1)] = -np.inf
    return softmax(scores) @ v


def shard_positions(t_total: int, world: int, zigzag: bool):
    """每个 rank 持有哪些 token 位置。"""
    if not zigzag:
        return np.split(np.arange(t_total), world)                       # D × [T/D], 连续
    chunks = np.split(np.arange(t_total), 2 * world)                     # 2D × [T/2D]
    return [np.concatenate([chunks[r], chunks[2 * world - 1 - r]]) for r in range(world)]   # 一头一尾配对


def ring_attention(q, k, v, world: int, zigzag: bool = False):
    t_total, d = q.shape
    pos = shard_positions(t_total, world, zigzag)
    q_loc = [q[p] for p in pos]                                          # D × [T/D, d], 常驻不动
    kv_blk = [np.concatenate([k[p], v[p]], axis=1) for p in pos]         # D × [T/D, 2d], 沿环流动
    kv_pos = [p.copy() for p in pos]                                     # KV 块自带位置, 因果 mask 要用

    n_loc = t_total // world
    m = [np.full(n_loc, -1e30) for _ in range(world)]                    # 已见分数的逐行最大值 (有限值, 防全 mask 行出 NaN)
    den = [np.zeros(n_loc) for _ in range(world)]                        # softmax 分母
    acc = [np.zeros((n_loc, d)) for _ in range(world)]                   # 未归一化的 Σ p·v
    work = np.zeros((world, world), dtype=int)                           # [step, rank] → 本轮算的 q·k 对数

    for step in range(world):
        for r in range(world):
            mask = pos[r][:, None] >= kv_pos[r][None, :]                 # [T/D, T/D] 因果: q 位置 ≥ k 位置
            if not mask.any():
                continue                                                 # 整块在未来: 跳过 (但这一轮别的卡还在算)
            work[step, r] = mask.sum()
            kb, vb = kv_blk[r][:, :d], kv_blk[r][:, d:]
            s = np.where(mask, q_loc[r] @ kb.T / np.sqrt(d), -np.inf)
            m_new = np.maximum(m[r], s.max(axis=1))
            fix = np.exp(m[r] - m_new)                                   # 旧累计量换到新基准
            p = np.exp(s - m_new[:, None])
            den[r] = den[r] * fix + p.sum(axis=1)
            acc[r] = acc[r] * fix[:, None] + p @ vb
            m[r] = m_new
        if step < world - 1:
            kv_blk = ring_shift(kv_blk)                                  # rank r 发给 r+1: [T/D, 2d]
            kv_pos = [kv_pos[(r - 1) % world] for r in range(world)]     # 位置元数据跟着走 (字节可忽略)

    out = np.empty_like(q)
    for r in range(world):
        out[pos[r]] = acc[r] / den[r][:, None]                           # 按原位置写回
    return out, work


def main() -> None:
    banner("M12 - Context Parallel (Ring Attention + zigzag)")

    rs = make_rng(11)
    world, T, d = 4, 32, 16
    q, k, v = rs.randn(T, d), rs.randn(T, d), rs.randn(T, d)
    base = full_attention(q, k, v)

    results = {}
    for name, zz in [("连续切分", False), ("zigzag", True)]:
        comm.reset()
        out, work = ring_attention(q, k, v, world, zigzag=zz)
        results[name] = (max_abs_diff(base, out), work, comm.total)

    print("\n[1] 正确性 (online softmax 跨块合并是精确的, 不是近似)")
    for name, (diff, _, _) in results.items():
        kv(f"max |full - ring| {name}", f"{diff:.1e}")
        assert diff < 1e-12

    print("\n[2] 因果注意力的每卡工作量 (q·k 对数; 行 = 环上第几轮, 列 = rank)")
    block = (T // world) ** 2
    for name, (_, work, _) in results.items():
        print(f"    {name}:")
        for step, row in enumerate(work):
            print(f"      step {step}: {row.tolist()}   本轮耗时 ∝ max = {row.max()}")
    wall = {name: int(work.max(axis=1).sum()) for name, (_, work, _) in results.items()}
    total = {name: int(work.sum()) for name, (_, work, _) in results.items()}
    per_rank = {name: work.sum(axis=0) for name, (_, work, _) in results.items()}
    kv("总计算量 (两者相同)", f"{total['连续切分']} 对 (非因果为 {T * T})")
    kv("每卡总工作量 连续", per_rank["连续切分"].tolist())
    kv("每卡总工作量 zigzag", per_rank["zigzag"].tolist())
    kv("墙钟 Σ_step max_rank", f"连续 {wall['连续切分']} → zigzag {wall['zigzag']}  ({wall['连续切分'] / wall['zigzag']:.2f}x)")
    kv("不利用因果性的墙钟", world * block)

    assert total["连续切分"] == total["zigzag"] == T * (T + 1) // 2
    assert wall["连续切分"] == per_rank["连续切分"].max() > 0.85 * world * block, \
        "连续切分: 计算省了一半, 墙钟却只省 ~10% —— 最后一张卡每轮都在算整块, 所有人等它"
    assert per_rank["zigzag"].max() == per_rank["zigzag"].min(), "zigzag: 每卡工作量完全相同"
    assert per_rank["连续切分"].max() > 3 * per_rank["连续切分"].min()
    assert wall["zigzag"] < 0.6 * wall["连续切分"]

    print("\n[3] 显存与通信")
    kv("每卡常驻 KV", f"{T // world * d * 2} / {T * d * 2} floats (1/{world})")
    kv("通信量", f"{results['zigzag'][2]:.0f} B/rank = (D-1) 轮 × 一个 KV 块; 与是否 zigzag 无关")
    assert results["zigzag"][2] == results["连续切分"][2] == (world - 1) * (T // world) * 2 * d * 8

    print("\n  OK: Ring Attention 精确等于完整注意力; 因果场景必须 zigzag (或 striped) 才能把省下的计算变成省下的时间。")


if __name__ == "__main__":
    main()
