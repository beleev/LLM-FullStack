"""
M05 — ZeRO-1/2/3 与 FSDP

是什么: DDP 每卡都存一整份 "模型状态"; ZeRO 把它按 rank 切开, 每卡只留 1/N。
解决的瓶颈: 显存。混合精度 Adam 每个参数 16 字节 = 2 (fp16 参数) + 2 (fp16 梯度) + 12 (fp32 master+m+v):
    DDP 16Ψ → ZeRO-1  4Ψ+12Ψ/N → ZeRO-2  2Ψ+14Ψ/N → ZeRO-3/FSDP  16Ψ/N
代价: 通信。stage 1/2 与 DDP 同量级; stage 3 每层 fwd/bwd 都要 all-gather 参数, 约 1.5×。
读代码盯住: 每个 rank 的 `state` 字典里 **哪些数组是整份、哪些是 1/N** —— 显存账直接用它们的 nbytes 算。
为什么能切: Adam 逐元素更新, rank r 只更新第 r 片, 结果与整体更新逐位相同 (见 core.adam_update)。
"""
from __future__ import annotations

import numpy as np

from llm_train.core import (
    adam_update, all_gather, all_reduce_sum, banner, bytes_of, comm, kv, max_abs_diff,
    reduce_scatter_sum,
)

F16, F32 = np.float16, np.float32
LR = 1e-2


# ---- 2 层 tanh MLP, 参数按层存成一维向量 (FSDP 的 "flat parameter") ---- #

def layer_fwd(h, w_flat, shape):
    return np.tanh(h @ w_flat.reshape(shape))                        # [B, d_in] -> [B, d_out]


def layer_bwd(d_out, h_in, h_out, w_flat, shape):
    d_pre = d_out * (1 - h_out * h_out)                              # tanh' = 1 - tanh²
    return (h_in.T @ d_pre).reshape(-1), d_pre @ w_flat.reshape(shape).T     # (dW 拍平, d_h_in)


def local_grads(params16, shapes, x, y):
    """一个 rank 上的 fp16 前向+反向, 需要 **完整** 参数。返回每层的 fp16 梯度。"""
    hs = [x]
    for w, s in zip(params16, shapes):
        hs.append(layer_fwd(hs[-1], w, s))
    d = ((hs[-1] - y) * F16(2.0 / y.size)).astype(F16)
    grads = [None] * len(shapes)
    for i in reversed(range(len(shapes))):
        grads[i], d = layer_bwd(d, hs[i], hs[i + 1], params16[i], shapes[i])
    return grads


def shard(a, world, r):
    return np.split(a, world)[r].copy()                              # [S] -> [S/N]


def init_rank_state(master_full, world, r, stage):
    """stage 决定哪些东西存整份、哪些只存第 r 片。这就是 ZeRO 的全部 "数据结构"。"""
    cut = (lambda a: shard(a, world, r)) if stage >= 1 else (lambda a: a.copy())
    opt = [cut(w) for w in master_full]                              # fp32 master: stage>=1 起只留 1/N
    return {
        "p16": [(shard(w, world, r) if stage == 3 else w).astype(F16) for w in master_full],
        "g16": None,                                                 # 反向后填入
        "master": opt,
        "m": [np.zeros_like(w) for w in opt],
        "v": [np.zeros_like(w) for w in opt],
    }


def train(stage: int, master_full, shapes, xs, ys, steps: int):
    """stage 0 = DDP。返回 (最终完整 fp16 参数, 每 rank 常驻字节, stage 3 瞬时峰值字节, 每步通信字节)。"""
    world, L = len(xs), len(shapes)
    ranks = [init_rank_state(master_full, world, r, stage) for r in range(world)]
    resident = transient_peak = 0

    for t in range(1, steps + 1):
        # ---------------- 前向 + 反向 ---------------- #
        if stage < 3:
            g_local = [local_grads(rk["p16"], shapes, xs[r], ys[r]) for r, rk in enumerate(ranks)]
        else:
            # FSDP: 用到哪层才 all-gather 哪层 → 算完立刻 free; 瞬时多出的只是 "一层" 的完整参数
            hs = [[x] for x in xs]
            for i in range(L):
                full = all_gather([rk["p16"][i] for rk in ranks])             # N × [S/N] -> [S]
                transient_peak = max(transient_peak, full[0].nbytes)
                for r in range(world):
                    hs[r].append(layer_fwd(hs[r][-1], full[r], shapes[i]))
                del full                                                      # ← free: 前向后不保留完整参数
            d = [((hs[r][-1] - ys[r]) * F16(2.0 / ys[r].size)).astype(F16) for r in range(world)]
            g_local = [[None] * L for _ in range(world)]
            for i in reversed(range(L)):
                full = all_gather([rk["p16"][i] for rk in ranks])             # 反向再 gather 一次 (用通信换显存)
                for r in range(world):
                    g_local[r][i], d[r] = layer_bwd(d[r], hs[r][i], hs[r][i + 1], full[r], shapes[i])
                del full

        # ---------------- 梯度同步: 三个 stage 的分水岭 ---------------- #
        for i in range(L):
            layer_g = [g_local[r][i] for r in range(world)]                   # N × [S] fp16
            if stage <= 1:
                synced = all_reduce_sum(layer_g)                              # 每 rank 留整份梯度 [S]
            else:
                synced = reduce_scatter_sum(layer_g)                          # 每 rank 只留自己那片 [S/N]
            for r in range(world):
                g_local[r][i] = synced[r] / F16(world)
        for r, rk in enumerate(ranks):
            rk["g16"] = g_local[r]

        if t == steps:
            resident = bytes_of(ranks[0])                                     # 此刻参数/梯度/优化器状态都在

        # ---------------- 优化器: 只更新自己持有的那片 ---------------- #
        for r, rk in enumerate(ranks):
            for i in range(L):
                g = rk["g16"][i].astype(F32)
                if stage == 1:
                    g = shard(g, world, r)                                    # 整份梯度里只用到第 r 片
                adam_update(rk["master"][i], g, rk["m"][i], rk["v"][i], t, LR)

        # ---------------- 把新参数发回 fp16 计算副本 ---------------- #
        for i in range(L):
            if stage == 0:
                for rk in ranks:
                    rk["p16"][i] = rk["master"][i].astype(F16)
            elif stage < 3:
                full = all_gather([rk["master"][i].astype(F16) for rk in ranks])   # N × [S/N] -> [S]
                for r, rk in enumerate(ranks):
                    rk["p16"][i] = full[r]
            else:
                for rk in ranks:
                    rk["p16"][i] = rk["master"][i].astype(F16)               # stage 3 参数本来就只存分片

    wire = comm.total / steps                                         # 下面为了比对而做的 gather 不算训练通信
    if stage == 3:
        final = [all_gather([rk["p16"][i] for rk in ranks])[0] for i in range(L)]
    else:
        final = ranks[0]["p16"]
    return final, resident, transient_peak, wire


def dense_fp32_adam(master_full, shapes, x, y, steps):
    """单卡、全 fp32、整 batch 的教科书 Adam —— 最终的对照组。"""
    p = [w.copy() for w in master_full]
    m = [np.zeros_like(w) for w in p]
    v = [np.zeros_like(w) for w in p]
    for t in range(1, steps + 1):
        hs = [x]
        for w, s in zip(p, shapes):
            hs.append(layer_fwd(hs[-1], w, s))
        d = (hs[-1] - y) * F32(2.0 / y.size)
        for i in reversed(range(len(p))):
            g, d = layer_bwd(d, hs[i], hs[i + 1], p[i], shapes[i])
            adam_update(p[i], g, m[i], v[i], t, LR)
    return p


def main() -> None:
    banner("M05 - ZeRO-1/2/3 & FSDP")

    rs = np.random.RandomState(5)
    world, steps = 4, 5
    shapes = [(8, 16), (16, 4)]                                       # 两层: 128 + 64 = 192 个参数
    master = [(rs.randn(a * b) * 0.3).astype(F32) for a, b in shapes]
    psi = sum(w.size for w in master)
    x = rs.randn(16, 8).astype(F32)
    y = np.tanh(rs.randn(16, 4)).astype(F32)
    xs = [a.astype(F16) for a in np.split(x, world)]                  # [16,8] -> 4 × [4,8]
    ys = [a.astype(F16) for a in np.split(y, world)]

    formula = {
        0: 16 * psi,
        1: 4 * psi + 12 * psi // world,
        2: 2 * psi + 14 * psi // world,
        3: 16 * psi // world,
    }
    names = {0: "DDP", 1: "ZeRO-1", 2: "ZeRO-2", 3: "ZeRO-3/FSDP"}

    print(f"  Ψ = {psi} 参数, N = {world} ranks, 混合精度 Adam, {steps} 步\n")
    print(f"  {'':<14}{'常驻 B/rank':>12}{'公式':>8}{'通信 B/rank/step':>20}   vs DDP")
    finals, comm_per_step = {}, {}
    for stage in range(4):
        comm.reset()
        finals[stage], resident, transient, comm_per_step[stage] = train(stage, master, shapes, xs, ys, steps)
        extra = f"  (+{transient} B 瞬时: 一层的完整 fp16 参数)" if stage == 3 else ""
        print(f"  {names[stage]:<14}{resident:>12}{formula[stage]:>8}{comm_per_step[stage]:>20.0f}"
              f"   {comm_per_step[stage] / comm_per_step[0]:.2f}x{extra}")
        assert resident == formula[stage], (stage, resident, formula[stage])

    print()
    for stage in (1, 2, 3):
        diff = max(max_abs_diff(a, b) for a, b in zip(finals[0], finals[stage]))
        kv(f"max |{names[stage]} - DDP| (fp16 参数)", f"{diff:.1e}")
        assert diff == 0.0, "分片只改变 '谁存哪一片', 不改变任何一个数"

    ref = dense_fp32_adam(master, shapes, x, y, steps)
    diff32 = max(max_abs_diff(a, b) for a, b in zip(ref, finals[3]))
    moved = max(max_abs_diff(a, b) for a, b in zip(ref, master))
    kv("max |ZeRO-3 - 单卡 fp32 Adam|", f"{diff32:.1e}  (参数本身移动了 {moved:.1e})")
    assert diff32 < 0.05 * moved, "与单卡 fp32 基线只差 fp16 舍入"
    assert comm_per_step[2] == comm_per_step[0], "ZeRO-2: reduce-scatter + all-gather == 一次 all-reduce"
    assert abs(comm_per_step[3] / comm_per_step[0] - 1.5) < 1e-9, "ZeRO-3: 多一次参数 all-gather → 1.5×"

    print("\n  OK: 三个 stage 与 DDP 逐位相同, 常驻显存精确等于论文公式; stage 3 通信 1.5×。")
    print("      本实现的 ZeRO-1 用 all-reduce + all-gather (1.5×); DeepSpeed 改用 reduce-scatter 后与 DDP 持平。")


if __name__ == "__main__":
    main()
