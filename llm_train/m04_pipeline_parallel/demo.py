"""
M04 — 流水线并行: GPipe / 1F1B / Interleaved 1F1B

是什么: 模型按层切成 PP 个 stage, batch 切成 M 个 micro-batch 在 stage 间流动。
解决的瓶颈: 整个模型放不下 (层间切, 只传边界激活, 通信最省); 新问题是 **气泡** 和 **在途激活**。
关键公式: 气泡占比 = (PP-1) / (v·M + PP-1)          v = 每卡的虚拟 stage 数 (非交错时 v=1)
          在途激活峰值 (stage s): GPipe = M,  1F1B = PP - s   ← 1F1B 省的是显存, 不是时间
读代码盯住: `device_order` 里的 warmup —— 它就是 "最多允许几个 micro-batch 只 F 没 B"。
说明: 反向 ≈ 2× 前向, 所以 B 默认占 2 个时间格 (t_b 可调)。
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from llm_train.core import banner, kv

Op = Tuple[str, int, int]          # ("F"|"B", micro-batch, chunk);  chunk = 该卡上第几个虚拟 stage


def device_order(kind: str, s: int, PP: int, M: int, v: int = 1) -> List[Op]:
    """stage s 这张卡按什么顺序执行自己的 M·v 个 F 和 M·v 个 B。"""
    # F 的顺序: 每 PP 个 micro-batch 一组, 组内先把 chunk 0 跑完再跑 chunk 1 (Megatron 交错顺序)
    fwd = [("F", g + i, c) for g in range(0, M, PP) for c in range(v) for i in range(min(PP, M - g))]
    bwd = [("B", g + i, c) for g in range(0, M, PP) for c in reversed(range(v)) for i in range(min(PP, M - g))]
    if kind == "gpipe":
        return fwd + bwd                                  # 全部 F 完才开始 B → 在途 = M
    # 1F1B: 先做 warmup 个 F, 之后严格 F/B 交替 → 在途恒为 warmup+1
    warmup = (PP - 1 - s) if v == 1 else (PP - 1 - s) * 2 + (v - 1) * PP
    warmup = min(warmup, len(fwd))
    order, f_i, b_i = list(fwd[:warmup]), warmup, 0
    while b_i < len(bwd):
        if f_i < len(fwd):
            order.append(fwd[f_i]); f_i += 1
        order.append(bwd[b_i]); b_i += 1
    return order


def simulate(kind: str, PP: int, M: int, v: int = 1, t_f: int = 1, t_b: int = 2):
    """每张卡按固定顺序执行; 一个 op 在 "卡空闲 且 依赖完成" 的最早时刻开始。

    依赖: F(m, 虚拟stage k) 等 F(m, k-1);  B(m, k) 等 B(m, k+1);  最后一个 stage 的 B 等自己的 F。
    虚拟 stage k = chunk·PP + s, 共 PP·v 个, 第 k 个住在卡 k % PP 上。
    """
    if v > 1:
        assert M % PP == 0, "交错调度要求 M 是 PP 的整数倍"
    orders = [device_order(kind, s, PP, M, v) for s in range(PP)]
    end: Dict[Tuple[str, int, int], int] = {}             # (F/B, m, 虚拟stage) -> 完成时刻
    free, ptr = [0] * PP, [0] * PP
    rows: List[List[str]] = [[] for _ in range(PP)]
    live, peak = [0] * PP, [0] * PP
    last = PP * v - 1

    while any(p < len(o) for p, o in zip(ptr, orders)):
        progressed = False
        for s in range(PP):
            if ptr[s] == len(orders[s]):
                continue
            op, m, c = orders[s][ptr[s]]
            k = c * PP + s
            dep = ("F", m, k - 1) if op == "F" and k > 0 else \
                  ("B", m, k + 1) if op == "B" and k < last else \
                  ("F", m, k) if op == "B" else None
            if dep is not None and dep not in end:
                continue                                  # 依赖还没被调度, 这张卡先等
            start = max(free[s], end[dep] if dep else 0)
            dur = t_f if op == "F" else t_b
            rows[s] += ["."] * (start - len(rows[s])) + [f"{op}{m}"] * dur
            end[(op, m, k)] = free[s] = start + dur
            live[s] += 1 if op == "F" else -1             # F 产生一份待反向的激活, B 释放一份
            peak[s] = max(peak[s], live[s])
            ptr[s] += 1
            progressed = True
        assert progressed, "调度死锁"

    total = max(free)
    table = [r + ["."] * (total - len(r)) for r in rows]
    busy = M * v * (t_f + t_b)                            # 每张卡的有效工作量
    return table, peak, 1.0 - busy / total


def print_table(table) -> None:
    for i, row in enumerate(table):
        print(f"  s{i}: " + " ".join(f"{c:>2}" for c in row))


def main() -> None:
    banner("M04 - Pipeline Parallel: GPipe / 1F1B / Interleaved")

    PP, M, v = 3, 6, 2
    results = {}
    for name, kind, vv in [("GPipe", "gpipe", 1), ("1F1B", "1f1b", 1), (f"Interleaved v={v}", "1f1b", v)]:
        table, peak, bubble = simulate(kind, PP, M, vv)
        results[name] = (peak, bubble)
        unit = "1 格 = 1 个 stage 的前向" if vv == 1 else f"1 格 = 1/{vv} 个 stage 的前向 (时间轴被放大 {vv} 倍)"
        print(f"\n[{name}]  PP={PP}, M={M}; {unit}; B 占 2 格")
        print_table(table)

    print()
    for name, (peak, bubble) in results.items():
        formula = (PP - 1) / ((v if "Inter" in name else 1) * M + PP - 1)
        kv(f"{name} 气泡 (实测 / 公式)", f"{bubble:.1%} / {formula:.1%}")
    for name, (peak, _) in results.items():
        unit = " (单位: 1/v 个 stage 的激活)" if "Inter" in name else ""
        kv(f"{name} 在途激活峰值 / stage", f"{peak}{unit}")

    # ---- 精确断言: 审计里点名的两组配置 ----
    for pp, m in [(3, 4), (4, 8)]:
        _, peak_g, bub_g = simulate("gpipe", pp, m)
        _, peak_1, bub_1 = simulate("1f1b", pp, m)
        assert peak_g == [m] * pp
        assert peak_1 == [pp - s for s in range(pp)], peak_1          # [3,2,1] / [4,3,2,1]
        assert abs(bub_g - (pp - 1) / (m + pp - 1)) < 1e-12
        assert abs(bub_1 - bub_g) < 1e-12, "1F1B 不减少气泡, 只减少在途激活"
        kv(f"PP={pp}, M={m}: 1F1B 峰值", f"{peak_1}  (GPipe {peak_g})")
    _, peak_i, bub_i = simulate("1f1b", 4, 8, v=2)
    assert abs(bub_i - 3 / (2 * 8 + 3)) < 1e-12
    assert bub_i < 3 / (8 + 3)
    assert max(peak_i) / 2 > 4, "交错的代价: 在途激活 (折算成整 stage) 比 1F1B 多"
    kv("PP=4, M=8: 气泡 1F1B → 交错(v=2)", f"{3 / 11:.1%} → {bub_i:.1%}")
    kv("PP=4, M=8: 交错在途峰值", f"{peak_i} 份 1/2-stage 激活 = {max(peak_i) / 2:.1f} 个 stage (1F1B 为 4)")

    print("\n  OK: GPipe 与 1F1B 气泡相同、显存不同; 交错把气泡再除以 v, 代价是 v 倍的 P2P 次数和更多在途激活。")


if __name__ == "__main__":
    main()
