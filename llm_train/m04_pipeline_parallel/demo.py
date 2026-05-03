"""
M04 — Pipeline Parallelism (GPipe vs 1F1B)

把模型按层切成多个 stage。一个 batch 再切成 micro-batch 后, stage 之间
像流水线一样交错执行, 用吞吐换取跨 stage 通信和 pipeline bubble。

本 demo 用离散时间表对比两种调度:
    - GPipe (Huang et al. 2018): 所有 micro-batch 先全部前向, 再全部反向
                                  实现最简, 但激活峰值 = O(micro_batches) 显存高
    - 1F1B (Narayanan et al. 2019, PipeDream / Megatron-LM 默认):
                                  warmup 后交错 1 forward 1 backward
                                  wall-clock 时间与 GPipe 接近, 激活峰值 = O(num_stages)
                                  → 大模型训练首选

两者的 bubble 比例公式相同: (PP-1)/(PP+M-1), 区别在显存。
"""

from __future__ import annotations

from typing import List

from llm_train.core import banner, kv

Table = List[List[str]]


# --------------------------------------------------------------------- #
# 调度生成                                                              #
# --------------------------------------------------------------------- #

def gpipe_schedule(num_stages: int, micro_batches: int) -> Table:
    """GPipe: 全部 forward → 全部 backward。

    Forward wave : stage s 在时间 m+s 处理 micro m
    Backward wave: 反向阶段顺序倒过来, 从最后 stage 最大 micro 开始
    总时长 = 2 * (num_stages + micro_batches - 1)
    """
    total_slots = 2 * (num_stages + micro_batches - 1)
    table: Table = [["." for _ in range(total_slots)] for _ in range(num_stages)]

    for m in range(micro_batches):
        for s in range(num_stages):
            table[s][m + s] = f"F{m}"

    offset = num_stages + micro_batches - 1
    for m in reversed(range(micro_batches)):
        for s in reversed(range(num_stages)):
            t = offset + (micro_batches - 1 - m) + (num_stages - 1 - s)
            table[s][t] = f"B{m}"
    return table


def one_f_one_b_schedule(num_stages: int, micro_batches: int) -> Table:
    """1F1B: warmup 后每个 stage 交错执行一次 F + 一次 B。

    依赖关系:
        F(s, m) 需要 F(s-1, m) 已完成 (s>0)
        B(s, m) 需要 F(s, m) 已完成, 且 B(s+1, m) 已完成 (s<PP-1)
    调度策略:
        每个 stage s 先做 (PP-1-s) 个 warmup forward, 之后只要 backward 可执行
        就优先 backward (这是 "1F1B" 的核心, 让激活尽早释放)。
    """
    PP, M = num_stages, micro_batches
    f_done = [0] * PP
    b_done = [0] * PP
    rows: Table = [[] for _ in range(PP)]
    safety_cap = 4 * (PP + M)

    for _ in range(safety_cap):
        snap_f = list(f_done)
        snap_b = list(b_done)
        new_f = list(f_done)
        new_b = list(b_done)

        for s in range(PP):
            f_ready = snap_f[s] < M and (s == 0 or snap_f[s] < snap_f[s - 1])
            b_ready = snap_b[s] < snap_f[s] and (s == PP - 1 or snap_b[s] < snap_b[s + 1])
            warmup_done = snap_f[s] >= (PP - 1 - s)

            if b_ready and warmup_done:
                rows[s].append(f"B{snap_b[s]}")
                new_b[s] += 1
            elif f_ready:
                rows[s].append(f"F{snap_f[s]}")
                new_f[s] += 1
            elif b_ready:
                rows[s].append(f"B{snap_b[s]}")
                new_b[s] += 1
            else:
                rows[s].append(".")

        f_done = new_f
        b_done = new_b
        if all(b == M for b in b_done):
            break

    return rows


# --------------------------------------------------------------------- #
# 度量与打印                                                            #
# --------------------------------------------------------------------- #

def peak_activations(table: Table) -> List[int]:
    """各 stage 的激活峰值: F+1, B-1, 取过程中的最大值。

    GPipe → 峰值 ≈ M (前向全跑完才反向)
    1F1B  → 峰值 ≈ num_stages (反向尽早释放激活)
    """
    peaks: List[int] = []
    for row in table:
        live = 0
        cur_max = 0
        for cell in row:
            if cell.startswith("F"):
                live += 1
            elif cell.startswith("B"):
                live -= 1
            cur_max = max(cur_max, live)
        peaks.append(cur_max)
    return peaks


def bubble_ratio(table: Table) -> float:
    active = sum(cell != "." for row in table for cell in row)
    total = len(table) * len(table[0])
    return 1.0 - active / total


def print_table(table: Table) -> None:
    header = "time: " + " ".join(f"{i:>3}" for i in range(len(table[0])))
    print(header)
    for i, row in enumerate(table):
        print(f"s{i:<3}: " + " ".join(f"{cell:>3}" for cell in row))


# --------------------------------------------------------------------- #
# Demo                                                                  #
# --------------------------------------------------------------------- #

def main() -> None:
    banner("M04 - Pipeline Parallel: GPipe vs 1F1B")

    stages = 3
    micro_batches = 4

    print("\n[GPipe]  forward-then-backward, 实现最简")
    table_g = gpipe_schedule(stages, micro_batches)
    print_table(table_g)

    print("\n[1F1B]   warmup 后交错 F/B, 激活峰值降到 O(stages)")
    table_b = one_f_one_b_schedule(stages, micro_batches)
    print_table(table_b)

    no_pipeline_steps = 2 * stages * micro_batches
    peaks_g = peak_activations(table_g)
    peaks_b = peak_activations(table_b)

    kv("stages", stages)
    kv("micro-batches", micro_batches)
    kv("no-pipeline time slots", no_pipeline_steps)
    kv("GPipe time slots", len(table_g[0]))
    kv("1F1B  time slots", len(table_b[0]))
    kv("GPipe bubble ratio", f"{bubble_ratio(table_g):.1%}")
    kv("1F1B  bubble ratio", f"{bubble_ratio(table_b):.1%}")
    kv("GPipe peak activations / stage", peaks_g)
    kv("1F1B  peak activations / stage", peaks_b)

    assert len(table_g[0]) <= no_pipeline_steps
    assert max(peaks_b) <= max(peaks_g), "1F1B 应当激活峰值更低"

    print(
        "\n  关键对比:"
        "\n    GPipe : 实现最简, 激活峰值 O(M) — micro-batch 越多越省 bubble, 越费显存"
        "\n    1F1B  : Megatron 默认, wall-time 与 GPipe 相同, 激活峰值 O(stages)"
        "\n  两者 bubble 比例公式相同: (PP-1) / (PP + M - 1)。"
    )


if __name__ == "__main__":
    main()
