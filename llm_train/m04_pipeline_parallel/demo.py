"""
M04 — Pipeline Parallelism

把模型按层切成多个 stage。一个 batch 再切成 micro-batch 后, stage 之间
像流水线一样交错执行, 用吞吐换取跨 stage 通信和 pipeline bubble。

这里用离散时间表模拟 GPipe 的 forward-then-backward 调度。
"""
from __future__ import annotations

from llm_train.core import banner, kv


def gpipe_schedule(num_stages: int, micro_batches: int):
    total_slots = 2 * (num_stages + micro_batches - 1)
    table = [["." for _ in range(total_slots)] for _ in range(num_stages)]

    # Forward wave: stage s processes micro m at time m+s.
    for m in range(micro_batches):
        for s in range(num_stages):
            table[s][m + s] = f"F{m}"

    # Backward wave: reverse stage order after all forwards drain.
    offset = num_stages + micro_batches - 1
    for m in reversed(range(micro_batches)):
        for s in reversed(range(num_stages)):
            t = offset + (micro_batches - 1 - m) + (num_stages - 1 - s)
            table[s][t] = f"B{m}"
    return table


def print_table(table) -> None:
    header = "time: " + " ".join(f"{i:>3}" for i in range(len(table[0])))
    print(header)
    for i, row in enumerate(table):
        print(f"s{i:<3}: " + " ".join(f"{cell:>3}" for cell in row))


def main() -> None:
    banner("M04 - Pipeline Parallel Schedule")

    stages = 3
    micro_batches = 4
    table = gpipe_schedule(stages, micro_batches)
    print_table(table)

    active = sum(cell != "." for row in table for cell in row)
    total = len(table) * len(table[0])
    bubble = 1.0 - active / total

    no_pipeline_steps = 2 * stages * micro_batches
    pipeline_steps = len(table[0])
    kv("stages", stages)
    kv("micro-batches", micro_batches)
    kv("no pipeline time slots", no_pipeline_steps)
    kv("pipeline time slots", pipeline_steps)
    kv("bubble ratio", f"{bubble:.1%}")

    assert pipeline_steps < no_pipeline_steps
    print("\n  OK: micro-batch 越多, bubble 占比越低; 代价是更多激活驻留和调度复杂度。")


if __name__ == "__main__":
    main()

