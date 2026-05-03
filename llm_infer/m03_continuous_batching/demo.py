"""
m03 demo — Continuous Batching 调度器

用一个 mock model_runner (生成随机 token), 演示:
    1) 多请求并发的 prefill / decode 切换
    2) pool 紧张时的 preempt
    3) 调度统计

不接 TinyLM 也能跑, 关注点是"调度", 不是"模型输出"。
"""
from __future__ import annotations
import random

from llm_infer.core.utils import banner, kv
from llm_infer.m03_continuous_batching.scheduler import (
    Scheduler, SchedulerConfig,
)
from llm_infer.m03_continuous_batching.sequence import Stage


class MockModelRunner:
    """把 model.forward 替换成"返回随机 token id"。"""
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def run(self, batch, stage: Stage):
        """每条序列输出 1 个 token (prefill 也只关注最后一个 logit)。"""
        return [self.rng.randint(10, 99) for _ in batch]


def main():
    banner("M03 - Continuous Batching")

    cfg = SchedulerConfig(
        max_batch_seqs=4,
        max_batch_tokens=64,
        block_size=4,
        num_blocks=12,        # 故意小, 容易触发 preempt
    )
    sched = Scheduler(cfg)
    runner = MockModelRunner()

    # 提交 5 条请求, prompt 长度不同, max_new 不同
    requests = [
        ([1, 2, 3, 4, 5],          8),    # seq 0
        ([10, 20, 30],             4),    # seq 1
        ([7, 8, 9, 10, 11, 12, 13],10),   # seq 2
        ([100, 101],               6),    # seq 3
        ([50, 51, 52, 53, 54],     5),    # seq 4
    ]
    for prompt, max_new in requests:
        sid = sched.add_request(prompt, max_new)
        print(f"  add request seq_id={sid}, prompt_len={len(prompt)}, max_new={max_new}")

    print(f"\n初始 stats: {sched.stats()}")
    banner("开始循环 schedule")

    step = 0
    finished_all = []
    while sched.has_unfinished():
        step += 1
        # ---- prefill 优先 ---------------------------------- #
        picked, stage = sched.schedule()
        if stage == Stage.PREFILL and picked:
            outs = runner.run(picked, stage)
            done = sched.postprocess(picked, stage, outs)
            finished_all.extend(done)
            print(f"step {step:>3} [PREFILL] picked={str([s.seq_id for s in picked]):<25}"
                  f" finished={[s.seq_id for s in done]} pool={sched.bm.stats()['utilization']}")
            continue
        # ---- decode 阶段 ----------------------------------- #
        decoded = sched.schedule_decode()
        if not decoded:
            break
        outs = runner.run(decoded, Stage.DECODE)
        done = sched.postprocess(decoded, Stage.DECODE, outs)
        finished_all.extend(done)
        print(f"step {step:>3} [DECODE]  picked={str([s.seq_id for s in decoded]):<25}"
              f" finished={[s.seq_id for s in done]} pool={sched.bm.stats()['utilization']}"
              f" preempt={sched.preempt_count}")

    banner("结果")
    kv("总 step", step)
    kv("preempt 次数", sched.preempt_count)
    kv("完成请求", [s.seq_id for s in finished_all])
    print("\n  注意: 不同 prompt 长度的请求并发跑, 各自完成各自退出")
    print("        若 num_blocks 改成 5, 会观察到 preempt 触发")


if __name__ == "__main__":
    main()
