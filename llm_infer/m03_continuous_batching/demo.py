"""
m03 demo — Continuous Batching 调度器 (mock 模型: 只看调度, 不看 token 内容)

运行: python -m llm_infer.m03_continuous_batching.demo
看什么: 每步 batch 的组成 (P=prefill n 个 token, D=decode), pool 占用, 抢占; 以及两个回归:
    活锁 —— 队首请求拿不到 block 时必须落到 decode;  抢占 —— 不多生成也不丢 token。
"""
from __future__ import annotations

from llm_infer.core.utils import banner, kv
from llm_infer.m03_continuous_batching.scheduler import Scheduler, SchedulerConfig

REQUESTS = [([1, 2, 3, 4, 5], 8), ([10, 20, 30], 4), ([7, 8, 9, 10, 11, 12, 13], 10),
            ([100, 101], 6), ([50, 51, 52, 53, 54], 5)]          # (prompt, max_new)


def run(cfg: SchedulerConfig, verbose: bool = True, max_steps: int = 500):
    sched = Scheduler(cfg)
    seqs = [sched.add_request(p, m, eos_id=-1) for p, m in REQUESTS]
    step = 0
    while sched.has_unfinished():
        step += 1
        assert step <= max_steps, "活锁: 调度器空转"
        batch = sched.schedule()
        assert batch, "有未完成请求却调度出空 batch"
        # mock 模型: 第 k 个输出 token 的值就是 k → 抢占重算后内容是否连续一眼可查
        toks = [seq.num_output if seq.num_computed + n == seq.num_tokens else None for seq, n in batch]
        desc = " ".join(f"s{seq.seq_id}:{'P' + str(n) if n > 1 else 'D'}" for seq, n in batch)
        done = sched.postprocess(batch, toks)
        if verbose:
            print(f"  step {step:>2}  [{desc:<28}] pool={sched.bm.stats()['utilization']:>6} "
                  f"preempt={sched.preempt_count}" + (f"  ✓完成 {[s.seq_id for s in done]}" if done else ""))
    return sched, seqs, step


def main():
    banner("M03 - Continuous Batching")
    for p, m in REQUESTS:
        print(f"  request: prompt_len={len(p)} max_new={m}")

    print("\n[1] 宽松 pool (32 blocks × 4): 请求随到随进, 各自完成各自退出")
    sched, seqs, steps_loose = run(SchedulerConfig(max_batch_seqs=4, max_batch_tokens=64, block_size=4, num_blocks=32))
    assert sched.preempt_count == 0

    print("\n[2] 紧张 pool (7 blocks × 4 = 28 token): 触发抢占; 队首进不来时 running 照常 decode (活锁回归)")
    sched, seqs, steps_tight = run(SchedulerConfig(max_batch_seqs=4, max_batch_tokens=64, block_size=4, num_blocks=7))

    banner("结果")
    kv("总 step (宽松 / 紧张)", f"{steps_loose} / {steps_tight}")
    kv("抢占次数", sched.preempt_count)
    kv("各序列被抢占次数", [s.num_preempted for s in seqs])
    assert sched.preempt_count > 0
    for seq, (_, max_new) in zip(seqs, REQUESTS):
        # 抢占后: 输出一个不多 (max_new 不被重置)、一个不少、顺序不乱
        assert seq.output_ids == list(range(max_new)), (seq.seq_id, seq.output_ids)
    assert sched.bm.num_free_blocks() == 7 and sum(sched.bm.ref_count) == 0, "block 泄漏"
    print("  ✓ 无活锁; 抢占前后输出完整且恰好 max_new 个; block 全部归还")


if __name__ == "__main__":
    main()
