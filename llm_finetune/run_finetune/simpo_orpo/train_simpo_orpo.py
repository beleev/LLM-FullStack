#!/usr/bin/env python
"""
SimPO / ORPO vs DPO: 同一个 SFT 起点、同样的偏好数据和步数。比两件事 ——
(1) 不要 reference 省了什么 (前向次数 / 常驻权重 / 实测耗时);  (2) 留出集上各自把模型带去了哪里。

    python -m llm_finetune.run_finetune.simpo_orpo.train_simpo_orpo
"""

import copy
import time

import torch

from llm_finetune import (
    DPOLoss, InstructionDataGenerator, ORPOLoss, PairwiseForward, PreferenceDataGenerator, SeqTask,
    SFTLoss, SimPOLoss, weight_bytes,
)
from llm_finetune.run_finetune.common import fit, make_model, preference_accuracy, sft_warmup

SFT_STEPS, STEPS, LR, SCRATCH_LR = 100, 200, 3e-4, 3e-3   # 从零训练用 SFT 的 lr, 对齐阶段用小 lr


def main() -> None:
    torch.manual_seed(0)
    task = SeqTask("sort")
    sft = make_model(task)
    em_sft = sft_warmup(sft, task, SFT_STEPS)
    rows = {"SFT 起点": (preference_accuracy(sft, task), em_sft, 0, 0.0, weight_bytes(sft))}

    forwards = [0]                                            # 数 LLaMA.forward 被调了几次 (policy 与 ref 都算)
    hook = lambda *_: forwards.__setitem__(0, forwards[0] + 1)

    def run(name, policy, wrapper, loss, steps=STEPS, data=None, lr=LR):
        handles = [m.register_forward_hook(hook) for m in {policy, wrapper._ref[0]} - {None}] if wrapper else []
        forwards[0], t0 = 0, time.perf_counter()
        fit(wrapper or policy, data or PreferenceDataGenerator(task), loss, steps, lr, log_interval=steps)
        dt = time.perf_counter() - t0
        n_fwd = forwards[0] / steps
        for h in handles:
            h.remove()
        resident = weight_bytes(policy) * (2 if wrapper and wrapper._ref[0] is not None else 1)
        rows[name] = (preference_accuracy(policy, task), task.exact_match(policy), n_fwd, dt, resident)

    for name, loss, with_ref in [("DPO β=0.5", DPOLoss(0.5), True), ("SimPO β=2 γ=1", SimPOLoss(2.0, 1.0), False),
                                 ("ORPO λ=0.5", ORPOLoss(0.5), False)]:
        torch.manual_seed(1)
        policy = copy.deepcopy(sft)
        run(name, policy, PairwiseForward.with_frozen_copy(policy) if with_ref else PairwiseForward(policy), loss)

    # ORPO 的卖点是 "不需要先 SFT": 从**随机初始化**出发, 与同样步数的纯 SFT 对照
    for name, loss in [("纯 SFT (从零)", None), ("ORPO (从零)", ORPOLoss(0.5))]:
        torch.manual_seed(1)
        policy = make_model(task)
        if loss is None:
            run(name, policy, None, SFTLoss(), SFT_STEPS + STEPS, InstructionDataGenerator(task), lr=SCRATCH_LR)
        else:
            run(name, policy, PairwiseForward(policy), loss, SFT_STEPS + STEPS, lr=SCRATCH_LR)

    print(f"\n{'':<16}{'偏好准确率':>8}{'logπ(chosen)':>14}{'logπ(rejected)':>16}{'EM':>8}{'前向/步':>9}{'耗时':>8}{'常驻权重':>10}")
    for name, (p, em, n_fwd, dt, mem) in rows.items():
        print(f"{name:<16}{p['accuracy']:>12.3f}{p['logp_chosen']:>14.2f}{p['logp_rejected']:>16.2f}"
              f"{em:>8.3f}{n_fwd:>10.0f}{dt:>8.1f}s{mem / 1024:>8.0f}KB")

    dpo, simpo, orpo = rows["DPO β=0.5"], rows["SimPO β=2 γ=1"], rows["ORPO λ=0.5"]
    assert (dpo[2], simpo[2], orpo[2]) == (2, 1, 1), "DPO 每步多一次 ref 前向"
    assert dpo[4] == 2 * simpo[4], "DPO 常驻两份权重"
    for name in ("DPO β=0.5", "SimPO β=2 γ=1", "ORPO λ=0.5"):
        assert rows[name][0]["accuracy"] >= rows["SFT 起点"][0]["accuracy"], f"{name}: 留出集偏好准确率不应下降"
    # NLL 项是 ORPO 的锚: chosen 的概率只升不降; DPO / SimPO 只约束差值, 没有这个保证
    assert orpo[0]["logp_chosen"] > rows["SFT 起点"][0]["logp_chosen"] > max(dpo[0]["logp_chosen"], simpo[0]["logp_chosen"])
    assert orpo[1] > em_sft, "ORPO 应同时提升 exact-match"
    scratch_sft, scratch_orpo = rows["纯 SFT (从零)"], rows["ORPO (从零)"]
    assert scratch_orpo[1] > 0.5, "ORPO 应能不经 SFT 阶段直接学会任务"
    assert scratch_orpo[0]["logp_rejected"] < scratch_sft[0]["logp_rejected"], "odds-ratio 项应把 rejected 压得比纯 SFT 更低"


if __name__ == "__main__":
    main()
