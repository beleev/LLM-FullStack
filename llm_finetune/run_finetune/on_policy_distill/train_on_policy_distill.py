#!/usr/bin/env python
"""
On-policy 蒸馏 (reverse KL) vs off-policy 蒸馏 (forward KL): 同一个 teacher、同一个热身后的 student、同样的步数。

    python -m llm_finetune.run_finetune.on_policy_distill.train_on_policy_distill
"""

import copy

import torch

from llm_finetune import DistillLoss, PromptDataGenerator, TeacherStudent, on_policy_distill_step
from llm_finetune.run_finetune.common import fit
from llm_finetune.run_finetune.distill.train_distill import (
    MAJOR, MixtureData, evaluate, make_student, report, train_teacher,
)

WARMUP_STEPS, STEPS, LR = 200, 300, 1e-3


def main() -> None:
    torch.manual_seed(0)
    teacher = train_teacher()
    forward_kl = DistillLoss(temperature=1.0, alpha=0.0)       # 纯 KL(teacher‖student), 不掺硬标签, 与 reverse KL 对称比较

    # on-policy 需要一个 "已经能说出点东西" 的 student (真实流程里它先经过 SFT / off-policy 蒸馏)
    torch.manual_seed(1)
    warm = make_student()
    fit(TeacherStudent(warm, teacher), MixtureData(), forward_kl, WARMUP_STEPS, 3e-3, log_interval=WARMUP_STEPS)
    rows = {"teacher": evaluate(teacher, teacher), "热身后的 student": evaluate(warm, teacher)}

    off = copy.deepcopy(warm)
    fit(TeacherStudent(off, teacher), MixtureData(), forward_kl, STEPS, LR, log_interval=STEPS)
    rows["+ off-policy forward KL"] = evaluate(off, teacher)

    on = copy.deepcopy(warm)
    optimizer = torch.optim.AdamW(on.parameters(), lr=LR)
    prompts = PromptDataGenerator(MAJOR, 64)
    for step in range(1, STEPS + 1):
        m = on_policy_distill_step(on, teacher, prompts.generate_batch()["prompts"], MAJOR.response_len, optimizer)
        if step % 100 == 0:
            print(f"on-policy step {step}: 自己样本上的 reverse KL {m['reverse_kl']:.3f}")
    rows["+ on-policy reverse KL"] = evaluate(on, teacher)
    report(rows)

    off_m, on_m = rows["+ off-policy forward KL"], rows["+ on-policy reverse KL"]
    # 各自在自己优化的那个指标上赢
    assert on_m["reverse_kl"] < off_m["reverse_kl"] and off_m["forward_kl"] < on_m["forward_kl"]
    # mode-seeking: 自己采样出来的回复合格率更高 (容量不够时宁可只做好一种答案) ……
    assert on_m["valid"] > off_m["valid"] + 0.1
    # …… 代价是 teacher 的少数派答案 (copy, 30%) 被放弃得更彻底; forward KL 则两头都留着概率
    assert on_m["logp_major"] > off_m["logp_major"] and on_m["logp_minor"] < off_m["logp_minor"]


if __name__ == "__main__":
    main()
