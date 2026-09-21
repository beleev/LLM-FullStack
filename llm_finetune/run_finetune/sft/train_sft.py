#!/usr/bin/env python
"""
SFT: 在 "反转 prompt" 任务上全参微调, 用**留出集** exact-match 验收; 顺带复现 prompt-mask 差一位的后果。

    python -m llm_finetune.run_finetune.sft.train_sft
"""

import math

import torch

from llm_finetune.data import InstructionDataGenerator, SeqTask
from llm_finetune.methods.sft import SFTLoss
from llm_finetune.run_finetune.common import fit, make_model
from llm_finetune.utils import print_trainable_parameters

STEPS, LR = 600, 3e-3


class OffByOneMask(InstructionDataGenerator):
    """反面教材: 把 labels[:, :P] (而不是 [:, :P-1]) 置 -100 —— 第一个回复 token 从此没有监督。"""

    def _sample(self):
        batch = super()._sample()
        batch["labels"][:, self.task.prompt_len - 1] = -100
        return batch


def main() -> None:
    torch.manual_seed(0)
    task = SeqTask("reverse")

    model = make_model(task)
    print_trainable_parameters(model, "SFT (全参)")
    em_before = task.exact_match(model)
    metrics = fit(model, InstructionDataGenerator(task), SFTLoss(), STEPS, LR)
    em = task.exact_match(model)

    torch.manual_seed(0)
    buggy = make_model(task)
    fit(buggy, OffByOneMask(task), SFTLoss(), STEPS, LR, log_interval=STEPS)
    em_buggy = task.exact_match(buggy)

    first = metrics[0]["total_loss"]
    print(f"初始 loss {first:.3f} (ln V = {math.log(task.vocab_size):.3f})")
    print(f"留出集 exact-match: 训练前 {em_before:.3f} → 训练后 {em:.3f}")
    print(f"prompt mask 多盖一位 (labels[:, :P]) 的同配置模型: exact-match {em_buggy:.3f}")

    assert abs(first - math.log(task.vocab_size)) < 0.5, "初始 loss 应 ≈ ln V"
    assert em > 0.9, f"SFT 没学会任务: 留出集 exact-match {em:.3f}"
    assert em_buggy < 0.5 < em, "差一位的 mask 应当让第一个回复 token 学不到"


if __name__ == "__main__":
    main()
