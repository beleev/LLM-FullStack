#!/usr/bin/env python
"""
DPO: SFT 过的 policy + 偏好对 (正确回复 vs 损坏回复)。用通用 Trainer 训练, 在**留出**偏好对上验收;
同时如实展示 DPO 的著名副作用: 差值变大了, 但 chosen 自己的概率也掉了。

    python -m llm_finetune.run_finetune.dpo.train_dpo
"""

import torch

from llm_finetune import DPOLoss, PairwiseForward, PreferenceDataGenerator, SeqTask
from llm_finetune.run_finetune.common import fit, make_model, preference_accuracy, sft_warmup

SFT_STEPS, DPO_STEPS, LR, BETA = 100, 200, 3e-4, 0.5


def main() -> None:
    torch.manual_seed(0)
    task = SeqTask("sort")
    policy = make_model(task)
    em_sft = sft_warmup(policy, task, SFT_STEPS)              # 故意只热身到 "半会": 留出提升空间
    before = preference_accuracy(policy, task)

    # ref = 此刻 policy 的冻结副本; PairwiseForward 把 policy×2 + ref×2 的前向包成一个 Module 交给通用 Trainer
    hist = fit(PairwiseForward.with_frozen_copy(policy), PreferenceDataGenerator(task), DPOLoss(BETA),
               DPO_STEPS, LR, log_interval=50)
    after, em_dpo = preference_accuracy(policy, task), task.exact_match(policy)

    print(f"第 1 步 loss {hist[0]['total_loss']:.4f} (policy = ref ⇒ 恰为 ln 2 = 0.6931)")
    print(f"{'留出集':<10}{'偏好准确率':>10}{'log π(chosen)':>16}{'log π(rejected)':>18}{'exact-match':>14}")
    print(f"{'SFT 后':<10}{before['accuracy']:>14.3f}{before['logp_chosen']:>16.2f}{before['logp_rejected']:>18.2f}{em_sft:>14.3f}")
    print(f"{'DPO 后':<10}{after['accuracy']:>14.3f}{after['logp_chosen']:>16.2f}{after['logp_rejected']:>18.2f}{em_dpo:>14.3f}")

    gap = lambda m: m["logp_chosen"] - m["logp_rejected"]
    assert abs(hist[0]["total_loss"] - 0.6931) < 1e-3
    assert after["accuracy"] >= before["accuracy"] and after["accuracy"] > 0.97, "留出集偏好准确率应提升"
    assert gap(after) > gap(before) + 2, "chosen 与 rejected 的 log-prob 差应被拉开"
    # DPO 的 loss 只看差值: 两边一起降、rejected 降得更多, 也算 "优化成功" (likelihood displacement)
    assert after["logp_chosen"] < before["logp_chosen"], "本配置下应能观察到 chosen 的 log-prob 下降"


if __name__ == "__main__":
    main()
