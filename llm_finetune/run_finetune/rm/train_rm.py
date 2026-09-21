#!/usr/bin/env python
"""
Reward Model: SFT 主干 + 标量头, Bradley-Terry loss, 通用 Trainer 训练; 在**留出**偏好对上验收,
并单测 "右 pad 时分数取自最后一个真 token"。

    python -m llm_finetune.run_finetune.rm.train_rm
"""

import torch
import torch.nn.functional as F

from llm_finetune import BradleyTerryLoss, PairwiseForward, PreferenceDataGenerator, RewardModel, SeqTask
from llm_finetune.data import PAD
from llm_finetune.run_finetune.common import fit, make_model, sft_warmup

SFT_STEPS, STEPS, LR = 100, 300, 1e-3


@torch.no_grad()
def heldout_accuracy(rm: RewardModel, batch) -> float:
    rm.eval()
    r_w = rm(batch["chosen_input_ids"], batch["chosen_attention_mask"])
    r_l = rm(batch["rejected_input_ids"], batch["rejected_attention_mask"])
    return float((r_w > r_l).float().mean())


def main() -> None:
    torch.manual_seed(0)
    task = SeqTask("sort")
    backbone = make_model(task)
    sft_warmup(backbone, task, SFT_STEPS)                     # 业界惯例: RM 从 SFT checkpoint 起步
    rm = RewardModel(backbone)                                # 注意: backbone 的 lm_head 从此被摘掉

    heldout = PreferenceDataGenerator(task, 512, split="test", fixed=True).generate_batch()
    acc_before = heldout_accuracy(rm, heldout)
    hist = fit(PairwiseForward(rm), PreferenceDataGenerator(task), BradleyTerryLoss(), STEPS, LR, log_interval=100)
    acc = heldout_accuracy(rm, heldout)
    print(f"第 1 步 loss {hist[0]['total_loss']:.4f} (≈ ln 2); 留出集偏好准确率 {acc_before:.3f} → {acc:.3f}")

    # ---- 右 pad 单测: 同一条序列, 后面多垫 5 个 PAD, 分数必须不变 ----
    ids, mask = heldout["chosen_input_ids"][:8], heldout["chosen_attention_mask"][:8]
    with torch.no_grad():
        clean = rm(ids, mask)
        padded = rm(F.pad(ids, (0, 5), value=PAD), F.pad(mask, (0, 5), value=0))
        naive = rm(F.pad(ids, (0, 5), value=PAD))             # 不给 mask = 旧实现的 scores[:, -1], 读到 pad 位置
    print(f"右 pad 5 位后的分数偏移: 取最后一个真 token {float((padded - clean).abs().max()):.1e} | "
          f"取 [:, -1] {float((naive - clean).abs().max()):.2f}")

    # rejected 里 "漏一个 token" 的那一半天然带右 pad, 训练 / 评估全程都在走 gather 分支
    assert (heldout["rejected_attention_mask"].sum(1) < heldout["rejected_attention_mask"].size(1)).any()
    assert abs(hist[0]["total_loss"] - 0.6931) < 0.05
    assert acc > 0.9 > acc_before, f"RM 没学会在留出集上排序: {acc:.3f}"
    assert torch.allclose(padded, clean, atol=1e-4), "pad 不应改变分数"
    assert (naive - clean).abs().max() > 1e-2, "读 pad 位置的分数应当明显不同 (这就是 B2)"


if __name__ == "__main__":
    main()
