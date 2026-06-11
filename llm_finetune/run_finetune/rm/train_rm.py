#!/usr/bin/env python
"""
Reward Model 训练示例 (RLHF 第二阶段)
======================================

教学目标:
    - 复用 DPO 的 PreferenceDataGenerator: RM 与 DPO 吃**同一种**偏好对数据,
      差别只在用法 (RM 学打分函数, DPO 直接学策略)
    - Bradley-Terry 损失: 只用 "A 比 B 好" 的序关系, 学出标量分
    - 验证: 训练后 RM 给 chosen 的分应稳定高于 rejected (准确率 → 1)

运行:
    python -m llm_finetune.run_finetune.rm.train_rm
"""

import torch

from llm_models.models.language_models.llama import LLaMA

from llm_finetune import (
    PreferenceDataGenerator,
    RewardModel,
    bradley_terry_loss,
    print_trainable_parameters,
)


def main() -> None:
    torch.manual_seed(42)
    vocab_size, steps = 1000, 60

    # ---- 1) backbone + value head (业界: backbone 从 SFT checkpoint 起步) ----
    backbone = LLaMA(
        vocab_size=vocab_size, d_model=256, n_heads=4, num_kv_heads=2,
        num_layers=2, max_len=128, dropout=0.0,
    )
    rm = RewardModel(backbone)
    print_trainable_parameters(rm, name="RewardModel (LLaMA + value head)")

    # ---- 2) 偏好对数据: 与 DPO 完全同源 ----
    data_gen = PreferenceDataGenerator(
        vocab_size=vocab_size, batch_size=4, seq_len=32, seed=42,
    )
    optimizer = torch.optim.AdamW(rm.parameters(), lr=3e-4)

    # ---- 3) 训练: r_chosen 与 r_rejected 之间拉开分差 ----
    first_acc = None
    for step in range(1, steps + 1):
        batch = data_gen.generate_batch()
        r_chosen = rm(batch["chosen_input_ids"])
        r_rejected = rm(batch["rejected_input_ids"])
        loss = bradley_terry_loss(r_chosen, r_rejected)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            acc = float((r_chosen > r_rejected).float().mean())
            margin = float((r_chosen - r_rejected).mean())
        if first_acc is None:
            first_acc = acc
        if step % 15 == 0 or step == 1:
            print(f"step {step:>3} | loss {float(loss):.4f} | "
                  f"acc {acc:.2f} | margin {margin:+.3f}")

    # ---- 4) 验证: 偏好准确率应到 1.0 (固定 batch 上可记忆) ----
    assert acc == 1.0, f"RM 未学会偏好排序: acc={acc}"
    print(f"\nRM 通过: 偏好准确率 {first_acc:.2f} → {acc:.2f}, "
          f"分差 margin {margin:+.3f}")
    print("这个标量分数就是 GRPO/PPO 阶段的奖励来源之一 (另一来源是规则验证 RLVR)。")


if __name__ == "__main__":
    main()
