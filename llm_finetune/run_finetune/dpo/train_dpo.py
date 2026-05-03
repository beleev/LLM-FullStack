#!/usr/bin/env python
"""
LLaMA DPO (Direct Preference Optimization) 对齐示例
=====================================================

教学目标:
    - 演示 DPO 跳过 reward model 与 PPO, 仅用 (prompt, chosen, rejected) 三元组对齐
    - 演示 policy / reference 双模型架构: ref 自动 deepcopy + 冻结 + eval()
    - 跟踪 DPO 特有的监控指标: reward_chosen / reward_rejected / reward_margin / accuracy
    - 验证: reward_margin 应当随训练扩大, accuracy 应当趋近 1.0

约定 (与论文一致):
    - policy 起点应是 SFT 终态; 这里为简化教学, 直接用未训练的 LLaMA 也能看到信号
    - β = 0.1 是 DPO 论文常用值, 越大越保守贴近 ref

运行:
    python -m llm_finetune.run_finetune.dpo.train_dpo
"""

import torch

from llm_models.models.language_models.llama import LLaMA
from llm_models.training import TrainingConfig

from llm_finetune import (
    DPOLoss,
    DPOTrainer,
    PreferenceDataGenerator,
    print_trainable_parameters,
)


def main() -> None:
    cfg = TrainingConfig(
        learning_rate=5e-5,   # DPO 通常用更小的 lr (论文 1e-6 ~ 5e-5),
                              # 因为目标是"在 SFT 终态附近做小调整", 大 lr 会破坏 SFT 习得的能力
        batch_size=2,
        seq_len=32,
        num_steps=80,         # DPO 信号比 SFT 弱 (二分类 logsigmoid), 多走几步
        warmup_steps=10,
        log_interval=10,
        seed=42,
    )
    torch.manual_seed(cfg.seed)

    # ---- 1) 构造 policy 模型 ----
    vocab_size = 1000
    policy = LLaMA(
        vocab_size=vocab_size,
        d_model=256,
        n_heads=4,
        num_kv_heads=2,
        num_layers=2,
        max_len=128,
        dropout=0.0,
    )
    print_trainable_parameters(policy, name="DPO policy")

    # ---- 2) 数据: 偏好对 ----
    data_gen = PreferenceDataGenerator(
        vocab_size=vocab_size,
        batch_size=cfg.batch_size,
        seq_len=cfg.seq_len,
        prompt_len=cfg.seq_len // 2,
        seed=cfg.seed,
    )

    # ---- 3) DPOLoss + DPOTrainer (后者会自动 deepcopy + 冻结 ref) ----
    loss_fn = DPOLoss(beta=0.1)
    trainer = DPOTrainer(
        model=policy,
        config=cfg,
        data_generator=data_gen,
        loss_computer=loss_fn,
        ref_model=None,   # None → 自动用 deepcopy(policy) 作 ref
    )

    metrics = trainer.train()

    # ---- 4) 验证: DPO 训练应当让 reward_margin 扩大 ----
    first, last = metrics[0], metrics[-1]
    margin_first = first["reward_margin"]
    margin_last = last["reward_margin"]
    acc_last = last["accuracy"]

    print(
        f"\nReward margin: {margin_first:+.4f} → {margin_last:+.4f}"
        f"   accuracy: {acc_last:.2f}"
    )
    assert margin_last > margin_first, (
        f"reward_margin 未扩大 (chosen 没被相对拉高): "
        f"first={margin_first:.4f}  last={margin_last:.4f}"
    )
    print("DPO 训练通过: chosen 被相对 reference 拉高于 rejected")


if __name__ == "__main__":
    main()
