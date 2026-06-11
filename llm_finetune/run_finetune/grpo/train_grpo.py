#!/usr/bin/env python
"""
GRPO 训练示例 (DeepSeek-R1 同款算法, 教学尺寸)
===============================================

教学目标:
    - 在线 RL 数据形态: 只有 prompt, completion 由 policy 实时采样
    - 组内相对优势替代 critic (GRPO 的核心创新, 对比 PPO 少一个 value 网络)
    - 可验证奖励 (RLVR): 规则给分, 无 reward model, 无 reward hacking 空间
    - KL 惩罚把 policy 锚在 ref 附近, 防止分布跑飞

任务 (玩具版可验证任务):
    completion 中落在"答案区" [V/2, V) 的 token 比例 = 奖励。
    随机初始化的 policy 奖励 ≈ 0.5; GRPO 应把它推向 1.0。

运行:
    python -m llm_finetune.run_finetune.grpo.train_grpo
"""

import torch

from llm_models.models.language_models.llama import LLaMA

from llm_finetune import (
    GRPOTrainer,
    PromptDataGenerator,
    make_region_reward,
)


def main() -> None:
    torch.manual_seed(42)
    vocab_size, steps = 64, 80

    # 小词表 + 小模型: 让 CPU 上 80 步内就能看到奖励显著上升
    policy = LLaMA(
        vocab_size=vocab_size, d_model=64, n_heads=4, num_kv_heads=2,
        num_layers=2, max_len=64, dropout=0.0,
    )
    # 关键细节: nn.Embedding 默认 N(0,1) 初始化, 叠加权重绑定 + sqrt(d) 缩放后
    # 初始 logits 方差巨大 → softmax 接近 one-hot → 同一 prompt 采 8 条全相同
    # → 组内方差为 0, GRPO 完全没有梯度 (在线 RL 依赖采样多样性!)。
    # 真实 LLM 的标准做法就是小 std 初始化 (GPT-2: 0.02), 这里补上。
    torch.nn.init.normal_(policy.token_embedding.weight, std=0.02)

    prompt_gen = PromptDataGenerator(
        vocab_size=vocab_size, batch_size=4, prompt_len=4, seed=42,
    )
    trainer = GRPOTrainer(
        policy,
        reward_fn=make_region_reward(vocab_size),
        group_size=8,          # 每个 prompt 采 8 条, 组内比较
        beta=0.01,
        lr=1e-3,
        max_new=8,
        temperature=1.0,
    )

    history = []
    for step in range(1, steps + 1):
        prompts = prompt_gen.generate_batch()["prompts"]
        metrics = trainer.step(prompts)
        history.append(metrics)
        if step % 20 == 0 or step == 1:
            print(f"step {step:>3} | reward {metrics['reward_mean']:.3f} "
                  f"± {metrics['reward_std']:.3f} | kl {metrics['kl']:.4f} "
                  f"| pg_loss {metrics['pg_loss']:+.4f}")

    first, last = history[0]["reward_mean"], history[-1]["reward_mean"]
    assert last > first + 0.2, f"GRPO 奖励未显著上升: {first:.3f} → {last:.3f}"
    print(f"\nGRPO 通过: 平均奖励 {first:.3f} → {last:.3f} (随机基线 0.5)")
    print("没有 critic, 没有 reward model —— 组内排名 + 规则验证就够了 (R1 配方)。")


if __name__ == "__main__":
    main()
