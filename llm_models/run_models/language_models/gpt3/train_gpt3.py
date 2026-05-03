#!/usr/bin/env python
"""
GPT-3 训练示例

演示使用通用 Trainer 训练 GPT-3 (Decoder-only) 模型。
使用合成数据 + 标准交叉熵 loss。

训练目标：next-token prediction（自回归语言建模）
- 输入: tokens[:, :-1]
- 标签: tokens[:, 1:]（左移一位）
- 因果掩码已在模型内部启用，确保位置 t 不能看见 t+1..T-1

数据是随机生成的 token id 序列，本身没有语义；这里仅验证训练循环
能跑通、loss 能下降，并不会得到能用的语言模型。
"""

import torch
from llm_models.models import GPT3
from llm_models.training import (
    Trainer,
    TrainingConfig,
    StandardLMLoss,
    DecoderOnlyDataGenerator,
)


def main():
    # --- 训练配置 ---
    # 全部用迷你尺寸，方便 CPU 秒级验证
    config = TrainingConfig(
        learning_rate=3e-4,
        batch_size=2,
        seq_len=32,
        num_steps=50,
        warmup_steps=5,        # 学习率 warmup 步数（前几步线性升温避免训练不稳）
        log_interval=10,
        seed=42,
    )
    torch.manual_seed(config.seed)

    # --- 模型配置 (GPT-Mini，便于 CPU 运行) ---
    vocab_size = 1000
    model = GPT3(
        vocab_size=vocab_size,
        d_model=256,
        n_heads=4,
        num_layers=2,
        max_len=128,
        dropout=0.1,
        use_rope=False,        # 用经典 sinusoidal 位置编码
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"GPT-3 Mini | 参数量: {num_params:,}")

    # --- 数据生成器 + 损失函数 ---
    # DecoderOnlyDataGenerator: 随机 token，自动构造 next-token label
    data_gen = DecoderOnlyDataGenerator(
        vocab_size=vocab_size,
        batch_size=config.batch_size,
        seq_len=config.seq_len,
    )
    # StandardLMLoss: 标准交叉熵 + 自动 ignore_index=-100（忽略 pad/特殊位置）
    loss_fn = StandardLMLoss()

    # --- 训练 ---
    trainer = Trainer(model, config, data_gen, loss_fn)
    metrics = trainer.train()

    # --- 验证 loss 下降 ---
    # 在合成随机数据上模型至少要能拟合统计先验，loss 必须能降
    assert metrics[-1]["total_loss"] < metrics[0]["total_loss"], "Loss 未下降!"
    print("GPT-3 训练验证通过!")


if __name__ == "__main__":
    main()
