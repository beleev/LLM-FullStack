#!/usr/bin/env python
"""
Mamba 训练示例 (SSM 替代 attention, 无需 mask)

教学注意: Mamba 的 scan 用纯 Python for-loop 实现, 序列越长越慢。
本示例把 seq_len 压到 16 秒级跑完。
"""

import torch
from llm_models.models.language_models.mamba import Mamba
from llm_models.training import (
    Trainer, TrainingConfig, StandardLMLoss, DecoderOnlyDataGenerator,
)


def main():
    cfg = TrainingConfig(
        learning_rate=3e-4, batch_size=2, seq_len=16,
        num_steps=30, warmup_steps=3, log_interval=5, seed=42,
    )
    torch.manual_seed(cfg.seed)

    vocab_size = 500
    model = Mamba(
        vocab_size=vocab_size, d_model=64, num_layers=2, d_state=8, d_conv=3,
    )
    print(f"Mamba Mini | 参数量: {sum(p.numel() for p in model.parameters()):,}")

    data_gen = DecoderOnlyDataGenerator(
        vocab_size=vocab_size, batch_size=cfg.batch_size, seq_len=cfg.seq_len,
    )
    loss_fn = StandardLMLoss()

    trainer = Trainer(model, cfg, data_gen, loss_fn)
    metrics = trainer.train()

    # Mamba 在合成随机数据 + 30 步的规模上 loss 通常在 ~log(vocab) 附近抖动,
    # 不强制严格下降; 仅验证训练链路 (数值不崩 + loss 有限) 可用。
    assert not (metrics[-1]["total_loss"] != metrics[-1]["total_loss"]), "Loss 变成 nan!"
    print(f"初始 loss: {metrics[0]['total_loss']:.4f} | 终态: {metrics[-1]['total_loss']:.4f}")
    print("Mamba 训练链路验证通过!")


if __name__ == "__main__":
    main()
