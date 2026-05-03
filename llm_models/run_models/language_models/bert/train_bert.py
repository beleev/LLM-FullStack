#!/usr/bin/env python
"""
BERT MLM 训练示例

数据: 随机 token 序列, 按 15% 概率做 mask (80% → [MASK], 10% → 随机, 10% 保持)
loss: 只在 mask 位置算 cross_entropy
"""

import torch
from llm_models.models.language_models.bert import BERT
from llm_models.training import (
    Trainer, TrainingConfig, MaskedLMLoss, MaskedLMDataGenerator,
)


def main():
    cfg = TrainingConfig(
        learning_rate=1e-4, batch_size=4, seq_len=32,
        num_steps=50, warmup_steps=5, log_interval=10, seed=42,
    )
    torch.manual_seed(cfg.seed)

    vocab_size = 500
    model = BERT(
        vocab_size=vocab_size, d_model=128, n_heads=4,
        num_layers=2, max_len=64, dropout=0.1,
    )
    print(f"BERT Mini | 参数量: {sum(p.numel() for p in model.parameters()):,}")

    data_gen = MaskedLMDataGenerator(
        vocab_size=vocab_size, batch_size=cfg.batch_size, seq_len=cfg.seq_len,
    )
    loss_fn = MaskedLMLoss()

    trainer = Trainer(model, cfg, data_gen, loss_fn)
    metrics = trainer.train()

    assert metrics[-1]["total_loss"] < metrics[0]["total_loss"], "Loss 未下降!"
    print("BERT MLM 训练验证通过!")


if __name__ == "__main__":
    main()
