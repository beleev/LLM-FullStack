#!/usr/bin/env python
"""
Mistral 训练示例 — 与 LLaMA 训练代码完全一致, 只替换模型类。
带状 mask 对 Trainer 透明: 数据、loss、优化器都不需要知道 SWA 的存在。
"""

import torch

from llm_models.models.language_models.mistral import Mistral
from llm_models.training import (
    Trainer, TrainingConfig, StandardLMLoss, DecoderOnlyDataGenerator,
)


def main():
    cfg = TrainingConfig(
        learning_rate=3e-4, batch_size=2, seq_len=32,
        num_steps=50, warmup_steps=5, log_interval=10, seed=42,
    )
    torch.manual_seed(cfg.seed)

    vocab_size = 1000
    model = Mistral(
        vocab_size=vocab_size, d_model=256, n_heads=4, num_kv_heads=2,
        num_layers=2, max_len=128, window_size=8, dropout=0.0,
    )
    print(f"Mistral Mini | 参数量: {sum(p.numel() for p in model.parameters()):,} "
          f"(SWA 窗口 W=8)")

    data_gen = DecoderOnlyDataGenerator(
        vocab_size=vocab_size, batch_size=cfg.batch_size, seq_len=cfg.seq_len,
    )
    loss_fn = StandardLMLoss()

    trainer = Trainer(model, cfg, data_gen, loss_fn)
    metrics = trainer.train()

    assert metrics[-1]["total_loss"] < metrics[0]["total_loss"], "Loss 未下降!"
    print("Mistral 训练验证通过!")


if __name__ == "__main__":
    main()
