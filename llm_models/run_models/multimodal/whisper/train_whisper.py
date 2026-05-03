#!/usr/bin/env python
"""
Whisper 训练示例 — mel → text 的 teacher forcing 训练
"""

import torch
from llm_models.models.multimodal.whisper import Whisper
from llm_models.training import (
    Trainer, TrainingConfig, StandardLMLoss, WhisperDataGenerator,
)


def main():
    cfg = TrainingConfig(
        learning_rate=3e-4, batch_size=2, seq_len=16,
        num_steps=30, warmup_steps=3, log_interval=5, seed=42,
    )
    torch.manual_seed(cfg.seed)

    vocab_size = 500
    model = Whisper(
        vocab_size=vocab_size, n_mels=80, d_model=128, n_heads=4,
        encoder_layers=2, decoder_layers=2,
        max_source_len=100, max_target_len=64,
    )
    print(f"Whisper Mini | 参数量: {sum(p.numel() for p in model.parameters()):,}")

    data_gen = WhisperDataGenerator(
        vocab_size=vocab_size, batch_size=cfg.batch_size,
        tgt_len=cfg.seq_len, n_mels=80, t_mel=50,
    )
    loss_fn = StandardLMLoss()

    trainer = Trainer(model, cfg, data_gen, loss_fn)
    metrics = trainer.train()

    assert metrics[-1]["total_loss"] < metrics[0]["total_loss"], "Loss 未下降!"
    print("Whisper 训练验证通过!")


if __name__ == "__main__":
    main()
