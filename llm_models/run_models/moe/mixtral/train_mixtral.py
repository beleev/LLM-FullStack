#!/usr/bin/env python
"""
Mixtral MoE 训练示例 — 与 DeepSeekV3 训练完全同接口 (都返回 routing_info)。
"""

import torch
from llm_models.models.moe.mixtral import Mixtral
from llm_models.training import (
    Trainer, TrainingConfig, MoELMLoss, DecoderOnlyDataGenerator,
)


def main():
    cfg = TrainingConfig(
        learning_rate=3e-4, batch_size=2, seq_len=32,
        num_steps=50, warmup_steps=5, aux_loss_weight=0.01,
        log_interval=10, seed=42,
    )
    torch.manual_seed(cfg.seed)

    vocab_size = 1000
    model = Mixtral(
        vocab_size=vocab_size, d_model=128, n_heads=4, num_kv_heads=2,
        num_layers=2, num_experts=4, top_k=2, max_len=64,
    )
    print(f"Mixtral Mini | 参数量: {sum(p.numel() for p in model.parameters()):,}")

    data_gen = DecoderOnlyDataGenerator(
        vocab_size=vocab_size, batch_size=cfg.batch_size, seq_len=cfg.seq_len,
    )
    loss_fn = MoELMLoss(aux_loss_weight=cfg.aux_loss_weight)

    trainer = Trainer(model, cfg, data_gen, loss_fn)
    metrics = trainer.train()

    assert metrics[-1]["total_loss"] < metrics[0]["total_loss"], "Loss 未下降!"
    print(f"最终 lm_loss: {metrics[-1]['lm_loss']:.4f} | aux_loss: {metrics[-1]['aux_loss']:.4f}")
    print("Mixtral 训练验证通过!")


if __name__ == "__main__":
    main()
