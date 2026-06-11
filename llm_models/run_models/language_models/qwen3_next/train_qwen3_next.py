#!/usr/bin/env python
"""
Qwen3-Next 训练示例 — 混合层对 Trainer 完全透明。

DeltaNet 的递推是可微的 (einsum 链), autograd 直接穿过时间步循环,
不需要任何自定义反向 —— 这正是"线性注意力可以当普通层用"的含义。
"""

import torch

from llm_models.models.language_models.qwen3_next import Qwen3Next
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
    model = Qwen3Next(
        vocab_size=vocab_size, d_model=128, n_heads=4, num_kv_heads=2,
        num_layers=4, max_len=128, linear_ratio=3, dropout=0.0,
    )
    pattern = "".join("Δ" if t == "delta" else "A" for t in model.layer_types)
    print(f"Qwen3-Next Mini | 参数量: {sum(p.numel() for p in model.parameters()):,} "
          f"| 层排布 [{pattern}]")

    data_gen = DecoderOnlyDataGenerator(
        vocab_size=vocab_size, batch_size=cfg.batch_size, seq_len=cfg.seq_len,
    )
    loss_fn = StandardLMLoss()

    trainer = Trainer(model, cfg, data_gen, loss_fn)
    metrics = trainer.train()

    assert metrics[-1]["total_loss"] < metrics[0]["total_loss"], "Loss 未下降!"
    print("Qwen3-Next 训练验证通过! (梯度穿过 DeltaNet 递推 + 全注意力两类层)")


if __name__ == "__main__":
    main()
