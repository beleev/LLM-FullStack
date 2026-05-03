#!/usr/bin/env python
"""
LLaMA SFT (全参监督微调) 示例
==================================

教学目标:
    - 演示 SFT 与预训练在 **数据形态** 上的唯一差异: prompt 段 label = -100
    - 复用 ``llm_models.training.Trainer`` 的训练循环, 把 finetune 视为
      "新 LossComputer + 新 DataGenerator", 体现策略模式的扩展性
    - 验证 loss 单调下降, 证明 SFT 通路可工作

运行:
    python -m llm_finetune.run_finetune.sft.train_sft
"""

import torch

from llm_models.models.language_models.llama import LLaMA
from llm_models.training import Trainer, TrainingConfig

from llm_finetune import (
    SFTLoss,
    InstructionDataGenerator,
    print_trainable_parameters,
)


def main() -> None:
    cfg = TrainingConfig(
        learning_rate=3e-4,
        batch_size=2,
        seq_len=32,
        num_steps=50,
        warmup_steps=5,
        log_interval=10,
        seed=42,
    )
    torch.manual_seed(cfg.seed)

    # ---- 1) 构造 LLaMA Mini (CPU 友好规模) ----
    vocab_size = 1000
    model = LLaMA(
        vocab_size=vocab_size,
        d_model=256,
        n_heads=4,
        num_kv_heads=2,
        num_layers=2,
        max_len=128,
        dropout=0.0,
    )

    # SFT 默认全参可训, 打印一行用于和后续 LoRA 示例对照
    print_trainable_parameters(model, name="LLaMA-Mini SFT")

    # ---- 2) 数据 / Loss ----
    # 区别于预训练的 DecoderOnlyDataGenerator: 这里 prompt 区 label = -100
    data_gen = InstructionDataGenerator(
        vocab_size=vocab_size,
        batch_size=cfg.batch_size,
        seq_len=cfg.seq_len,
        prompt_len=cfg.seq_len // 2,  # 一半 prompt 一半 response
        seed=cfg.seed,
    )
    loss_fn = SFTLoss()

    # ---- 3) 训练 ----
    trainer = Trainer(model, cfg, data_gen, loss_fn)
    metrics = trainer.train()

    # ---- 4) 验证: loss 应当下降 ----
    first, last = metrics[0]["total_loss"], metrics[-1]["total_loss"]
    assert last < first, f"SFT loss 未下降: first={first:.4f}  last={last:.4f}"
    print(f"\nSFT 通过: loss {first:.4f} → {last:.4f}")


if __name__ == "__main__":
    main()
