#!/usr/bin/env python
"""
VAR 训练示例

教学假设: tokenizer 已独立预训练并冻结 (真实做法), 本示例跳过 VQ-VAE 预训练,
直接把随机 tokenizer 冻结, 让 GPT 学合成图像 → 随机 token 序列 的分布。
合成数据不保证 loss 单调下降, 但能验证整条训练链路。
"""

import torch
from llm_models.models.generative.var import ImageTokenizer, VARModel
from llm_models.training import (
    Trainer, TrainingConfig, VARLoss, VARImageDataGenerator,
)


def main():
    cfg = TrainingConfig(
        learning_rate=3e-4, batch_size=4, num_steps=30,
        warmup_steps=3, log_interval=5, seed=42,
    )
    torch.manual_seed(cfg.seed)

    tokenizer = ImageTokenizer(
        image_size=32, codebook_size=128, latent_dim=32,
        base_channels=16, levels=2,
    )
    model = VARModel(
        tokenizer=tokenizer,
        gpt_d_model=96, gpt_n_heads=4, gpt_num_layers=2,
    )
    print(f"VAR Mini | 参数量: {sum(p.numel() for p in model.parameters()):,}")

    data_gen = VARImageDataGenerator(batch_size=cfg.batch_size, image_size=32)
    loss_fn = VARLoss()

    trainer = Trainer(model, cfg, data_gen, loss_fn)
    metrics = trainer.train()

    print(f"初始 loss: {metrics[0]['total_loss']:.4f} | 终态: {metrics[-1]['total_loss']:.4f}")
    print("VAR 训练链路验证通过!")


if __name__ == "__main__":
    main()
