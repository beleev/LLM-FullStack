#!/usr/bin/env python
"""
CLIP 训练示例 — 对称对比 loss

合成数据: 随机图像 + 随机文本, 对角线作为正样本。
注意: 合成数据下相似度分布是随机的, loss 收敛幅度有限, 主要验证训练链路。
"""

import torch
from llm_models.models.multimodal.clip import CLIPModel
from llm_models.training import (
    Trainer, TrainingConfig, ContrastiveLoss, CLIPDataGenerator,
)


def main():
    cfg = TrainingConfig(
        learning_rate=5e-4, batch_size=8, seq_len=16,
        num_steps=30, warmup_steps=3, log_interval=5, seed=42,
    )
    torch.manual_seed(cfg.seed)

    vocab_size = 500
    model = CLIPModel(
        embed_dim=64, vocab_size=vocab_size,
        text_d_model=64, text_n_heads=4, text_num_layers=2, text_max_len=32,
        image_size=56, patch_size=14,
        vision_d_model=64, vision_n_heads=4, vision_num_layers=2,
    )
    print(f"CLIP Mini | 参数量: {sum(p.numel() for p in model.parameters()):,}")

    data_gen = CLIPDataGenerator(
        vocab_size=vocab_size, batch_size=cfg.batch_size,
        text_len=cfg.seq_len, image_size=56,
    )
    loss_fn = ContrastiveLoss()

    trainer = Trainer(model, cfg, data_gen, loss_fn)
    metrics = trainer.train()

    # 合成数据不保证严格下降, 只验证前向 + 反向链路通畅
    print(f"初始 loss: {metrics[0]['total_loss']:.4f} | 终态: {metrics[-1]['total_loss']:.4f}")
    print(f"logit_scale: {metrics[-1]['logit_scale']:.3f}")
    print("CLIP 训练链路验证通过!")


if __name__ == "__main__":
    main()
