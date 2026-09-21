#!/usr/bin/env python
"""
CLIP 训练: 初始对比 loss ≈ ln B, 训练后 [B, B] 相似度矩阵的对角线成为每行每列的最大值

注意: 8 对 (随机图, 随机文本) 是固定的, 这里验证的是"能把它们配上对" (记忆), 不是语义对齐。
"""

import math

import torch

from llm_models.models.multimodal.clip import CLIPModel
from llm_models.training import Trainer, TrainingConfig, ContrastiveLoss, CLIPDataGenerator


@torch.no_grad()
def diag_accuracy(model, batch) -> float:
    """image→text 检索 top-1: 第 i 张图最相似的文本是不是第 i 条。"""
    model.eval()
    out = model(batch["images"], batch["input_ids"], eos_token_id=batch["eos_token_id"])
    sim = out["image_features"] @ out["text_features"].t()             # [B, B]
    return (sim.argmax(-1) == torch.arange(sim.size(0))).float().mean().item()


def main():
    cfg = TrainingConfig(
        learning_rate=5e-4, batch_size=8, seq_len=16,
        num_steps=30, warmup_steps=3, log_interval=5, seed=42,
    )
    torch.manual_seed(cfg.seed)

    V, B = 500, cfg.batch_size
    model = CLIPModel(
        embed_dim=64, vocab_size=V,
        text_d_model=64, text_n_heads=4, text_num_layers=2, text_max_len=32,
        image_size=56, patch_size=14,
        vision_d_model=64, vision_n_heads=4, vision_num_layers=2,
    )
    print(f"CLIP Mini | 参数量: {sum(p.numel() for p in model.parameters()):,}")

    data_gen = CLIPDataGenerator(vocab_size=V, batch_size=B, text_len=cfg.seq_len, image_size=56)
    acc0 = diag_accuracy(model, data_gen.generate_batch())
    metrics = Trainer(model, cfg, data_gen, ContrastiveLoss()).train()
    acc1 = diag_accuracy(model, data_gen.generate_batch())

    first, last = metrics[0]["total_loss"], metrics[-1]["total_loss"]
    print(f"初始 loss {first:.3f} (ln B = {math.log(B):.3f}) -> 终态 {last:.4f} | "
          f"对角线 top-1: {acc0:.2f} -> {acc1:.2f} | logit_scale {metrics[0]['logit_scale']:.2f} -> {metrics[-1]['logit_scale']:.2f}")
    assert abs(first - math.log(B)) < 0.5, "未训练时对比 loss 应 ≈ ln B"
    assert last < 0.1 and acc1 == 1.0 and acc1 > acc0


if __name__ == "__main__":
    main()
