#!/usr/bin/env python
"""
Qwen3-Next 训练示例 — 验证两件事, 都用 assert 写死:
    1. 初始 loss ≈ ln V: 刚初始化的模型应当 "均匀瞎猜"。偏离很多说明初始化有问题
       (本库曾因 N(0,1) embedding + weight tying 得到 ~250)。
    2. loss 明显下降: 数据是 **固定的一个随机 batch** (SyntheticDataGenerator.fixed=True),
       随机 token 没有规律可学, 所以这里的 "下降" = 模型背下了这个 batch —— 只证明
       forward / backward / 优化器通路正确, 不代表学到了语言。
"""

import math

import torch

from llm_models.models.language_models.qwen3_next import Qwen3Next
from llm_models.training import (
    Trainer, TrainingConfig, StandardLMLoss, DecoderOnlyDataGenerator,
)


def main():
    cfg = TrainingConfig(
        learning_rate=1e-3, batch_size=2, seq_len=32,
        num_steps=60, warmup_steps=5, log_interval=10, seed=42,
    )
    torch.manual_seed(cfg.seed)

    vocab_size = 1000
    model = Qwen3Next(
        vocab_size=vocab_size, d_model=256, n_heads=4, num_layers=4, max_len=128, dropout=0.0, num_kv_heads=2, linear_ratio=3,
    )
    print(f"Qwen3-Next Mini | 参数量: {sum(p.numel() for p in model.parameters()):,}")

    data_gen = DecoderOnlyDataGenerator(
        vocab_size=vocab_size, batch_size=cfg.batch_size, seq_len=cfg.seq_len,
    )
    metrics = Trainer(model, cfg, data_gen, StandardLMLoss()).train()

    # step 1 的 lr = 0 (warmup 起点), 所以第一条 loss 就是未训练模型的 loss
    first, last = metrics[0]["total_loss"], metrics[-1]["total_loss"]
    ln_v = math.log(vocab_size)
    print(f"初始 loss {first:.3f}  vs  ln V = {ln_v:.3f}   |   最终 loss {last:.3f}")
    assert abs(first - ln_v) < 0.5, f"初始 loss {first:.2f} 应 ≈ ln V = {ln_v:.2f}"
    assert last < 0.5 * first, f"loss 未明显下降: {first:.3f} -> {last:.3f}"


if __name__ == "__main__":
    main()
