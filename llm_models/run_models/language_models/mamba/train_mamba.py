#!/usr/bin/env python
"""
Mamba 训练: 初始 loss ≈ ln V, 并能背下一个固定 batch

注意: 数据是固定的随机 token (无规律可学), "loss 下降" = 记忆, 只验证梯度链路通。
"""

import math

import torch

from llm_models.models.language_models.mamba import Mamba
from llm_models.training import Trainer, TrainingConfig, StandardLMLoss, DecoderOnlyDataGenerator


def main():
    cfg = TrainingConfig(
        learning_rate=3e-3, batch_size=2, seq_len=16,
        num_steps=60, warmup_steps=3, log_interval=10, seed=42,
    )
    torch.manual_seed(cfg.seed)

    V = 500
    model = Mamba(vocab_size=V, d_model=64, num_layers=2, d_state=8, d_conv=3)
    print(f"Mamba Mini | 参数量: {sum(p.numel() for p in model.parameters()):,}")

    # Δ 的专用初始化没有被 init_weights 清掉: softplus(bias) ∈ [1e-3, 1e-1]
    dt = torch.nn.functional.softplus(model.layers[0].layer.ssm.dt_proj.bias)
    assert 1e-3 * 0.99 <= dt.min() and dt.max() <= 1e-1 * 1.01

    data_gen = DecoderOnlyDataGenerator(vocab_size=V, batch_size=cfg.batch_size, seq_len=cfg.seq_len)
    metrics = Trainer(model, cfg, data_gen, StandardLMLoss()).train()

    first, last = metrics[0]["total_loss"], metrics[-1]["total_loss"]
    print(f"初始 loss {first:.3f} (ln V = {math.log(V):.3f}) -> 终态 {last:.3f}")
    assert abs(first - math.log(V)) < 0.5, "初始 loss 应 ≈ ln V"
    assert last < 0.5 * first, "固定 batch 上 loss 应明显下降"


if __name__ == "__main__":
    main()
