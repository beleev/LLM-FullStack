#!/usr/bin/env python
"""
Transformer (Encoder-Decoder) 训练: teacher forcing + 交叉熵

注意: (src, tgt) 是固定的随机 token 对, 两者没有任何对应关系; "loss 下降" = 背下这个 batch, 只验证梯度链路。
"""

import math

import torch

from llm_models.models.foundation.transformer import Transformer
from llm_models.training import Trainer, TrainingConfig, StandardLMLoss, EncoderDecoderDataGenerator


def main():
    config = TrainingConfig(
        learning_rate=1e-3, batch_size=2, seq_len=32,
        num_steps=60, warmup_steps=5, log_interval=10, seed=42,
    )
    torch.manual_seed(config.seed)

    V_src, V_tgt = 800, 1000
    model = Transformer(
        src_vocab_size=V_src, tgt_vocab_size=V_tgt,
        d_model=256, n_heads=4, num_layers=2, d_ff=512, max_len=128, dropout=0.0,
    )
    print(f"Transformer | 参数量: {sum(p.numel() for p in model.parameters()):,}")

    data_gen = EncoderDecoderDataGenerator(
        src_vocab_size=V_src, tgt_vocab_size=V_tgt,
        batch_size=config.batch_size, src_len=config.seq_len, tgt_len=config.seq_len,
    )
    metrics = Trainer(model, config, data_gen, StandardLMLoss()).train()

    first, last = metrics[0]["total_loss"], metrics[-1]["total_loss"]
    print(f"初始 loss {first:.3f} (ln V_tgt = {math.log(V_tgt):.3f}) -> 终态 {last:.3f}")
    assert abs(first - math.log(V_tgt)) < 0.5, "初始 loss 应 ≈ ln V"
    assert last < 0.5 * first, "固定 batch 上 loss 应明显下降"


if __name__ == "__main__":
    main()
