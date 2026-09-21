#!/usr/bin/env python
"""
MTP 训练示例 — L = CE(main) + λ · mean_k CE(mtp_k)

数据侧零改动: MTPLoss 自己把 next-token 标签左移 k 位对齐到第 k 级 MTP 模块。
数据是固定的一个随机 batch (SyntheticDataGenerator.fixed=True): 随机 token 上 "下下个 token"
没有任何规律, 所以这里的下降 = 两条支路都能背下这个 batch —— 证明的是级联通路梯度可达。
"""

import math

import torch

from llm_models.models.language_models.mtp import MTPLLaMA
from llm_models.training import Trainer, TrainingConfig, DecoderOnlyDataGenerator
from llm_models.training.loss import MTPLoss


def main():
    cfg = TrainingConfig(
        learning_rate=1e-3, batch_size=2, seq_len=32,
        num_steps=60, warmup_steps=5, log_interval=10, seed=42,
    )
    torch.manual_seed(cfg.seed)

    vocab_size = 1000
    model = MTPLLaMA(
        vocab_size=vocab_size, d_model=256, n_heads=4, num_kv_heads=2,
        num_layers=2, max_len=128, mtp_depth=1, dropout=0.0,
    )
    print(f"MTPLLaMA Mini | 参数量: {sum(p.numel() for p in model.parameters()):,} (mtp_depth=1)")

    data_gen = DecoderOnlyDataGenerator(
        vocab_size=vocab_size, batch_size=cfg.batch_size, seq_len=cfg.seq_len,
    )
    metrics = Trainer(model, cfg, data_gen, MTPLoss(mtp_lambda=0.3)).train()

    first, last, ln_v = metrics[0], metrics[-1], math.log(vocab_size)
    print(f"main {first['main_loss']:.3f} → {last['main_loss']:.3f}   "
          f"mtp {first['mtp_loss']:.3f} → {last['mtp_loss']:.3f}   (ln V = {ln_v:.3f})")
    # 两条支路初始都应是 "均匀瞎猜"
    assert abs(first["main_loss"] - ln_v) < 0.5 and abs(first["mtp_loss"] - ln_v) < 0.5
    assert abs(first["total_loss"] - (first["main_loss"] + 0.3 * first["mtp_loss"])) < 1e-4
    assert last["main_loss"] < 0.5 * first["main_loss"], "main loss 未明显下降"
    assert last["mtp_loss"] < 0.5 * first["mtp_loss"], "mtp loss 未明显下降 (级联通路梯度不通?)"


if __name__ == "__main__":
    main()
