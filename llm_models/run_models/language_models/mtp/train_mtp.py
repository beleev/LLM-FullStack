#!/usr/bin/env python
"""
MTP 训练示例 — L = CE(main) + λ · mean_k CE(mtp_k)

复用标准的 DecoderOnlyDataGenerator: MTPLoss 自己负责把 next-token 标签
左移 k 位对齐到第 k 级 MTP 模块, 数据侧不需要任何改动。
"""

import torch

from llm_models.models.language_models.mtp import MTPLLaMA, MTPLoss
from llm_models.training import Trainer, TrainingConfig, DecoderOnlyDataGenerator


class FixedBatchDataGenerator(DecoderOnlyDataGenerator):
    """固定单 batch 的合成数据: 每步返回同一份样本 (模型可记忆)。

    纯随机数据上 next-next-token 不可学 (没有任何规律), MTP 分支的 loss
    只会原地抖动; 固定一个 batch 让模型记忆, 才能直观看到 main / mtp
    两路 loss 同时下降, 证明级联通路梯度可达。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cached = super().generate_batch()

    def generate_batch(self):
        return dict(self._cached)  # 浅拷贝: Trainer 会 pop("labels")


def main():
    cfg = TrainingConfig(
        learning_rate=3e-4, batch_size=2, seq_len=32,
        num_steps=80, warmup_steps=5, log_interval=20, seed=42,
    )
    torch.manual_seed(cfg.seed)

    vocab_size = 1000
    model = MTPLLaMA(
        vocab_size=vocab_size, d_model=256, n_heads=4, num_kv_heads=2,
        num_layers=2, max_len=128, mtp_depth=1, dropout=0.0,
    )
    print(f"MTPLLaMA Mini | 参数量: {sum(p.numel() for p in model.parameters()):,} "
          f"(mtp_depth=1)")

    data_gen = FixedBatchDataGenerator(
        vocab_size=vocab_size, batch_size=cfg.batch_size, seq_len=cfg.seq_len,
    )
    loss_fn = MTPLoss(mtp_lambda=0.3)

    trainer = Trainer(model, cfg, data_gen, loss_fn)
    metrics = trainer.train()

    first, last = metrics[0], metrics[-1]
    assert last["total_loss"] < first["total_loss"], "total loss 未下降!"
    assert last["main_loss"] < first["main_loss"], "main loss 未下降!"
    assert last["mtp_loss"] < first["mtp_loss"], "mtp loss 未下降!"
    print(
        f"MTP 训练验证通过! main {first['main_loss']:.3f}→{last['main_loss']:.3f}  "
        f"mtp {first['mtp_loss']:.3f}→{last['mtp_loss']:.3f}"
    )


if __name__ == "__main__":
    main()
