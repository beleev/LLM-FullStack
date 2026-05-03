#!/usr/bin/env python
"""
Transformer (Encoder-Decoder) 训练示例

演示使用通用 Trainer 训练标准 Encoder-Decoder Transformer
(原版 "Attention Is All You Need" 架构，常用于翻译类 seq2seq 任务)。

训练方式：teacher forcing + 标准交叉熵 loss
- teacher forcing：训练时给 Decoder 的输入是真实的 ground-truth tgt
  （而非模型自己上一步的预测），收敛更快、训练更稳
- 因果掩码 + tgt 错位：保证模型只能用过去 token 预测下一个 token

数据为随机 token 对 (src, tgt)，仅用于打通流程并验证 loss 收敛。
"""

import torch
from llm_models.models import Transformer
from llm_models.training import (
    Trainer,
    TrainingConfig,
    StandardLMLoss,
    EncoderDecoderDataGenerator,
)


def main():
    # --- 训练配置 ---
    config = TrainingConfig(
        learning_rate=3e-4,
        batch_size=2,
        seq_len=32,
        num_steps=50,
        warmup_steps=5,
        log_interval=10,
        seed=42,
    )
    torch.manual_seed(config.seed)

    # --- 模型配置 ---
    # src/tgt 词表大小不同，模拟翻译场景（如英→中）
    src_vocab_size = 800
    tgt_vocab_size = 1000
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=256,
        n_heads=4,
        num_layers=2,
        d_ff=512,                  # FFN 中间层维度，常为 d_model 的 2-4 倍
        max_len=128,
        dropout=0.1,
        use_rope=False,
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Transformer | 参数量: {num_params:,}")

    # --- 数据生成器 + 损失函数 ---
    # EncoderDecoderDataGenerator 同步生成 (src, tgt) 对，并构造好掩码
    data_gen = EncoderDecoderDataGenerator(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        batch_size=config.batch_size,
        src_len=config.seq_len,
        tgt_len=config.seq_len,
    )
    loss_fn = StandardLMLoss()

    # --- 训练 ---
    trainer = Trainer(model, config, data_gen, loss_fn)
    metrics = trainer.train()

    # --- 验证 loss 下降 ---
    assert metrics[-1]["total_loss"] < metrics[0]["total_loss"], "Loss 未下降!"
    print("Transformer 训练验证通过!")


if __name__ == "__main__":
    main()
