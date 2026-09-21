#!/usr/bin/env python
"""
Whisper 训练: 构造一个 "只有听音频才能做对" 的固定 batch

batch 里 2 条样本的 decoder 输入**完全相同**, 但 labels 不同, 唯一的区别是 mel。
只看文本的模型最多做到每个位置二选一 (loss = ln 2 ≈ 0.693); loss 能降到 ln 2 以下 ⇒ cross-attention 确实在读音频。
(数据仍是随机的, 这是记忆而非语音识别。)
"""

import math

import torch

from llm_models.models.multimodal.whisper import Whisper
from llm_models.training import Trainer, TrainingConfig, StandardLMLoss, WhisperDataGenerator


class SameTextDifferentAudio(WhisperDataGenerator):
    def _sample(self):
        batch = super()._sample()
        batch["decoder_input_ids"] = batch["decoder_input_ids"][:1].repeat(self.batch_size, 1)
        return batch


def main():
    cfg = TrainingConfig(
        learning_rate=1e-3, batch_size=2, seq_len=16,
        num_steps=150, warmup_steps=3, log_interval=25, seed=42,
    )
    torch.manual_seed(cfg.seed)

    V = 500
    model = Whisper(
        vocab_size=V, n_mels=80, d_model=128, n_heads=4,
        encoder_layers=2, decoder_layers=2, max_source_len=100, max_target_len=64,
    )
    print(f"Whisper Mini | 参数量: {sum(p.numel() for p in model.parameters()):,}")

    data_gen = SameTextDifferentAudio(vocab_size=V, batch_size=cfg.batch_size, tgt_len=cfg.seq_len, n_mels=80, t_mel=50)
    metrics = Trainer(model, cfg, data_gen, StandardLMLoss()).train()

    first, last = metrics[0]["total_loss"], metrics[-1]["total_loss"]
    print(f"初始 loss {first:.3f} (ln V = {math.log(V):.3f}) -> 终态 {last:.3f} (不听音频的下界 ln 2 = {math.log(2):.3f})")
    assert abs(first - math.log(V)) < 0.5, "初始 loss 应 ≈ ln V"
    assert last < 0.5 * math.log(2), "loss 低于 ln 2 才说明 decoder 用上了音频"


if __name__ == "__main__":
    main()
