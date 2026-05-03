#!/usr/bin/env python
"""
Whisper 前向示例 (音频 encoder-decoder)

展示:
- mel-spectrogram → 2 层 1D Conv stem → Transformer encoder
- 文本 decoder 通过 cross-attention 读 encoder 输出
"""

import torch
from llm_models.models.multimodal.whisper import Whisper


def main():
    torch.manual_seed(42)

    vocab_size = 1000
    model = Whisper(
        vocab_size=vocab_size, n_mels=80,
        d_model=128, n_heads=4,
        encoder_layers=2, decoder_layers=2,
        max_source_len=100, max_target_len=32,
    ).eval()
    print(f"Whisper Mini | 参数量: {sum(p.numel() for p in model.parameters()):,}")

    batch = 1
    t_mel = 50    # conv2 stride=2 → encoder 序列约 25
    mel = torch.randn(batch, 80, t_mel)
    tgt = torch.randint(0, vocab_size, (batch, 8))

    with torch.inference_mode():
        logits = model(mel, tgt)

    print(f"mel: {tuple(mel.shape)}  decoder_input: {tuple(tgt.shape)}")
    print(f"logits: {tuple(logits.shape)}")
    assert logits.shape == (batch, 8, vocab_size)
    print("✅ Whisper 前向通过")


if __name__ == "__main__":
    main()
