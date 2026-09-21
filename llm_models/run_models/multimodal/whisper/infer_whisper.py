#!/usr/bin/env python
"""
Whisper 推理: 验证 Conv stem 的 2× 下采样、cross-attention 读到了音频、decoder 因果; 再做贪心解码
"""

import torch

from llm_models.models.multimodal.whisper import Whisper


def main():
    torch.manual_seed(42)
    V, B, T_mel, T = 1000, 2, 100, 8
    model = Whisper(
        vocab_size=V, n_mels=80, d_model=128, n_heads=4,
        encoder_layers=2, decoder_layers=2, max_source_len=200, max_target_len=64,
    ).eval()
    print(f"Whisper Mini | 参数量: {sum(p.numel() for p in model.parameters()):,}")

    mel = torch.randn(B, 80, T_mel)
    ids = torch.randint(1, V, (B, T))
    with torch.inference_mode():
        enc = model.encoder(mel)
        assert enc.shape == (B, T_mel // 2, 128)                       # stride=2: 100 帧 → 50 帧
        logits = model(mel, ids)
        assert logits.shape == (B, T, V)

        # 1) 换一段音频, 同样的文本前缀 → logits 必须变 (音频只能经 cross-attn 进入 decoder)
        d_audio = (model(torch.randn(B, 80, T_mel), ids) - logits).abs().max().item()
        # 2) decoder 因果: 改最后一个 token, 前面位置不变
        ids2 = ids.clone(); ids2[:, -1] = ids2[:, -1] % (V - 1) + 1
        d_causal = (model(mel, ids2)[:, :-1] - logits[:, :-1]).abs().max().item()
        print(f"mel {tuple(mel.shape)} -> encoder {tuple(enc.shape)} | 换音频 logits 变化 {d_audio:.4f} | 改未来 token 过去变化 {d_causal:.1e}")
        assert d_audio > 1e-3 and d_causal < 1e-5

        # 3) 贪心解码: encoder 只跑一次
        ys = ids[:, :2]                                                # 假装是 <|startoftranscript|><|zh|> 之类的 task prompt
        for _ in range(6):
            nxt = model.decoder(ys, encoder_hidden=enc)[:, -1].argmax(-1, keepdim=True)
            ys = torch.cat([ys, nxt], dim=1)
        assert ys.shape == (B, 8)
    print(f"贪心解码 (未训练, 无意义): {ys[0].tolist()}")


if __name__ == "__main__":
    main()
