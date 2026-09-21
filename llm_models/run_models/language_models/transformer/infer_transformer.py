#!/usr/bin/env python
"""
Transformer (Encoder-Decoder) 推理: 用断言验证三种 mask 与 cross-attention, 再做一次贪心解码

盯住: encode() 只跑一次得到 memory; decode() 每步重跑 (教学版无 KV cache)。
"""

import torch

from llm_models.models.foundation.transformer import Transformer
from llm_models.utils.masks import get_pad_mask, get_subsequent_mask


def main():
    torch.manual_seed(42)
    V_src, V_tgt, B, S, T = 100, 120, 2, 7, 6
    model = Transformer(V_src, V_tgt, d_model=64, n_heads=4, num_layers=2, d_ff=128, use_rope=True).eval()
    print(f"Transformer Mini | 参数量: {sum(p.numel() for p in model.parameters()):,}")

    src = torch.randint(1, V_src, (B, S)); src[:, -2:] = 0             # 末尾 2 个 pad (pad_idx=0)
    tgt = torch.randint(1, V_tgt, (B, T))
    src_mask = get_pad_mask(src, pad_idx=0)                            # [B, 1, S]
    tgt_mask = get_pad_mask(tgt, pad_idx=0) & get_subsequent_mask(tgt)  # [B,1,T] & [1,T,T] -> [B,T,T]
    assert src_mask.shape == (B, 1, S) and tgt_mask.shape == (B, T, T)

    with torch.inference_mode():
        out = model(src, tgt, src_mask, tgt_mask)
        assert out.shape == (B, T, V_tgt)

        # 1) 因果: 改 tgt 最后一个 token, 前面位置的 logits 不变
        tgt2 = tgt.clone(); tgt2[:, -1] = tgt2[:, -1] % (V_tgt - 1) + 1
        d_causal = (model(src, tgt2, src_mask, tgt_mask)[:, :-1] - out[:, :-1]).abs().max().item()
        # 2) cross-attention: 改 src 的有效 token, 所有 tgt 位置的 logits 都变
        src2 = src.clone(); src2[:, 0] = src2[:, 0] % (V_src - 1) + 1
        d_cross = (model(src2, tgt, src_mask, tgt_mask) - out).abs().amax(dim=(0, 2)).min().item()
        # 3) 源 padding: 改 pad 位置上的 token (mask 不变), 输出不变
        src3 = src.clone(); src3[:, -2:] = 5
        d_pad = (model(src3, tgt, src_mask, tgt_mask) - out).abs().max().item()
    print(f"改未来 tgt → 过去 logits 变化 {d_causal:.2e} | 改 src → 每个 tgt 位置至少变化 {d_cross:.4f} | 改 src pad → {d_pad:.2e}")
    assert d_causal < 1e-5 and d_cross > 1e-4 and d_pad < 1e-5

    # 4) 贪心解码: encoder 只跑一次; BOS=1
    with torch.inference_mode():
        memory = model.encode(src, src_mask)                           # [B, S, D]
        ys = torch.ones(B, 1, dtype=torch.long)
        for _ in range(5):
            logits = model.decode(ys, memory, src_mask, get_subsequent_mask(ys))
            ys = torch.cat([ys, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
    print(f"贪心解码输出: {ys.tolist()}")
    assert ys.shape == (B, 6)


if __name__ == "__main__":
    main()
