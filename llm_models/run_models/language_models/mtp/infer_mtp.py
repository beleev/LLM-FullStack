#!/usr/bin/env python
"""
MTP 前向示例 — 一次前向, 多步预测

重点观察:
    1. forward 返回 main logits (预测 t+1) + K 路 MTP logits (预测 t+1+k)
    2. MTP 模块的参数开销很小 (1 个拼接投影 + 1 个 Block, head/embedding 共享)
    3. "免费草稿": 主 head 出 t+1, MTP head 出 t+2 —— 这正是投机解码的 draft
       (对应 llm_infer/m07_speculative_decoding 的 draft 模型来源之一)
"""

import torch

from llm_models.models.language_models.llama import LLaMA
from llm_models.models.language_models.mtp import MTPLLaMA


def main():
    torch.manual_seed(42)

    vocab_size, T = 1000, 16
    common = dict(
        vocab_size=vocab_size, d_model=256, n_heads=8, num_kv_heads=2,
        num_layers=2, max_len=128, dropout=0.0,
    )
    llama = LLaMA(**common)
    mtp = MTPLLaMA(**common, mtp_depth=1).eval()

    n_llama = sum(p.numel() for p in llama.parameters())
    n_mtp = sum(p.numel() for p in mtp.parameters())
    print(f"LLaMA    参数量: {n_llama:,}")
    print(f"MTPLLaMA 参数量: {n_mtp:,}  (+{(n_mtp - n_llama) / n_llama:.1%}: "
          f"1 个拼接投影 + 1 个 Block)")
    print("(玩具模型只有 2 层主干所以占比大; DeepSeek-V3 61 层主干上 MTP 仅 ~1.5%)")

    idx = torch.randint(0, vocab_size, (2, T))
    with torch.inference_mode():
        out = mtp(idx)
    print(f"\n输入: {tuple(idx.shape)}")
    print(f"main logits:  {tuple(out['logits'].shape)}   (位置 i 预测 t_i+1)")
    for k, lg in enumerate(out["mtp_logits"], start=1):
        print(f"mtp-{k} logits: {tuple(lg.shape)}   (位置 i 预测 t_i+{k + 1})")

    # ---- "免费草稿" 演示: 最后一个位置同时给出未来 2 个 token 的猜测 ----
    last = T - 1
    draft_1 = int(out["logits"][0, last].argmax())
    draft_2 = int(out["mtp_logits"][0][0, last].argmax())
    print(f"\n一次前向拿到 2 个草稿 token: t+1={draft_1}, t+2={draft_2}")
    print("投机解码可先接受草稿、再批量验证 (DeepSeek-V3: 接受率 85%+, 解码 ~1.8x)")

    print("\n✅ MTP 前向通过 (部署时 MTP 模块可整体丢弃, 或留作 draft head)")


if __name__ == "__main__":
    main()
