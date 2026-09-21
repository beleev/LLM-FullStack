#!/usr/bin/env python
"""
MTP 推理示例 — 一次前向, 多步预测

    1. 参数开销: 每级 MTP = 1 个拼接投影 [2D→D] + 1 个 Block + 3 个 RMSNorm (embedding / lm_head 共享)
    2. 输出: main logits 预测 t+1, 第 k 路 mtp logits 预测 t+1+k
    3. MTP 支路不改变主干: 主 logits 与 "同权重的纯 LLaMA" 完全相同 → 部署时可整体丢弃
    4. 生成 (KV cache) 只跑主干; 有/无 cache 输出一致
"""

import torch

from llm_models.models.language_models.llama import LLaMA
from llm_models.models.language_models.mtp import MTPLLaMA
from llm_models.utils.generation import benchmark_kv_cache


@torch.inference_mode()
def main():
    torch.manual_seed(42)
    vocab_size, T, D = 1000, 16, 256
    common = dict(vocab_size=vocab_size, d_model=D, n_heads=8, num_kv_heads=2,
                  num_layers=2, max_len=256, dropout=0.0)
    llama, mtp = LLaMA(**common).eval(), MTPLLaMA(**common, mtp_depth=1).eval()

    n_llama = sum(p.numel() for p in llama.parameters())
    n_mtp = sum(p.numel() for p in mtp.parameters())
    n_block = sum(p.numel() for p in llama.layers[0].parameters())
    assert n_mtp - n_llama == 2 * D * D + n_block + 3 * D
    print(f"[1] LLaMA {n_llama:,} → MTPLLaMA {n_mtp:,} (+{(n_mtp - n_llama) / n_llama:.1%}; "
          f"玩具主干只有 2 层所以占比大, DeepSeek-V3 61 层上仅 ~2%)")

    idx = torch.randint(1, vocab_size, (2, T))
    out = mtp(idx)
    assert out["logits"].shape == (2, T, vocab_size)
    assert [tuple(l.shape) for l in out["mtp_logits"]] == [(2, T, vocab_size)]
    draft_1 = int(out["logits"][0, -1].argmax())
    draft_2 = int(out["mtp_logits"][0][0, -1].argmax())
    print(f"[2] 一次前向拿到 2 个草稿 token: t+1={draft_1}, t+2={draft_2} (投机解码的 draft 来源)")

    llama.load_state_dict(mtp.state_dict(), strict=False)      # 只拷主干 (多出来的 mtp_modules.* 被忽略)
    assert torch.equal(llama(idx), out["logits"])
    print("[3] 主 logits 与同权重 LLaMA 完全相同 → MTP 模块可零成本丢弃")

    speedup = benchmark_kv_cache(mtp, idx[:, :8], max_new_tokens=200)
    print(f"[4] 生成 200 token: 有/无 cache 输出一致, 加速 {speedup:.1f}x")


if __name__ == "__main__":
    main()
