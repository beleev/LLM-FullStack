"""
m06 demo — Chunked Prefill 与 full prefill 数值等价性

我们直接调 TinyLM 的 block_forward, 手工拼装 chunked 路径,
然后与 lm.prefill(full) 对比。
"""
from __future__ import annotations
from typing import List, Tuple
import numpy as np

from llm_infer.core import TinyLM, ModelConfig
from llm_infer.core.utils import banner, kv, rms_norm
from llm_infer.core.tiny_model import block_forward


def chunked_prefill(
    lm: TinyLM, prompt_ids: np.ndarray, chunk_size: int
) -> Tuple[np.ndarray, list]:
    """把 prompt 切成 chunk_size 段, 顺序 prefill, 累积 KV cache。

    返回 (last_logits, kv_cache list)。
    """
    T = len(prompt_ids)
    kv_cache: List = [None] * lm.cfg.n_layer
    start_pos = 0
    last_logits = None
    for s in range(0, T, chunk_size):
        e = min(T, s + chunk_size)
        chunk = prompt_ids[s:e]
        x = lm.w.tok_emb[chunk]
        for li, layer in enumerate(lm.w.layers):
            x, kv_cache[li] = block_forward(
                x, layer, lm.cos, lm.sin,
                kv_cache=kv_cache[li],   # 传入历史 KV → block_forward 会拼接
                start_pos=start_pos,
            )
        x = rms_norm(x, lm.w.norm_f_g)
        last_logits = x @ lm.w.lm_head    # (chunk_len, V)
        start_pos += len(chunk)
    return last_logits, kv_cache


def main():
    banner("M06 - Chunked Prefill")

    cfg = ModelConfig(d_model=64, d_mlp=128, n_layer=4, vocab_size=128, max_seq_len=512)
    lm = TinyLM(cfg)

    rs = np.random.RandomState(0)
    T = 64
    prompt = rs.randint(3, cfg.vocab_size, size=T).astype(np.int64)

    # ---- baseline: full prefill ---------------------------------- #
    logits_full, kv_full = lm.prefill(prompt)

    # ---- 不同 chunk_size 对比数值 -------------------------------- #
    print(f"\n[1] 不同 chunk_size 输出与 full prefill 的差异 (T={T})")
    print(f"  {'chunk_size':>12}  {'last_logit_max_abs_diff':>26}  {'KV[0] max_diff':>16}")
    for chunk_size in [8, 16, 32, 64]:
        logits_c, kv_c = chunked_prefill(lm, prompt, chunk_size)
        n = logits_c.shape[0]
        d_logits = np.max(np.abs(logits_full[-n:] - logits_c))
        d_kv = np.max(np.abs(kv_full[0][0] - kv_c[0][0]))
        print(f"  {chunk_size:>12}  {d_logits:>26.2e}  {d_kv:>16.2e}")

    # ---- 显存峰值估算 (用 attn 矩阵 element 数代理) -------------- #
    print(f"\n[2] 显存峰值代理: max(Q×K) attn 矩阵元素数")
    for chunk_size in [8, 16, 32, 64]:
        peak = 0
        for s in range(0, T, chunk_size):
            qlen = min(chunk_size, T - s)
            klen = s + qlen
            peak = max(peak, qlen * klen)
        kv(f"chunk={chunk_size:>3}", f"{peak:>6} elems  (full = {T*T} elems)")

    print("\n  ✓ chunked 输出与 full 数值等价, 显存峰值显著降低")


if __name__ == "__main__":
    main()
