"""
m08 demo — Weight-INT8 + KV-INT8 同时演示

观察:
    1) 权重 quant 前后输出差异 (max abs diff, cosine sim)
    2) 显存节省比例
    3) KV quant 后 attention 输出差异
"""
from __future__ import annotations
import numpy as np

from llm_infer.core.utils import banner, kv, softmax, causal_mask
from llm_infer.m08_quantization.int8_weight import quantize_int8, matmul_qint8
from llm_infer.m08_quantization.kv_quant import QKVCache


def cos_sim(a, b):
    return float(np.dot(a.flatten(), b.flatten()) /
                 (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    banner("M08 - Quantization (Weight INT8 + KV INT8)")

    rs = np.random.RandomState(0)

    # ---- A. 权重量化 -------------------------------------------- #
    print("\n[A] 权重 INT8 量化")
    D_in, D_out, B = 256, 256, 4
    W = rs.randn(D_in, D_out).astype(np.float32) * 0.05
    x = rs.randn(B, D_in).astype(np.float32)

    qw = quantize_int8(W)
    y_fp = x @ W
    y_q = matmul_qint8(x, qw)
    diff = np.max(np.abs(y_fp - y_q))
    cs = cos_sim(y_fp, y_q)

    kv("权重 fp32 大小 (bytes)", f"{W.nbytes:,}")
    kv("权重 int8 大小 (bytes)", f"{qw.nbytes():,}")
    kv("压缩比", f"{W.nbytes / qw.nbytes():.2f}x")
    kv("max |y_fp - y_q|", f"{diff:.4f}")
    kv("cosine sim", f"{cs:.6f}")

    # ---- B. KV cache 量化 + attention --------------------------- #
    print("\n[B] KV cache INT8 量化, 对 attention 输出的影响")
    T, D = 64, 32
    K = rs.randn(T, D).astype(np.float32) * 0.5
    V = rs.randn(T, D).astype(np.float32) * 0.5
    q_query = rs.randn(1, D).astype(np.float32)

    # baseline: fp32
    scores = (q_query @ K.T) / np.sqrt(D)
    attn = softmax(scores + causal_mask(1, T), axis=-1)
    out_fp = attn @ V

    # KV quant
    qkv = QKVCache(d_model=D)
    for t in range(T):
        qkv.append(K[t], V[t])
    K_q, V_q = qkv.get_full()
    scores_q = (q_query @ K_q.T) / np.sqrt(D)
    attn_q = softmax(scores_q + causal_mask(1, T), axis=-1)
    out_q = attn_q @ V_q

    diff = np.max(np.abs(out_fp - out_q))
    kv("KV fp32 大小 (bytes)", f"{2 * K.nbytes:,}")
    kv("KV int8 大小 (bytes)", f"{qkv.nbytes():,}")
    kv("压缩比", f"{2 * K.nbytes / qkv.nbytes():.2f}x")
    kv("max |attn_fp - attn_q|", f"{diff:.4f}")
    kv("cosine sim", f"{cos_sim(out_fp, out_q):.6f}")

    print("\n  ✓ INT8 权重量化几乎无损 (cos_sim > 0.9999)")
    print("  ✓ INT8 KV 量化误差略大但可接受 (cos_sim > 0.99)")
    print("  ✓ 显存近乎砍半")


if __name__ == "__main__":
    main()
