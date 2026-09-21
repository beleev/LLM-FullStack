"""
m08 demo — 量化的核心问题只有一个: min/max 的 "组" 怎么划

[A] weight-only: RTN INT8 / RTN INT4 per-channel / RTN INT4 group-wise / AWQ INT4 group-wise
    激活带少数离群通道; 比较输出误差 ‖XW−XŴ‖/‖XW‖ 与含 scale/zero 开销的等效 bit 数。
[B] KV cache: per-tensor / per-token / KIVI (K per-channel + V per-token), INT8 / INT4 / INT2
    K 带固定离群通道; 比较 K 重建误差与 attention 输出 vs core.dense_attention。
数据是人工合成的 (离群通道的位置与倍数都打印出来), 用来把真实 LLM 里观察到的现象放大到肉眼可见。
"""
from __future__ import annotations
import numpy as np

from llm_infer.core import dense_attention
from llm_infer.core.utils import banner, kv
from llm_infer.m08_quantization.int8_weight import quantize_int8
from llm_infer.m08_quantization.int4_awq import quantize_groupwise, awq_quantize, output_err
from llm_infer.m08_quantization.kv_quant import SCHEMES, quantize_kv, dequantize_kv

ATTN_BOUND = {8: 1e-2, 4: 5e-2}          # KIVI 的 attention 输出 max-abs-diff 上界 (INT2 不设界, 只看排序)


def rel_err(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(a))


def weight_demo(rs):
    print("\n[A] weight-only 量化: 输出误差 ‖XW−XŴ‖/‖XW‖ (held-out 激活)")
    D_in, D_out, N, g = 256, 256, 256, 32
    outliers, mult = [7, 50, 131, 200], 30.0
    W = (rs.randn(D_in, D_out) * 0.05).astype(np.float32)
    X = rs.randn(2 * N, D_in).astype(np.float32)
    X[:, outliers] *= mult                                # 离群输入通道: 每个 token 上都大 (LLM 激活的典型形态)
    X_calib, X_test = X[:N], X[N:]                        # AWQ 只在 calib 上选 α, 误差在 test 上报告
    kv("W / X", f"({D_in},{D_out}) / ({N},{D_in}), 离群输入通道 {outliers} × {mult:g}")

    q8 = quantize_int8(W)
    q4c = quantize_groupwise(W, 4, None)
    q4g = quantize_groupwise(W, 4, g)
    W_awq, alpha, s, errs = awq_quantize(W, X_calib, 4, g)
    rows = [("RTN INT8 per-channel (对称)", q8.dequantize(), q8.nbytes()),
            ("RTN INT4 per-channel (非对称)", q4c.dequantize().reshape(W.shape), q4c.nbytes()),
            (f"RTN INT4 group={g} (非对称)", q4g.dequantize().reshape(W.shape), q4g.nbytes()),
            (f"AWQ INT4 group={g} (α={alpha})", W_awq, q4g.nbytes())]        # AWQ 存储格式与 group-wise 相同, 1/s 折进上一层
    print(f"  {'方案':<30} {'输出误差':>9} {'bytes':>9} {'bit/权重':>9} {'vs FP16':>8}")
    errs_test = [output_err(X_test, W, W_hat) for _, W_hat, _ in rows]
    for (name, _, nbytes), err in zip(rows, errs_test):
        print(f"  {name:<30} {err:>9.4f} {nbytes:>9,} {nbytes * 8 / W.size:>9.2f} {W.size * 2 / nbytes:>7.2f}x")
    kv("FP16 bytes", f"{W.size * 2:,}")
    kv("α 网格 (calib 误差)", "  ".join(f"{a:g}:{v:.4f}" for a, v in errs.items() if a in (0, 0.2, 0.4, 0.6, 0.8, 1.0)))
    kv("s 在离群通道 / 其余通道(中位数)", f"{s[outliers].mean():.2f} / {np.median(s):.2f}")
    e8, e4c, e4g, eawq = errs_test
    assert e8 < eawq < e4g < e4c, (e8, eawq, e4g, e4c)
    assert eawq < 0.7 * e4g                               # AWQ 在 4-bit 下至少再降 30% 输出误差
    assert 0 < alpha < 1                                  # α=0 (RTN) 与 α=1 (只顾离群通道) 都不是最优


def kv_demo(rs):
    print("\n[B] KV cache 量化: K 有固定离群通道")
    T, Tq, D, group = 128, 8, 64, 32
    outliers = [3, 17, 40]
    K = (rs.randn(T, D) * 0.5).astype(np.float32)
    K[:, outliers] = (rs.randn(T, len(outliers)) * 2 + np.array([8.0, -8.0, 8.0])).astype(np.float32)  # 所有 token 上都大
    V = (rs.randn(T, D) * 0.5).astype(np.float32)
    q = rs.randn(Tq, D).astype(np.float32)
    ref = dense_attention(q, K, V)                        # (Tq,D) 浮点基线; 最后 Tq 个位置的 causal attention
    kv("K / V / q", f"({T},{D}) / ({T},{D}) / ({Tq},{D}), K 离群通道 {outliers} ≈ ±8, 其余 std 0.5")
    fp32 = K.nbytes + V.nbytes

    print(f"  {'bits':>4} {'scheme':<11} {'K 相对误差':>10} {'V 相对误差':>10} {'attn max|Δ|':>12} {'bytes':>7} {'vs FP32':>8} {'vs FP16':>8}")
    res = {}
    for bits in (8, 4, 2):
        for scheme in SCHEMES:
            qK, qV = quantize_kv(K, V, bits, scheme, group)
            K_hat, V_hat = dequantize_kv(qK, qV)
            nbytes = qK.nbytes() + qV.nbytes()
            r = res[bits, scheme] = (rel_err(K, K_hat), rel_err(V, V_hat),
                                     float(np.max(np.abs(ref - dense_attention(q, K_hat, V_hat)))))
            print(f"  {bits:>4} {scheme:<11} {r[0]:>10.4f} {r[1]:>10.4f} {r[2]:>12.4f} {nbytes:>7,} "
                  f"{fp32 / nbytes:>7.2f}x {fp32 / 2 / nbytes:>7.2f}x")
    kv("FP32 / FP16 bytes", f"{fp32:,} / {fp32 // 2:,}")

    for bits in (8, 4, 2):
        assert res[bits, "kivi"][0] < res[bits, "per-token"][0] < res[bits, "per-tensor"][0]   # K 误差: 组划得越对越小
        assert res[bits, "kivi"][2] < res[bits, "per-token"][2]                               # attention 输出同样受益
    for bits, bound in ATTN_BOUND.items():
        assert res[bits, "kivi"][2] < bound, (bits, res[bits, "kivi"][2])


def main():
    banner("M08 - Quantization (RTN / group-wise INT4 / AWQ / KIVI)")
    rs = np.random.RandomState(0)
    weight_demo(rs)
    kv_demo(rs)
    print("\n  ✓ 4-bit 权重: AWQ < RTN group-wise < RTN per-channel (输出误差); AWQ 与 RTN group-wise 存储相同")
    print(f"  ✓ KV: KIVI 的 K 误差 < per-token < per-tensor; attention max|Δ| 上界 INT8 {ATTN_BOUND[8]:g} / INT4 {ATTN_BOUND[4]:g}")
    print("  ✓ KV INT8 ≈ FP32 的 1/4、FP16 的 1/2 (扣掉 scale 开销后略少)")


if __name__ == "__main__":
    main()
