"""
int4_awq.py — group-wise INT4 weight-only 量化 + AWQ 式激活感知缩放

是什么: 权重压到 4 bit。RTN 只看权重本身; AWQ (Lin et al. 2023) 还看激活: 输出误差 = Σ_i x_i·ΔW_i,
    激活大的输入通道 (salient, 约 1%) 上的权重误差被放大得最厉害, 应该优先保护。
解决的瓶颈: 权重显存与 decode 带宽 (FP16 → INT4 约 3.2~3.6×, 含 scale/zero 开销)。
做法: 量化前把第 i 个输入通道的权重行放大 s_i = mean|x_i|^α, 激活同步除以 s_i (数学上 XW 不变):
        X W = (X / s) · (s ⊙ W)   →   X Ŵ,   Ŵ = Q(s ⊙ W) / s
    放大后该行的相对舍入误差缩小约 s_i 倍; 代价是它撑大了所在 group 的 range, 同组其它行变粗
    → α 不能无脑取 1, 用校准集网格搜索 (α=0 即 RTN)。
量化器: 非对称 (min/max + zero), group 沿输入维划分 (每个输出通道内每 g 个相邻输入通道共用一组 scale/lo)。
读代码盯住: `s` 以及 `quant_dequant(W * s) / s`。
对应真实系统: AutoAWQ / vLLM `quantization="awq"`; 1/s 离线折进上一层 (RMSNorm 的 gamma 或前一个 Linear), 推理时零开销。
"""
from __future__ import annotations
import numpy as np

from llm_infer.m08_quantization.int8_weight import QTensor, quantize_affine

ALPHAS = tuple(np.round(np.arange(0, 1.01, 0.1), 1))


def quantize_groupwise(W: np.ndarray, bits: int = 4, group: int | None = 32) -> QTensor:
    """W (D_in, D_out)。group=None → per-channel (整列一组); 否则 reshape 成 (D_in/g, g, D_out) 在 g 上统计。"""
    D_in, D_out = W.shape
    g = D_in if group is None else group
    assert D_in % g == 0
    return quantize_affine(W.reshape(D_in // g, g, D_out), bits, axis=1)  # scale/lo: (D_in/g, 1, D_out)


def quant_dequant(W: np.ndarray, bits: int = 4, group: int | None = 32) -> np.ndarray:
    return quantize_groupwise(W, bits, group).dequantize().reshape(W.shape)


def output_err(X: np.ndarray, W: np.ndarray, W_hat: np.ndarray) -> float:
    """相对输出误差 ‖XW − XŴ‖_F / ‖XW‖_F —— 权重量化真正要最小化的量 (而不是 ‖W − Ŵ‖)。"""
    Y = X @ W                                                             # (N,D_in)@(D_in,D_out) → (N,D_out)
    return float(np.linalg.norm(Y - X @ W_hat) / np.linalg.norm(Y))


def awq_quantize(W: np.ndarray, X_calib: np.ndarray, bits: int = 4, group: int = 32):
    """返回 (Ŵ 等效浮点权重, 最优 α, s, {α: 校准集误差})。只用校准激活的逐通道统计量 mean|x|。"""
    act = np.mean(np.abs(X_calib), axis=0)                                # (D_in,) 每个输入通道的激活幅度
    errs, best = {}, None
    for alpha in ALPHAS:
        s = act ** alpha
        s = (s / np.sqrt(s.max() * s.min())).astype(np.float32)           # 归一化到几何中心为 1, 不整体放大 W
        W_hat = quant_dequant(W * s[:, None], bits, group) / s[:, None]   # (D_in,D_out): 行 i 放大 s_i → 量化 → 缩回
        errs[float(alpha)] = output_err(X_calib, W, W_hat)
        if best is None or errs[float(alpha)] < errs[best[1]]:
            best = (W_hat, float(alpha), s)
    return (*best, errs)
