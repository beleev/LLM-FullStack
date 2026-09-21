"""
kv_quant.py — KV cache 量化: per-tensor / per-token / KIVI (K per-channel + V per-token)

是什么: KV cache 存成 INT8/INT4/INT2, 做 attention 前反量化回浮点。
解决的瓶颈: 长上下文 / 大 batch 下 KV cache 比权重还大 (显存), decode 每步还要把它整个读一遍 (带宽)。
关键数字: INT8 ≈ FP16 的 1/2、FP32 的 1/4; INT4 再减半。bit 越低, "组怎么划" 越决定成败。
为什么 K 要 per-channel (KIVI, Liu et al. 2024 的观察):
    K 里少数固定通道在**所有 token** 上都是大值 (离群通道)。per-token 分组时每一行都含这些离群值,
    scale 被它们撑大, 其余通道只剩一两个格点; 按通道分组则把离群值关在自己的组里。
    V 没有这种固定离群通道, 沿用 per-token (每个 token 写入时即可独立量化, 对流式追加友好)。
读代码盯住: `quantize_kv` 里三种 scheme 只差 reshape 与 `axis`。
对应真实系统: vLLM `--kv-cache-dtype fp8` (per-tensor scale); KIVI / KVQuant 的 2~4 bit 方案; LMDeploy 的 INT4/INT8 KV。
"""
from __future__ import annotations
import numpy as np

from llm_infer.m08_quantization.int8_weight import QTensor, quantize_affine

SCHEMES = ("per-tensor", "per-token", "kivi")


def quantize_kv(K: np.ndarray, V: np.ndarray, bits: int, scheme: str, group: int = 32):
    """K, V (T,D) → (QTensor_K, QTensor_V)。

    per-tensor: K、V 各一组 scale
    per-token : 每个 token (行) 一组, scale 形状 (T,1)
    kivi      : K 每 `group` 个 token 内按通道分组, scale 形状 (T/group,1,D); V 同 per-token
    """
    assert scheme in SCHEMES
    if scheme == "per-tensor":
        return quantize_affine(K, bits, None), quantize_affine(V, bits, None)
    qV = quantize_affine(V, bits, axis=1)                                 # 在 D 上统计 → (T,1)
    if scheme == "per-token":
        return quantize_affine(K, bits, axis=1), qV
    T, D = K.shape
    # 简化 (ponytail): 要求 T 是 group 的整数倍。真实 KIVI 把最近不足一组的 token 留在 fp16 "residual" 里, 凑满一组再量化
    assert T % group == 0
    return quantize_affine(K.reshape(T // group, group, D), bits, axis=1), qV   # 在组内 token 维上统计


def dequantize_kv(qK: QTensor, qV: QTensor):
    V = qV.dequantize()                                                   # (T,D)
    return qK.dequantize().reshape(V.shape[0], -1), V                     # kivi 的 (T/g,g,D) → (T,D)
