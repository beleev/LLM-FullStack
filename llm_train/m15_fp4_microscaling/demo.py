"""
M15 — Microscaling FP4: MXFP4 与 NVFP4

是什么: 每个元素只用 4 位浮点 E2M1 (±{0, 0.5, 1, 1.5, 2, 3, 4, 6}, 共 15 个值), 靠 **小 block 共享 scale** 撑住精度。
解决的瓶颈: 显存带宽 / 显存容量 / 算力 (Blackwell 原生 FP4 GEMM; gpt-oss 以 MXFP4 发布权重)。
两种格式只差在 scale 怎么存:
    MXFP4 (OCP 标准): block = 32, scale = E8M0 (纯 2 的幂, 8 位)                → 4 + 8/32 = 4.25 bit/元素
    NVFP4 (NVIDIA)  : block = 16, scale = E4M3 (带 3 位尾数) × 一个 per-tensor FP32 scale → 4 + 8/16 = 4.5 bit/元素
关键直觉: 2 的幂 scale 对不准 amax (amax/scale 落在 [4, 8) 的哪里全凭运气, >6 还会饱和); E4M3 scale 让 amax 精确贴到 6。
读代码盯住: 两个函数里 `scale` 那一行 —— 其余完全一样。
说明: E2M1 / E4M3 都用 core.fake_quant_float 模拟; INT4 作为同位宽的整数对照。
"""
from __future__ import annotations

import numpy as np

from llm_train.core import banner, fake_quant_float, kv, make_rng, quant_blockwise

E2M1_MAX = 6.0


def _blocks(x: np.ndarray, block: int) -> np.ndarray:
    assert x.size % block == 0
    return x.reshape(-1, block)                                          # [n_blocks, block]


def mxfp4(x: np.ndarray, block: int = 32) -> np.ndarray:
    b = _blocks(x, block)
    amax = np.abs(b).max(axis=1, keepdims=True)
    # E8M0: scale = 2^(⌊log2 amax⌋ − 2); 2 是 E2M1 最大指数 (6 = 1.5·2²)。amax/scale ∈ [4, 8) → 大于 6 的会饱和
    scale = 2.0 ** (np.floor(np.log2(np.maximum(amax, 1e-30))) - 2)
    return (fake_quant_float(b / scale, "e2m1") * scale).reshape(x.shape)


def nvfp4(x: np.ndarray, block: int = 16) -> np.ndarray:
    b = _blocks(x, block)
    amax = np.abs(b).max(axis=1, keepdims=True)
    tensor_scale = np.abs(x).max() / (448.0 * E2M1_MAX)                  # FP32, 整个张量 1 个: 把 block scale 搬进 E4M3 的范围
    scale = fake_quant_float(amax / E2M1_MAX / tensor_scale, "e4m3") * tensor_scale   # block scale 本身只有 8 位
    scale[scale == 0] = 1.0
    return (fake_quant_float(b / scale, "e2m1") * scale).reshape(x.shape)


def int4(x: np.ndarray, block: int = 32) -> np.ndarray:
    b = _blocks(x, block)
    scale = np.abs(b).max(axis=1, keepdims=True) / 7.0                   # 对称 INT4: 整数 −7..7 均匀网格
    scale[scale == 0] = 1.0
    return (np.clip(np.round(b / scale), -7, 7) * scale).reshape(x.shape)


def rel_err(q, x) -> float:
    return float(np.linalg.norm(q - x) / np.linalg.norm(x))


def main() -> None:
    banner("M15 - Microscaling FP4 (MXFP4 / NVFP4)")
    rs = make_rng(15)

    grid = np.unique(np.abs(fake_quant_float(np.linspace(-8, 8, 3201), "e2m1")))
    kv("E2M1 的全部非负取值", grid.tolist())
    assert grid.tolist() == [0, 0.5, 1, 1.5, 2, 3, 4, 6]

    n = 1 << 14
    tensors = {
        "高斯 (权重)": rs.randn(n) * 0.02,
        "重尾 (激活/梯度)": rs.standard_t(df=3, size=n) * 0.02,          # t 分布: 偶发大值
    }
    methods = {
        "FP8 E4M3 block128 (8.25 bit)": lambda t: quant_blockwise(t, "e4m3", 128),
        "NVFP4  block16    (4.50 bit)": nvfp4,
        "MXFP4  block32    (4.25 bit)": mxfp4,
        "INT4   block32    (4.50 bit)": int4,                             # scale 按 FP16 计
    }
    print(f"\n  {'相对量化误差 ‖Q(x)−x‖/‖x‖':<34}" + "".join(f"{k:>18}" for k in tensors))
    err = {}
    for name, fn in methods.items():
        err[name] = [rel_err(fn(t), t) for t in tensors.values()]
        print(f"  {name:<34}" + "".join(f"{e:>20.4f}" for e in err[name]))

    fp8, nv, mx, i4 = (np.array(err[k]) for k in methods)
    assert (fp8 < nv).all() and (nv < mx).all(), "FP8 < NVFP4 < MXFP4: 位宽, 然后是 scale 的精细度"
    print()
    kv("FP8 → NVFP4 误差放大", f"{(nv / fp8).round(1).tolist()}x  (位宽砍半的代价)")
    kv("MXFP4 / NVFP4", f"{(mx / nv).round(2).tolist()}x  (2 的幂 scale + 大 block 的代价)")
    kv("INT4 / NVFP4 (高斯, 重尾)", f"{(i4 / nv).round(2).tolist()}x")
    kv("INT4 / MXFP4 (高斯, 重尾)", f"{(i4 / mx).round(2).tolist()}x  ← 高斯上 INT4 反而赢: FP4 不是白送的")
    assert i4[1] / nv[1] > i4[0] / nv[0], "重尾数据上浮点网格 (对数间距) 相对整数网格 (均匀间距) 的优势更大"
    assert i4[0] < mx[0] and mx[1] < i4[1], "同 block 下: 高斯 → 均匀网格更准; 重尾 → 对数网格更准"

    # ---- scale 格式单独消融: 同为 block 16, 只换 scale ----
    x = tensors["高斯 (权重)"]
    mx16 = rel_err(mxfp4(x, block=16), x)
    kv("同为 block16: E8M0 scale vs E4M3 scale", f"{mx16:.4f} vs {nv[0]:.4f}")
    assert nv[0] < mx16, "block 大小相同时, 差距全部来自 scale 的精度"

    # ---- MXFP4 的饱和: amax/scale ∈ (6, 8) 的 block 最大值被截到 6 ----
    b = _blocks(x, 32)
    amax = np.abs(b).max(1)
    ratio = amax / 2.0 ** (np.floor(np.log2(amax)) - 2)
    kv("MXFP4 中最大值被饱和截断的 block", f"{np.mean(ratio > 6):.0%}")
    assert 0.2 < np.mean(ratio > 6) < 0.7

    print("\n  OK: 误差排序 FP8 < NVFP4 < MXFP4; FP4 目前主要用于推理权重和 QAT, 全程 FP4 预训练仍需额外技巧")
    print("      (随机舍入、Hadamard 旋转去 outlier、关键层保留 BF16)。")


if __name__ == "__main__":
    main()
