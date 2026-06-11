"""
M13 — FP8 训练 (DeepSeek-V3 同款思路, numpy 模拟)

m06 讲了 FP16/BF16 混合精度; FP8 把"半精度"再砍一半:
    E4M3 (4 位指数 3 位尾数):  范围 ±448,    精度高  → 权重 / 激活
    E5M2 (5 位指数 2 位尾数):  范围 ±57344,  范围大  → 梯度 (动态范围更野)

天上不会掉算力: 尾数只剩 2~3 位, 量化误差比 FP16 大一个量级。
FP8 训练能 work 靠的是三件套 (DeepSeek-V3 公开配方):
    1. **block-wise scaling**: 不是整个张量共享一个缩放因子, 而是每
       128 个元素一组各自缩放 —— 单个 outlier 只毁掉自己那一小块,
       而不是把全张量的有效精度拖下水
    2. **FP32 master weights + 高精度累加**: 乘法用 FP8, 加法 (GEMM
       累加 / 梯度累积 / 优化器更新) 留在高精度
    3. 关键路径 (norm / softmax / 优化器状态) 不量化

说明: numpy 没有原生 fp8 dtype, 本 demo 用 "encode→decode" 假量化
      (fake quantization) 模拟数值效应, 与真卡上的舍入行为一致。
"""
from __future__ import annotations

import numpy as np

from llm_train.core import LinearModel, ToyDataStream, banner, kv


# --------------------------------------------------------------------- #
# FP8 假量化: 把 float64 舍入到 E4M3 / E5M2 可表示的最近值              #
# --------------------------------------------------------------------- #

FP8_FORMATS = {
    # (尾数位数, 最大可表示值, 最小 normal 指数)
    # E4M3: 最小 normal 2^-6 ≈ 0.0156, subnormal 步长 2^-9 ≈ 0.00195
    # E5M2: 最小 normal 2^-14,         subnormal 步长 2^-16
    "e4m3": (3, 448.0, -6),
    "e5m2": (2, 57344.0, -14),
}


def quantize_fp8(x: np.ndarray, fmt: str = "e4m3") -> np.ndarray:
    """逐元素舍入到 FP8 网格 (含上界饱和、subnormal 区与下溢冲零)。

    浮点数的相对精度在 normal 区内与量级无关 (恒为 ~2^-(m+1)),
    但跌破最小 normal 后进入 subnormal 区: 网格变成**固定步长**,
    越小的数相对误差越大, 低于半步长直接下溢成 0 —— 这正是
    "为什么 FP8 必须配 scaling" 的数值根源。
    """
    m_bits, max_val, e_min = FP8_FORMATS[fmt]
    out = np.clip(x, -max_val, max_val)

    mant, exp = np.frexp(out)                       # x = mant * 2^exp, mant∈[0.5,1)
    step = 2.0 ** (m_bits + 1)                      # 尾数网格密度
    q_normal = np.ldexp(np.round(mant * step) / step, exp)

    quantum = 2.0 ** (e_min - m_bits)               # subnormal 固定步长
    q_subnormal = np.round(out / quantum) * quantum  # 下溢: |x| < quantum/2 → 0

    return np.where(np.abs(out) >= 2.0 ** e_min, q_normal, q_subnormal)


def quantize_blockwise(x: np.ndarray, fmt: str = "e4m3", block: int = 128) -> np.ndarray:
    """block-wise scaling 假量化: 每 block 个元素一组, 缩放到满量程再量化。

    真卡上 scale 以 FP32 单独存储 (DeepSeek-V3: 激活 1x128, 权重 128x128 tile)。
    """
    _, max_val, _ = FP8_FORMATS[fmt]
    flat = x.reshape(-1)
    pad = (-len(flat)) % block
    padded = np.concatenate([flat, np.zeros(pad)]) if pad else flat
    blocks = padded.reshape(-1, block)

    scale = np.abs(blocks).max(axis=1, keepdims=True) / max_val   # 每块一个 scale
    scale = np.where(scale == 0, 1.0, scale)
    deq = quantize_fp8(blocks / scale, fmt) * scale               # 量化→反缩放
    return deq.reshape(-1)[: len(flat)].reshape(x.shape)


def rel_err(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def main() -> None:
    banner("M13 - FP8 Training (E4M3 / E5M2 + block scaling)")

    rs = np.random.RandomState(0)

    # ---- 1) 两种 FP8 格式的取舍: 精度 vs 范围 ----
    print("\n[1] 格式取舍 (同一正态张量的量化相对误差)")
    x = rs.randn(4096)
    kv("E4M3 (3 位尾数, ±448)", f"{rel_err(quantize_fp8(x, 'e4m3'), x):.4f}  ← 权重/激活")
    kv("E5M2 (2 位尾数, ±57344)", f"{rel_err(quantize_fp8(x, 'e5m2'), x):.4f}  ← 梯度")
    big = x * 1e5                                   # 模拟梯度尖峰
    kv("E4M3 遇到 1e5 量级尖峰", f"{rel_err(quantize_fp8(big, 'e4m3'), big):.4f}  (饱和截断!)")
    kv("E5M2 遇到 1e5 量级尖峰", f"{rel_err(quantize_fp8(big, 'e5m2'), big):.4f}")

    # ---- 2) outlier 毁掉 per-tensor scaling, block-wise 救回来 ----
    print("\n[2] 为什么必须 block-wise scaling (重尾梯度 + 0.1% 尖峰)")
    # 真实梯度跨多个数量级 (重尾), 偶发尖峰再大 3~4 个数量级
    grad = rs.randn(4096) * np.exp(rs.randn(4096))
    grad[rs.choice(4096, 4, replace=False)] = 3e4    # 梯度尖峰

    # per-tensor: 为容纳尖峰, scale = 3e4/448 ≈ 67 → 普通值被压进
    # subnormal 区 (固定步长网格), 小梯度直接下溢成 0
    scale_t = np.abs(grad).max() / FP8_FORMATS["e4m3"][1]
    per_tensor = quantize_fp8(grad / scale_t, "e4m3") * scale_t
    per_block = quantize_blockwise(grad, "e4m3", block=128)

    def elem_err_median(q: np.ndarray) -> float:
        nz = np.abs(grad) > 0
        return float(np.median(np.abs(q[nz] - grad[nz]) / np.abs(grad[nz])))

    def flushed(q: np.ndarray) -> float:
        return float(np.mean((q == 0) & (grad != 0)))

    kv("per-tensor 逐元素中位误差", f"{elem_err_median(per_tensor):.1%}")
    kv("per-tensor 被冲成 0 的比例", f"{flushed(per_tensor):.1%}")
    kv("block-wise(128) 中位误差", f"{elem_err_median(per_block):.1%}")
    kv("block-wise(128) 冲零比例", f"{flushed(per_block):.1%}")
    assert elem_err_median(per_block) < elem_err_median(per_tensor) / 2
    assert flushed(per_block) < flushed(per_tensor)

    # ---- 3) 端到端: FP8 训练 vs FP32 基线 ----
    print("\n[3] 训练对比 (线性回归 60 步, 同一数据流)")

    def train(mode: str, steps: int = 60, lr: float = 0.1) -> list[float]:
        """mode: fp32 | fp8_block (FP32 master) | fp8_naive (无 master, per-tensor)"""
        stream = ToyDataStream(d_in=8, d_out=4, batch_size=32, seed=123)
        model = LinearModel.init(d_in=8, d_out=4, seed=3)
        master_W = model.W.astype(np.float64).copy()   # FP32 master 权重
        losses = []
        for _ in range(steps):
            x, y = stream.next_batch()
            if mode != "fp32":
                blockwise = mode == "fp8_block"
                q = (lambda t, f: quantize_blockwise(t, f, 128)) if blockwise \
                    else (lambda t, f: quantize_fp8(t, f))
                # 前向/反向的乘法输入量化: 权重&激活走 E4M3, 梯度走 E5M2
                model.W[...] = q(master_W if blockwise else model.W, "e4m3")
                x = q(x, "e4m3")
            loss, grads = model.loss_and_grads(x, y)
            losses.append(loss)
            if mode == "fp8_block":
                g = quantize_blockwise(grads["W"], "e5m2", 128)
                master_W -= lr * g                    # 高精度 master 上更新
                model.b -= lr * grads["b"]
            elif mode == "fp8_naive":
                g = quantize_fp8(grads["W"], "e5m2")
                model.W -= lr * g                     # 直接在 FP8 权重上更新
                model.b -= lr * grads["b"]
            else:
                model.apply_grads(grads, lr)
        return losses

    base = train("fp32")
    fp8_good = train("fp8_block")
    fp8_naive = train("fp8_naive")
    kv("FP32 基线  final loss", f"{base[-1]:.6f}")
    kv("FP8 三件套 final loss", f"{fp8_good[-1]:.6f}  (block scaling + master)")
    kv("FP8 裸跑   final loss", f"{fp8_naive[-1]:.6f}  (per-tensor, 无 master)")
    kv("三件套比裸跑好", f"{fp8_naive[-1] / fp8_good[-1]:.0f}x (基本贴住 FP32 曲线)")

    assert abs(fp8_good[-1] - base[-1]) < abs(fp8_naive[-1] - base[-1]), \
        "带 scaling+master 的 FP8 应显著好于裸跑 FP8"

    print("\n  OK: FP8 不是免费午餐 —— 乘法省一半, 但 scaling 粒度和高精度")
    print("      累加/master 一个都不能少 (DeepSeek-V3 是首个全程 FP8 的前沿模型)。")


if __name__ == "__main__":
    main()
