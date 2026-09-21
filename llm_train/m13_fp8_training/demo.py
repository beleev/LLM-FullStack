"""
M13 — FP8 训练

是什么: 矩阵乘的输入 (权重 / 激活 / 输出梯度) 量化到 8 位浮点, 累加和参数更新留在高精度。
解决的瓶颈: 算力 + 显存带宽 (GEMM 约 2× 于 BF16)。代价: 尾数只剩 2~3 位, 动态范围很窄。
    E4M3: ±448,   最小 normal 2^-6,  相对误差 ~3%   |   E5M2: ±57344, 最小 normal 2^-14, 相对误差 ~6%
两套公开配方 (不要混为一谈):
    Transformer Engine: 前向 E4M3 / 反向梯度 E5M2, **per-tensor** scaling (用 E5M2 的范围兜住梯度)
    DeepSeek-V3:        **全程 E4M3**, 细粒度 scaling —— 激活 1×128 tile, 权重 128×128 block
                        (用更细的 scale 兜住范围, 于是梯度也能享受 E4M3 的精度)
共同点: FP32/BF16 master weights + 高精度优化器状态; norm / softmax / embedding / 输出头不量化。
读代码盯住: `make_quantizer(scaling, fmt)` 的两个旋钮, 以及 `master` 开关 —— 三个消融臂各只动一个变量。
说明: numpy 无 fp8 dtype, 用 core.fake_quant_float 做 "量化→反量化" 模拟舍入效应。
"""
from __future__ import annotations

import numpy as np

from llm_train.core import adam_update, banner, fake_quant_float, kv, quant_blockwise

F32 = np.float32
BLOCK = 128


def make_quantizer(scaling: str, fmt):
    """scaling ∈ {none, tensor, block}; fmt=None 表示不量化 (FP32 基线)。"""
    if fmt is None:
        return lambda t: t
    if scaling == "none":
        return lambda t: fake_quant_float(t, fmt).astype(F32)              # 直接舍入到 FP8 原生网格
    block = BLOCK if scaling == "block" else None
    return lambda t: quant_blockwise(t, fmt, block or t.size).astype(F32)  # block=整个张量 即 per-tensor


def rel_err(a, b) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def train(scaling: str, fwd_fmt, bwd_fmt, master: bool = True, steps: int = 200) -> float:
    """线性回归, FP8 GEMM: Y = Q(X)·Q(W),  dW = Q(X)ᵀ·Q(dY)。返回干净验证集上的相对 loss。

    量级刻意取真实 LLM 的水平: 权重 std 0.004, 逐元素梯度 ~1e-5 —— 都落在 FP8 原生网格的
    subnormal 区以下, 这正是 "不 scaling 就不能用" 的原因。
    """
    rs = np.random.RandomState(0)
    d_in, d_out, B = 128, 32, 128
    W_true = (rs.randn(d_in, d_out) * 0.004).astype(F32)
    W = np.zeros((d_in, d_out), dtype=F32)                                # master (真的是 float32)
    m, v = np.zeros_like(W), np.zeros_like(W)                             # Adam 状态: 始终高精度
    q_fwd, q_bwd = make_quantizer(scaling, fwd_fmt), make_quantizer(scaling, bwd_fmt)
    x_val = rs.randn(256, d_in).astype(F32)
    y_val = x_val @ W_true

    for t in range(1, steps + 1):
        x = rs.randn(B, d_in).astype(F32)                                 # [B, 128]: block=128 恰好是 1×128 per-token tile
        y = x @ W_true
        xq, wq = q_fwd(x), q_fwd(W)                                       # GEMM 的两个输入都量化
        diff = xq @ wq - y                                                # 输出与 loss 留在高精度
        g = q_bwd((diff * F32(2.0 / diff.size)).astype(F32))              # 输出梯度 [B, 32], 量级 ~1e-5
        dW = (xq.T @ g).astype(F32)                                       # [128, 32]
        adam_update(W, dW, m, v, t, lr=4e-4 * (1 - t / steps))
        if not master:
            W = q_fwd(W)                                                  # 无 master: 权重只以 FP8 形式存在, 小更新被舍掉
    return float(np.mean((x_val @ W - y_val) ** 2) / np.mean(y_val ** 2))


def main() -> None:
    banner("M13 - FP8 Training")
    rs = np.random.RandomState(0)

    # ---- 1) 格式取舍: 精度 vs 范围 (无 scaling, 看原生网格) ----
    print("\n[1] E4M3 vs E5M2 (同一个 N(0,1) 张量; 另有 4 个孤立的 3e4 尖峰)")
    x = rs.randn(4096)
    spikes = rs.choice(4096, 4, replace=False)
    x_spiky = x.copy()
    x_spiky[spikes] = 3e4                                                 # 只改 4 个元素, 其余不动
    for fmt in ("e4m3", "e5m2"):
        body = rel_err(fake_quant_float(x, fmt), x)
        peak = fake_quant_float(x_spiky, fmt)[spikes][0]
        kv(f"{fmt}: 主体相对误差 / 尖峰 3e4 →", f"{body:.3f} / {peak:.0f}")
    assert rel_err(fake_quant_float(x, "e4m3"), x) < rel_err(fake_quant_float(x, "e5m2"), x), "E4M3 多 1 位尾数 → 更准"
    assert fake_quant_float(x_spiky, "e4m3")[spikes][0] == 448, "E4M3 饱和: 3e4 被截成 448"
    assert rel_err(fake_quant_float(x_spiky, "e5m2")[spikes], x_spiky[spikes]) < 0.07, "E5M2 装得下"

    # ---- 2) scaling 粒度: 孤立 outlier 越大, per-tensor 越惨; 1×128 tile 不受影响 ----
    print("\n[2] 3 个孤立 outlier 激活对 **其余 29 个 token** 的 GEMM 输出误差 (X[32,512], 全程 E4M3)")
    X0 = rs.randn(32, 512)
    Wm = rs.randn(512, 32) * 0.02
    rows = np.array([3, 11, 20])
    bystander = np.setdiff1d(np.arange(32), rows)                         # 自己没有 outlier 的 token
    print(f"      {'outlier 幅度':<14}{'per-tensor':>12}{'1×128 tile':>12}{'per-tensor 冲零比例':>20}")
    err = {}
    for mag in (1e2, 1e3, 1e4, 1e5, 1e6):
        X = X0.copy()
        X[rows, [17, 200, 450]] = mag                                     # 只改 3 个元素
        Y = X @ Wm
        q_t, q_b = quant_blockwise(X, "e4m3", X.size), quant_blockwise(X, "e4m3", BLOCK)
        err[mag] = (rel_err((q_t @ Wm)[bystander], Y[bystander]), rel_err((q_b @ Wm)[bystander], Y[bystander]))
        print(f"      {mag:<14.0e}{err[mag][0]:>12.3f}{err[mag][1]:>12.3f}{np.mean(q_t[bystander] == 0):>20.1%}")
    assert err[1e2][0] < 1.2 * err[1e2][1], "outlier 不大时两者打平: E4M3 自带 2^15 的动态范围"
    assert err[1e4][0] < 1.2 * err[1e4][1], "诚实结论: 1e4× 以内 per-tensor 扛得住"
    assert err[1e5][0] > 4 * err[1e5][1] and err[1e6][0] > 0.9, "再大就把所有人压进 subnormal, 直到全部冲成 0"
    assert all(e_b < 0.05 for _, e_b in err.values()), "tile scaling: outlier 只影响它自己那 128 个元素"

    # ---- 3) 端到端消融: 每个臂只动一个变量 ----
    print("\n[3] 训练消融 (200 步 Adam; 数值 = 验证 loss / ‖y‖², 越小越好)")
    arms = {
        "FP32 基线":                          ("none", None, None, True),
        "A  无 scaling        + master":      ("none", "e4m3", "e5m2", True),
        "B  per-tensor (TE 式) + master":     ("tensor", "e4m3", "e5m2", True),
        "C  block-128 (V3 式)  + master":     ("block", "e4m3", "e4m3", True),
        "D  block-128 (V3 式), 无 master":    ("block", "e4m3", "e4m3", False),
    }
    loss = {name: train(*cfg) for name, cfg in arms.items()}
    for name, val in loss.items():
        kv(name, f"{val:.2e}")
    a, b, c, d = (loss[k] for k in list(arms)[1:])
    kv("scaling 的收益 A/B", f"{a / b:.0f}x")
    kv("master 的收益 D/C", f"{d / c:.0f}x")
    kv("粒度的收益 B/C (本数据无 outlier)", f"{b / c:.2f}x  ← 约等于 1: 粒度只在有 outlier 时才起作用, 见 [2]")

    assert a > 10 * b, "不 scaling: 权重和梯度都掉进 subnormal/下溢区"
    assert d > 100 * c, "无 master: 小于 FP8 步长的更新全部丢失"
    assert 0.8 < b / c < 1.25, "干净数据上 per-tensor 与 block 基本打平 (诚实结论)"
    assert c < 1e-3, "FP8 的终点 = 量化噪声底 (~5e-4); 无噪声的玩具问题上 FP32 能到 1e-8, 真实 LLM 的 loss 噪声远大于此"

    print("\n  OK: 三件事各自独立 —— scaling 决定能不能用, master 决定能不能收敛, 粒度决定扛不扛得住 outlier。")


if __name__ == "__main__":
    main()
