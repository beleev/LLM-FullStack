"""
M14 — Muon 优化器 (Kimi K2 / Moonlight 等在用)

是什么: 对 **2-D 权重矩阵**, 把动量矩阵 M 正交化后再更新:  W ← W − lr · NS(M),  NS(M) ≈ U·Vᵀ  (M = UΣVᵀ)。
        embedding / 输出头 / bias / norm 等非矩阵参数仍用 AdamW。
解决的瓶颈: 样本效率 (同样 loss 约省 ~1/2 FLOPs) 和优化器显存 (只存 1 份动量, Adam 要 m+v 两份)。
关键公式: Newton–Schulz 五次迭代  X ← aX + (bA + cA²)X,  A = XXᵀ,  (a,b,c) = (3.4445, −4.7750, 2.0315)
          只用矩阵乘 → GPU 友好; 5 步把所有奇异值推到 ~1 附近 (不求精确, 0.7~1.2 即可)。
直觉: 梯度的奇异值谱极不均匀, SGD/Adam 的更新被少数大奇异方向主导; 正交化让每个方向步长相同。
读代码盯住: `newton_schulz` 前后的奇异值, 和 `0.2·√max(n,m)` 这个让 Muon 复用 AdamW 学习率的缩放。
"""
from __future__ import annotations

import numpy as np

from llm_train.core import adam_update, banner, kv, make_rng, softmax


def newton_schulz(G: np.ndarray, steps: int = 5) -> np.ndarray:
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (np.linalg.norm(G) + 1e-7)               # Frobenius 归一 → 奇异值全部 ≤ 1, 迭代才收敛
    tall = X.shape[0] > X.shape[1]
    if tall:
        X = X.T                                      # 让 A = XXᵀ 是较小的那个方阵
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X          # 对每个奇异值 σ 作用多项式 aσ + bσ³ + cσ⁵
    return X.T if tall else X


class Muon:
    def __init__(self, shape, lr, momentum=0.95, wd=0.0):
        self.lr, self.mu, self.wd = lr, momentum, wd
        self.buf = np.zeros(shape)                   # 唯一的优化器状态
        self.rms_match = 0.2 * np.sqrt(max(shape))   # Moonlight: 让更新的 RMS ≈ AdamW 的 ~0.2, lr 可直接复用

    def step(self, W, G):
        self.buf = self.mu * self.buf + G
        O = newton_schulz(G + self.mu * self.buf)    # Nesterov 式动量, 再正交化
        W *= 1 - self.lr * self.wd
        W -= self.lr * self.rms_match * O


def qk_clip(Wq, Wk, x, tau: float):
    """MuonClip 的 QK-clip: 若最大 attention logit 超过 τ, 把 Wq/Wk 各缩 √(τ/S_max)。

    Muon 的更新是满秩的, 大规模训练时 QK logit 容易爆; Kimi K2 每步后做这个裁剪。
    """
    s_max = np.abs((x @ Wq) @ (x @ Wk).T).max() / np.sqrt(Wq.shape[1])
    eta = min(1.0, tau / s_max)
    return Wq * np.sqrt(eta), Wk * np.sqrt(eta), s_max


def run(opt_name: str, lr: float, rotate: bool = True, steps: int = 150):
    """病态的矩阵回归: 输入协方差条件数 1e4 且 **不与坐标轴对齐** (乘了随机旋转 Q)。

    Adam 的逐元素缩放只能修 "对角" 的病态; 旋转之后它无能为力, 而 NS 正交化与基无关。
    W 走 Muon 或 AdamW, 1-D 的 b 两边都用 Adam。"""
    rs = make_rng(14)
    d_in, d_out, B = 32, 32, 64
    feat_scale = np.logspace(0, -2, d_in)            # 主轴标准差 1 → 0.01
    Q = np.linalg.qr(rs.randn(d_in, d_in))[0] if rotate else np.eye(d_in)   # 随机旋转: 病态方向不再是单个坐标
    W_true, b_true = rs.randn(d_in, d_out), rs.randn(d_out)
    W, b = np.zeros((d_in, d_out)), np.zeros(d_out)
    muon = Muon(W.shape, lr)
    mW, vW, mb, vb = np.zeros_like(W), np.zeros_like(W), np.zeros_like(b), np.zeros_like(b)
    losses = []
    for t in range(1, steps + 1):
        x = (rs.randn(B, d_in) * feat_scale) @ Q     # [B, d_in]
        diff = x @ W + b - (x @ W_true + b_true)     # [B, d_out]
        losses.append(float(np.mean(diff**2)))
        gW, gb = x.T @ diff * (2 / diff.size), diff.sum(0) * (2 / diff.size)
        lr_t = lr * (1 - t / steps)
        if opt_name == "muon":
            muon.lr = lr_t
            muon.step(W, gW)
        else:
            adam_update(W, gW, mW, vW, t, lr_t)
        adam_update(b, gb, mb, vb, t, 0.05 * (1 - t / steps))     # 1-D 参数: 两边都用 Adam
    return losses


def main() -> None:
    banner("M14 - Muon Optimizer")
    rs = make_rng(0)

    # ---- 1) Newton–Schulz 到底做了什么 ----
    print("\n[1] Newton–Schulz 正交化 (条件数 1e4 的 16×32 矩阵)")
    U, _ = np.linalg.qr(rs.randn(16, 16))
    V, _ = np.linalg.qr(rs.randn(32, 32))
    sigma = np.logspace(0, -4, 16)
    G = U @ np.diag(sigma) @ V[:16]
    sv = {k: np.linalg.svd(newton_schulz(G, steps=k), compute_uv=False) for k in (1, 3, 5)}
    kv("输入奇异值 max / min", f"{sigma.max():.0e} / {sigma.min():.0e}")
    for k, s in sv.items():
        kv(f"NS {k} 步后奇异值 max / min", f"{s.max():.3f} / {s.min():.3f}")
    O = newton_schulz(G)
    exact = U @ V[:16]                               # 精确的 U·Vᵀ (需要 SVD, GPU 上很慢)
    big = sigma > 1e-2                               # 只在 "来得及被推到 1" 的方向上比较
    kv("与精确 UVᵀ 的方向余弦", f"{np.sum(O * exact) / np.linalg.norm(O) / np.linalg.norm(exact):.3f}")
    assert sv[5][: big.sum()].min() > 0.6 and sv[5].max() < 1.25, "σ ≥ 1e-2 的方向被推到 [0.6, 1.25]"
    assert sv[5].min() > 100 * sigma.min(), "哪怕 σ=1e-4 的方向也被放大了两个数量级以上"

    # ---- 2) Muon vs AdamW ----
    print("\n[2] 病态矩阵回归 150 步, 各自扫 lr 取最好")
    grid = [0.03, 0.1, 0.3, 1.0, 3.0]
    best = {}
    for rotate in (True, False):
        for name in ("adamw", "muon"):
            finals = {lr: run(name, lr, rotate)[-1] for lr in grid}
            best[name, rotate] = min(finals.values())
            if rotate:
                kv(f"{name}: final loss @ lr", {lr: float(f"{v:.2e}") for lr, v in finals.items()})
    a, m_ = best["adamw", True], best["muon", True]
    kv("旋转过的病态 (非轴对齐)", f"AdamW {a:.2e} / Muon {m_:.2e}  → Muon 好 {a / m_:.1f}x")
    a0, m0 = best["adamw", False], best["muon", False]
    kv("轴对齐的病态 (Adam 的主场)", f"AdamW {a0:.2e} / Muon {m0:.2e}  → Adam 好 {m0 / a0:.1f}x")
    assert m_ < 0.5 * a, "病态方向不与坐标轴对齐时, 各自最优 lr 下 Muon 明显更低"
    assert a / m_ > a0 / m0, "轴对齐时 Adam 的逐元素缩放本来就够用, Muon 的优势缩小"
    kv("优化器状态 (每个矩阵参数)", "AdamW 2 份 (m, v) / Muon 1 份 (动量)")

    # ---- 3) QK-clip ----
    print("\n[3] MuonClip 的 QK-clip (τ = 100)")
    x = rs.randn(16, 32)
    Wq, Wk = rs.randn(32, 8) * 3, rs.randn(32, 8) * 3
    Wq2, Wk2, s_before = qk_clip(Wq, Wk, x, tau=100.0)
    _, _, s_after = qk_clip(Wq2, Wk2, x, tau=100.0)
    kv("max attention logit 裁剪前 → 后", f"{s_before:.1f} → {s_after:.1f}")
    p = softmax((x @ Wq) @ (x @ Wk).T / np.sqrt(8))
    kv("裁剪前 softmax 最大概率均值", f"{p.max(1).mean():.3f}  (≈1 即 one-hot, 梯度消失)")
    assert s_before > 100 and abs(s_after - 100) < 1e-6

    print("\n  OK: NS 把梯度谱拉平; 非轴对齐的病态问题上 Muon 胜过调好 lr 的 AdamW (轴对齐时反而 Adam 赢); 大规模时配 QK-clip。")


if __name__ == "__main__":
    main()
