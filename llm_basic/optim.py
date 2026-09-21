"""
optim.py — 手写 Adam / AdamW + 梯度裁剪 + warmup-cosine 学习率（纯 numpy）。

解决什么问题：SGD 对所有参数用同一个步长，而 Transformer 各参数梯度量级差几个数量级；
Adam 用 g 的一阶/二阶矩做"逐参数自适应步长"，每步更新量 ≈ lr，与梯度绝对大小无关。
    m ← β1·m + (1−β1)·g            v ← β2·v + (1−β2)·g²
    m̂ = m/(1−β1ᵗ)   v̂ = v/(1−β2ᵗ)   （偏置修正：m、v 从 0 起步，前几步被低估）
    W ← W − lr·( m̂/(√v̂+ε) + λ·W )   （λ = weight_decay；解耦 = 不经过 m/v，即 AdamW）

三个可选项默认全关（= 原始 Adam + 恒定 lr）：weight_decay=0、clip_grad_norm 不调用、
cosine_lr 不调用。读代码时盯住：weight_decay 那一项加在哪——加进 g 是 L2，加在更新量上才是 AdamW。
自检：python optim.py
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def adam_init(W: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        "m": {k: np.zeros_like(v) for k, v in W.items()},
        "v": {k: np.zeros_like(v) for k, v in W.items()},
        "t": 0,
    }


def adam_step(
    W: dict[str, np.ndarray],
    grads: dict[str, np.ndarray],
    state: dict[str, Any],
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """返回新的 (W, state)，不就地修改。weight_decay 只作用于 ≥2 维的矩阵（gain/bias 不衰减）。"""
    t = state["t"] + 1
    bc1 = 1.0 - beta1 ** t
    bc2 = 1.0 - beta2 ** t
    new_m, new_v, new_W = {}, {}, {}

    for k, w in W.items():
        g = grads[k]
        m = beta1 * state["m"][k] + (1.0 - beta1) * g
        v = beta2 * state["v"][k] + (1.0 - beta2) * g * g
        update = (m / bc1) / (np.sqrt(v / bc2) + eps)
        if weight_decay and w.ndim >= 2:
            update = update + weight_decay * w     # 解耦衰减：不进 m/v，不被 1/√v 缩放
        new_m[k], new_v[k] = m, v
        new_W[k] = w - lr * update

    return new_W, {"m": new_m, "v": new_v, "t": t}


def clip_grad_norm(
    grads: dict[str, np.ndarray], max_norm: float
) -> tuple[dict[str, np.ndarray], float]:
    """全局范数裁剪：把所有梯度当成一个长向量，‖g‖ > max_norm 时整体等比缩小（方向不变）。

    返回 (裁剪后的 grads, 裁剪前的范数)。
    """
    norm = math.sqrt(sum(float((g * g).sum()) for g in grads.values()))
    scale = min(1.0, max_norm / (norm + 1e-6))
    return {k: g * scale for k, g in grads.items()}, norm


def cosine_lr(step: int, max_lr: float, min_lr: float, warmup: int, max_steps: int) -> float:
    """step 从 1 计。前 warmup 步线性升到 max_lr，之后半个余弦降到 min_lr（step=max_steps 时）。"""
    if step <= warmup:
        return max_lr * step / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)            # 0 → 1
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def _self_check() -> None:
    rng = np.random.default_rng(0)
    grads = {"a": rng.standard_normal((3, 4)) * 10, "b": rng.standard_normal(5) * 10}
    norm_of = lambda g: math.sqrt(sum(float((x * x).sum()) for x in g.values()))  # noqa: E731

    # 裁剪：范数被压到 max_norm，方向不变；小于阈值时原样返回
    clipped, norm = clip_grad_norm(grads, 1.0)
    assert norm > 1.0 and abs(norm_of(clipped) - 1.0) < 1e-5
    assert np.allclose(clipped["a"] / grads["a"], 1.0 / norm, atol=1e-6)
    same, _ = clip_grad_norm(grads, 1e9)
    assert all(np.array_equal(same[k], grads[k]) for k in grads)

    # cosine：warmup 终点 = max_lr，最后一步 = min_lr，中点 = 平均值，warmup 后单调不增
    lrs = [cosine_lr(s, 1e-3, 1e-4, 10, 110) for s in range(1, 111)]
    assert abs(lrs[0] - 1e-4) < 1e-12 and abs(lrs[9] - 1e-3) < 1e-12
    assert abs(lrs[-1] - 1e-4) < 1e-12 and abs(lrs[59] - 5.5e-4) < 1e-12
    assert all(a >= b for a, b in zip(lrs[9:], lrs[10:]))

    # Adam 第 1 步：偏置修正后 m̂/√v̂ = sign(g) → 每个元素恰好移动 lr
    W = {"a": np.ones((3, 4)), "b": np.ones(5)}
    W1, _ = adam_step(W, grads, adam_init(W), lr=0.1)
    assert np.allclose(W1["a"], 1.0 - 0.1 * np.sign(grads["a"]))
    # AdamW：矩阵多减 lr·λ·w，1 维参数不衰减；λ=0 时与 Adam 完全一致
    W2, _ = adam_step(W, grads, adam_init(W), lr=0.1, weight_decay=0.5)
    assert np.allclose(W2["a"], W1["a"] - 0.1 * 0.5 * W["a"])
    assert np.array_equal(W2["b"], W1["b"])
    print(f"optim self-check OK: clip {norm:.2f} → {norm_of(clipped):.6f}, "
          f"cosine lr {lrs[0]:.1e} → {max(lrs):.1e} → {lrs[-1]:.1e}")


if __name__ == "__main__":
    _self_check()
