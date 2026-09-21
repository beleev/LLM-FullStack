"""
M10 — 训练稳定性: LR 调度 (cosine / WSD) · 全局梯度裁剪 · NaN guard

是什么: 让长跑任务不炸的三件朴素工具。
解决的瓶颈: 不是显存也不是通信, 是 "loss spike 之后要不要回滚 3 天" 的工程风险。
关键公式: cosine  lr = ½·base·(1 + cos(π·progress))            需要预先知道总步数
          WSD     warmup → 恒定 (stable) → 最后 ~10% 快速衰减     稳定段与总步数无关, 可随时续训/分叉
          clip    g ← g · min(1, c / ‖g‖₂),  ‖·‖ 是 **所有参数拼在一起** 的范数 (方向不变)
读代码盯住: `wsd_lr` 里 stable 段不依赖 total_steps; `guarded_step` 里坏 step 对参数零影响。
"""
from __future__ import annotations

import math
import numpy as np

from llm_train.core import banner, clip_by_global_norm, global_norm, has_overflow, kv


def warmup_cosine_lr(step: int, total_steps: int, base_lr: float, warmup_steps: int, min_ratio: float = 0.0) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def wsd_lr(step: int, total_steps: int, base_lr: float, warmup_steps: int,
           decay_frac: float = 0.1, min_ratio: float = 0.0) -> float:
    """Warmup-Stable-Decay (MiniCPM / DeepSeek-V3 / Kimi 同类调度)。"""
    decay_start = total_steps - int(total_steps * decay_frac)
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    if step < decay_start:
        return base_lr                                     # 稳定段: 与 total_steps 无关
    progress = (step - decay_start) / max(1, total_steps - decay_start)
    return base_lr * (1 - (1 - min_ratio) * progress)      # 线性退火 (也常用 1-sqrt / 指数)


def guarded_step(params, grads, lr: float, max_norm: float) -> bool:
    """NaN guard + clip + SGD。坏 step 返回 False 且不碰参数。"""
    if has_overflow(grads):
        return False
    clipped, _, _ = clip_by_global_norm(grads, max_norm)
    for k in params:
        params[k] -= lr * clipped[k]
    return True


def main() -> None:
    banner("M10 - Training Stability")

    # ---- 1) 两种调度 ----
    total, warm, base = 100, 10, 1e-3
    cos = [warmup_cosine_lr(s, total, base, warm) for s in range(total)]
    wsd = [wsd_lr(s, total, base, warm) for s in range(total)]
    marks = [0, 9, 30, 60, 89, 95, 99]
    print("\n[1] LR 调度 (total=100, warmup=10, base=1e-3)")
    kv("step", marks)
    kv("cosine ×1e-3", [round(cos[s] * 1e3, 3) for s in marks])
    kv("WSD    ×1e-3", [round(wsd[s] * 1e3, 3) for s in marks])

    # "想多训一倍" 时: WSD 已经走过的稳定段 lr 完全不变, cosine 整条曲线都变了
    cos200 = [warmup_cosine_lr(s, 200, base, warm) for s in range(90)]
    wsd200 = [wsd_lr(s, 200, base, warm) for s in range(90)]
    kv("总步数 100→200, 前 90 步 lr 最大变化", f"cosine {max(abs(a - b) for a, b in zip(cos, cos200)):.2e},  "
                                                 f"WSD {max(abs(a - b) for a, b in zip(wsd, wsd200)):.2e}")
    assert wsd[:90] == wsd200, "WSD: 稳定段 checkpoint 可以直接续训或分叉出一个 decay 分支"
    assert max(abs(a - b) for a, b in zip(cos, cos200)) > 0.3 * base
    assert all(lr == base for lr in wsd[warm:90]) and wsd[-1] < 0.11 * base
    assert abs(cos[warm] - base) < 1e-12 and cos[-1] < 0.01 * base
    assert all(a < b for a, b in zip(cos[:warm - 1], cos[1:warm])), "warmup 单调上升"

    # ---- 2) 全局范数裁剪 ----
    print("\n[2] global-norm clipping")
    grads = {
        "W": np.array([[1.0, 2.0], [100.0, -50.0]], dtype=np.float32),      # 一个 spike
        "b": np.array([0.5, -0.25], dtype=np.float32),
    }
    clipped, old_norm, scale = clip_by_global_norm(grads, max_norm=5.0)
    cosine_sim = float(np.sum(grads["W"] * clipped["W"]) / (np.linalg.norm(grads["W"]) * np.linalg.norm(clipped["W"])))
    kv("‖g‖ 裁剪前 → 后", f"{old_norm:.2f} → {global_norm(clipped):.2f}  (scale={scale:.4f})")
    kv("方向 cos 相似度", f"{cosine_sim:.6f}")
    assert abs(global_norm(clipped) - 5.0) < 1e-5 and abs(cosine_sim - 1) < 1e-6
    small = {"W": np.array([0.1], dtype=np.float32)}
    assert clip_by_global_norm(small, 5.0)[2] == 1.0, "范数没超阈值就原样放行"

    # ---- 3) NaN guard ----
    print("\n[3] NaN/Inf guard")
    params = {"W": np.ones((2, 2), dtype=np.float32), "b": np.zeros(2, dtype=np.float32)}
    before = {k: v.copy() for k, v in params.items()}
    bad = {"W": np.array([[np.nan, 1.0], [1.0, 1.0]], dtype=np.float32), "b": np.zeros(2, dtype=np.float32)}
    ok_bad = guarded_step(params, bad, lr=0.1, max_norm=5.0)
    untouched = all(np.array_equal(before[k], params[k]) for k in params)
    ok_good = guarded_step(params, grads, lr=0.1, max_norm=5.0)
    kv("坏 step: 执行? / 参数被改?", f"{ok_bad} / {not untouched}")
    kv("好 step: 执行?", ok_good)
    assert not ok_bad and untouched and ok_good
    assert np.isfinite(params["W"]).all()
    # clip 救不了 NaN: 范数是 NaN, 任何缩放系数乘上去还是 NaN (torch 里系数本身也变 NaN, 污染全部参数)
    assert has_overflow(clip_by_global_norm(bad, 5.0)[0]), "所以 guard 必须排在 clip 之前"

    print("\n  OK: WSD 稳定段与总步数解耦; clip 只缩长度不改方向; NaN guard 必须排在 clip 之前。")


if __name__ == "__main__":
    main()
