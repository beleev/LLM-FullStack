"""
M10 — Training Stability

规模化训练常见稳定性手段:
    - warmup + cosine decay: 避免开局大步长炸掉
    - global grad clipping: 控制偶发梯度尖峰
    - NaN/Inf 检测: 跳过坏 step, 降低 loss scale 或回滚
"""
from __future__ import annotations

import math
import numpy as np

from llm_train.core import banner, clip_by_global_norm, global_norm, kv


def warmup_cosine_lr(step: int, total_steps: int, base_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


def has_bad_number(grads) -> bool:
    return any(not np.all(np.isfinite(v)) for v in grads.values())


def main() -> None:
    banner("M10 - Training Stability")

    lrs = [warmup_cosine_lr(s, total_steps=10, base_lr=1e-3, warmup_steps=3) for s in range(10)]
    grads = {
        "W": np.array([[1.0, 2.0], [100.0, -50.0]], dtype=np.float32),
        "b": np.array([0.5, -0.25], dtype=np.float32),
    }
    clipped, old_norm, scale = clip_by_global_norm(grads, max_norm=5.0)

    bad_grads = {"W": np.array([np.nan], dtype=np.float32)}

    kv("lr schedule", [round(x, 6) for x in lrs])
    kv("grad norm before clip", f"{old_norm:.2f}")
    kv("clip scale", f"{scale:.4f}")
    kv("grad norm after clip", f"{global_norm(clipped):.2f}")
    kv("bad grad detected", has_bad_number(bad_grads))
    kv("bad step action", "skip optimizer.step(); lower loss scale; reload if needed")

    assert abs(global_norm(clipped) - 5.0) < 1e-5
    assert has_bad_number(bad_grads)
    print("\n  OK: 稳定性工程通常很朴素, 但决定长跑任务能不能跑完。")


if __name__ == "__main__":
    main()

