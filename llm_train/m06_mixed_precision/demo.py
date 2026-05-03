"""
M06 — Mixed Precision

训练常用 FP16/BF16 做矩阵计算, FP32 master weights 保存可更新精度。
FP16 的主要风险是小梯度下溢为 0, 所以需要 loss scaling:
    scaled_loss = loss * scale
    scaled_grad = grad * scale
    unscaled_grad = scaled_grad / scale

如果 scale 设得过大, 反向后梯度会出现 NaN/Inf, 这时必须:
    1. 跳过当前 step (不要污染 master weight)
    2. 把 scale 缩小, 准备下一步重试
真实训练框架 (PyTorch GradScaler / DeepSpeed) 会在反向后自动检测梯度
中的 NaN/Inf。本 demo 用 detect_overflow() 演示该自动检测路径。
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from llm_train.core import banner, kv


def detect_overflow(grads: Iterable[np.ndarray]) -> bool:
    """检查任一梯度张量是否含 NaN 或 Inf。

    真实训练里在反向后调用一次, 决定本 step 是否跳过 + 是否回退 scale。
    """
    for g in grads:
        if not np.isfinite(g).all():
            return True
    return False


class LossScaler:
    """Tiny dynamic loss scaler with optional auto overflow detection."""

    def __init__(self, init_scale: float = 2**10, growth: float = 2.0, backoff: float = 0.5):
        self.scale = init_scale
        self.growth = growth
        self.backoff = backoff
        self.good_steps = 0

    def update(self, has_overflow: bool) -> None:
        """根据是否溢出更新 scale (调用方传入)。"""
        if has_overflow:
            self.scale = max(1.0, self.scale * self.backoff)
            self.good_steps = 0
        else:
            self.good_steps += 1
            if self.good_steps >= 2:
                self.scale *= self.growth
                self.good_steps = 0

    def update_from_grads(self, grads: Iterable[np.ndarray]) -> bool:
        """自动检测梯度是否溢出并更新 scale, 返回 has_overflow 让调用方决定是否跳过 step。"""
        has_overflow = detect_overflow(grads)
        self.update(has_overflow)
        return has_overflow


def main() -> None:
    banner("M06 - Mixed Precision + Loss Scaling")

    tiny_grad = np.array([1e-8, 3e-8, 1e-7], dtype=np.float32)
    fp16_grad = tiny_grad.astype(np.float16)

    scale = 2**15
    scaled_fp16 = (tiny_grad * scale).astype(np.float16)
    recovered = scaled_fp16.astype(np.float32) / scale

    # Master weight update: compute/store model copy in fp16, update fp32 master.
    master_w = np.array([1.0], dtype=np.float32)
    model_w_fp16 = master_w.astype(np.float16)
    update_grad = np.array([1e-3], dtype=np.float32)
    master_w -= 1e-2 * update_grad
    model_w_fp16[...] = master_w.astype(np.float16)

    # ---- 手动 update (历史接口, 调用方需自己判断) ---------------------- #
    scaler = LossScaler()
    scaler.update(has_overflow=False)
    scaler.update(has_overflow=False)  # grow after 2 good steps
    grown = scaler.scale
    scaler.update(has_overflow=True)

    kv("tiny grad fp32", tiny_grad.tolist())
    kv("cast directly to fp16", fp16_grad.tolist())
    kv("after scale -> fp16 -> unscale", recovered.tolist())
    kv("fp32 master weight", float(master_w[0]))
    kv("fp16 model copy", float(model_w_fp16[0]))
    kv("dynamic scale after good steps", grown)
    kv("dynamic scale after overflow (manual)", scaler.scale)

    # ---- 自动 NaN/Inf 检测路径 (真实框架默认行为) --------------------- #
    print("\n[auto-overflow]  反向后自动检测梯度中的 NaN/Inf, 决定是否跳过 step")
    auto_scaler = LossScaler(init_scale=2**14)

    healthy_grads = [
        np.array([0.1, -0.2, 0.05], dtype=np.float32),
        np.array([[1e-3, 2e-3]], dtype=np.float32),
    ]
    overflowed_grads = [
        np.array([0.1, np.inf, 0.05], dtype=np.float32),  # ← scale 设得过大引发的 Inf
        np.array([[1e-3, 2e-3]], dtype=np.float32),
    ]

    of_healthy = auto_scaler.update_from_grads(healthy_grads)
    kv("step 1 healthy detected as overflow?", of_healthy)
    kv("scale after healthy step", auto_scaler.scale)

    auto_scaler.update_from_grads(healthy_grads)  # 第二个 healthy step 触发 grow
    kv("scale after 2 healthy steps (grow)", auto_scaler.scale)

    of_bad = auto_scaler.update_from_grads(overflowed_grads)
    kv("step 3 overflow detected?", of_bad)
    kv("scale after overflow (auto backoff)", auto_scaler.scale)
    print("  → 真实训练里 of_bad=True 时, optimizer.step() 应被跳过, master weight 保持不变。")

    assert fp16_grad[0] == 0.0
    assert recovered[-1] > 0.0
    assert detect_overflow(healthy_grads) is False
    assert detect_overflow(overflowed_grads) is True
    assert of_bad is True

    print("\n  OK: 混合精度省显存/提吞吐; loss scaling + 自动 NaN/Inf 检测保证训练稳定。")


if __name__ == "__main__":
    main()
