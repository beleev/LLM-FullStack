"""
M06 — Mixed Precision

训练常用 FP16/BF16 做矩阵计算, FP32 master weights 保存可更新精度。
FP16 的主要风险是小梯度下溢为 0, 所以需要 loss scaling:
    scaled_loss = loss * scale
    scaled_grad = grad * scale
    unscaled_grad = scaled_grad / scale
"""
from __future__ import annotations

import numpy as np

from llm_train.core import banner, kv


class LossScaler:
    """Tiny dynamic loss scaler."""

    def __init__(self, init_scale: float = 2**10, growth: float = 2.0, backoff: float = 0.5):
        self.scale = init_scale
        self.growth = growth
        self.backoff = backoff
        self.good_steps = 0

    def update(self, has_overflow: bool) -> None:
        if has_overflow:
            self.scale = max(1.0, self.scale * self.backoff)
            self.good_steps = 0
        else:
            self.good_steps += 1
            if self.good_steps >= 2:
                self.scale *= self.growth
                self.good_steps = 0


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
    kv("dynamic scale after overflow", scaler.scale)

    assert fp16_grad[0] == 0.0
    assert recovered[-1] > 0.0
    print("\n  OK: 混合精度省显存/提吞吐; loss scaling 避免 FP16 小梯度直接归零。")


if __name__ == "__main__":
    main()
