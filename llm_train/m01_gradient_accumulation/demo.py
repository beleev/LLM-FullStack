"""
M01 — Gradient Accumulation

大 batch 放不进显存时, 把它拆成多个 micro-batch:
    1) 每个 micro-batch 前向/反向
    2) 梯度累加, 不立刻更新参数
    3) 累够 N 个 micro-batch 后再 optimizer.step()

等价条件: loss reduction 和梯度缩放要匹配。本 demo 验证:
    full_batch_grad == mean(micro_batch_grads)
"""
from __future__ import annotations

import numpy as np

from llm_train.core import LinearModel, banner, kv, max_abs_diff
from llm_train.core.utils import add_inplace, zeros_like


def main() -> None:
    banner("M01 - Gradient Accumulation")

    rs = np.random.RandomState(0)
    model = LinearModel.init(d_in=5, d_out=3, seed=1)
    x = rs.randn(8, 5).astype(np.float32)
    y = rs.randn(8, 3).astype(np.float32)

    # Baseline: one real large batch.
    full_loss, full_grads = model.loss_and_grads(x, y)

    # Accumulation: 4 micro-batches of size 2.
    micro_size = 2
    accum_grads = zeros_like(full_grads)
    micro_losses = []
    for start in range(0, len(x), micro_size):
        xb = x[start : start + micro_size]
        yb = y[start : start + micro_size]
        loss, grads = model.loss_and_grads(xb, yb)
        micro_losses.append(loss)

        # Each micro loss is averaged over its local samples, so average the
        # micro gradients to match the full-batch mean reduction.
        add_inplace(accum_grads, grads, scale=micro_size / len(x))

    diff = max_abs_diff(full_grads, accum_grads)
    kv("full batch loss", f"{full_loss:.6f}")
    kv("mean micro loss", f"{np.mean(micro_losses):.6f}")
    kv("max |full_grad - accum_grad|", f"{diff:.2e}")
    kv("optimizer.step frequency", "1 step per 4 micro-batches")

    assert diff < 1e-6
    print("\n  OK: 梯度累积在数学上等价于更大的 batch, 但峰值激活显存按 micro-batch 计算。")


if __name__ == "__main__":
    main()

