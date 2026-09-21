"""
M01 — 梯度累积 (Gradient Accumulation)

是什么: 大 batch 拆成 K 个 micro-batch 依次前向/反向, 梯度累加, 累够再 step 一次。
解决的瓶颈: 显存。激活峰值只按 micro-batch 计, 代价是 K 倍的串行时间, 不省通信也不省算力。
关键公式: loss 取 mean 时  g_full = Σ_k (n_k / N) · g_k   —— 权重是样本数占比, 不是 1。
读代码盯住: `add_inplace(..., scale=len(xb) / N)` 这个 scale; 漏掉它等于把学习率放大 K 倍。
"""
from __future__ import annotations

import numpy as np

from llm_train.core import LinearModel, banner, kv, max_abs_diff
from llm_train.core.utils import add_inplace, zeros_like


def accumulate(model: LinearModel, x, y, sizes, weighted: bool = True):
    """按 sizes 切 micro-batch 并累积梯度。weighted=False 演示 "忘了缩放" 的错误写法。"""
    accum = zeros_like(model.params())
    start = 0
    for n in sizes:
        xb, yb = x[start : start + n], y[start : start + n]          # [n, d_in], [n, d_out]
        _, grads = model.loss_and_grads(xb, yb)                       # 每个 micro 的 loss 是自己 n 个样本的 mean
        add_inplace(accum, grads, scale=n / len(x) if weighted else 1.0)
        start += n
    return accum


def main() -> None:
    banner("M01 - Gradient Accumulation")

    rs = np.random.RandomState(0)
    model = LinearModel.init(d_in=5, d_out=3, seed=1)
    x = rs.randn(8, 5).astype(np.float32)
    y = rs.randn(8, 3).astype(np.float32)

    _, full = model.loss_and_grads(x, y)                              # 基线: 真的一次喂 8 个样本

    even = accumulate(model, x, y, sizes=[2, 2, 2, 2])
    ragged = accumulate(model, x, y, sizes=[3, 3, 2])                 # 最后一个 micro 不满: 权重必须按样本数
    naive = accumulate(model, x, y, sizes=[2, 2, 2, 2], weighted=False)

    kv("max |full - accum(2,2,2,2)|", f"{max_abs_diff(full, even):.2e}")
    kv("max |full - accum(3,3,2)|", f"{max_abs_diff(full, ragged):.2e}")
    kv("忘记缩放: |naive| / |full|", f"{np.linalg.norm(naive['W']) / np.linalg.norm(full['W']):.2f}x")
    kv("激活峰值 (样本数)", "8 -> 2")
    kv("optimizer.step 频率", "每 4 个 micro-batch 一次")

    assert max_abs_diff(full, even) < 1e-6
    assert max_abs_diff(full, ragged) < 1e-6
    assert np.allclose(naive["W"], 4 * full["W"], atol=1e-5), "不缩放 = 梯度放大 K 倍"
    print("\n  OK: 累积梯度 == 大 batch 梯度 (含不等长 micro-batch); 不缩放则恰好放大 K=4 倍。")


if __name__ == "__main__":
    main()
