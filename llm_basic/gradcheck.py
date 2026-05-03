"""
gradcheck.py — 数值梯度 vs 解析梯度的逐参数比对。

为什么必须做 gradcheck？
  手写反向传播极易出 bug：一个转置写反、一个 sum 维度错、softmax Jacobian
  漏一项……这些 bug 不会让训练崩溃，但会让模型学得比应有的差很多，
  调起来非常痛苦。
  gradcheck 用最朴素的中心差分 ( f(W+ε) - f(W-ε) ) / 2ε 算"真实"梯度，
  跟我们写的反向传播比对，相对误差应该 < 1e-5。

工作原理
========
1) 构造一个非常小的模型 (vocab=8, dim=4, hidden=8, T=3, B=2)
2) 随机生成输入 / 标签
3) 对每个参数张量，随机抽 N 个位置：
     - 解析梯度 g_a = backward(...)[k][idx]
     - 数值梯度 g_n = (loss(W+ε at idx) - loss(W-ε at idx)) / 2ε
     - 用 numpy.allclose 风格的"绝对 + 相对"组合判定：
         pass ⇔  |g_a - g_n|  <  atol + rtol * max(|g_a|, |g_n|)
4) 报告每个参数的最大绝对误差与最大相对误差

为什么不用纯相对误差？
  随机抽样常常落到"真实梯度本来就≈0"的位置（比如某个 vocab 的
  tok_emb 行根本没在 batch 里出现）。两个数都是 ~1e-8，相对误差
  能放大到 1e-3，但其实根本没有 bug。所以引入 atol 来"对零附近的
  位置宽容、对量级正常的梯度严格"。

注意：
  - 必须用 float64（model.DTYPE 已经是）
  - eps 选 1e-4：经验上 float64 + 平滑函数的最佳折中
    (太小被 roundoff 主导，太大被三阶截断误差主导)
  - 模型小才能跑得快；逻辑上跟大模型一模一样
"""
from __future__ import annotations

import numpy as np

from model import (
    cross_entropy_forward_backward,
    init_weights,
    transformer_backward,
    transformer_forward,
)


def loss_only(W: dict[str, np.ndarray], ids: np.ndarray, targets: np.ndarray) -> float:
    """前向 + loss，不需要 cache（数值梯度时反复调用）。"""
    logits, _ = transformer_forward(W, ids)
    loss, _ = cross_entropy_forward_backward(logits, targets)
    return loss


def numeric_grad_at(
    W: dict[str, np.ndarray],
    key: str,
    multi_idx: tuple[int, ...],
    ids: np.ndarray,
    targets: np.ndarray,
    eps: float,
) -> float:
    """中心差分。临时改 W[key][multi_idx]，算完恢复。"""
    arr = W[key]
    orig = arr[multi_idx]

    arr[multi_idx] = orig + eps
    lp = loss_only(W, ids, targets)

    arr[multi_idx] = orig - eps
    lm = loss_only(W, ids, targets)

    arr[multi_idx] = orig                # 还原
    return (lp - lm) / (2.0 * eps)


def gradcheck(
    num_samples: int = 8,
    eps: float = 1e-4,
    rtol: float = 1e-4,
    atol: float = 1e-7,
    seed: int = 0,
) -> dict[str, dict]:
    """
    对每个参数随机抽 num_samples 个位置做 gradcheck。

    返回 {param_name: {"max_abs": ..., "max_rel": ..., "ok": bool}}。
    判定标准：所有抽样点都满足 |g_a - g_n| < atol + rtol * max(|g_a|, |g_n|)
    """
    rng = np.random.default_rng(seed)

    # 故意小：让 gradcheck 在 1 秒内跑完
    config = {
        "vocab_size": 8,
        "dim": 4,
        "hidden_dim": 8,
        "max_seq_len": 3,
    }
    B, T = 2, 3
    V = config["vocab_size"]

    W = init_weights(config, rng)
    ids = rng.integers(0, V, size=(B, T))
    targets = rng.integers(0, V, size=(B, T))

    # 算一次解析梯度
    logits, cache = transformer_forward(W, ids)
    _, dlogits = cross_entropy_forward_backward(logits, targets)
    grads = transformer_backward(dlogits, cache)

    results: dict[str, dict] = {}

    for key in sorted(W.keys()):
        arr = W[key]
        size = arr.size
        n = min(num_samples, size)
        flat_indices = rng.choice(size, size=n, replace=False)

        max_abs = 0.0
        max_rel = 0.0
        ok = True
        for fi in flat_indices:
            multi = np.unravel_index(int(fi), arr.shape)
            g_a = float(grads[key][multi])
            g_n = numeric_grad_at(W, key, multi, ids, targets, eps)

            abs_err = abs(g_a - g_n)
            denom = max(abs(g_a), abs(g_n), 1e-12)
            rel_err = abs_err / denom

            tol = atol + rtol * max(abs(g_a), abs(g_n))
            if abs_err > tol:
                ok = False

            max_abs = max(max_abs, abs_err)
            max_rel = max(max_rel, rel_err)

        results[key] = {"max_abs": max_abs, "max_rel": max_rel, "ok": ok}

    return results


def main() -> None:
    print("running gradcheck on a tiny model (vocab=8, dim=4, T=3, B=2) ...")
    print("criterion: |g_a - g_n| < atol + rtol * max(|g_a|, |g_n|)")
    print("           atol=1e-7, rtol=1e-4, eps=1e-4\n")
    results = gradcheck()

    longest_name = max(len(k) for k in results)
    all_ok = True
    for k in sorted(results):
        r = results[k]
        all_ok &= r["ok"]
        flag = "OK " if r["ok"] else "BAD"
        print(
            f"  [{flag}] {k:<{longest_name}}  "
            f"max_abs={r['max_abs']:.2e}  max_rel={r['max_rel']:.2e}"
        )

    print()
    if all_ok:
        print("all gradients within tolerance — analytical backward looks correct.")
    else:
        print("some gradients out of tolerance — check the corresponding *_backward.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
