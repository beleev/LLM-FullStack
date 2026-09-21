"""
gradcheck.py — 用数值梯度验证手写反向传播（先逐算子，再端到端）。

解决什么问题：手写 backward 的 bug（转置写反、sum 维度错、softmax Jacobian 漏项）
不会让训练崩溃，只会让模型悄悄学得更差。中心差分给出"不可能写错"的参照：
    g_num = ( f(w+ε) − f(w−ε) ) / 2ε        截断误差 O(ε²)，float64 下 ε≈1e-5~1e-4 最准

两级检查：
  1) 逐算子：L = Σ(out ⊙ R)（R 为固定随机张量，于是 dL/dout = R），对每个输入的
     每个元素做差分，断言 ‖g_a − g_n‖ / max(‖g_a‖, ‖g_n‖) < 1e-6 —— 哪个算子错一目了然
  2) 端到端：整个模型 + CE，n_layer = 1 和 2 各跑一遍，每个参数抽样若干位置

读代码时盯住：check_op 里的 R —— 它就是喂给 backward 的 dout。
用法：cd llm_basic && python gradcheck.py
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from model import (
    attention_backward,
    attention_forward,
    cross_entropy_forward_backward,
    embedding_backward,
    embedding_forward,
    init_weights,
    linear_backward,
    linear_forward,
    mlp_backward,
    mlp_forward,
    relu_backward,
    relu_forward,
    rmsnorm_backward,
    rmsnorm_forward,
    transformer_backward,
    transformer_forward,
)

OP_TOL = 1e-6


# ============================================================
# 1. 逐算子检查
# ============================================================
def numeric_grad(f: Callable[[], float], arr: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """对 arr 的每个元素做中心差分（就地改、算完还原）。f() 返回标量。"""
    g = np.zeros_like(arr)
    for idx in np.ndindex(arr.shape):
        orig = arr[idx]
        arr[idx] = orig + eps
        lp = f()
        arr[idx] = orig - eps
        lm = f()
        arr[idx] = orig
        g[idx] = (lp - lm) / (2.0 * eps)
    return g


def check_op(
    name: str,
    forward: Callable[[], tuple[np.ndarray, object]],
    backward: Callable[[np.ndarray, object], tuple],
    inputs: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> float:
    """forward() 读 inputs 里的数组；backward 返回与 inputs 同序的梯度。返回最大相对误差。"""
    out, cache = forward()
    R = rng.standard_normal(out.shape)             # dL/dout，L = Σ(out ⊙ R)
    analytic = backward(R, cache)

    worst = 0.0
    for (key, arr), g_a in zip(inputs.items(), analytic):
        g_n = numeric_grad(lambda: float((forward()[0] * R).sum()), arr)
        rel = np.linalg.norm(g_a - g_n) / max(np.linalg.norm(g_a), np.linalg.norm(g_n), 1e-12)
        assert rel < OP_TOL, f"{name}: d{key} 相对误差 {rel:.2e} ≥ {OP_TOL}"
        worst = max(worst, rel)
    print(f"  [OK ] {name:<14} max_rel={worst:.2e}")
    return worst


def check_ops(seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    B, T, D, H, V = 2, 3, 4, 8, 6
    r = rng.standard_normal                       # 用 std=1 的输入，让非线性真正起作用

    x, Wl, b = r((B, T, D)), r((D, H)), r(H)
    check_op("linear", lambda: linear_forward(x, Wl, b), linear_backward,
             {"x": x, "W": Wl, "b": b}, rng)

    ids = np.array([[0, 1, 1], [5, 1, 0]])        # 故意含重复 id：专门考 np.add.at 的累加
    E = r((V, D))
    check_op("embedding", lambda: embedding_forward(ids, E),
             lambda d, c: (embedding_backward(d, c),), {"W": E}, rng)

    g = r(D)
    check_op("rmsnorm", lambda: rmsnorm_forward(x, g), rmsnorm_backward,
             {"x": x, "g": g}, rng)

    xr = r((B, T, D))
    xr[np.abs(xr) < 1e-3] = 0.5                   # 避开 ReLU 在 0 处的折点（那里差分无意义）
    check_op("relu", lambda: relu_forward(xr),
             lambda d, c: (relu_backward(d, c),), {"x": xr}, rng)

    Wq, Wk, Wv, Wo = (r((D, D)) for _ in range(4))
    check_op("attention", lambda: attention_forward(x, Wq, Wk, Wv, Wo), attention_backward,
             {"x": x, "Wq": Wq, "Wk": Wk, "Wv": Wv, "Wo": Wo}, rng)

    W1, b1, W2, b2 = r((D, H)), r(H), r((H, D)), r(D)
    check_op("mlp", lambda: mlp_forward(x, W1, b1, W2, b2), mlp_backward,
             {"x": x, "W1": W1, "b1": b1, "W2": W2, "b2": b2}, rng)

    # cross-entropy 自己就输出标量 loss，单独处理
    logits, targets = r((B, T, V)) * 3.0, rng.integers(0, V, size=(B, T))
    _, g_a = cross_entropy_forward_backward(logits, targets)
    g_n = numeric_grad(lambda: cross_entropy_forward_backward(logits, targets)[0], logits)
    rel = np.linalg.norm(g_a - g_n) / np.linalg.norm(g_n)
    assert rel < OP_TOL, f"cross_entropy: 相对误差 {rel:.2e}"
    assert abs(g_a.sum()) < 1e-12                 # 每行 Σ(p − onehot) = 0
    print(f"  [OK ] {'cross_entropy':<14} max_rel={rel:.2e}")


# ============================================================
# 2. 端到端检查
# ============================================================
def loss_only(W: dict[str, np.ndarray], ids: np.ndarray, targets: np.ndarray) -> float:
    logits, _ = transformer_forward(W, ids)
    return cross_entropy_forward_backward(logits, targets)[0]


def gradcheck(
    n_layer: int = 1,
    num_samples: int = 8,
    eps: float = 1e-5,
    rtol: float = 1e-4,
    atol: float = 1e-7,
    seed: int = 0,
) -> dict[str, dict]:
    """每个参数随机抽 num_samples 个位置。返回 {name: {"max_abs", "max_rel", "ok"}}。

    判定用 |g_a − g_n| < atol + rtol·max(|g_a|, |g_n|) 而不是纯相对误差：抽样常落在
    真实梯度≈0 的位置（如 batch 里没出现的 token 那一行 tok_emb），两个 ~1e-10 的数
    相对误差可以很大，但并不是 bug。

    eps=1e-5 而不是 1e-4：初始化 std=0.02 → 残差流 rms≈0.02，RMSNorm 在这里曲率极大，
    ε=1e-4 时 pos_emb 的截断误差就到 1e-4（实测误差随 ε² 缩小，说明是截断误差不是 bug）。
    """
    rng = np.random.default_rng(seed)
    config = {"vocab_size": 8, "dim": 4, "hidden_dim": 8, "max_seq_len": 3, "n_layer": n_layer}
    B, T, V = 2, 3, config["vocab_size"]

    W = init_weights(config, rng)
    ids = rng.integers(0, V, size=(B, T))
    targets = rng.integers(0, V, size=(B, T))

    logits, cache = transformer_forward(W, ids)
    _, dlogits = cross_entropy_forward_backward(logits, targets)
    grads = transformer_backward(dlogits, cache)
    assert grads.keys() == W.keys(), "backward 必须给每个参数都返回梯度"

    results: dict[str, dict] = {}
    for key in sorted(W):
        arr = W[key]
        max_abs = max_rel = 0.0
        ok = True
        for fi in rng.choice(arr.size, size=min(num_samples, arr.size), replace=False):
            idx = np.unravel_index(int(fi), arr.shape)
            orig = arr[idx]
            arr[idx] = orig + eps
            lp = loss_only(W, ids, targets)
            arr[idx] = orig - eps
            lm = loss_only(W, ids, targets)
            arr[idx] = orig                        # 还原

            g_a, g_n = float(grads[key][idx]), (lp - lm) / (2.0 * eps)
            abs_err = abs(g_a - g_n)
            scale = max(abs(g_a), abs(g_n))
            ok &= abs_err <= atol + rtol * scale
            max_abs = max(max_abs, abs_err)
            max_rel = max(max_rel, abs_err / max(scale, 1e-12))
        results[key] = {"max_abs": max_abs, "max_rel": max_rel, "ok": ok}
    return results


def main() -> None:
    print(f"[1] 逐算子 gradcheck（全元素，断言 rel < {OP_TOL}）")
    check_ops()

    for n_layer in (1, 2):
        print(f"\n[2] 端到端 gradcheck  n_layer={n_layer}  (vocab=8, dim=4, T=3, B=2; eps=1e-5, atol=1e-7, rtol=1e-4)")
        results = gradcheck(n_layer=n_layer)
        width = max(len(k) for k in results)
        for k, r in results.items():
            flag = "OK " if r["ok"] else "BAD"
            print(f"  [{flag}] {k:<{width}}  max_abs={r['max_abs']:.2e}  max_rel={r['max_rel']:.2e}")
        bad = [k for k, r in results.items() if not r["ok"]]
        assert not bad, f"梯度超出容差，去查对应的 *_backward: {bad}"

    print("\nall gradients within tolerance — analytical backward looks correct.")


if __name__ == "__main__":
    main()
