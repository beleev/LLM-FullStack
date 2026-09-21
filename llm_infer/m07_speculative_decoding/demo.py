"""
m07 demo — 投机解码的两个承诺, 都用 assert 验:
    [1][2] greedy: 输出与 target 单独 greedy 逐 token 相同, target 调用数更少
    [3]    sampling: 输出分布 == target 单独采样的分布 (经验 TV / 卡方, 对照一个故意写错的规则)
算法全部在 speculative.py; 这里只有实验。
"""
from __future__ import annotations

import time

import numpy as np

from llm_infer.core import ModelConfig, TinyLM, softmax
from llm_infer.core.utils import banner, kv
from llm_infer.m07_speculative_decoding.speculative import (
    ModelDrafter, make_draft, sample, speculative_decode)


def always_accept(d_tokens, d_probs, t_probs, rng):
    """故意写错的接受规则: 不看 p_t/p_d, 全收 draft → 输出分布被 draft 污染。"""
    return len(d_tokens), sample(t_probs[-1], rng)


def exact_marginals(target: TinyLM, prompt, n_new: int, temperature: float) -> np.ndarray:
    """枚举所有前缀, 精确算出 target 采样时第 1..n_new 个新 token 的边缘分布 (n_new, V)。"""
    V = target.cfg.vocab_size
    marg = np.zeros((n_new, V))
    prefixes = [((), 1.0)]
    for d in range(n_new):
        nxt = []
        for pre, p_pre in prefixes:
            logits, _ = target.forward(list(prompt) + list(pre))
            p = softmax(logits[-1].astype(np.float64) / temperature)       # (V,)
            marg[d] += p_pre * p
            if d + 1 < n_new:
                nxt += [(pre + (v,), p_pre * p[v]) for v in range(V)]
        prefixes = nxt
    return marg


def plain_sampling(target: TinyLM, prompt, max_new: int, temperature: float, rng) -> list:
    logits, cache = target.prefill(np.asarray(prompt))
    out, logits = [], logits[-1]
    for _ in range(max_new):
        out.append(sample(softmax(logits.astype(np.float64) / temperature), rng))
        logits, cache = target.decode_step(out[-1], cache)
    return out


def tv_and_chi2(samples: np.ndarray, exact: np.ndarray):
    """samples (N, n_new) int, exact (n_new, V) → 每个位置的 TV 距离与卡方统计量。"""
    N, V = len(samples), exact.shape[1]
    tv, chi2 = [], []
    for d in range(exact.shape[0]):
        obs = np.bincount(samples[:, d], minlength=V)
        tv.append(0.5 * np.abs(obs / N - exact[d]).sum())
        chi2.append(((obs - N * exact[d]) ** 2 / (N * exact[d])).sum())
    return np.array(tv), np.array(chi2)


def main() -> None:
    banner("M07 - Speculative Decoding (KV 回滚 + 分布无损检验)")

    target = TinyLM(ModelConfig(d_model=64, d_mlp=128, n_layer=4, vocab_size=128))
    prompt = np.array([1, 5, 10, 15, 20, 25], dtype=np.int64)
    max_new, K = 48, 4
    ref = target.generate_greedy(prompt, max_new)         # baseline: max_new 次 target forward

    print(f"\n[1][2] greedy, K={K}, 生成 {max_new} token (baseline target 调用 = {max_new})")
    drafts = {
        "draft == target (上限)": target,
        "权重加噪 10% (模拟蒸馏 draft)": make_draft(target, noise=0.1),
        "只用前 2 层 (LayerSkip 式)": make_draft(target, n_layer=2),
        "独立随机 1 层小模型": TinyLM(ModelConfig(d_model=64, d_mlp=128, n_layer=1, vocab_size=128)),
    }
    calls_by_name = {}
    for name, d in drafts.items():
        drafter = ModelDrafter(d)
        out, calls, acc = speculative_decode(target, drafter, prompt, max_new, K)
        assert out == ref, f"{name}: greedy 投机输出必须与 target greedy 逐 token 相同"
        assert calls == 1 + len(acc)                      # 1 次 prefill + 每轮 1 次验证
        calls_by_name[name] = calls
        kv(name, f"target 调用 {calls:>2} ({max_new / calls:.2f}x), 每轮接受 {np.mean(acc):.2f}/{K}, "
                 f"draft 调用 {drafter.calls}")
    print("  (加速按 target 调用数算; 没计 draft 自身开销 —— draft 越贵, 真实加速越打折)")
    assert calls_by_name["draft == target (上限)"] == 1 + -(-(max_new - 1) // (K + 1))
    assert calls_by_name["权重加噪 10% (模拟蒸馏 draft)"] < max_new
    # 猜不中的 draft 每轮仍白送 1 个纠错 token → 调用数不超过 baseline + 1 (只亏 draft 开销)
    assert calls_by_name["独立随机 1 层小模型"] <= max_new + 1

    print("\n[3] sampling: 投机采样的输出分布 == target 单独采样? (小词表便于精确枚举)")
    tgt = TinyLM(ModelConfig(vocab_size=16, d_model=32, d_mlp=64, n_layer=2))
    drf = make_draft(tgt, n_layer=1, noise=0.3)
    p_small, n_new, K3, T, N = [1, 5, 9], 4, 2, 1.0, 2500
    t0 = time.perf_counter()
    exact = exact_marginals(tgt, p_small, n_new, T)       # (4, 16), 枚举 1+16+256+4096 个前缀
    rng = np.random.default_rng(0)
    runs = {"plain target 采样": [], "投机采样 (min(1,p_t/p_d)+残差)": [], "错误规则: 全收 draft": []}
    acc_all = []
    for _ in range(N):
        runs["plain target 采样"].append(plain_sampling(tgt, p_small, n_new, T, rng))
        out, _, acc = speculative_decode(tgt, ModelDrafter(drf), p_small, n_new, K3, T, rng)
        runs["投机采样 (min(1,p_t/p_d)+残差)"].append(out[len(p_small):])
        acc_all += acc
        out, _, _ = speculative_decode(tgt, ModelDrafter(drf), p_small, n_new, K3, T, rng,
                                       accept=always_accept)
        runs["错误规则: 全收 draft"].append(out[len(p_small):])
    kv("设置", f"V=16, T={T}, K={K3}, N={N} 条 × {n_new} token, 每轮接受 {np.mean(acc_all):.2f}/{K3}")
    res = {}
    for name, s in runs.items():
        res[name] = tv_and_chi2(np.array(s), exact)
        kv(name, "TV " + " ".join(f"{x:.3f}" for x in res[name][0])
           + " | χ² " + " ".join(f"{x:5.1f}" for x in res[name][1]))
    kv("耗时", f"{time.perf_counter() - t0:.1f} s  (第 1 个 token 来自 prefill, 第 2~4 个走接受/残差/bonus)")

    CHI2_CRIT = 37.70                                      # χ²(df=15) 的 99.9% 分位
    tv_plain, _ = res["plain target 采样"]
    tv_spec, chi_spec = res["投机采样 (min(1,p_t/p_d)+残差)"]
    tv_bad, chi_bad = res["错误规则: 全收 draft"]
    assert (chi_spec < CHI2_CRIT).all(), "投机采样的边缘分布应与精确 target 分布无法区分"
    assert (tv_spec < 2 * tv_plain.max()).all(), "TV 应与同样 N 的 plain 采样噪声同量级"
    assert chi_bad[1:].max() > 10 * CHI2_CRIT and tv_bad[1:].max() > 3 * tv_plain.max(), \
        "全收 draft 的错误规则必须被同一个检验抓出来"
    print("  ✓ 投机采样通过 (χ² < 37.7, TV ≈ 采样噪声); 错误规则在同一检验下被抓出")


if __name__ == "__main__":
    main()
