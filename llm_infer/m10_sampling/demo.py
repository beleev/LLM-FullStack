"""
m10 demo — 采样策略: 先看直方图建立直觉, 再用断言钉死每个策略承诺的性质。

[1] 固定 logits (V=10) 上各策略 10000 次采样的频次
[2] 性质断言 (200 组随机 logits): top-k 恰好 k 个 / top-p 质量 ≥p 且最小 / min-p 阈值 / T→0 == greedy
[3] Gumbel-max 经验分布 vs softmax 的 total-variation 距离
[4] repetition penalty 让被罚 token 概率下降 (正、负 logit 都成立)

运行: python -m llm_infer.m10_sampling.demo
"""
from __future__ import annotations
import numpy as np

from llm_infer.core.utils import banner, kv, softmax
from llm_infer.m10_sampling.samplers import (
    greedy, temperature_sample, top_k_filter, top_p_filter, min_p_filter,
    repetition_penalty, gumbel_max, sample, SamplingParams,
)

V = 10
LOGITS = np.array([3.0, 2.5, 2.0, 1.0, 0.5, 0.2, 0.1, 0.0, -0.5, -1.0])
N = 10000
TV_TOL = 0.02        # N=10000, V=10 时采样噪声造成的 TV 约 0.01


def histogram(fn) -> np.ndarray:
    return np.bincount([fn() for _ in range(N)], minlength=V)          # (V,)


def check_filter_properties(rng) -> None:
    for _ in range(200):
        logits = rng.normal(0, 2, size=50)
        logits[:3] = logits[0]                                         # 故意制造并列, 专门考 top-k
        probs = softmax(logits)
        k, p, mp = int(rng.integers(1, 50)), float(rng.uniform(0.05, 0.99)), float(rng.uniform(0.01, 0.5))

        kept = np.isfinite(top_k_filter(logits, k))
        assert kept.sum() == k                                         # 并列时也恰好 k 个
        assert logits[kept].min() >= logits[~kept].max()               # 留下的确实是最大的 k 个

        kept = np.isfinite(top_p_filter(logits, p))
        mass = probs[kept].sum()
        assert mass >= p - 1e-12                                       # nucleus 质量 ≥ p
        assert mass - probs[kept].min() < p                            # 最小性: 再去掉留下的最小者就 < p
        assert probs[kept].min() >= probs[~kept].max(initial=0.0)      # 留下的是概率最大的那批

        kept = np.isfinite(min_p_filter(logits, mp))
        assert np.array_equal(kept, probs >= mp * probs.max())

        assert sample(logits, SamplingParams(temperature=0.0), rng=rng) == greedy(logits)
        logits[:3] += [0.3, 0.2, 0.1]                                  # 拆掉并列, 否则 T→0 时并列 token 本来就该平分
        assert sample(logits, SamplingParams(temperature=1e-4), rng=rng) == greedy(logits)


def main():
    banner("M10 - Sampling Strategies")
    rng = np.random.RandomState(0)
    probs = softmax(LOGITS)

    print("\n[1] 采样频次 (每行 10000 次, seed=0); 第一行是 softmax 概率 ×10000")
    rows = {
        "softmax(logits)·N": (probs * N).round().astype(int),
        "greedy": histogram(lambda: greedy(LOGITS)),
        "temp=1.0": histogram(lambda: temperature_sample(LOGITS, 1.0, rng)),
        "temp=0.3 (更尖)": histogram(lambda: temperature_sample(LOGITS, 0.3, rng)),
        "temp=2.0 (更平)": histogram(lambda: temperature_sample(LOGITS, 2.0, rng)),
        "top_k=3": histogram(lambda: sample(LOGITS, SamplingParams(top_k=3), rng=rng)),
        "top_p=0.5": histogram(lambda: sample(LOGITS, SamplingParams(top_p=0.5), rng=rng)),
        "min_p=0.1": histogram(lambda: sample(LOGITS, SamplingParams(min_p=0.1), rng=rng)),
    }
    print(f"  {'token id':<20}" + "".join(f"{i:>6}" for i in range(V)))
    for name, h in rows.items():
        print(f"  {name:<20}" + "".join(f"{c:>6}" for c in h))
    assert rows["greedy"][0] == N
    assert rows["top_k=3"][3:].sum() == 0                              # 被砍的 token 一次都不出现
    assert rows["top_p=0.5"][2:].sum() == 0                            # p0=0.416 <0.5, p0+p1=0.668 ≥0.5 → 只留 2 个
    assert rows["min_p=0.1"][4:].sum() == 0                            # 阈值 0.1·0.416=0.0416; p3=0.056 留, p4=0.034 砍

    print("\n[2] 性质断言: 200 组随机 logits (V=50, 含并列值)")
    check_filter_properties(np.random.default_rng(1))
    print("  top-k 恰好 k 个有限值 / top-p 质量≥p 且去掉最小者就<p / min-p 集合 == {p_i ≥ min_p·p_max} / T=0 与 T=1e-4 == greedy  ✓")

    print("\n[3] Gumbel-max 与 softmax 同分布? total-variation = ½·Σ|经验频率 − p|")
    tv_gumbel = 0.5 * np.abs(histogram(lambda: gumbel_max(probs, rng)) / N - probs).sum()
    tv_multi = 0.5 * np.abs(rows["temp=1.0"] / N - probs).sum()
    kv("TV(gumbel-max, softmax)", f"{tv_gumbel:.4f}")
    kv("TV(multinomial, softmax)", f"{tv_multi:.4f}  ← 同样 N 下的采样噪声水平, 作参照")
    assert tv_gumbel < TV_TOL and tv_multi < TV_TOL

    print("\n[4] repetition penalty=1.5, history=[0, 0, 9] (token 0 logit=+3.0, token 9 logit=-1.0)")
    pen = softmax(repetition_penalty(LOGITS, [0, 0, 9], 1.5))
    for t in (0, 9):
        kv(f"token {t} 概率", f"{probs[t]:.4f} → {pen[t]:.4f}")
        assert pen[t] < probs[t]
    wrong = LOGITS.copy(); wrong[[0, 9]] /= 1.5                        # 常见错误写法: 不分正负一律除
    kv("错误写法 (一律除) token 9 概率", f"{probs[9]:.4f} → {softmax(wrong)[9]:.4f}  ← 反而变大")
    assert softmax(wrong)[9] > probs[9]


if __name__ == "__main__":
    main()
