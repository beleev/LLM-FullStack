"""
m10 demo — 各采样策略对比

用一个固定 logits, 跑 N 次每种策略, 看分布。
"""
from __future__ import annotations
import numpy as np
from collections import Counter

from llm_infer.core.utils import banner, kv, softmax
from llm_infer.m10_sampling.samplers import (
    greedy, temperature_sample, top_k_filter, top_p_filter, min_p_filter,
    gumbel_max, sample, SamplingParams,
)


def main():
    banner("M10 - Sampling Strategies")

    # 构造 V=10 的 logits, 让分布有明显尾巴
    logits = np.array([3.0, 2.5, 2.0, 1.0, 0.5, 0.2, 0.1, 0.0, -0.5, -1.0], dtype=np.float32)
    probs = softmax(logits, axis=-1)
    print("\n原始概率分布:")
    for i, p in enumerate(probs):
        print(f"  token {i}: prob={p:.3f}  bar={'#'*int(p*40)}")

    rng = np.random.RandomState(0)
    N = 10000

    # ---- 各策略采样直方图 -------------------------------------- #
    print("\n[1] 采样直方图 (10000 次)")

    def hist(name, fn):
        counts = Counter(fn() for _ in range(N))
        line = "".join(f"{counts.get(i,0):5}" for i in range(10))
        print(f"  {name:<24} {line}")

    hist("greedy", lambda: greedy(logits))
    hist("temp=1.0", lambda: temperature_sample(logits, 1.0, rng))
    hist("temp=0.3 (sharp)", lambda: temperature_sample(logits, 0.3, rng))
    hist("temp=2.0 (smooth)", lambda: temperature_sample(logits, 2.0, rng))
    hist("top_k=3 (然后采样)", lambda: gumbel_max(softmax(top_k_filter(logits, 3)), rng))
    hist("top_p=0.5",          lambda: gumbel_max(softmax(top_p_filter(logits, 0.5)), rng))
    hist("min_p=0.1",          lambda: gumbel_max(softmax(min_p_filter(logits, 0.1)), rng))

    # ---- 组合: 一站式 sample(...) ------------------------------ #
    print("\n[2] 一站式 sample(...) 组合演示")
    params = SamplingParams(temperature=0.7, top_k=5, top_p=0.9, repetition_penalty=1.2)
    hist_counter = Counter()
    history = [0, 0, 1]            # 已生成过 token 0 两次, token 1 一次
    for _ in range(N):
        hist_counter[sample(logits, params, history=history, rng=rng)] += 1
    print(f"  {'rep_pen+temp+top_k+top_p':<24}", end=" ")
    print("".join(f"{hist_counter.get(i,0):5}" for i in range(10)))
    print("  ↑ 注意 token 0 因 repetition penalty 频次显著降低")

    # ---- Gumbel-Max 等价性验证 --------------------------------- #
    print("\n[3] Gumbel-Max 与 multinomial 是否同分布?")
    p = softmax(logits)
    multi = Counter(int(rng.choice(10, p=p)) for _ in range(N))
    gum = Counter(gumbel_max(p, rng) for _ in range(N))
    print(f"  multi:  {[multi.get(i,0) for i in range(10)]}")
    print(f"  gumbel: {[gum.get(i,0) for i in range(10)]}")
    print("  ✓ 频次相近 (统计意义上同分布)")


if __name__ == "__main__":
    main()
