"""
m01 demo — 对比"暴力解码"与"KV cache 增量解码"

运行:
    python -m llm_infer.m01_kv_cache.demo

输出三段:
    1) 正确性: 两条路径输出的 logits 应该数值相同
    2) 性能:   每一步解码耗时, cache 版本单步成本与 t 无关
    3) 总结:   累计加速比
"""
from __future__ import annotations
import numpy as np

from llm_infer.core import TinyLM, ModelConfig, Timer
from llm_infer.core.utils import banner, kv


# --------------------------------------------------------------------- #
# 暴力解码: 每一步都把"prefix + 已生成"重新 prefill 一遍                #
# --------------------------------------------------------------------- #

def generate_no_cache(lm: TinyLM, prompt_ids: np.ndarray, max_new: int):
    """每一步都重新 prefill, 模拟无 KV cache 的朴素实现。"""
    ids = list(prompt_ids)
    times_ms = []
    for _ in range(max_new):
        with Timer() as t:
            logits, _ = lm.prefill(np.array(ids, dtype=np.int64))
            next_id = int(np.argmax(logits[-1]))
        times_ms.append(t.ms)
        ids.append(next_id)
    return ids, times_ms


# --------------------------------------------------------------------- #
# KV cache 解码: prefill 一次, 后续单 token decode_step                 #
# --------------------------------------------------------------------- #

def generate_with_cache(lm: TinyLM, prompt_ids: np.ndarray, max_new: int):
    """正经的 prefill + 增量 decode。"""
    ids = list(prompt_ids)
    times_ms = []
    with Timer() as t0:
        logits, kv_cache = lm.prefill(np.array(ids, dtype=np.int64))
        next_id = int(np.argmax(logits[-1]))
    ids.append(next_id)
    times_ms.append(t0.ms)  # prefill 时间也算进来
    for _ in range(max_new - 1):
        with Timer() as t:
            logits, kv_cache = lm.decode_step(next_id, kv_cache)
            next_id = int(np.argmax(logits))
        times_ms.append(t.ms)
        ids.append(next_id)
    return ids, times_ms


def main():
    cfg = ModelConfig(d_model=64, d_mlp=128, n_layer=4, vocab_size=128, max_seq_len=512)
    lm = TinyLM(cfg)
    prompt = np.array([1, 10, 20, 30, 40, 50], dtype=np.int64)
    max_new = 32

    banner("M01 - KV Cache: brute force vs incremental")

    ids_a, times_a = generate_no_cache(lm, prompt, max_new)
    ids_b, times_b = generate_with_cache(lm, prompt, max_new)

    # ---- 1) 正确性 -------------------------------------------------- #
    print("\n[1] 正确性")
    same = ids_a == ids_b
    kv("无 cache 输出 ids", ids_a)
    kv("有 cache 输出 ids", ids_b)
    kv("两条路径完全一致", same)
    assert same, "KV cache 实现有 bug, 输出不应不同"

    # ---- 2) 单步耗时分布 ------------------------------------------- #
    print("\n[2] 单步耗时 (ms)")
    print(f"  step:  {'no_cache':>10}  {'with_cache':>10}")
    for i in range(0, max_new, max(1, max_new // 8)):
        print(f"  {i:>4}:  {times_a[i]:>10.3f}  {times_b[i]:>10.3f}")

    # ---- 3) 总耗时与加速比 ----------------------------------------- #
    sum_a, sum_b = sum(times_a), sum(times_b)
    print("\n[3] 总结")
    kv("无 cache 总耗时 (ms)", f"{sum_a:.2f}")
    kv("有 cache 总耗时 (ms)", f"{sum_b:.2f}")
    kv("加速比", f"{sum_a / sum_b:.2f}x")
    print("\n结论: 即使在 T=38 这样小的尺度, cache 已带来明显加速; ")
    print("      实际 LLaMA T=4096 时差距是数百倍。")


if __name__ == "__main__":
    main()
