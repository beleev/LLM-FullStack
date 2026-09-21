"""
m01 KV Cache — 把历史 token 的 K/V 存下来, decode 每步只算 1 个新 token。

瓶颈: 无 cache 时第 t 步要重算 t 个 token 的全部层 (延迟随 t 线性涨, 累计 O(T²) 次 token 前向);
      有 cache 后每步只过 1 个 token, 代价是显存: KV 字节 = 2·n_layer·T·D·sizeof(dtype)。
关键数字: 本 demo T=6+32, 累计 token 前向次数 688 → 37 (18.6×); LLaMA-7B fp16 每 token 0.5 MB KV。
读代码盯住: generate_with_cache 里的 `kv_cache` — 每步 K.shape[0] 只 +1, 喂给模型的永远是 1 个 token。
真实系统: HF `past_key_values`; vLLM / SGLang 的 KV block pool 就是这块内存的分页管理 (见 m02)。

运行: python -m llm_infer.m01_kv_cache.demo
"""
from __future__ import annotations
import numpy as np

from llm_infer.core import TinyLM, ModelConfig, Timer
from llm_infer.core.utils import banner, kv


def generate_no_cache(lm: TinyLM, prompt_ids: np.ndarray, max_new: int):
    """朴素基线: 每步把 prompt+已生成 整段重新 prefill。返回 (ids, 每步末位 logits, 每步 ms)。"""
    ids, step_logits, times_ms = list(prompt_ids), [], []
    for _ in range(max_new):
        with Timer() as t:
            logits, _ = lm.prefill(np.array(ids, dtype=np.int64))   # (t, V), 第 t 步重算 t 个 token
        times_ms.append(t.ms)
        step_logits.append(logits[-1])                              # (V,) 只有最后一行有用, 前 t-1 行全是浪费
        ids.append(int(np.argmax(logits[-1])))
    return ids, np.stack(step_logits), times_ms                     # logits: (max_new, V)


def generate_with_cache(lm: TinyLM, prompt_ids: np.ndarray, max_new: int):
    """prefill 一次建 KV, 之后每步 decode_step 只喂 1 个 token。"""
    ids, step_logits, times_ms = list(prompt_ids), [], []
    with Timer() as t:
        logits, kv_cache = lm.prefill(np.asarray(prompt_ids, dtype=np.int64))  # kv: n_layer × (K (T,D), V (T,D))
    last = logits[-1]                                               # (V,)
    times_ms.append(t.ms)
    for _ in range(max_new - 1):
        step_logits.append(last)
        ids.append(int(np.argmax(last)))
        with Timer() as t:
            last, kv_cache = lm.decode_step(ids[-1], kv_cache)      # (V,); 每层 K/V 行数 +1
        times_ms.append(t.ms)
    step_logits.append(last)
    ids.append(int(np.argmax(last)))
    return ids, np.stack(step_logits), times_ms, kv_cache


def main():
    cfg = ModelConfig(d_model=64, d_mlp=128, n_layer=4, vocab_size=128, max_seq_len=512)
    lm = TinyLM(cfg)
    prompt = np.array([1, 10, 20, 30, 40, 50], dtype=np.int64)
    max_new = 32

    banner("M01 - KV Cache: brute force vs incremental")
    ids_a, logits_a, times_a = generate_no_cache(lm, prompt, max_new)
    ids_b, logits_b, times_b, kv_cache = generate_with_cache(lm, prompt, max_new)

    print("\n[1] 正确性: cache 解码 vs 每步重算")
    diff = float(np.abs(logits_a - logits_b).max())
    kv("生成 ids (前 10 个)", ids_b[len(prompt):len(prompt) + 10])
    kv("token 逐个一致", ids_a == ids_b)
    kv("logits max-abs-diff (32 步)", f"{diff:.2e}")
    assert ids_a == ids_b, "KV cache 解码与重算结果不一致"
    assert diff < 1e-4, diff

    print("\n[2] KV 显存: 公式 vs 实际 nbytes")
    T = kv_cache[0][0].shape[0]                       # cache 里的 token 数 = prompt + max_new - 1 (最后一个 token 还没喂回)
    itemsize = kv_cache[0][0].itemsize
    formula = 2 * cfg.n_layer * T * cfg.d_model * itemsize
    actual = sum(K.nbytes + V.nbytes for K, V in kv_cache)
    kv("cache 内 token 数 T", T)
    kv(f"2·L·T·D·{itemsize} (L={cfg.n_layer}, D={cfg.d_model})", f"{formula} B")
    kv("实际 sum(K.nbytes+V.nbytes)", f"{actual} B")
    assert T == len(prompt) + max_new - 1 and formula == actual
    # LLaMA-7B: 32 层, D=4096, fp16 → 每 token 2·32·4096·2 B
    per_tok = 2 * 32 * 4096 * 2
    kv("同公式代入 LLaMA-7B fp16", f"{per_tok / 2**20:.2f} MiB/token, T=4096 → {per_tok * 4096 / 2**30:.2f} GiB")

    print("\n[3] 计算量: 累计过模型的 token 数 (与机器无关)")
    n_prompt = len(prompt)
    work_a = sum(n_prompt + i for i in range(max_new))   # 第 i 步重算 n_prompt+i 个
    work_b = n_prompt + (max_new - 1)                    # prefill 一次 + 每步 1 个
    kv("无 cache", work_a)
    kv("有 cache", work_b)
    kv("比值", f"{work_a / work_b:.1f}x")
    assert work_a == 688 and work_b == 37

    print("\n[4] 实测单步耗时 ms (numpy/CPU, 有噪声, 仅看趋势)")
    print(f"  step:  {'no_cache':>10}  {'with_cache':>10}")
    for i in range(0, max_new, max_new // 8):
        print(f"  {i:>4}:  {times_a[i]:>10.3f}  {times_b[i]:>10.3f}")
    kv("总耗时 无/有 cache (ms)", f"{sum(times_a):.2f} / {sum(times_b):.2f}  → {sum(times_a) / sum(times_b):.1f}x")
    print("  注: T=38 太小, 实测加速比远低于 token 数比值 — 每次 numpy 调用的固定开销占主导 (正是 m12 的主题)。")


if __name__ == "__main__":
    main()
