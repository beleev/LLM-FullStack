"""
m07 demo — Speculative Decoding (greedy 版)

draft 与 target 用同一个 TinyLM 类, 但 draft 用更小的配置 (n_layer=1)
模拟"便宜模型"。教学场景下两者其实输出无关 (随机权重), 但接受规则的
逻辑流程完全正确。
"""
from __future__ import annotations
from typing import List, Tuple
import numpy as np

from llm_infer.core import TinyLM, ModelConfig
from llm_infer.core.utils import banner, kv


def baseline_greedy(target: TinyLM, prompt: np.ndarray, max_new: int) -> Tuple[List[int], int]:
    """target 单独 greedy, 记录 forward 次数。"""
    out = list(prompt)
    n_calls = 0
    logits, kv_t = target.prefill(prompt); n_calls += 1
    nxt = int(np.argmax(logits[-1])); out.append(nxt)
    for _ in range(max_new - 1):
        logits, kv_t = target.decode_step(nxt, kv_t); n_calls += 1
        nxt = int(np.argmax(logits)); out.append(nxt)
    return out, n_calls


def spec_decode(
    target: TinyLM, draft: TinyLM, prompt: np.ndarray, max_new: int, K: int = 4
) -> Tuple[List[int], int, List[int]]:
    """
    Algorithm:
        1) draft 从当前位置 autoregressive 出 K 个候选
        2) target 把 (prefix + K candidates) prefill 一次, 拿到 K+1 个 logits
        3) 从前往后比对: target.argmax(i) == draft[i]? 接受; 否则用 target 替换并停
        4) 接受 m 个 (0 <= m <= K), 还能 +1 (target 在第 m 位的预测一定接受)
        5) 把已确认的 token 写回 KV cache, 继续

    我们简化: 每轮 target 都全量 prefill (prompt + 已确认 + K candidates)。
    真实 vLLM 用 KV cache 增量, 但等价。
    """
    confirmed = list(prompt)
    target_calls = 0
    accept_history: List[int] = []

    while len(confirmed) - len(prompt) < max_new:
        # 1) draft 出 K 个 token (从 confirmed 末尾开始)
        d_logits, d_kv = draft.prefill(np.array(confirmed, dtype=np.int64))
        d_tokens: List[int] = []
        cur_logits = d_logits[-1]
        for _ in range(K):
            t = int(np.argmax(cur_logits))
            d_tokens.append(t)
            cur_logits, d_kv = draft.decode_step(t, d_kv)

        # 2) target 一次 prefill (confirmed + d_tokens)
        ext = confirmed + d_tokens
        t_logits, _ = target.prefill(np.array(ext, dtype=np.int64))
        target_calls += 1

        # 3) 逐位接受
        # t_logits[i] 对应"看到 ext[:i+1] 之后的预测",
        # 我们要校验位置 [len(confirmed)-1 .. len(confirmed)+K-1]
        n_accept = 0
        base = len(confirmed) - 1     # confirmed 最后一个 token 的下标
        for i in range(K):
            target_pred = int(np.argmax(t_logits[base + i]))
            if target_pred == d_tokens[i]:
                confirmed.append(d_tokens[i])
                n_accept += 1
            else:
                confirmed.append(target_pred)   # 用 target 的纠正
                break
        else:
            # 全部接受, 还能再 bonus +1: target 在最后位置的预测
            confirmed.append(int(np.argmax(t_logits[base + K])))
        accept_history.append(n_accept)

        if confirmed[-1] == 2:        # eos
            break

    # 截到 max_new
    out = confirmed[:len(prompt) + max_new]
    return out, target_calls, accept_history


def main():
    banner("M07 - Speculative Decoding (greedy)")

    # 同 cfg 时, draft 与 target 输出完全一致 → 接受率应 = 100%
    cfg_t = ModelConfig(d_model=64, d_mlp=128, n_layer=4, vocab_size=128)
    target = TinyLM(cfg_t)
    # draft 用同一权重模拟"小模型也很准"的极端情况
    draft = target

    prompt = np.array([1, 5, 10, 15, 20, 25], dtype=np.int64)
    max_new = 32

    print("\n[1] draft == target 时 (理论加速极限)")
    out_a, calls_a = baseline_greedy(target, prompt, max_new)
    out_b, calls_b, accepts = spec_decode(target, draft, prompt, max_new, K=4)
    kv("baseline ids", out_a)
    kv("spec     ids", out_b[:len(out_a)])
    kv("一致?", out_a == out_b[:len(out_a)])
    kv("baseline target_calls", calls_a)
    kv("spec     target_calls", calls_b)
    kv("加速 (按 target call 数)", f"{calls_a / calls_b:.2f}x")
    kv("接受 token 数 / 轮", accepts)

    # ---- draft != target: 用更小的 n_layer 模拟 ------------------ #
    print("\n[2] draft 是更小的模型 (n_layer=1), 接受率 < 100%")
    draft_small = TinyLM(ModelConfig(d_model=64, d_mlp=128, n_layer=1, vocab_size=128))
    out_c, calls_c, accepts_c = spec_decode(target, draft_small, prompt, max_new, K=4)
    kv("spec ids (前 16)", out_c[:16])
    kv("target_calls", calls_c)
    kv("接受 token 数 / 轮", accepts_c)
    avg_accept = sum(accepts_c) / len(accepts_c)
    kv("平均接受 / K", f"{avg_accept:.2f} / 4 = {100*avg_accept/4:.0f}%")
    print(f"\n  随权重不同, 接受率从 0% (完全没用) 到 100% (完全没浪费); ")
    print(f"  真实场景 draft = target 的小蒸馏版本时, 典型 60-80%。")


if __name__ == "__main__":
    main()
