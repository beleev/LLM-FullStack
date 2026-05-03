"""
m07 demo — Speculative Decoding (greedy + sampling 双版本)

draft 与 target 用同一个 TinyLM 类, 但 draft 用更小的配置 (n_layer=1)
模拟"便宜模型"。教学场景下两者其实输出无关 (随机权重), 但接受规则的
逻辑流程完全正确。

本 demo 提供两个变体:
    - spec_decode_greedy : argmax 比对接受 (实现简单, 但只在 greedy 解码下等价)
    - spec_decode_sampling: 标准 rejection sampling (Leviathan et al. 2023),
                            保证最终采样分布与 target 单独采样严格相同
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from llm_infer.core import ModelConfig, TinyLM
from llm_infer.core.utils import banner, kv


# --------------------------------------------------------------------- #
# 工具                                                                  #
# --------------------------------------------------------------------- #

def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """数值稳定的 softmax (按最后一维)。"""
    if temperature != 1.0:
        logits = logits / temperature
    x = logits - np.max(logits, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def _sample_categorical(probs: np.ndarray, rng: np.random.Generator) -> int:
    """从离散分布采样一个 token id。"""
    return int(rng.choice(probs.shape[-1], p=probs))


# --------------------------------------------------------------------- #
# Baseline                                                              #
# --------------------------------------------------------------------- #

def baseline_greedy(target: TinyLM, prompt: np.ndarray, max_new: int) -> Tuple[List[int], int]:
    """target 单独 greedy, 记录 forward 次数。"""
    out = list(prompt)
    n_calls = 0
    logits, kv_t = target.prefill(prompt)
    n_calls += 1
    nxt = int(np.argmax(logits[-1]))
    out.append(nxt)
    for _ in range(max_new - 1):
        logits, kv_t = target.decode_step(nxt, kv_t)
        n_calls += 1
        nxt = int(np.argmax(logits))
        out.append(nxt)
    return out, n_calls


# --------------------------------------------------------------------- #
# 变体 A: greedy 比对 (实现最简, 仅在 temperature=0 时等价)             #
# --------------------------------------------------------------------- #

def spec_decode_greedy(
    target: TinyLM, draft: TinyLM, prompt: np.ndarray, max_new: int, K: int = 4
) -> Tuple[List[int], int, List[int]]:
    """Greedy 简化版 — argmax 比对一致就接受。

    适用前提: 用户只想要 target 的 greedy 输出 (temperature=0)。
    在 greedy 设定下, target 与 draft 都退化为 Dirac 分布, rejection 概率
    min(1, p_target/p_draft) 退化为「token 是否相同」, 因此本变体与
    rejection sampling 在 greedy 极限下等价。

    若需要随机采样输出, 必须改用 spec_decode_sampling 才能保证分布正确。
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

        # 3) 逐位 argmax 比对
        n_accept = 0
        base = len(confirmed) - 1
        for i in range(K):
            target_pred = int(np.argmax(t_logits[base + i]))
            if target_pred == d_tokens[i]:
                confirmed.append(d_tokens[i])
                n_accept += 1
            else:
                confirmed.append(target_pred)
                break
        else:
            # 全部接受, bonus +1
            confirmed.append(int(np.argmax(t_logits[base + K])))
        accept_history.append(n_accept)

        if confirmed[-1] == 2:        # eos
            break

    out = confirmed[: len(prompt) + max_new]
    return out, target_calls, accept_history


# --------------------------------------------------------------------- #
# 变体 B: 标准 rejection sampling (Leviathan et al. 2023, Algorithm 1) #
# --------------------------------------------------------------------- #

def spec_decode_sampling(
    target: TinyLM,
    draft: TinyLM,
    prompt: np.ndarray,
    max_new: int,
    K: int = 4,
    temperature: float = 1.0,
    seed: int = 0,
) -> Tuple[List[int], int, List[int]]:
    """标准 speculative decoding 的随机采样版本。

    与 greedy 比对的唯一区别在第 3 步:
        - 对每个 draft token t_i, 计算 p_target(t_i) 和 p_draft(t_i)
        - 以 min(1, p_target/p_draft) 概率接受
        - 拒绝时, 从修正分布 max(0, p_target - p_draft) / Z 重采样
    可证明: 最终输出 token 序列的分布严格等于 target 单独采样得到的分布
            (Leviathan et al. 2023 引理 3.5)。
    """
    rng = np.random.default_rng(seed)
    confirmed = list(prompt)
    target_calls = 0
    accept_history: List[int] = []

    while len(confirmed) - len(prompt) < max_new:
        # 1) draft 采样 K 个 token, 同时记录 draft 在每个位置的完整分布
        d_logits, d_kv = draft.prefill(np.array(confirmed, dtype=np.int64))
        d_tokens: List[int] = []
        d_probs_list: List[np.ndarray] = []
        cur_logits = d_logits[-1]
        for _ in range(K):
            p_d = _softmax(cur_logits, temperature)
            t = _sample_categorical(p_d, rng)
            d_tokens.append(t)
            d_probs_list.append(p_d)
            cur_logits, d_kv = draft.decode_step(t, d_kv)

        # 2) target 一次 prefill 拿到 K+1 个位置的 logits
        ext = confirmed + d_tokens
        t_logits, _ = target.prefill(np.array(ext, dtype=np.int64))
        target_calls += 1

        # 3) 逐位 rejection sampling
        n_accept = 0
        base = len(confirmed) - 1
        rejected = False
        for i in range(K):
            p_target = _softmax(t_logits[base + i], temperature)
            p_draft = d_probs_list[i]
            t_i = d_tokens[i]
            ratio = p_target[t_i] / max(p_draft[t_i], 1e-12)
            accept_prob = min(1.0, float(ratio))
            if rng.random() < accept_prob:
                confirmed.append(t_i)
                n_accept += 1
            else:
                # 从修正分布 max(0, p_target - p_draft) 重采样
                residual = np.maximum(p_target - p_draft, 0.0)
                Z = residual.sum()
                if Z > 0:
                    residual = residual / Z
                    corrected = _sample_categorical(residual, rng)
                else:
                    # 罕见数值边界: 退化为直接采 target
                    corrected = _sample_categorical(p_target, rng)
                confirmed.append(corrected)
                rejected = True
                break

        if not rejected:
            # 全部接受, bonus 从 p_target 末位采样
            p_bonus = _softmax(t_logits[base + K], temperature)
            confirmed.append(_sample_categorical(p_bonus, rng))

        accept_history.append(n_accept)
        if confirmed[-1] == 2:        # eos
            break

    out = confirmed[: len(prompt) + max_new]
    return out, target_calls, accept_history


# --------------------------------------------------------------------- #
# Demo                                                                  #
# --------------------------------------------------------------------- #

def main() -> None:
    banner("M07 - Speculative Decoding (greedy + sampling)")

    cfg_t = ModelConfig(d_model=64, d_mlp=128, n_layer=4, vocab_size=128)
    target = TinyLM(cfg_t)
    draft_same = target  # draft == target → 接受率 100%
    draft_small = TinyLM(ModelConfig(d_model=64, d_mlp=128, n_layer=1, vocab_size=128))

    prompt = np.array([1, 5, 10, 15, 20, 25], dtype=np.int64)
    max_new = 32

    # ---- [1] greedy 变体: argmax 比对 --------------------------------- #
    print("\n[1] greedy 变体 (argmax 比对) — draft == target, 加速极限")
    out_a, calls_a = baseline_greedy(target, prompt, max_new)
    out_b, calls_b, accepts = spec_decode_greedy(target, draft_same, prompt, max_new, K=4)
    kv("baseline ids", out_a)
    kv("spec ids    ", out_b[: len(out_a)])
    kv("一致?", out_a == out_b[: len(out_a)])
    kv("baseline target_calls", calls_a)
    kv("spec     target_calls", calls_b)
    kv("加速 (按 target call 数)", f"{calls_a / calls_b:.2f}x")
    kv("接受 token 数 / 轮", accepts)

    # ---- [2] greedy 变体: draft 是更小模型 ---------------------------- #
    print("\n[2] greedy 变体 — draft 是更小模型 (n_layer=1), 接受率 < 100%")
    out_c, calls_c, accepts_c = spec_decode_greedy(target, draft_small, prompt, max_new, K=4)
    kv("spec ids (前 16)", out_c[:16])
    kv("target_calls", calls_c)
    kv("接受 token 数 / 轮", accepts_c)
    avg_accept = sum(accepts_c) / len(accepts_c)
    kv("平均接受 / K", f"{avg_accept:.2f} / 4 = {100 * avg_accept / 4:.0f}%")

    # ---- [3] sampling 变体: 标准 rejection sampling ------------------- #
    print("\n[3] sampling 变体 (rejection sampling) — 保证分布正确性")
    out_d, calls_d, accepts_d = spec_decode_sampling(
        target, draft_small, prompt, max_new, K=4, temperature=1.0, seed=42
    )
    kv("spec ids (前 16)", out_d[:16])
    kv("target_calls", calls_d)
    kv("接受 token 数 / 轮", accepts_d)
    avg_accept_s = sum(accepts_d) / len(accepts_d)
    kv("平均接受 / K", f"{avg_accept_s:.2f} / 4 = {100 * avg_accept_s / 4:.0f}%")

    print(
        "\n  关键差异:"
        "\n    greedy 变体  : 仅在 temperature=0 时与 target 一致, 实现最简"
        "\n    sampling 变体: 输出分布严格等于 target 单独采样 (论文核心保证)"
        "\n  真实 vLLM 实现走 sampling 路径, draft 用蒸馏小模型时典型接受率 60-80%。"
    )


if __name__ == "__main__":
    main()
