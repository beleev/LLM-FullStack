"""
samplers.py — 推理采样策略

输入: logits (V,) numpy array
输出: 采样到的 token id (int)

每个函数都是无状态的; 设置 seed 用 np.random.seed 或传 rng。
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence
import numpy as np

from llm_infer.core.utils import softmax


# --------------------------------------------------------------------- #
# 单一策略                                                              #
# --------------------------------------------------------------------- #

def greedy(logits: np.ndarray) -> int:
    return int(np.argmax(logits))


def temperature_sample(
    logits: np.ndarray, temperature: float, rng: Optional[np.random.RandomState] = None
) -> int:
    if temperature <= 0:
        return greedy(logits)
    rng = rng or np.random
    probs = softmax(logits / temperature, axis=-1)
    return int(rng.choice(len(probs), p=probs))


def top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
    """只保留 top-k logits, 其他置 -inf。"""
    if k >= len(logits):
        return logits
    threshold = np.partition(logits, -k)[-k]
    return np.where(logits >= threshold, logits, -np.inf)


def top_p_filter(logits: np.ndarray, p: float) -> np.ndarray:
    """nucleus: 按概率从大到小累加, 累计 > p 的之后全砍。"""
    if p >= 1.0:
        return logits
    probs = softmax(logits, axis=-1)
    sorted_idx = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_idx]
    cum = np.cumsum(sorted_probs)
    # 第一个累计 ≥ p 的位置之后全砍 (含自己)
    cutoff = np.searchsorted(cum, p) + 1
    keep = sorted_idx[:cutoff]
    out = np.full_like(logits, -np.inf)
    out[keep] = logits[keep]
    return out


def min_p_filter(logits: np.ndarray, min_p: float) -> np.ndarray:
    """凡概率 < min_p × max_prob 的全砍。"""
    probs = softmax(logits, axis=-1)
    threshold = min_p * float(np.max(probs))
    return np.where(probs >= threshold, logits, -np.inf)


def repetition_penalty(logits: np.ndarray, history: Sequence[int], penalty: float) -> np.ndarray:
    """对 history 中出现过的 token, logits 除以 penalty (penalty>1 抑制)。"""
    out = logits.copy()
    for tid in set(history):
        if 0 <= tid < len(out):
            if out[tid] > 0:
                out[tid] /= penalty
            else:
                out[tid] *= penalty
    return out


# --------------------------------------------------------------------- #
# Gumbel-Max (nano-vllm 风格, 一次 kernel)                               #
# --------------------------------------------------------------------- #

def gumbel_max(probs: np.ndarray, rng: Optional[np.random.RandomState] = None) -> int:
    """从 probs 采样: 等价于 multinomial, 但纯 element-wise + argmax。
    
    数学:  argmax(log p_i + Gumbel_i)  ~  multinomial(p)
    实现:  -log(-log(U)) 是 Gumbel(0,1)
    nano-vllm: argmax(probs / Gumbel(0,1))   ← 数值另一种但等价
    """
    rng = rng or np.random
    g = rng.gumbel(0, 1, size=probs.shape)
    return int(np.argmax(np.log(probs + 1e-30) + g))


# --------------------------------------------------------------------- #
# 组合: 一站式采样器                                                    #
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 1.0
    top_k: int = 0                  # 0 = 不开
    top_p: float = 1.0
    min_p: float = 0.0
    repetition_penalty: float = 1.0


def sample(
    logits: np.ndarray,
    params: SamplingParams,
    history: Optional[Sequence[int]] = None,
    rng: Optional[np.random.RandomState] = None,
) -> int:
    """vLLM/SGLang 标准组合顺序: rep_pen → temp → top_k → top_p → min_p → sample"""
    if params.repetition_penalty != 1.0 and history is not None:
        logits = repetition_penalty(logits, history, params.repetition_penalty)
    if params.temperature == 0:
        return greedy(logits)
    logits = logits / params.temperature
    if params.top_k > 0:
        logits = top_k_filter(logits, params.top_k)
    if params.top_p < 1.0:
        logits = top_p_filter(logits, params.top_p)
    if params.min_p > 0:
        logits = min_p_filter(logits, params.min_p)
    probs = softmax(logits, axis=-1)
    return gumbel_max(probs, rng)
