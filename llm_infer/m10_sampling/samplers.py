"""
samplers.py — decode 的最后一步: logits (V,) → 1 个 token id。

瓶颈: 不在算力而在质量与延迟 — 纯 greedy 会复读, 纯采样会从长尾里抽到垃圾 token;
      GPU 上 multinomial 需要 cumsum + 搜索, Gumbel-max 只要 element-wise + argmax。
关键数字: V=128k 时长尾 token 单个概率 ~1e-6, 但总质量可达几个百分点 → 每几十个 token 就抽到一次垃圾; top-k/top-p/min-p 就是砍尾巴。
读代码盯住: 每个 filter 都是 "logits → logits, 被砍的置 -inf", 所以可以任意串联; 顺序有意义 (top_p 看到的是温度缩放后的分布)。
真实系统: vLLM `Sampler` / SGLang `sampler.py` / HF `LogitsProcessor` 链; nano-vllm 用 Gumbel-max 采样。

对外 API (full_engine 依赖, 保持稳定): `SamplingParams`, `sample(logits, params, history=..., rng=...)`, temperature=0 即 greedy。
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence
import numpy as np

from llm_infer.core.utils import softmax


def greedy(logits: np.ndarray) -> int:
    return int(np.argmax(logits))


def temperature_sample(logits: np.ndarray, temperature: float,
                       rng: Optional[np.random.RandomState] = None) -> int:
    """softmax(logits/T) 后 multinomial。T<1 更尖, T>1 更平, T→0 退化为 greedy。"""
    if temperature <= 0:
        return greedy(logits)
    rng = rng or np.random
    probs = softmax(logits / temperature)                # (V,)
    return int(rng.choice(len(probs), p=probs))


def top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
    """恰好保留 k 个最大的 logit, 其余置 -inf。"""
    if k <= 0 or k >= len(logits):
        return logits
    keep = np.argpartition(logits, -k)[-k:]              # (k,) 用下标而不是 ">= 阈值": 并列时后者会留下多于 k 个
    out = np.full_like(logits, -np.inf)
    out[keep] = logits[keep]
    return out


def top_p_filter(logits: np.ndarray, p: float) -> np.ndarray:
    """nucleus: 按概率降序累加, 保留累计质量首次 ≥ p 的最小集合。"""
    if p >= 1.0:
        return logits
    probs = softmax(logits)                              # (V,)
    order = np.argsort(-probs)                           # (V,) 降序下标
    cum = np.cumsum(probs[order])                        # (V,)
    n_keep = int(np.searchsorted(cum, p)) + 1            # searchsorted 给出首个 cum≥p 的位置, +1 把它自己也留下 (至少留 1 个)
    out = np.full_like(logits, -np.inf)
    out[order[:n_keep]] = logits[order[:n_keep]]
    return out


def min_p_filter(logits: np.ndarray, min_p: float) -> np.ndarray:
    """砍掉 p_i < min_p · p_max 的 token。阈值随分布的尖锐程度自动伸缩: 模型很确定时砍得狠, 犹豫时留得多。"""
    probs = softmax(logits)
    return np.where(probs >= min_p * probs.max(), logits, -np.inf)


def repetition_penalty(logits: np.ndarray, history: Sequence[int], penalty: float) -> np.ndarray:
    """CTRL 论文的做法: 出现过的 token, 正 logit 除以 penalty, 负 logit 乘以 penalty。

    分正负是因为目标是"让 logit 变小": 负数除以 >1 的数反而变大 (更可能被选)。与出现次数无关, 只看是否出现过。
    """
    out = logits.copy()
    ids = [t for t in set(history) if 0 <= t < len(out)]
    out[ids] = np.where(out[ids] > 0, out[ids] / penalty, out[ids] * penalty)
    return out


def gumbel_max(probs: np.ndarray, rng: Optional[np.random.RandomState] = None) -> int:
    """argmax_i(log p_i + G_i), G_i ~ Gumbel(0,1) i.i.d.  与 multinomial(p) 同分布。

    好处: 全程 element-wise + 一次 argmax, 没有 cumsum / 二分搜索, batch 维天然并行, GPU 上一个 kernel。
    nano-vllm 写成 argmax(p / E), E~Exp(1): 取 log 后 -log E 正是 Gumbel(0,1), 两者等价。
    """
    rng = rng or np.random
    g = rng.gumbel(0, 1, size=probs.shape)               # (V,)
    with np.errstate(divide="ignore"):
        return int(np.argmax(np.log(probs) + g))         # 被砍 token: log 0 = -inf, 永远不会被选中


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 1.0        # 0 = greedy
    top_k: int = 0                  # 0 = 不开
    top_p: float = 1.0              # 1 = 不开
    min_p: float = 0.0              # 0 = 不开
    repetition_penalty: float = 1.0 # 1 = 不开


def sample(
    logits: np.ndarray,                                  # (V,)
    params: SamplingParams,
    history: Optional[Sequence[int]] = None,
    rng: Optional[np.random.RandomState] = None,
) -> int:
    """rep_penalty → temperature → top_k → top_p → min_p → Gumbel-max。

    penalty 在最前 (greedy 也受影响); 温度在 filter 之前, 所以 top_p/min_p 看到的是缩放后的分布。
    各框架顺序不完全一致 (例如 min_p 放在 top_k/top_p 之前还是之后), 同一组参数跨框架结果可能不同。
    """
    if params.repetition_penalty != 1.0 and history is not None:
        logits = repetition_penalty(logits, history, params.repetition_penalty)
    if params.temperature <= 0:
        return greedy(logits)
    logits = logits / params.temperature
    if params.top_k > 0:
        logits = top_k_filter(logits, params.top_k)
    if params.top_p < 1.0:
        logits = top_p_filter(logits, params.top_p)
    if params.min_p > 0:
        logits = min_p_filter(logits, params.min_p)
    return gumbel_max(softmax(logits), rng)
