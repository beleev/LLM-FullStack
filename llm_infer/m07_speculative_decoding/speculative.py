"""
m07 speculative.py — 投机解码: draft 猜 K 个, target 一次 forward 验 K+1 个位置

瓶颈: decode 每步只产 1 token, 却要把整份权重从显存读一遍 (带宽受限, 延迟高)。
做法: 便宜的 draft 连猜 K 个 token → target 一次 forward 并行验证 → 接受最长合法前缀,
      再白送 1 个 (纠错或 bonus)。每次 target 调用产出 1 + n_accept 个 token。
关键数字: 每 token 接受率 α → 每次 target 调用期望产出 (1-α^(K+1))/(1-α) 个 token。
读代码盯住: speculative_decode 里的 kv (target KV 只覆盖 out[:-1], 拒绝后 truncate_kv
      回滚, 从不重新 prefill) 和 n (本轮接受数)。
对应真实系统: vLLM spec_decode 的 rejection sampler / SGLang speculative;
      m17 (EAGLE drafter) 与 m19 (树形验证) 复用本文件的接受规则与循环。
"""
from __future__ import annotations

import dataclasses
from typing import List, Optional, Tuple

import numpy as np

from llm_infer.core import TinyLM, softmax, truncate_kv
from llm_infer.core.tiny_model import LayerWeights, ModelWeights


def make_draft(target: TinyLM, n_layer: Optional[int] = None, noise: float = 0.0, seed: int = 1) -> TinyLM:
    """从 target 造一个"有时猜得对"的 draft (随机权重的独立小模型 argmax 命中率只有 1/V)。

    n_layer: 只保留前 n 层, 共享 embedding / lm_head (LayerSkip 式自投机, 真的更便宜);
    noise  : 给矩阵权重加相对标准差为 noise 的高斯噪声 (模拟"蒸馏得不完美"的 draft, 计算量不变)。
    """
    rs = np.random.RandomState(seed)

    def perturb(w: np.ndarray) -> np.ndarray:
        if noise == 0.0 or w.ndim == 1:                   # norm 的 gamma 不动
            return w
        return (w + noise * w.std() * rs.randn(*w.shape)).astype(np.float32)

    layers = [LayerWeights(**{f.name: perturb(getattr(l, f.name)) for f in dataclasses.fields(l)})
              for l in target.w.layers[:n_layer]]
    w = ModelWeights(tok_emb=target.w.tok_emb, layers=layers,
                     norm_f_g=target.w.norm_f_g, lm_head=target.w.lm_head)
    return TinyLM(dataclasses.replace(target.cfg, n_layer=len(layers)), w)


def sample(p: np.ndarray, rng: np.random.Generator) -> int:
    """从离散分布 p (V,) 采一个 id (逆 CDF)。"""
    return int(min(np.searchsorted(np.cumsum(p), rng.random(), side="right"), len(p) - 1))


def pick(logits: np.ndarray, temperature: float, rng) -> Tuple[int, Optional[np.ndarray]]:
    """temperature=0 → (argmax, None); 否则 → (采样 id, 完整分布 p (V,))。"""
    if temperature == 0:
        return int(np.argmax(logits)), None
    p = softmax(logits.astype(np.float64) / temperature)
    return sample(p, rng), p


# --------------------------------------------------------------------- #
# 接受规则: 输入 K 个 draft token + target 在 K+1 个槽位的输出          #
# 输出 (n_accept, next_token); next_token 是纠错 token (n<K) 或 bonus   #
# --------------------------------------------------------------------- #

def accept_greedy(d_tokens: List[int], t_logits: np.ndarray) -> Tuple[int, int]:
    """t_logits (K+1, V): 第 i 行 = target 看完 [ctx, d_0..d_{i-1}] 后对槽位 i 的预测。"""
    t_pred = t_logits.argmax(-1)                          # (K+1,)
    n = 0
    while n < len(d_tokens) and d_tokens[n] == t_pred[n]:
        n += 1
    return n, int(t_pred[n])


def accept_sampling(d_tokens: List[int], d_probs: np.ndarray, t_probs: np.ndarray,
                    rng: np.random.Generator) -> Tuple[int, int]:
    """Leviathan et al. 2023 的 rejection sampling。d_probs (K, V), t_probs (K+1, V)。

    以 min(1, p_t/p_d) 接受 d_i; 拒绝则从残差 max(0, p_t - p_d) 重采样并停止。
    两步合起来, 槽位 i 的输出分布恰好是 p_t (README 有推导) —— 这是"无损"的全部来源。
    """
    for i, tok in enumerate(d_tokens):
        if rng.random() < t_probs[i, tok] / d_probs[i, tok]:     # tok 采自 p_d, 分母 > 0
            continue
        residual = np.maximum(t_probs[i] - d_probs[i], 0.0)      # (V,)
        z = residual.sum()
        return i, sample(residual / z if z > 0 else t_probs[i], rng)
    return len(d_tokens), sample(t_probs[-1], rng)                # 全接受: bonus 直接采自 target


# --------------------------------------------------------------------- #
# Drafter: propose(out, K, temperature, rng, hidden) → (tokens, probs)   #
# --------------------------------------------------------------------- #

class ModelDrafter:
    """独立小模型当 draft, 自带 KV cache。回滚靠"已喂 token 与 out 的公共前缀"。"""

    def __init__(self, lm: TinyLM):
        self.lm, self.kv, self.fed, self.calls = lm, None, [], 0

    def propose(self, out: List[int], K: int, temperature: float, rng, hidden=None):
        n = 0                                             # 公共前缀长度 = 仍然有效的 draft KV
        while n < min(len(self.fed), len(out) - 1) and self.fed[n] == out[n]:
            n += 1
        kv = truncate_kv(self.kv, n) if n else None       # 被拒绝的 draft token 的 KV 在这里丢掉
        feed = out[n:]                                    # 追平: 首轮是整个 prompt, 之后 1~2 个 token
        tokens, probs = [], []
        for _ in range(K):
            logits, kv = self.lm.forward(feed, kv)        # (len(feed), V)
            self.calls += 1
            tok, p = pick(logits[-1], temperature, rng)
            tokens.append(tok)
            probs.append(p)
            feed = [tok]
        self.kv, self.fed = kv, out + tokens[:-1]         # 最后一个 draft token 还没喂进去
        return tokens, (None if temperature == 0 else np.stack(probs))   # (K, V)


# --------------------------------------------------------------------- #
# 主循环 (chain 形 draft)。target_calls 如实计数: 1 次 prefill + 每轮 1 次 #
# --------------------------------------------------------------------- #

def speculative_decode(target: TinyLM, drafter, prompt, max_new: int, K: int = 4,
                       temperature: float = 0.0, rng: Optional[np.random.Generator] = None,
                       accept=accept_sampling) -> Tuple[List[int], int, List[int]]:
    """返回 (prompt+新 token, target forward 次数, 每轮接受数)。temperature=0 走 greedy 规则。

    不变量: 每轮开始时 target 的 kv 恰好覆盖 out[:-1]; out[-1] 已确定但还没喂给 target。
    hidden = 产生 out[-1] 的那个位置的 target 特征 (D,), 只有 EAGLE 式 drafter 用。
    """
    logits, kv, hid = target.forward(prompt, return_hidden=True)      # prefill, (T, V)
    target_calls = 1
    out = list(map(int, prompt)) + [pick(logits[-1], temperature, rng)[0]]
    hidden = hid[-1]
    accepts: List[int] = []

    while len(out) - len(prompt) < max_new:
        n_ctx = len(out) - 1                              # 当前 target KV 长度
        d_tokens, d_probs = drafter.propose(out, K, temperature, rng, hidden)

        # 一次 forward 验 K+1 个槽位: 喂 [out[-1], d_0..d_{K-1}], 因果 mask 保证
        # 第 i 行只看到 d_{<i}, 等价于 K+1 次独立的 decode_step
        logits, kv, hid = target.forward([out[-1]] + d_tokens, kv, return_hidden=True)   # (K+1, V)
        target_calls += 1

        if temperature == 0:
            n, nxt = accept_greedy(d_tokens, logits)
        else:
            t_probs = softmax(logits.astype(np.float64) / temperature)                  # (K+1, V)
            n, nxt = accept(d_tokens, d_probs, t_probs, rng)

        out += d_tokens[:n] + [nxt]
        kv = truncate_kv(kv, n_ctx + 1 + n)               # 回滚: 丢掉被拒 draft 的 KV, 不重新 prefill
        hidden = hid[n]
        accepts.append(n)

    return out[: len(prompt) + max_new], target_calls, accepts
