"""
m17 eagle.py — EAGLE 式 draft: 在 target 的特征空间里自回归, 而不是另养一个小 LM

瓶颈: m07 的独立 draft 只看得到 token id, target 算好的 hidden state 全被扔掉 → 接受率低,
      target 调用省不下来 (延迟)。
做法: ĥ_{t+1} = Draft(h_t, emb(x_{t+1})), 再过 **target 自己的 lm_head** 得到 draft token。
      h_t 是上一轮验证时 target 白送的真特征 (TinyLM.forward(..., return_hidden=True))。
关键数字: 论文接受率 ~80% / 端到端 ~3x; 本 demo 的线性 draft 只有 (2D+1)×D 个参数, 数字小得多,
      但"只差一个输入 h"的对照组足以看出差距。
读代码盯住: EagleDrafter.propose 里的 h —— 第 1 步是 target 的真特征, 之后是 draft 自己的 ĥ
      (误差逐步累积, 所以越靠后的槽位越难接受)。
对应真实系统: vLLM / SGLang 的 EAGLE / EAGLE-3 speculator; DeepSeek-V3 的 MTP head 同源。
验证循环与接受规则不在这里 —— 直接复用 m07 的 speculative_decode, 只换 drafter。
"""
from __future__ import annotations

from typing import List

import numpy as np

from llm_infer.core import TinyLM
from llm_infer.m07_speculative_decoding.speculative import pick


def collect_pairs(lm: TinyLM, n_seq: int, seq_len: int, seed: int, p_greedy: float = 0.8):
    """跑 target 收集监督对 (h_t, emb(x_{t+1})) → h_{t+1}。

    轨迹 80% 走 target 自己的 greedy (贴近推理分布, 真实 EAGLE 也在 target 输出上训练),
    20% 走随机 token (扩大覆盖, 否则 greedy 很快掉进循环)。
    返回 H_prev (N, D), E_next (N, D), H_next (N, D)。
    """
    rs = np.random.RandomState(seed)
    V = lm.cfg.vocab_size
    H_prev, E_next, H_next = [], [], []
    for _ in range(n_seq):
        ids = [int(rs.randint(V))]
        logits, cache, hid = lm.forward(ids, return_hidden=True)
        hs = [hid[-1]]
        for _ in range(seq_len):
            tok = int(np.argmax(logits[-1])) if rs.rand() < p_greedy else int(rs.randint(V))
            ids.append(tok)
            logits, cache, hid = lm.forward([tok], cache, return_hidden=True)
            hs.append(hid[-1])
        hs = np.stack(hs)                                 # (seq_len+1, D)
        H_prev.append(hs[:-1])
        E_next.append(lm.w.tok_emb[ids[1:]])              # (seq_len, D)
        H_next.append(hs[1:])
    return np.concatenate(H_prev), np.concatenate(E_next), np.concatenate(H_next)


def fit_draft(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """一步最小二乘 = 本 demo 的"训练"。X (N, d_in), Y (N, D) → A (d_in+1, D), 末行是 bias。"""
    Xb = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    A, *_ = np.linalg.lstsq(Xb, Y, rcond=None)
    return A


class EagleDrafter:
    """m07 drafter 协议的特征级实现。use_feature=False 是只看 token 的对照组。

    不需要自己的 KV: 线性 draft 对 (h, token) 是马尔可夫的, 回滚 = 什么都不用做。
    draft 一步只是一次 (2D+1)→D 的矩阵乘, 不产生任何 target 调用。
    """

    def __init__(self, lm: TinyLM, A: np.ndarray, use_feature: bool = True):
        self.lm, self.A, self.use_feature = lm, A, use_feature

    def propose(self, out: List[int], K: int, temperature: float, rng, hidden: np.ndarray):
        h, tok = hidden, out[-1]              # h: 产生 out[-1] 的位置的 target 真特征 (D,)
        tokens, probs = [], []
        for _ in range(K):
            e = self.lm.w.tok_emb[tok]                                    # (D,)
            x = np.concatenate([h, e, [1.0]]) if self.use_feature else np.concatenate([e, [1.0]])
            h = x @ self.A                                                # ĥ (D,), 下一步的输入
            tok, p = pick(h @ self.lm.w.lm_head, temperature, rng)        # 共享 target 的输出头
            tokens.append(tok)
            probs.append(p)
        return tokens, (None if temperature == 0 else np.stack(probs))
