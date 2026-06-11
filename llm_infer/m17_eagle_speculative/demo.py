"""
m17 demo — EAGLE 式投机解码: draft 看特征, 不只看 token

m07 的经典投机解码用"独立小模型"当 draft, 有两个天然短板:
    1. draft 只能看到 token id —— target 内部丰富的 hidden state 全被扔掉
    2. 要单独维护/蒸馏一个小模型

EAGLE (Li et al., 2024) 的两个关键改动:
    1. **在特征空间做自回归**: draft 预测 target 的下一个 hidden state
           ĥ_{t+1} = Draft(h_t, emb(x_{t+1}))
       h_t 是 target 真实算出的特征 (上一轮验证时白拿的), 信息量远大于
       离散 token —— 这就是 EAGLE 接受率高的根本原因
    2. **复用 target 的输出头**: ĥ 过同一个 lm_head 得到 draft 分布,
       draft 本体只是一层很小的网络

效果 (论文数字): 接受率 ~80%, 端到端 2.7-3.5x, 是 vLLM/SGLang 默认
投机方案; EAGLE-2/3 进一步引入动态草稿树。DeepSeek-V3 的 MTP head
(llm_models mtp.py) 与此同源 —— 都是"特征级多步预测"。

本 demo (numpy 自包含):
    target = 一个递归小 LM (h_t = tanh(W_h h_{t-1} + W_e e_t), 共享头),
    天然暴露特征 h。两个 draft 都用最小二乘一步拟合 (零梯度训练):
        EAGLE 式:   [h_t ; e_{t+1}] → ĥ_{t+1}     (特征 + token)
        token-only: [e_{t+1}]       → ĥ_{t+1}     (只有 token, 对照组)
    然后跑与 m07 相同的 greedy 接受循环, 对比接受率与 target 调用数。
"""
from __future__ import annotations

import numpy as np

from llm_infer.core.utils import banner, kv


# --------------------------------------------------------------------- #
# 一个把特征摆在明面上的递归小 LM (target)                              #
# --------------------------------------------------------------------- #

class TinyRecurrentLM:
    """h_t = tanh(W_h h_{t-1} + W_e e_{x_t});  logits_t = h_t @ E^T (共享头)。"""

    def __init__(self, vocab: int = 64, d: int = 32, seed: int = 0) -> None:
        rs = np.random.RandomState(seed)
        self.embed = rs.randn(vocab, d) * 0.8
        self.w_h = rs.randn(d, d) * (0.9 / np.sqrt(d))   # 谱半径<1, 递推稳定
        self.w_e = rs.randn(d, d) * (1.0 / np.sqrt(d))
        self.d = d

    def step(self, h: np.ndarray, token: int) -> np.ndarray:
        return np.tanh(h @ self.w_h + self.embed[token] @ self.w_e)

    def forward(self, tokens: list[int], h0: np.ndarray | None = None):
        """顺序前向: 返回每个位置的特征 hs[T, d] (一次调用 = 一次"并行验证")。"""
        h = np.zeros(self.d) if h0 is None else h0
        hs = []
        for t in tokens:
            h = self.step(h, t)
            hs.append(h)
        return np.stack(hs)

    def head(self, h: np.ndarray) -> np.ndarray:
        return h @ self.embed.T                          # 共享输出头


# --------------------------------------------------------------------- #
# 两个 lstsq 拟合的 draft (特征级 vs token-only)                        #
# --------------------------------------------------------------------- #

def collect_pairs(lm: TinyRecurrentLM, n_seq: int, seq_len: int, seed: int):
    """收集 (h_t, e_{t+1}, h_{t+1}) 监督对。

    用 **ε-greedy 混合驱动**: 一半步走模型自己的 greedy 输出 (贴近推理时
    的真实分布 —— 真实 EAGLE 就在 target 的输出上训练 draft), 一半步走
    随机 token (扩大状态覆盖, 否则 greedy 很快掉进循环吸引子)。
    """
    rs = np.random.RandomState(seed)
    vocab = lm.embed.shape[0]
    H_prev, E_next, H_next = [], [], []
    for _ in range(n_seq):
        h = np.zeros(lm.d)
        tok = int(rs.randint(vocab))
        for step in range(seq_len):
            h_new = lm.step(h, tok)
            if step > 0:
                H_prev.append(h)
                E_next.append(lm.embed[tok])
                H_next.append(h_new)
            h = h_new
            tok = int(np.argmax(lm.head(h))) if rs.rand() < 0.5 \
                else int(rs.randint(vocab))
    return np.stack(H_prev), np.stack(E_next), np.stack(H_next)


def fit_draft(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """一步最小二乘 = 本 demo 的"训练": 返回线性映射 A (含 bias 列)。"""
    Xb = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    A, *_ = np.linalg.lstsq(Xb, Y, rcond=None)
    return A


# --------------------------------------------------------------------- #
# 与 m07 相同的 greedy 接受循环, draft 换成特征级自回归                 #
# --------------------------------------------------------------------- #

def spec_decode(lm: TinyRecurrentLM, draft_A: np.ndarray, use_feature: bool,
                prompt: list[int], max_new: int, K: int = 4):
    confirmed = list(prompt)
    hs = lm.forward(confirmed)                 # prefill, 顺便拿到真实特征
    target_calls = 1
    accept_history: list[int] = []

    while len(confirmed) - len(prompt) < max_new:
        # ---- 1) draft: 从最后一个"真实特征"出发, 在特征空间滚 K 步 ----
        h_hat = hs[-1]
        d_tokens: list[int] = []
        for _ in range(K):
            nxt = int(np.argmax(lm.head(h_hat)))          # 共享 target 的头
            d_tokens.append(nxt)
            e = lm.embed[nxt]
            x = np.concatenate([h_hat, e, [1.0]]) if use_feature \
                else np.concatenate([e, [1.0]])
            h_hat = x @ draft_A                            # 线性 draft 一步

        # ---- 2) target 一次并行验证 (= 一次调用) ----
        hs_ext = lm.forward(d_tokens, h0=hs[-1])
        target_calls += 1
        # 位置 i 的 target 预测来自 "看完 confirmed + 前 i 个 draft" 的特征:
        # 槽位 0 用 hs[-1], 槽位 i>0 用 hs_ext[i-1], 槽位 K 是全接受后的 bonus
        target_preds = [int(np.argmax(lm.head(hs[-1])))] + [
            int(np.argmax(lm.head(h))) for h in hs_ext
        ]

        # ---- 3) 逐位 argmax 比对 (m07 同款规则) ----
        n_accept = 0
        new_hs = []
        for i in range(K):
            if target_preds[i] == d_tokens[i]:
                confirmed.append(d_tokens[i])
                new_hs.append(hs_ext[i])
                n_accept += 1
            else:
                confirmed.append(target_preds[i])          # 纠错: 用 target 的
                new_hs.append(lm.step(hs[-1] if not new_hs else new_hs[-1],
                                      target_preds[i]))
                break
        else:
            confirmed.append(target_preds[K])               # 全接受, bonus +1
            new_hs.append(lm.step(new_hs[-1], target_preds[K]))
        hs = np.concatenate([hs, np.stack(new_hs)])
        accept_history.append(n_accept)

    return confirmed[: len(prompt) + max_new], target_calls, accept_history


def main() -> None:
    banner("M17 - EAGLE 式投机解码 (特征级 draft)")

    lm = TinyRecurrentLM(vocab=128, d=32, seed=0)

    # ---- "训练" 两个 draft: 同一批数据, 唯一差别是输入有没有特征 h ----
    H_prev, E_next, H_next = collect_pairs(lm, n_seq=24, seq_len=40, seed=7)
    A_eagle = fit_draft(np.concatenate([H_prev, E_next], axis=1), H_next)
    A_token = fit_draft(E_next, H_next)
    kv("draft 训练对 (lstsq 一步拟合)", f"{len(H_prev)} 条 (h_t, e_t+1) → h_t+1")

    def fit_err(A: np.ndarray, X: np.ndarray) -> float:
        pred = np.concatenate([X, np.ones((len(X), 1))], 1) @ A
        return float(np.linalg.norm(pred - H_next) / np.linalg.norm(H_next))

    kv("特征拟合误差: EAGLE 式", f"{fit_err(A_eagle, np.concatenate([H_prev, E_next], 1)):.1%}")
    kv("特征拟合误差: token-only", f"{fit_err(A_token, E_next):.1%}")

    # ---- 16 个随机 prompt 上对比 (瞬态段, 避开 greedy 轨迹后期的循环吸引子) ----
    rs = np.random.RandomState(123)
    max_new, K, n_prompts = 16, 4, 16
    sum_acc = {"eagle": 0.0, "token": 0.0}
    sum_calls = {"eagle": 0, "token": 0}
    for _ in range(n_prompts):
        prompt = [int(t) for t in rs.randint(lm.embed.shape[0], size=4)]
        out_e, calls_e, acc_e = spec_decode(lm, A_eagle, True, prompt, max_new, K)
        out_t, calls_t, acc_t = spec_decode(lm, A_token, False, prompt, max_new, K)
        assert out_e == out_t, "greedy 投机解码输出必须与 target 一致 (与谁当 draft 无关)"
        sum_acc["eagle"] += sum(acc_e)
        sum_acc["token"] += sum(acc_t)
        sum_calls["eagle"] += calls_e
        sum_calls["token"] += calls_t

    baseline_calls = n_prompts * max_new            # target 单独 greedy: 一步一调用
    # 每轮验证最多接受 K 个; 减去 n_prompts 次 prefill 调用得到验证轮数
    rate_e = sum_acc["eagle"] / (sum_calls["eagle"] - n_prompts) / K
    rate_t = sum_acc["token"] / (sum_calls["token"] - n_prompts) / K
    print(f"\n[对比] {n_prompts} 个随机 prompt × {max_new} 新 token, 只换 draft 的输入:")
    kv("EAGLE 式  平均接受率", f"{rate_e:.0%}  (每轮草稿 K={K})")
    kv("token-only 平均接受率", f"{rate_t:.0%}")
    kv("EAGLE 式  target 调用", f"{sum_calls['eagle']} (baseline {baseline_calls}, "
                              f"{baseline_calls / sum_calls['eagle']:.2f}x)")
    kv("token-only target 调用", f"{sum_calls['token']} (baseline {baseline_calls}, "
                               f"{baseline_calls / sum_calls['token']:.2f}x)")
    assert rate_e > rate_t + 0.1, "特征级 draft 的接受率应显著更高"

    print("\n结论: 同样大小的 draft, 喂它 target 的 hidden state, 接受率立刻拉开 —")
    print("      特征里已经写好了 target '接下来想说什么', token id 只是它的影子。")
    print("      (EAGLE 论文 ~80% 接受率 / 3x 加速; MTP head 是它的训练时孪生兄弟)")


if __name__ == "__main__":
    main()
