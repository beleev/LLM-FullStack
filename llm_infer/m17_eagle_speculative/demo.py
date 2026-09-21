"""
m17 demo — 同一个 target、同一批训练数据、同一个验证循环 (m07.speculative_decode),
唯一变量是 draft 的输入里有没有 target 特征 h。
assert: (1) 两种 draft 的输出都与 target greedy 逐 token 相同;
        (2) target 调用数 = 1 次 prefill + 验证轮数, 没有任何漏记的调用;
        (3) 特征级 draft 每轮接受更多、target 调用更少。
"""
from __future__ import annotations

import numpy as np

from llm_infer.core import ModelConfig, TinyLM
from llm_infer.core.utils import banner, kv
from llm_infer.m07_speculative_decoding.speculative import speculative_decode
from llm_infer.m17_eagle_speculative.eagle import EagleDrafter, collect_pairs, fit_draft


class CountingLM(TinyLM):
    """数 forward 次数, 用来对账 speculative_decode 自报的 target_calls。"""
    n_forward = 0

    def forward(self, *a, **k):
        self.n_forward += 1
        return super().forward(*a, **k)


def main() -> None:
    banner("M17 - EAGLE 式投机解码 (特征级 draft)")
    lm = CountingLM(ModelConfig())                        # V=128, D=32, 4 层
    V = lm.cfg.vocab_size

    H_prev, E_next, H_next = collect_pairs(lm, n_seq=150, seq_len=40, seed=7)
    X = {"EAGLE 式 [h_t; e_t+1]": np.concatenate([H_prev, E_next], axis=1),   # (N, 2D)
         "token-only [e_t+1]": E_next}                                         # (N, D)
    A = {name: fit_draft(x, H_next) for name, x in X.items()}
    kv("draft 训练对 (lstsq 一步拟合)", f"{len(H_next)} 条")
    for name, x in X.items():
        pred = np.concatenate([x, np.ones((len(x), 1))], 1) @ A[name]
        kv(f"特征拟合相对误差: {name}", f"{np.linalg.norm(pred - H_next) / np.linalg.norm(H_next):.1%}")

    n_prompts, max_new, K = 32, 24, 4
    rs = np.random.RandomState(123)
    prompts = [rs.randint(V, size=6) for _ in range(n_prompts)]
    refs = [lm.generate_greedy(p, max_new) for p in prompts]
    baseline_calls = n_prompts * max_new

    print(f"\n[对比] {n_prompts} 个随机 prompt × {max_new} 新 token, K={K}, baseline target 调用 = {baseline_calls}")
    stats = {}
    for name in X:
        drafter = EagleDrafter(lm, A[name], use_feature=name.startswith("EAGLE"))
        calls, accepts = 0, []
        lm.n_forward = 0
        for p, ref in zip(prompts, refs):
            out, c, acc = speculative_decode(lm, drafter, p, max_new, K)
            assert out == ref, f"{name}: 输出必须与 target greedy 逐 token 相同"
            calls += c
            accepts += acc
        assert calls == lm.n_forward == n_prompts + len(accepts), "target 调用必须全部入账"
        stats[name] = (np.mean(accepts), np.mean(np.array(accepts) > 0), calls)
        kv(name, f"每轮接受 {stats[name][0]:.2f}/{K}, 首槽命中 {stats[name][1]:.0%}, "
                 f"target 调用 {calls} ({baseline_calls / calls:.2f}x)")

    (acc_e, first_e, calls_e), (acc_t, first_t, calls_t) = stats.values()
    assert acc_e > 1.3 * acc_t and first_e > 1.5 * first_t and calls_e < calls_t < baseline_calls

    print("\n结论: target、训练数据、验证循环全都一样, draft 只多喂一个 target 特征 h, 接受就拉开。")
    print("      数字远低于论文的 ~80%: 这里的 draft 是一个线性映射, 真 EAGLE 是一层带注意力的")
    print("      decoder 且经过训练; 这里的 target 也是随机权重, 没有语言规律可学。")


if __name__ == "__main__":
    main()
