#!/usr/bin/env python
"""
LLaDA 采样示例 — 把 "从全 [MASK] 逐步去噪" 的过程一步步打印出来。

独立可跑: 内部先训练 400 步 (~10s, 复用 train_llada.train), 再采样; 不依赖任何 checkpoint。
看什么: 每一步 `_` ([MASK]) 的个数按线性日程减少; 先定下来的是模型最有把握的位置 (不一定挨着已知 token),
        后面的步骤把它们当作新的已知条件; 已定的 token 之后不再变。
"""

import torch

from llm_models.run_models.language_models.llada.train_llada import L, P, make_batch, train


def show(row: torch.Tensor, mask_id: int) -> str:
    return " ".join(" _" if v == mask_id else f"{v:2d}" for v in row.tolist())


def main():
    model, perm = train()
    x = make_batch(perm, 256, torch.Generator().manual_seed(7))  # [B, L] held-out
    M = model.mask_id

    # 只给中间一个 token, 两头都要填 — 自回归模型做不了 "往左填"
    noisy = torch.full_like(x, M)
    noisy[:, 8] = x[:, 8]
    steps = 5
    out, history = model.sample(noisy, steps, return_history=True)

    print(f"\n真值     : {show(x[0], M)}")
    print(f"输入     : {show(noisy[0], M)}")
    for s, h in enumerate(history, 1):
        print(f"第 {s} 步后 : {show(h[0], M)}")

    n_gen = L - 1
    prev = noisy
    for s, h in enumerate(history, 1):
        n_masked = (h == M).sum(1)  # [B]
        assert (n_masked == round(n_gen * (1 - s / steps))).all(), "[MASK] 个数应服从线性日程"
        committed = prev != M
        assert (h[committed] == prev[committed]).all(), "已定下来的 token 不应再变 (含 prompt)"
        prev = h
    assert not (out == M).any(), "最终输出不应含 [MASK]"
    assert (out < P).all(), "只应生成数据 token"

    # 双向 + padding mask: 改动被 attention_mask 屏蔽的位置, 不应影响其余位置的 logits
    am = torch.ones_like(x)
    am[:, -4:] = 0
    x_pad = x.clone()
    x_pad[:, -4:] = 0
    with torch.inference_mode():
        diff = (model(x, am)[:, :-4] - model(x_pad, am)[:, :-4]).abs().max().item()
        leak = (model(x)[:, :-4] - model(x_pad)[:, :-4]).abs().max().item()
    assert diff < 1e-4 and leak > 1e-2, "pad 应被屏蔽; 不传 mask 时右侧 token 应能影响左侧 (双向)"

    # temperature > 0 走采样分支, 同样不能采出 [MASK]
    assert not (model.sample(noisy, steps, temperature=1.0) == M).any()

    valid = (perm[out[:, :-1]] == out[:, 1:]).float().mean().item()  # 相邻对满足 x[i+1] = π(x[i]) 的比例
    acc = (out == x).float().mean().item()
    print(f"\n{steps} 步 low-confidence: 与真值一致 {acc:.3f} | 链规则成立 {valid:.3f} | chance {1 / P:.3f}")
    assert acc > 0.8, "应远高于 chance"
    print("LLaDA 采样验证通过")


if __name__ == "__main__":
    main()
