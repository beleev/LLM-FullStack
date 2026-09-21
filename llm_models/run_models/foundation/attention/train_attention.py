#!/usr/bin/env python
"""
单层 attention 能学会什么: 联想检索 (associative recall)

任务: 序列里有 T 个 (key_i, value_i) 对, 最后一个 token 是一个查询 key_j; 要求在最后位置输出 value_j。
     答案在哪个位置每条样本都不同 ⇒ 逐 token 的 MLP 做不到 (只能输出均值, MSE ≈ 1), 必须"按内容寻址" —— 这正是 attention。
每步都是新采样的数据, 所以这里的 loss 下降是真的学会了, 不是背 batch。
"""

import torch
import torch.nn as nn

from llm_models.layers.core.attention import MultiHeadAttention

D_KEY, D_VAL = 32, 31
D_MODEL = D_KEY + D_VAL + 1                                            # 最后 1 维是 "我是查询" 标志位


def make_batch(B: int, T: int):
    keys, vals = torch.randn(B, T, D_KEY), torch.randn(B, T, D_VAL)
    j = torch.randint(0, T, (B,))                                      # 每条样本要检索的位置
    rows = torch.arange(B)
    x = torch.zeros(B, T + 1, D_MODEL)
    x[:, :T, :D_KEY], x[:, :T, D_KEY:-1] = keys, vals                  # 前 T 个 token: [key | value | 0]
    x[:, T, :D_KEY], x[:, T, -1] = keys[rows, j], 1.0                  # 查询 token:   [key_j | 0 | 1]
    return x, vals[rows, j]                                            # [B, T+1, D], [B, D_VAL]


class Recall(nn.Module):
    def __init__(self, use_attention: bool = True):
        super().__init__()
        self.attn = MultiHeadAttention(D_MODEL, num_heads=4) if use_attention else nn.Linear(D_MODEL, D_MODEL)
        self.proj = nn.Linear(D_MODEL, D_VAL)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.attn(x))[:, -1]                          # 只读查询位置 [B, D_VAL]


def train(model: nn.Module, steps: int = 400) -> tuple:
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    for step in range(1, steps + 1):
        x, y = make_batch(64, 8)
        loss = nn.functional.mse_loss(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        if step == 1 or step % 100 == 0:
            print(f"  step {step:>3d} | mse {loss.item():.4f}")
    return losses[0], sum(losses[-20:]) / 20


def main():
    torch.manual_seed(42)
    print("无 attention 的基线 (逐 token Linear):")
    _, base = train(Recall(use_attention=False))
    print("单层 MultiHeadAttention:")
    first, last = train(Recall(use_attention=True))
    print(f"基线 mse {base:.3f} (≈ value 的方差 1, 等于瞎猜) | attention: {first:.3f} -> {last:.3f}")
    assert base > 0.9, "逐 token 模型不可能知道该取哪个 value"
    assert last < 0.5 * base, "attention 应学会按 key 检索 value"


if __name__ == "__main__":
    main()
