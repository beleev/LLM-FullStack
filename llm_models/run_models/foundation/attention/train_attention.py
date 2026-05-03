#!/usr/bin/env python
"""
注意力机制训练示例

演示 MultiHeadAttention 的可训练性：
- 构造一个极简任务：输入 → Self-Attention → Linear → 输出
- 用 MSE loss 训练注意力层学习把随机输入映射到固定目标模式
- 验证 loss 能下降，证明注意力权重在被有效更新

为什么单独训练 Attention？
  在完整 Transformer 中 attention 只是一个子层，但理解它的可训练性
  有助于调试：如果 attention 单独训练 loss 不降，说明初始化或梯度流有问题。

对应论文：Attention Is All You Need (Vaswani et al., 2017)
"""

import torch
import torch.nn as nn
from llm_models.layers import MultiHeadAttention


class AttentionRegressor(nn.Module):
    """把 Attention 层包装成一个可训练的回归模型。"""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.attn(x))


def main():
    torch.manual_seed(42)

    # --- 配置 ---
    d_model = 128
    n_heads = 4
    batch_size = 4
    seq_len = 16
    num_steps = 100
    log_interval = 20
    lr = 1e-3

    print("=" * 50)
    print("MultiHeadAttention 训练测试")
    print("=" * 50)
    print(f"模型维度: {d_model}, 注意力头数: {n_heads}")
    print(f"训练步数: {num_steps}, 学习率: {lr}")

    model = AttentionRegressor(d_model, n_heads)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # 固定目标模式：attention 需要学会把随机输入映射到这个目标附近
    target_pattern = torch.randn(1, seq_len, d_model)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {num_params:,}")

    print(f"\n{'=' * 60}")
    print(f"开始训练 | 总步数: {num_steps} | LR: {lr}")
    print(f"{'=' * 60}")

    first_loss = None
    last_loss = None

    for step in range(1, num_steps + 1):
        model.train()
        x = torch.randn(batch_size, seq_len, d_model)
        target = target_pattern.expand(batch_size, -1, -1)

        output = model(x)
        loss = loss_fn(output, target)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        loss_val = loss.item()
        if step == 1:
            first_loss = loss_val
        last_loss = loss_val

        if step == 1 or step % log_interval == 0 or step == num_steps:
            print(f"  Step {step:>4d}/{num_steps} | loss: {loss_val:.6f}")

    assert last_loss < first_loss, "Loss 未下降!"
    print(f"\n初始 loss: {first_loss:.6f} → 最终 loss: {last_loss:.6f}")
    print("MultiHeadAttention 训练验证通过!")


if __name__ == "__main__":
    main()
