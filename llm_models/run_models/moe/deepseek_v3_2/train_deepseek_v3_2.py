#!/usr/bin/env python
"""
DeepSeek-V3.2 (DSA + MoE) 训练示例

在 DeepSeek-V3 训练流程基础上引入 DSA（稀疏注意力）。
训练目标与 V3 完全相同：next-token prediction + MoE 辅助负载均衡 loss。

DSA (DeepSeek Sparse Attention) 的训练影响：
- forward 与 V3 语义一致：输入 token id → 输出 logits + routing_info
- 但每一层的 attention 只在 top-k 个 key 上计算，梯度也只经过被选中的路径
- 辅助 loss 仍来自 MoE router，DSA 的 indexer 通过端到端梯度一起训练

合成数据 + MoE 辅助 loss，仅验证训练循环能跑通、loss 能下降。
"""

import torch
from llm_models.models import DeepSeekV3_2
from llm_models.training import (
    Trainer,
    TrainingConfig,
    MoELMLoss,
    DecoderOnlyDataGenerator,
)


def main():
    config = TrainingConfig(
        learning_rate=3e-4,
        batch_size=2,
        seq_len=32,
        num_steps=50,
        warmup_steps=5,
        aux_loss_weight=0.01,
        log_interval=10,
        seed=42,
    )
    torch.manual_seed(config.seed)

    vocab_size = 1000
    model = DeepSeekV3_2(
        vocab_size=vocab_size,
        d_model=256,
        n_heads=4,
        num_layers=2,
        num_routed_experts=4,
        num_shared_experts=1,
        top_k=2,
        dropout=0.1,
        sparse_top_k=16,
        indexer_heads=2,
    )

    param_info = model.get_num_active_params()
    print(f"DeepSeek-V3.2 Mini | 总参数: {param_info['total_params']:,} | "
          f"激活参数: {param_info['active_params']:,}")

    data_gen = DecoderOnlyDataGenerator(
        vocab_size=vocab_size,
        batch_size=config.batch_size,
        seq_len=config.seq_len,
    )
    loss_fn = MoELMLoss(aux_loss_weight=config.aux_loss_weight)

    trainer = Trainer(model, config, data_gen, loss_fn)
    metrics = trainer.train()

    assert metrics[-1]["total_loss"] < metrics[0]["total_loss"], "Loss 未下降!"
    print(f"最终 lm_loss: {metrics[-1]['lm_loss']:.4f} | "
          f"aux_loss: {metrics[-1]['aux_loss']:.4f}")
    print("DeepSeek-V3.2 训练验证通过!")


if __name__ == "__main__":
    main()
