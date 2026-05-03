#!/usr/bin/env python
"""
DeepSeek-V3 (MoE) 训练示例

演示使用通用 Trainer 训练 DeepSeek-V3 MoE 模型。
使用合成数据 + MoE 专用损失 (LM loss + 辅助负载均衡 loss)。

为什么需要辅助 loss（auxiliary load-balancing loss）：
- MoE 的 router 容易"塌缩"——所有 token 都被路由到少数几个专家，
  导致其他专家训不到、参数浪费
- 辅助 loss 鼓励 token 均匀分配到各路由专家，缓解负载不均衡
- 总 loss = lm_loss + aux_loss_weight * aux_loss
  aux_loss_weight 通常很小（如 0.01），避免压过主任务

合成数据：随机 token id 序列，仅用于跑通流程并验证 loss 能下降。
"""

import torch
from llm_models.models import DeepSeekV3
from llm_models.training import (
    Trainer,
    TrainingConfig,
    MoELMLoss,
    DecoderOnlyDataGenerator,
)


def main():
    # --- 训练配置 ---
    # 全部为极小值，可在 CPU 上几秒内跑完，仅做流程验证
    config = TrainingConfig(
        learning_rate=3e-4,
        batch_size=2,
        seq_len=32,
        num_steps=50,
        warmup_steps=5,             # 学习率 warmup，前 5 步线性升温
        aux_loss_weight=0.01,       # 辅助负载均衡 loss 权重
        log_interval=10,
        seed=42,
    )
    torch.manual_seed(config.seed)

    # --- 模型配置 (MoE-Mini) ---
    vocab_size = 1000
    model = DeepSeekV3(
        vocab_size=vocab_size,
        d_model=256,
        n_heads=4,
        num_layers=2,
        max_len=128,
        num_shared_experts=1,       # 1 个共享专家（始终激活）
        num_routed_experts=4,       # 4 个路由专家
        top_k=2,                    # 每 token 选 top-2 路由专家
        dropout=0.1,
    )

    # MoE 模型的关键观察指标：总参数 vs 单 token 实际激活参数
    param_info = model.get_num_active_params()
    print(f"DeepSeek-V3 Mini | 总参数: {param_info['total_params']:,} | "
          f"激活参数: {param_info['active_params']:,}")

    # --- 数据生成器 + MoE 损失函数 ---
    # 合成数据：每步采样随机 token id；输入和 label 错位 1（next-token prediction）
    data_gen = DecoderOnlyDataGenerator(
        vocab_size=vocab_size,
        batch_size=config.batch_size,
        seq_len=config.seq_len,
    )
    # MoELMLoss 会从模型输出中提取 routing_info，计算辅助负载均衡 loss
    loss_fn = MoELMLoss(aux_loss_weight=config.aux_loss_weight)

    # --- 训练 ---
    trainer = Trainer(model, config, data_gen, loss_fn)
    metrics = trainer.train()

    # --- 验证 ---
    # 最简单的训练有效性检查：最终 loss 应小于初始 loss
    assert metrics[-1]["total_loss"] < metrics[0]["total_loss"], "Loss 未下降!"
    print(f"最终 lm_loss: {metrics[-1]['lm_loss']:.4f} | "
          f"aux_loss: {metrics[-1]['aux_loss']:.4f}")
    print("DeepSeek-V3 训练验证通过!")


if __name__ == "__main__":
    main()
