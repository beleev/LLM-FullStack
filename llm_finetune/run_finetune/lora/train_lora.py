#!/usr/bin/env python
"""
LLaMA LoRA (Low-Rank Adaptation) 微调示例
============================================

教学目标:
    - 演示 PEFT 的核心卖点: 仅训练 ~0.5% 的参数仍能让 loss 下降
    - 对比 SFT 全参 (100% trainable) vs LoRA (<1% trainable) 的参数量差异
    - 演示推理路径: 训练后用 ``merge_lora_weights`` 合并权重 → 零额外开销
    - 演示部署路径: ``get_lora_state_dict`` 抽出仅 ~MB 的 adapter 落盘

LoRA 注入位点 (默认):
    LLaMA 的 GroupedQueryAttention 的四个投影矩阵 w_q / w_k / w_v / w_o
    (见 llm_models/layers/core/attention.py:201-204)

运行:
    python -m llm_finetune.run_finetune.lora.train_lora
"""

import torch

from llm_models.models.language_models.llama import LLaMA
from llm_models.training import Trainer, TrainingConfig

from llm_finetune import (
    SFTLoss,
    InstructionDataGenerator,
    apply_lora,
    mark_only_lora_as_trainable,
    merge_lora_weights,
    get_lora_state_dict,
    print_trainable_parameters,
)


def main() -> None:
    cfg = TrainingConfig(
        learning_rate=3e-3,   # LoRA 学习率通常比全参 SFT 高一个量级 (3e-4 → 3e-3),
                              # 因为可训参数仅占 1%, 等效梯度尺度小, 需更大 lr
        batch_size=2,
        seq_len=32,
        num_steps=50,
        warmup_steps=5,
        log_interval=10,
        seed=42,
    )
    torch.manual_seed(cfg.seed)

    # ---- 1) 构造与 SFT 示例同尺寸的 LLaMA Mini ----
    vocab_size = 1000
    model = LLaMA(
        vocab_size=vocab_size,
        d_model=256,
        n_heads=4,
        num_kv_heads=2,
        num_layers=2,
        max_len=128,
        dropout=0.0,
    )
    print_trainable_parameters(model, name="Before LoRA (full)")

    # ---- 2) 注入 LoRA + 冻结基座 ----
    # 默认 target_modules = ["w_q", "w_k", "w_v", "w_o"], 即 GQA 的四个投影
    apply_lora(model, r=8, alpha=16, dropout=0.0)
    mark_only_lora_as_trainable(model)
    print_trainable_parameters(model, name="After  LoRA (PEFT)")

    # ---- 3) 数据 / Loss / 训练 (与 SFT 同) ----
    data_gen = InstructionDataGenerator(
        vocab_size=vocab_size,
        batch_size=cfg.batch_size,
        seq_len=cfg.seq_len,
        prompt_len=cfg.seq_len // 2,
        seed=cfg.seed,
    )
    loss_fn = SFTLoss()
    trainer = Trainer(model, cfg, data_gen, loss_fn)
    metrics = trainer.train()

    first, last = metrics[0]["total_loss"], metrics[-1]["total_loss"]
    assert last < first, f"LoRA loss 未下降: first={first:.4f}  last={last:.4f}"
    print(f"\nLoRA 训练通过: loss {first:.4f} → {last:.4f}")

    # ---- 4) 演示部署: 抽出 adapter (落盘只需 ~KB) ----
    adapter = get_lora_state_dict(model)
    adapter_params = sum(t.numel() for t in adapter.values())
    print(
        f"Adapter 参数量: {adapter_params:,} "
        f"(可独立 torch.save 为 ~{adapter_params * 4 / 1024:.1f} KB)"
    )

    # ---- 5) 演示推理: 合并权重, 之后 forward 与原 nn.Linear 同速 ----
    merge_lora_weights(model)
    with torch.inference_mode():
        idx = torch.randint(1, vocab_size, (1, 8))
        logits = model(idx)
        assert logits.shape == (1, 8, vocab_size)
    print("LoRA 合并 + 推理通过")


if __name__ == "__main__":
    main()
