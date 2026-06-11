#!/usr/bin/env python
"""
QLoRA 训练示例: NF4 4-bit 基座 + LoRA 适配器
=============================================

教学目标:
    - NF4 量化误差有多大: 逐层对比 dequant(W) vs 原始 W
    - 显存对账: 基座从 4 字节/参数 → ~0.56 字节/参数 (真 4bit 打包)
    - 量化基座 + 高精度 LoRA 联合前向, loss 照常下降
      (梯度只流过 LoRA 旁路, 量化权重是 buffer, 天然冻结)

运行:
    python -m llm_finetune.run_finetune.qlora.train_qlora
"""

import torch

from llm_models.models.language_models.llama import LLaMA
from llm_models.training import Trainer, TrainingConfig

from llm_finetune import (
    InstructionDataGenerator,
    SFTLoss,
    apply_qlora,
    mark_only_lora_as_trainable,
    print_trainable_parameters,
)
from llm_finetune.methods.qlora import QLoRALinear


def main() -> None:
    torch.manual_seed(42)
    vocab_size = 1000

    cfg = TrainingConfig(
        learning_rate=3e-3,           # 与 LoRA 同理: 参数空间小, lr 可以更大
        batch_size=2, seq_len=32,
        num_steps=50, warmup_steps=5, log_interval=10, seed=42,
    )

    model = LLaMA(
        vocab_size=vocab_size, d_model=256, n_heads=4, num_kv_heads=2,
        num_layers=2, max_len=128, dropout=0.0,
    )

    # ---- 1) 量化误差: 先抽一层看 NF4 的保真度 ----
    w_orig = model.layers[0].attn.w_q.weight.detach().clone()

    apply_qlora(model, r=8, alpha=16, block_size=64)   # 替换 w_q/w_k/w_v/w_o
    mark_only_lora_as_trainable(model)                 # 其余参数全部冻结

    q_layer = model.layers[0].attn.w_q
    w_deq = q_layer.dequantized_weight()
    rel = (w_deq - w_orig).norm() / w_orig.norm()
    print(f"NF4 量化相对误差 (layers.0.attn.w_q): {rel:.2%} "
          f"(block_size=64, 16 个正态分位格点)")

    # ---- 2) 显存对账 ----
    nf4_total, fp32_total = 0, 0
    for m in model.modules():
        if isinstance(m, QLoRALinear):
            mem = m.memory_bytes()
            nf4_total += mem["nf4_bytes"]
            fp32_total += mem["fp32_bytes"]
    print(f"被量化的基座: FP32 {fp32_total / 1024:.0f} KB → NF4 {nf4_total / 1024:.0f} KB "
          f"({fp32_total / nf4_total:.1f}x 压缩, 含 scale 开销)")
    print_trainable_parameters(model, name="QLoRA (NF4 基座 + LoRA r=8)")

    # ---- 3) 训练: 梯度只流过 LoRA 旁路 ----
    data_gen = InstructionDataGenerator(
        vocab_size=vocab_size, batch_size=cfg.batch_size, seq_len=cfg.seq_len,
        seed=cfg.seed,
    )
    trainer = Trainer(model, cfg, data_gen, SFTLoss())
    metrics = trainer.train()

    first, last = metrics[0]["total_loss"], metrics[-1]["total_loss"]
    assert last < first, f"QLoRA loss 未下降: {first:.4f} → {last:.4f}"
    print(f"\nQLoRA 通过: loss {first:.4f} → {last:.4f}")
    print("65B 模型按此配方: 全参微调 780GB → QLoRA <48GB, 单卡可跑 (论文数字)。")


if __name__ == "__main__":
    main()
