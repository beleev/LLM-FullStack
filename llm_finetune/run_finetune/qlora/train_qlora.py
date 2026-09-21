#!/usr/bin/env python
"""
QLoRA: 把基座的全部线性层压成 NF4, 看 (1) 整个模型省了多少 (2) 量化伤了多少能力 (3) 还能不能照常适配新任务。

    python -m llm_finetune.run_finetune.qlora.train_qlora
"""

import torch

from llm_finetune import (
    InstructionDataGenerator, NF4Linear, SeqTask, SFTLoss, apply_qlora, count_parameters,
    mark_only_lora_as_trainable, merge_lora_weights, nf4_dequantize, nf4_quantize, weight_bytes,
)
from llm_finetune.run_finetune.common import fit, pretrained_base

STEPS, RANK, LR = 300, 8, 1e-2


def main() -> None:
    torch.manual_seed(0)
    old_task, task = SeqTask("copy"), SeqTask("sort")
    model = pretrained_base()
    probe = task.sample_prompts(8, "test")

    # ---- 0) 打包的边界情况: block_size 为奇数 → 索引个数为奇数 ----
    w = torch.randn(3, 5)
    packed, scales = nf4_quantize(w, block_size=5)                    # 15 个索引 → 8 字节
    w_hat = nf4_dequantize(packed, scales, w.shape, block_size=5)
    assert packed.numel() == 8 and w_hat.shape == w.shape
    assert (w_hat - w).norm() / w.norm() < 0.2, "奇数长度打包后反量化结果不对"

    # ---- 1) 量化全部线性层, 整模型对账 ----
    fp32_bytes, em_fp32 = weight_bytes(model), old_task.exact_match(model)
    w_orig = model.layers[0].ffn.w_up.weight.detach().clone()
    apply_qlora(model, r=RANK, alpha=2 * RANK, block_size=64)
    mark_only_lora_as_trainable(model)

    n_q = sum(isinstance(m, NF4Linear) for m in model.modules())
    adapter_bytes = count_parameters(model)["trainable"] * 4
    base_bytes = weight_bytes(model) - adapter_bytes
    rel_err = float((model.layers[0].ffn.w_up.base.weight - w_orig).norm() / w_orig.norm())
    em_nf4 = old_task.exact_match(model)                              # B = 0: 此刻测到的就是 "纯量化" 的影响
    print(f"量化了 {n_q} 个线性层; embedding / lm_head (共享) 与 RMSNorm 保持 fp32")
    print(f"整模型权重: fp32 {fp32_bytes / 1024:.0f} KB → NF4 基座 {base_bytes / 1024:.0f} KB "
          f"({fp32_bytes / base_bytes:.2f}×) + LoRA {adapter_bytes / 1024:.0f} KB")
    print(f"单层相对量化误差 {rel_err:.1%}; 基座原任务 (copy) 留出集 EM: fp32 {em_fp32:.3f} → NF4 {em_nf4:.3f}")

    # ---- 2) 在量化基座上训练 LoRA ----
    hist = fit(model, InstructionDataGenerator(task), SFTLoss(), STEPS, LR, log_interval=STEPS)
    em = task.exact_match(model)
    assert all(not p.requires_grad or "lora_" in n for n, p in model.named_parameters())
    print(f"QLoRA 适配到 sort: loss {hist[0]['total_loss']:.3f} → {hist[-1]['total_loss']:.3f}, 留出集 EM {em:.3f}")

    # ---- 3) 合并: 反量化 → 加 ΔW → 普通 nn.Linear ----
    with torch.no_grad():
        before = model(probe)
        merge_lora_weights(model)
        gap = float((before - model(probe)).abs().max())
    print(f"合并 (dequant → merge) 前后 logits 最大差 {gap:.1e}; 合并后模型回到 {weight_bytes(model) / 1024:.0f} KB 的高精度权重")

    assert n_q == 7 * len(model.layers), "每个 block 应量化 4 个注意力投影 + 3 个 SwiGLU 矩阵"
    assert fp32_bytes / base_bytes > 5, "整模型压缩比应 > 5× (理论上限 4 / 0.5625 = 7.1×)"
    assert em_nf4 > 0.9, "NF4 不应毁掉基座已有的能力"
    assert em > 0.2, f"QLoRA 没学到新任务: EM {em:.3f}"
    assert gap < 1e-4 and not any(isinstance(m, NF4Linear) for m in model.modules())


if __name__ == "__main__":
    main()
