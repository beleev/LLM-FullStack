#!/usr/bin/env python
"""
DoRA vs LoRA: 同一个基座、同样的 r / lr / 步数 / 随机种子, 只换适配器类型。

    python -m llm_finetune.run_finetune.dora.train_dora
"""

import copy

import torch

from llm_finetune import (
    ALL_LINEARS, DoRALinear, InstructionDataGenerator, LoRALinear, SeqTask, SFTLoss, apply_lora,
    count_parameters, mark_only_lora_as_trainable, merge_lora_weights,
)
from llm_finetune.run_finetune.common import fit, pretrained_base

STEPS, RANK, LR, SEEDS = 300, 8, 1e-2, (1, 2, 3)


def main() -> None:
    torch.manual_seed(0)
    task = SeqTask("sort")
    base = pretrained_base()
    probe = task.sample_prompts(8, "test")

    results = {}
    for cls in (LoRALinear, DoRALinear):
        runs = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            model = copy.deepcopy(base)
            apply_lora(model, r=RANK, alpha=2 * RANK, target_modules=ALL_LINEARS, layer_cls=cls)
            mark_only_lora_as_trainable(model)
            # m 初始化为 ‖W‖_row 且 B = 0 ⇒ W' = W: 与 LoRA 一样是无害启动 (只差浮点舍入)
            assert torch.allclose(model(probe), base(probe), atol=1e-5)
            hist = fit(model, InstructionDataGenerator(task), SFTLoss(), STEPS, LR, log_interval=STEPS)
            runs.append((hist[-1]["total_loss"], task.exact_match(model)))
        results[cls.__name__] = (count_parameters(model)["trainable"], runs)

    print(f"\n{'':<12}{'可训参数':>10}{'末步 loss (3 seeds)':>26}{'留出集 EM (3 seeds)':>26}")
    mean = {}
    for name, (n, runs) in results.items():
        mean[name] = [sum(x) / len(x) for x in zip(*runs)]
        print(f"{name:<12}{n:>10,}{'  '.join(f'{l:.3f}' for l, _ in runs):>26}{'  '.join(f'{e:.3f}' for _, e in runs):>26}")
    print(f"平均: LoRA loss {mean['LoRALinear'][0]:.3f} EM {mean['LoRALinear'][1]:.3f} | "
          f"DoRA loss {mean['DoRALinear'][0]:.3f} EM {mean['DoRALinear'][1]:.3f}")

    # 训练后 m 相对初值 ‖W₀‖ 的变化: DoRA 多出来的那个自由度到底动了多少
    layer = model.layers[0].attn.w_q
    drift = (layer.lora_magnitude.detach() / layer.base.weight.norm(dim=1) - 1).abs().mean()
    print(f"layers.0.attn.w_q: 长度向量 m 平均变化 {float(drift):.1%}")

    with torch.no_grad():
        before = model(probe)
        merge_lora_weights(model)
        gap = float((before - model(probe)).abs().max())

    extra = results["DoRALinear"][0] - results["LoRALinear"][0]
    assert extra == sum(m.out_features for m in base.modules() if isinstance(m, torch.nn.Linear)
                        and m is not base.lm_head), "DoRA 只比 LoRA 每层多 d_out 个参数"
    assert gap < 1e-4, "DoRA 合并前后输出应数值相等"
    assert mean["DoRALinear"][0] < mean["LoRALinear"][0], "同 r 下 DoRA 的平均 loss 应低于 LoRA"


if __name__ == "__main__":
    main()
