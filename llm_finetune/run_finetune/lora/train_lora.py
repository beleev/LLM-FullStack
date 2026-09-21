#!/usr/bin/env python
"""
LoRA vs 全参微调: 同一个基座 (会 copy) 适配到新任务 (sort), 同样的步数, 留出集 exact-match 说话。

    python -m llm_finetune.run_finetune.lora.train_lora
"""

import copy

import torch

from llm_finetune import (
    ALL_LINEARS, InstructionDataGenerator, SeqTask, SFTLoss, apply_lora, count_parameters,
    get_lora_state_dict, mark_only_lora_as_trainable, merge_lora_weights,
)
from llm_finetune.run_finetune.common import fit, make_model, pretrained_base

STEPS, RANK = 300, 8


def main() -> None:
    torch.manual_seed(0)
    task = SeqTask("sort")
    base = pretrained_base()
    probe = task.sample_prompts(8, "test")
    print(f"基座在新任务 sort 上的留出集 EM: {task.exact_match(base):.3f}")

    rows = {}
    for name, lr in [("全参 lr=3e-3", 3e-3), ("LoRA lr=3e-3", 3e-3), ("LoRA lr=1e-2", 1e-2)]:
        torch.manual_seed(1)
        model = copy.deepcopy(base)
        if name.startswith("LoRA"):
            apply_lora(model, r=RANK, alpha=2 * RANK, target_modules=ALL_LINEARS)
            mark_only_lora_as_trainable(model)
            # B = 0 ⇒ 注入后、训练前的输出必须与基座逐位相同
            assert torch.equal(model(probe), base(probe)), "LoRA 注入改变了初始输出"
        hist = fit(model, InstructionDataGenerator(task), SFTLoss(), STEPS, lr, log_interval=STEPS)
        rows[name] = (count_parameters(model)["trainable"], hist[-1]["total_loss"], task.exact_match(model))

    total = count_parameters(base)["total"]
    print(f"\n{'':<14}{'可训参数':>10}{'Adam 状态':>12}{'末步 loss':>11}{'留出集 EM':>11}")
    for name, (n, loss, em) in rows.items():
        print(f"{name:<14}{n:>10,}{n * 8 / 1024:>10.0f}KB{loss:>11.3f}{em:>11.3f}")
    print(f"(基座共 {total:,} 参数; 2r/d = {2 * RANK}/64 = 25%: d_model 太小, 比例不好看。d=4096 时同样的 r 只占 0.4%)")

    # ---- 合并: 换回普通 nn.Linear, 输出必须数值相等 ----
    adapter = get_lora_state_dict(model)
    with torch.no_grad():
        before = model(probe)
        merge_lora_weights(model)
        after = model(probe)
    gap = float((before - after).abs().max())
    print(f"adapter {sum(v.numel() for v in adapter.values()) * 4 / 1024:.0f} KB; 合并前后 logits 最大差 {gap:.1e}")

    full, lora_same_lr, lora = rows["全参 lr=3e-3"], rows["LoRA lr=3e-3"], rows["LoRA lr=1e-2"]
    assert gap < 1e-4, "合并后输出必须与合并前数值相等"
    assert not any("lora_" in n for n, _ in model.named_parameters()), "合并后不应再有 LoRA 参数"
    assert lora[2] > 0.2, f"LoRA 没学到新任务: EM {lora[2]:.3f}"
    assert lora[1] < lora_same_lr[1], "LoRA 需要比全参更大的 lr (B 从 0 起步, ΔW 的有效步长小)"
    assert full[1] < lora[1], "同样步数下全参应收敛得更快 —— LoRA 的卖点是显存 / 存储, 不是速度"


if __name__ == "__main__":
    main()
