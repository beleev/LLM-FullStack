"""
微调方法子包
================

每个文件实现一种 finetune 范式, 设计上独立, 互不依赖, 便于横向对比阅读。

    sft.py   全参监督微调 (Supervised Fine-Tuning)
    lora.py  Low-Rank Adaptation (PEFT)
    dpo.py   Direct Preference Optimization (alignment)
"""

from llm_finetune.methods.sft import SFTLoss
from llm_finetune.methods.lora import (
    LoRALinear,
    apply_lora,
    mark_only_lora_as_trainable,
    merge_lora_weights,
    get_lora_state_dict,
)
from llm_finetune.methods.dpo import (
    DPOLoss,
    DPOTrainer,
    compute_sequence_logprobs,
)

__all__ = [
    "SFTLoss",
    "LoRALinear",
    "apply_lora",
    "mark_only_lora_as_trainable",
    "merge_lora_weights",
    "get_lora_state_dict",
    "DPOLoss",
    "DPOTrainer",
    "compute_sequence_logprobs",
]
