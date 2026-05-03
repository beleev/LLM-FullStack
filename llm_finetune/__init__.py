"""
llm_finetune — 大模型微调教学章节
==================================

本章节是 ``llm_models`` 的后继: 已经能 "训练一个 LLM" 后, 真正落地到
下游应用还需要 "把通用语言模型变成对话模型 / 领域模型 / 对齐模型"。
这一步统称 **fine-tuning (微调)**, 是连接预训练与产品体验的关键环节。

教学路线 (从基础到现代):

    Full SFT       —— 全参监督微调, 概念最朴素 (loss-mask 在 prompt 上)
        ↓
    LoRA           —— 参数高效微调 (PEFT) 的事实标准, 在权重旁加低秩适配器
        ↓
    DPO            —— 对齐 (alignment) 的现代方法, 不需要 reward model

为什么挑这三个?
    - 覆盖三类 **目标差异**: 任务对齐 / 资源效率 / 偏好对齐
    - 各自代表了 finetune 的一个 "时代": SFT(2022) → LoRA(2021) → DPO(2023)
    - 都能在小尺寸 LLaMA 上 CPU 跑通, 教学闭环

模块结构:
    methods/        微调算法实现 (sft, lora, dpo)
    data/           合成数据生成器 (instruction / preference)
    utils/          参数管理工具 (冻结/统计/保存)
    run_finetune/   可运行的端到端示例脚本

设计原则:
    - 复用 ``llm_models.training`` 的 Trainer/Config 抽象, 把 fine-tune 视为
      "新的 LossComputer + 新的 DataGenerator", 保持开闭原则
    - 所有示例统一使用 LLaMA mini 作为 base model (业界 finetune 的事实标准)
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
from llm_finetune.data.instruction_data import InstructionDataGenerator
from llm_finetune.data.preference_data import PreferenceDataGenerator
from llm_finetune.utils.param_utils import (
    count_parameters,
    freeze_module,
    print_trainable_parameters,
)

__all__ = [
    # SFT
    "SFTLoss",
    "InstructionDataGenerator",
    # LoRA
    "LoRALinear",
    "apply_lora",
    "mark_only_lora_as_trainable",
    "merge_lora_weights",
    "get_lora_state_dict",
    # DPO
    "DPOLoss",
    "DPOTrainer",
    "PreferenceDataGenerator",
    "compute_sequence_logprobs",
    # 工具
    "count_parameters",
    "freeze_module",
    "print_trainable_parameters",
]
