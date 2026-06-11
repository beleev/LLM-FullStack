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
    DPO            —— 离线偏好对齐, 跳过 reward model 与 RL
        ↓
    Reward Model   —— 把人类相对偏好蒸馏成标量分 (RLHF 第二阶段)
        ↓
    GRPO           —— 在线 RL: 组内相对优势替代 critic (DeepSeek-R1 配方)
        ↓
    Distillation   —— 大模型能力压进小模型 (软标签 + 温度)

为什么挑这六个?
    - 覆盖四类 **目标差异**: 任务对齐 / 资源效率 / 偏好对齐 / 能力迁移
    - 各自代表了 finetune 的一个 "时代":
      LoRA(2021) → SFT(2022) → RM+RLHF(2022) → DPO(2023) → GRPO(2024-25)
      蒸馏(2015) 则贯穿始终, 在 R1 时代再次成为主角
    - 都能在小尺寸 LLaMA 上 CPU 跑通, 教学闭环

模块结构:
    methods/        微调算法实现 (sft, lora, dpo, reward_model, grpo, distill)
    data/           合成数据生成器 (instruction / preference / prompt)
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
from llm_finetune.methods.reward_model import RewardModel, bradley_terry_loss
from llm_finetune.methods.grpo import (
    GRPOTrainer,
    completion_logprobs,
    make_region_reward,
)
from llm_finetune.methods.distill import DistillLoss, soften_demo
from llm_finetune.methods.qlora import (
    QLoRALinear,
    apply_qlora,
    nf4_quantize,
    nf4_dequantize,
)
from llm_finetune.data.instruction_data import InstructionDataGenerator
from llm_finetune.data.preference_data import PreferenceDataGenerator
from llm_finetune.data.prompt_data import PromptDataGenerator
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
    # Reward Model
    "RewardModel",
    "bradley_terry_loss",
    # GRPO
    "GRPOTrainer",
    "PromptDataGenerator",
    "completion_logprobs",
    "make_region_reward",
    # Distillation
    "DistillLoss",
    "soften_demo",
    # QLoRA
    "QLoRALinear",
    "apply_qlora",
    "nf4_quantize",
    "nf4_dequantize",
    # 工具
    "count_parameters",
    "freeze_module",
    "print_trainable_parameters",
]
