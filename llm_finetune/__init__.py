"""
llm_finetune — 把预训练 LM 变成 "听指令 / 合偏好 / 会解题 / 更小" 的模型

路线 (每一步解决上一步留下的问题):
    SFT ─→ LoRA ─→ DoRA ─→ QLoRA          学会任务 → 省梯度与优化器显存 → 拆开长度与方向 → 连基座也压到 4 bit
    Reward Model ─→ DPO ─→ SimPO / ORPO   偏好变标量分 → 跳过 RM 和 RL → 连 reference 也不要
    GRPO ─→ DAPO / Dr.GRPO / GSPO         在线 RL + 可验证奖励, 组内均值替代 critic; 三个变体各修一个偏差
    蒸馏 (forward KL) ─→ on-policy 蒸馏 (reverse KL)

所有方法共用一个可学习、依赖 prompt 的合成任务 (data/tasks.py), 验收指标是**留出集**上的 exact-match / 偏好准确率。

与 llm_models.training.Trainer 的关系 (实话):
    - SFT / LoRA / DoRA / QLoRA: 纯粹是 "新数据 + 同一个 loss", Trainer 原样复用。
    - DPO / SimPO / ORPO / RM / 蒸馏: 一步里要跑多次前向 (chosen+rejected / ref / teacher)。
      把这些前向包进一个 nn.Module (dpo.PairwiseForward, distill.TeacherStudent) 后, 同样复用 Trainer, 不复制训练循环。
    - GRPO / on-policy 蒸馏: **不**用 Trainer。数据由当前策略现场采样, 且 GRPO 对同一批数据更新多次,
      "取 batch → 前向 → 更新一次" 的约定不成立, 硬塞只会更难读。
"""

from llm_finetune.data import (
    SeqTask, make_labels, completion_mask,
    InstructionDataGenerator, PreferenceDataGenerator, PromptDataGenerator,
)
from llm_finetune.methods.sft import SFTLoss
from llm_finetune.methods.lora import (
    LoRALinear, apply_lora, mark_only_lora_as_trainable, merge_lora_weights, get_lora_state_dict,
    ATTENTION_LINEARS, ALL_LINEARS,
)
from llm_finetune.methods.dora import DoRALinear
from llm_finetune.methods.qlora import NF4Linear, apply_qlora, nf4_quantize, nf4_dequantize, weight_bytes
from llm_finetune.methods.dpo import DPOLoss, PairwiseForward, compute_sequence_logprobs
from llm_finetune.methods.simpo import SimPOLoss
from llm_finetune.methods.orpo import ORPOLoss
from llm_finetune.methods.reward_model import RewardModel, BradleyTerryLoss
from llm_finetune.methods.grpo import (
    GRPOConfig, GRPOTrainer, VARIANTS, make_config, group_advantages, aggregate, completion_logprobs,
)
from llm_finetune.methods.distill import DistillLoss, TeacherStudent
from llm_finetune.methods.on_policy_distill import on_policy_distill_step, token_kl
from llm_finetune.utils.param_utils import count_parameters, freeze_module, print_trainable_parameters

__all__ = [
    # 数据
    "SeqTask", "make_labels", "completion_mask",
    "InstructionDataGenerator", "PreferenceDataGenerator", "PromptDataGenerator",
    # 监督 / 参数高效
    "SFTLoss", "LoRALinear", "DoRALinear", "NF4Linear",
    "apply_lora", "apply_qlora", "mark_only_lora_as_trainable", "merge_lora_weights", "get_lora_state_dict",
    "ATTENTION_LINEARS", "ALL_LINEARS", "nf4_quantize", "nf4_dequantize", "weight_bytes",
    # 离线偏好
    "PairwiseForward", "DPOLoss", "SimPOLoss", "ORPOLoss", "compute_sequence_logprobs",
    "RewardModel", "BradleyTerryLoss",
    # 在线 RL
    "GRPOConfig", "GRPOTrainer", "VARIANTS", "make_config", "group_advantages", "aggregate", "completion_logprobs",
    # 蒸馏
    "DistillLoss", "TeacherStudent", "on_policy_distill_step", "token_kl",
    # 工具
    "count_parameters", "freeze_module", "print_trainable_parameters",
]
