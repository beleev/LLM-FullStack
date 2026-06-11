# llm_finetune — 大模型微调教学章节

`llm_models` 教会了我们 **怎么训练一个 LLM**。本章回答下一个问题:
**怎么把它变成对话模型 / 领域模型 / 对齐模型?** —— 也就是 fine-tuning。

## 章节地图

```
llm_basic  →  llm_models  →  llm_finetune  →  llm_agent
                  ↓
              llm_infer  →  llm_train
```

`llm_finetune` 紧跟在 `llm_models` 之后, 复用其 `LLaMA` 与
`Trainer / TrainingConfig` 抽象, 把 finetune 看作 **新的 LossComputer +
新的 DataGenerator** —— 训练循环本身保持不变 (策略模式)。

## 七个最典型的 finetune 范式

| 方法 | 论文年代 | 解决什么问题 | 代码入口 |
|------|---------|------------|---------|
| **SFT**  | InstructGPT 2022 | 把"语言模型"变成"指令跟随者" | [methods/sft.py](methods/sft.py) |
| **LoRA** | Hu et al. 2021    | 全参微调显存爆炸; PEFT 用 0.5% 参数即可 | [methods/lora.py](methods/lora.py) |
| **QLoRA** | Dettmers 2023    | 冻结基座也占显存; NF4 4-bit 存放 + LoRA 旁路 | [methods/qlora.py](methods/qlora.py) |
| **DPO**  | Rafailov 2023     | 跳过 reward model + PPO, 直接做偏好对齐 | [methods/dpo.py](methods/dpo.py) |
| **RM**   | InstructGPT 2022  | 把人类相对偏好蒸馏成可优化的标量分 | [methods/reward_model.py](methods/reward_model.py) |
| **GRPO** | DeepSeek 2024-25  | 组内相对优势替代 critic, RLVR 可验证奖励 | [methods/grpo.py](methods/grpo.py) |
| **蒸馏** | Hinton 2015 / R1 2025 | 大模型能力压进小模型 (软标签 + 温度) | [methods/distill.py](methods/distill.py) |

## 一行差异对照表

| 维度 | SFT | LoRA | DPO | RM | GRPO | 蒸馏 |
|------|-----|------|-----|----|------|------|
| 训练数据 | (prompt, response) | (prompt, response) | (prompt, chosen, rejected) | (prompt, chosen, rejected) | 只有 prompt (在线采样) | 任意输入 + teacher logits |
| 可训参数 | 100% | <1% | 100% | backbone + value head | 100% | student 100% |
| 模型数量 | 1 | 1 | 2 (policy + ref) | 1 | 2 (policy + ref) | 2 (teacher 冻结) |
| Loss     | 交叉熵 (prompt mask) | 同 SFT, 仅 LoRA 有梯度 | -log σ(β·Δ logratio) | -log σ(r_w - r_l) | -Â·logπ + β·KL | α·CE + (1-α)·T²·KL |
| 教学重点 | label = -100 | 低秩矩阵 + 合并 | 双前向 + KL 锚定 | 序关系→标量分 | 组内归一化替代 critic | 温度放大暗知识 |

## 目录结构

```
llm_finetune/
├── methods/                微调算法核心
│   ├── sft.py              SFTLoss
│   ├── lora.py             LoRALinear / apply_lora / merge / get_state_dict
│   ├── qlora.py            NF4 量化 / QLoRALinear / apply_qlora (真 4bit 打包)
│   ├── dpo.py              DPOLoss / DPOTrainer / compute_sequence_logprobs
│   ├── reward_model.py     RewardModel / bradley_terry_loss
│   ├── grpo.py             GRPOTrainer / completion_logprobs / make_region_reward
│   └── distill.py          DistillLoss / soften_demo
├── data/                   合成数据生成器
│   ├── instruction_data.py InstructionDataGenerator     (SFT/LoRA 共用)
│   ├── preference_data.py  PreferenceDataGenerator      (DPO/RM 偏好对)
│   └── prompt_data.py      PromptDataGenerator          (GRPO 在线 RL)
├── utils/
│   └── param_utils.py      count / freeze / print_trainable
└── run_finetune/           可运行的端到端示例
    ├── sft/      train_sft.py
    ├── lora/     train_lora.py
    ├── qlora/    train_qlora.py
    ├── dpo/      train_dpo.py
    ├── rm/       train_rm.py
    ├── grpo/     train_grpo.py
    └── distill/  train_distill.py
```

## 快速运行

```bash
# 全参 SFT
python -m llm_finetune.run_finetune.sft.train_sft

# LoRA (PEFT)
python -m llm_finetune.run_finetune.lora.train_lora

# QLoRA (NF4 4-bit 基座 + LoRA)
python -m llm_finetune.run_finetune.qlora.train_qlora

# DPO 对齐
python -m llm_finetune.run_finetune.dpo.train_dpo

# Reward Model (RLHF 第二阶段)
python -m llm_finetune.run_finetune.rm.train_rm

# GRPO (R1 式在线 RL, 可验证奖励)
python -m llm_finetune.run_finetune.grpo.train_grpo

# 知识蒸馏 (4 层 teacher → 2 层 student)
python -m llm_finetune.run_finetune.distill.train_distill
```

每个脚本都基于 LLaMA Mini 尺寸, CPU 一分钟内跑完, 自带断言验证。

## 实测教学效果 (CPU, 默认配置)

| 指标 | SFT (全参, 50 步) | LoRA (1.6% 参数, 50 步) | DPO (β=0.1, 80 步) |
|------|-----|-----|-----|
| 起始 loss | 249.97 | 249.97 | 0.6931 (= log 2) |
| 终止 loss | 214.78 | **27.13** | **0.0018** |
| 可训参数 | 1,731,840 | 28,672 | 1,731,840 |
| 关键观察 | 显著下降 | 极快收敛, adapter 仅 112 KB | reward_margin 0 → +6.30, accuracy 100% |

LoRA 在同一份数据上比 SFT 收敛快得多 —— 这并不矛盾: LoRA 用了 10x 的学习率,
更小的可训参数空间让 lr 可以更大而不发散, 这正是 PEFT 的工程优势之一。

| 指标 | RM (60 步) | GRPO (80 步) | 蒸馏 (200 步) |
|------|-----|-----|-----|
| 核心指标 | 偏好准确率 0.25 → **1.00** | 平均奖励 0.488 → **0.953** | student CE 129 → **0.09** |
| 伴随观察 | margin +6.1 | KL 缓慢上升 (β 锚定) | KD 253 → 0.33 |
| 关键教学点 | 序关系学成标量分 | 采样多样性是 RL 的命门 (见脚本内 init 注释) | T 越大暗知识越显 |

## 如何拓展

- **想换别的量化格式?** 把 `methods/qlora.py` 的 NF4 码本换成 GPTQ/AWQ 的网格即可, 打包/反量化骨架可复用。
- **想加 ORPO / KTO?** 复制 `methods/dpo.py`, 改 loss 公式即可, `DPOTrainer`
  的双前向骨架可直接复用。
- **想接真实数据?** 把 `data/` 下的合成生成器换成读 jsonl 的 DataLoader, 别的不用动。
