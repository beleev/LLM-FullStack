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

## 三个最典型的 finetune 范式

| 方法 | 论文年代 | 解决什么问题 | 代码入口 |
|------|---------|------------|---------|
| **SFT**  | InstructGPT 2022 | 把"语言模型"变成"指令跟随者" | [methods/sft.py](methods/sft.py) |
| **LoRA** | Hu et al. 2021    | 全参微调显存爆炸; PEFT 用 0.5% 参数即可 | [methods/lora.py](methods/lora.py) |
| **DPO**  | Rafailov 2023     | 跳过 reward model + PPO, 直接做偏好对齐 | [methods/dpo.py](methods/dpo.py) |

## 一行差异对照表

| 维度 | SFT | LoRA | DPO |
|------|-----|------|-----|
| 训练数据 | (prompt, response) | (prompt, response) | (prompt, chosen, rejected) |
| 可训参数 | 100% | <1% | 100% (或基于 LoRA 之上) |
| 模型数量 | 1 | 1 | 2 (policy + ref) |
| Loss     | 交叉熵 (prompt mask) | 同 SFT, 但只有 LoRA 参数有梯度 | -log σ(β·Δ logratio) |
| 教学重点 | label = -100 | 低秩矩阵 + 注入位点 + 合并 | 双前向 + 对参考的 KL 约束 |

## 目录结构

```
llm_finetune/
├── methods/                微调算法核心
│   ├── sft.py              SFTLoss
│   ├── lora.py             LoRALinear / apply_lora / merge / get_state_dict
│   └── dpo.py              DPOLoss / DPOTrainer / compute_sequence_logprobs
├── data/                   合成数据生成器
│   ├── instruction_data.py InstructionDataGenerator     (SFT/LoRA 共用)
│   └── preference_data.py  PreferenceDataGenerator      (DPO 偏好对)
├── utils/
│   └── param_utils.py      count / freeze / print_trainable
└── run_finetune/           可运行的端到端示例
    ├── sft/   train_sft.py
    ├── lora/  train_lora.py
    └── dpo/   train_dpo.py
```

## 快速运行

```bash
# 全参 SFT
python -m llm_finetune.run_finetune.sft.train_sft

# LoRA (PEFT)
python -m llm_finetune.run_finetune.lora.train_lora

# DPO 对齐
python -m llm_finetune.run_finetune.dpo.train_dpo
```

每个脚本都使用同一个 LLaMA Mini 配置 (~1.7M 参数), CPU 30 秒内跑完, 自带断言验证。

## 实测教学效果 (CPU, 默认配置)

| 指标 | SFT (全参, 50 步) | LoRA (1.6% 参数, 50 步) | DPO (β=0.1, 80 步) |
|------|-----|-----|-----|
| 起始 loss | 249.97 | 249.97 | 0.6931 (= log 2) |
| 终止 loss | 214.78 | **27.13** | **0.0018** |
| 可训参数 | 1,731,840 | 28,672 | 1,731,840 |
| 关键观察 | 显著下降 | 极快收敛, adapter 仅 112 KB | reward_margin 0 → +6.30, accuracy 100% |

LoRA 在同一份数据上比 SFT 收敛快得多 —— 这并不矛盾: LoRA 用了 10x 的学习率,
更小的可训参数空间让 lr 可以更大而不发散, 这正是 PEFT 的工程优势之一。

## 如何拓展

- **想加 QLoRA?** 在 `apply_lora` 之前先把基座 quantize 到 int4。
- **想加 ORPO / KTO?** 复制 `methods/dpo.py`, 改 loss 公式即可, `DPOTrainer`
  的双前向骨架可直接复用。
- **想接真实数据?** 把 `data/` 下的合成生成器换成读 jsonl 的 DataLoader, 别的不用动。
