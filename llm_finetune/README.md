# llm_finetune — 微调与对齐

`llm_models` 教会了怎么训练一个 LM; 本章回答: 怎么把它变成 **听指令 / 合偏好 / 会解题 / 更小** 的模型。

```
llm_basic → llm_models → llm_finetune → llm_agent
                ↓
            llm_infer → llm_train
```

```bash
python -m llm_finetune.run_all                                   # 全部 10 个脚本, CPU 约 2.5 分钟
python -m llm_finetune.run_finetune.grpo.train_grpo              # 单个, 每个 < 35 s
pytest -m slow -k llm_finetune                                   # 同样的脚本, 走冒烟测试
```

## 一个任务贯穿全章

所有方法共用 [`data/tasks.py`](data/tasks.py) 的 `SeqTask`: prompt 是 6 个随机 token, 正确回复是它的 copy / reverse / sort + EOS。
答案**由 prompt 决定**, prompt 空间 13⁶ ≈ 480 万, 训练集与留出集按 token 和 mod 5 严格不相交 —— 所以每个脚本末尾断言的都是**留出集**指标, 不是 "背下了一个 batch"。

| 用法 | 怎么来 |
|---|---|
| SFT 数据 | (prompt, 正确回复) → 留出集 exact-match |
| 偏好对 | chosen = 正确回复, rejected = 改错 / 漏掉一个 token → 留出集偏好准确率 |
| RLVR 奖励 | `task.verify`: 全对得 1 分。策略不看 prompt 就拿不到分 |

## 路线与入口

| 方法 | 它解决上一步的什么问题 | 代码 | 讲义 (含实测数字 / 误区 / 自测题) |
|---|---|---|---|
| **SFT** | LM 只会续写 → 只在回复上算 loss | [sft.py](methods/sft.py) | [readme](run_finetune/sft/readme.md) |
| **LoRA** | 全参要存梯度 + Adam 状态 + 每任务一份权重 | [lora.py](methods/lora.py) | [readme](run_finetune/lora/readme.md) |
| **DoRA** | LoRA 把 "长度" 和 "方向" 绑在一起变 | [dora.py](methods/dora.py) | [readme](run_finetune/dora/readme.md) |
| **QLoRA** | 冻结的基座本身还占显存 → NF4 | [qlora.py](methods/qlora.py) | [readme](run_finetune/qlora/readme.md) |
| **Reward Model** | 相对偏好 → 可调用的标量分 | [reward_model.py](methods/reward_model.py) | [readme](run_finetune/rm/readme.md) |
| **DPO** | 跳过 RM 和 RL | [dpo.py](methods/dpo.py) | [readme](run_finetune/dpo/readme.md) |
| **SimPO / ORPO** | 连 reference model 也不要 | [simpo.py](methods/simpo.py) · [orpo.py](methods/orpo.py) | [readme](run_finetune/simpo_orpo/readme.md) |
| **GRPO** + DAPO / Dr.GRPO / GSPO | PPO 的 critic → 组内均值; 三个变体各修一个偏差 | [grpo.py](methods/grpo.py) | [readme](run_finetune/grpo/readme.md) |
| **蒸馏** (off-policy, forward KL) | 大模型 → 小模型 | [distill.py](methods/distill.py) | [readme](run_finetune/distill/readme.md) |
| **On-policy 蒸馏** (reverse KL) | student 没在自己会走到的前缀上被训练过 | [on_policy_distill.py](methods/on_policy_distill.py) | [readme](run_finetune/on_policy_distill/readme.md) |

## 各方法需要什么

| | 数据 | reference model | reward model / verifier | 在线采样 | 常驻权重 (policy = 1) | 每步 LM 前向 | 用通用 Trainer |
|---|---|---|---|---|---|---|---|
| SFT | (x, y) | – | – | – | 1 (+梯度 +2×Adam) | 1 | 是 |
| LoRA / DoRA | (x, y) | – | – | – | 1 冻结 + 适配器 (梯度 / Adam 只为适配器) | 1 | 是 |
| QLoRA | (x, y) | – | – | – | ≈0.15 (NF4) + 适配器 | 1 | 是 |
| Reward Model | (x, y_w, y_l) | – | 它自己就是 | – | 1 | 1 (2B 条) | 是, 经 `PairwiseForward` |
| DPO | (x, y_w, y_l) | **需要** | – | – | **2** | **2** (2B 条) | 是, 经 `PairwiseForward` |
| SimPO | (x, y_w, y_l) | – | – | – | 1 | 1 (2B 条) | 同上 |
| ORPO | (x, y_w, y_l), 无需先 SFT | – | – | – | 1 | 1 (2B 条) | 同上 |
| GRPO 系 | 只有 x | 仅 β>0 时 | **verifier 或 RM** | **需要** (G 条 / prompt) | 1 (β>0 时 2) | 采样 + 1 + μ | **否** |
| 蒸馏 | (x, y) + teacher | – (teacher 冻结) | – | – | 1 + teacher | 2 | 是, 经 `TeacherStudent` |
| On-policy 蒸馏 | 只有 x + teacher | – | – | **需要** | 1 + teacher | 采样 + 2 | **否** |

## 实测 (CPU, 默认种子; 全部为留出集指标)

| 脚本 | 结论 |
|---|---|
| sft | reverse 任务 EM 0.000 → **1.000** (600 步); prompt mask 多盖一位的同配置模型 EM = **0.000** |
| lora | copy→sort 适配 300 步: 全参 EM **0.809** / loss 0.119; LoRA(r=8) lr=1e-2 EM 0.352 / loss 0.384; lr=3e-3 EM 0.281。可训参数 99,648 → 19,456 |
| dora | 同 r=8、3 seeds 平均: LoRA loss 0.369 / EM 0.382; DoRA loss **0.296** / EM **0.522** (+1,280 参数) |
| qlora | 量化全部 14 个线性层: 整模型 389 KB → 59 KB (**6.57×**); 基座 EM 1.000 → 1.000; 适配 sort EM 0.402 |
| rm | 留出集偏好准确率 0.549 → **0.930**; 右 pad 5 位的分数偏移: 取最后真 token 5e-06, 取 `[:, -1]` 4.54 |
| dpo | 偏好准确率 0.965 → 0.996, 但 log π(chosen) −4.03 → −4.18、EM 0.332 → **0.137** (likelihood displacement) |
| simpo_orpo | DPO 2 次前向 / 778 KB / 5.4 s; SimPO、ORPO 1 次 / 389 KB / 4.0 s。EM: DPO 0.121, SimPO 0.023, ORPO **0.543**; ORPO 从零训练 EM 0.973 |
| grpo | 采样 pass@1 0.186 → 0.287 (GRPO) / 0.326 (DAPO) / 0.324 (Dr.GRPO) / 0.320 (GSPO); 贪心 EM 0.562 → 0.53 (基本不变)。ρ: epoch1 ≡ 1, epoch2 最大偏离 0.16~1.04 |
| distill | 128 条固定数据: forward KL 硬标签 2.292 vs 软标签 **1.860**; 无限数据时差距消失 (0.906 vs 0.959) |
| on_policy_distill | 样本合格率: off-policy 0.059 vs on-policy **0.402**; 但 on-policy 把 teacher 的少数派答案压到 log π = −33 (off-policy −17) |

**更正旧版结论**: 旧 README 写 "LoRA 在同一份数据上比 SFT 收敛快得多, 这是 PEFT 的工程优势"。那组数字 (loss 249.97 → 27 vs → 215) 来自病态初始化 (初始 CE ≈ 250 而非 ln V)、只背 2 条样本、LoRA 用了 10× lr。初始化修好后用留出集重测, **同样步数下全参更快更好**; LoRA 的优势是显存与存储, 不是收敛速度。

## 关于 "复用 Trainer" 的实话

`llm_models.training.Trainer` 的约定是 "取 batch → `model(**batch)` → `loss.compute(output, labels)` → 更新一次"。

- **SFT / LoRA / DoRA / QLoRA**: 只换数据 (和模型里的层), loss 就是 `StandardLMLoss` (`SFTLoss` 是它的别名, 不再复制一份)。
- **DPO / SimPO / ORPO / RM / 蒸馏**: 一步要跑多次前向。扩展点是**包一层 `nn.Module`**: [`PairwiseForward`](methods/dpo.py) (chosen + rejected [+ 冻结 ref]) 和 [`TeacherStudent`](methods/distill.py)。它们的 loss 都是普通 `LossComputer`, Trainer 一行不改, 旧版 `DPOTrainer` 里复制的 clip / step / scheduler 已删除。
- **GRPO / on-policy 蒸馏**: 不用 Trainer。数据由当前策略现场采样 (数据生成器得拿到模型), GRPO 还要在同一批 rollout 上更新 μ 次、并记住采样时刻的 log-prob。这与上面的约定有本质冲突, 硬塞进去只会更难读。

## 目录

```
llm_finetune/
├── data/           tasks.py (SeqTask, make_labels, verify) · instruction / preference / prompt 三种取数方式
├── methods/        sft · lora · dora · qlora · reward_model · dpo · simpo · orpo · grpo · distill · on_policy_distill
├── utils/          param_utils.py (count / freeze / print)
├── run_finetune/   common.py + <name>/train_<name>.py + <name>/readme.md
└── run_all.py
```

## 未实现
RFT / best-of-N → SFT (最简单的 RL 基线); QLoRA 的 double quantization 与 paged optimizer; DAPO 动态采样的 "补采到凑满 batch" 与 overlong 惩罚; KTO / IPO。
