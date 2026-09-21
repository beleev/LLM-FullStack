# On-policy 蒸馏 (reverse KL)

```bash
python -m llm_finetune.run_finetune.on_policy_distill.train_on_policy_distill   # ~19 s
```

## 直觉
off-policy 蒸馏只在 teacher / 数据集的前缀上教 student; student 推理时一旦走偏, 就进了从没被教过的状态。
on-policy: 让 student **自己采样**, teacher 在它走过的每个 token 位置上打分。像 RL 一样 on-policy, 又像蒸馏一样每个 token 都有稠密信号 (对整个词表解析求 KL, 不是 REINFORCE)。

## 核心公式
`y ~ π_s(·|x)` (no_grad)，`L = (1/|y|) Σ_t KL( π_s(·|x,y_<t) ‖ π_teacher(·|x,y_<t) )`
reverse KL = mode-seeking; forward KL = mode-covering。

## 运行后应该看到什么
同一个 teacher (70% 排序 / 30% 照抄)、同一个热身 200 步的 student (1 层 d=48), 再各训 300 步, 留出 prompt:

| | 样本合格率 | forward KL | reverse KL | log π(sort 答案) | log π(copy 答案) |
|---|---|---|---|---|---|
| teacher | 0.781 | 0 | 0 | −0.68 | −1.73 |
| 热身后的 student | 0.010 | 1.074 | 2.465 | −5.02 | −18.07 |
| + off-policy forward KL | 0.059 | **0.880** | 2.035 | −3.30 | **−17.26** |
| + on-policy reverse KL | **0.402** | 1.359 | **0.506** | **−1.34** | −33.32 |

- 各自在自己优化的指标上赢 (forward KL: 0.88 vs 1.36; reverse KL: 0.51 vs 2.04)。
- **mode-seeking**: on-policy student 采样出来的回复 40% 合格 (off-policy 只有 6%) —— 容量不够时它选择把多数派答案 (sort) 做好。
- **代价**: teacher 30% 概率的少数派答案 (copy) 被放弃得更彻底 (log π −33 vs −17)。forward KL 两头都留着概率, 结果两头都说不利索。

## 常见误区
- 对采样过程回传梯度 / 用 REINFORCE: 不需要。采样只决定 "在哪些状态上算 KL", 梯度来自解析的逐位置 KL。
- 从随机初始化的 student 开始 on-policy: 它采的全是垃圾, teacher 在这些分布外前缀上的输出也没意义。先 SFT / off-policy 热身。
- 把 reverse KL 写反: `KL(p‖q) = Σ p log(p/q)`, **谁在前, 期望就在谁的分布下取**。
- 认为 on-policy 全面更好: 它丢多样性。要覆盖 teacher 的全部模式就用 forward KL 或两者混合 (GKD)。

## 自测题
1. 为什么 reverse KL 不惩罚 "漏掉 teacher 的一个模式"? — 期望在 student 分布下取, student 不去的地方权重为 0。
2. 与 GRPO 相比, 每条样本提供多少监督信号? — GRPO: 整条回复 1 个标量; on-policy 蒸馏: 每个 token 位置一个 V 维分布。
3. 为什么不接通用 Trainer? — 与 GRPO 同理: 训练数据由当前 student 现场生成, 数据生成器得拿到模型本身。
