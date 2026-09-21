# 知识蒸馏 (off-policy, forward KL)

```bash
python -m llm_finetune.run_finetune.distill.train_distill   # ~26 s
```

## 直觉
硬标签只说 "答案是这个 token"; teacher 的分布还说了 "其它答案各有多合理"。当同一个输入有多个合理答案时, 一条软标签 ≈ 同一前缀上许多条采样硬标签。

## 核心公式
`L = α·CE(student, y) + (1−α)·T²·KL(p_T^teacher ‖ p_T^student)`，`p_T = softmax(z/T)`。KD 项与 CE 项用**同一个 mask** (label=−100 的位置都不算)。

## 运行后应该看到什么
teacher (2 层 d=64, 99,648 参数) 学的任务有**两种正确答案**: 70% 排序 / 30% 照抄。student 1 层 d=48 (26,256 参数), 各 300 步, 留出 prompt:

| | 样本合格率 | forward KL | reverse KL | log π(sort 答案) | log π(copy 答案) |
|---|---|---|---|---|---|
| teacher | 0.781 | 0 | 0 | −0.68 | −1.73 |
| 128 条固定数据 / 硬标签 | 0.053 | 2.292 | 2.707 | −11.06 | −32.24 |
| 128 条固定数据 / 软标签 T=2 | 0.059 | **1.860** | 2.429 | −8.15 | −29.11 |
| 无限新数据 / 硬标签 | 0.039 | 0.906 | 2.081 | −3.71 | −17.10 |
| 无限新数据 / 软标签 T=2 | 0.053 | 0.959 | 1.758 | −3.51 | −18.77 |

**如实解读**: 软标签的优势出现在**数据有限**时 (forward KL 1.86 vs 2.29, 有断言)。数据无限时硬标签自己就能把分布采出来, 两者都卡在 student 的容量上, 差距消失 (0.96 vs 0.91, 软标签甚至略差) —— 表里照列, 不设断言。
两种 student 的样本合格率都很低 (≈5%): 小 student 在 forward KL 下把概率摊在两种答案之间, 采样时半路串台。这正是 `on_policy_distill` 要解决的。

## 常见误区
- KD 项不做 mask: prompt / pad 位置也在被蒸馏, 而 CE 项没有 —— 两项优化的不是同一批位置。脚本里把 teacher 在 −100 位置的 logits 打乱, 断言 KD loss 不变。
- 忘了 ×T²: 调 T 会顺带改变 KD 与 CE 的相对权重。
- "蒸馏一定比硬标签好": 见上表后两行。

## 自测题
1. T→∞ 时 KD 项在学什么? — 两边都趋于均匀分布, 梯度 ∝ logits 之差: 退化成对 logits 的 MSE 匹配。
2. forward KL 为什么是 mode-covering? — 期望在 teacher 分布下取: teacher 有质量而 student 没有的地方, log(p_t/p_s) → ∞。
3. 这里的训练数据来自谁? 为什么叫 off-policy? — 来自数据集 / teacher, 不来自 student; student 从没在自己会走到的前缀上被训练。
