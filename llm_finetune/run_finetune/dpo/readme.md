# DPO — 直接偏好优化

```bash
python -m llm_finetune.run_finetune.dpo.train_dpo      # ~9 s
```

## 直觉
KL 约束下的最优策略满足 `r(x,y) = β·log π(y|x)/π_ref(y|x) + const`。把它代回 Bradley-Terry, RM 和 RL 两步就合并成了一个对偏好对的二分类 loss。

## 核心公式
`L = − log σ( β·[ (log π_θ(y_w) − log π_ref(y_w)) − (log π_θ(y_l) − log π_ref(y_l)) ] )`
`log π(y|x) = Σ_{t∈回复} log p(y_t|·)`，**包含第一个回复 token**。

## 运行后应该看到什么
起点 = SFT 100 步的 "半会" 模型; β=0.5, lr=3e-4, 200 步, 全部指标在**留出**偏好对上:

| 留出集 | 偏好准确率 | log π(chosen) | log π(rejected) | 贪心 exact-match |
|---|---|---|---|---|
| SFT 后 | 0.965 | −4.03 | −10.55 | 0.332 |
| DPO 后 | **0.996** | **−4.18 ↓** | −15.36 | **0.137 ↓** |

第 1 步 loss 恰为 0.6931 = ln 2 (policy = ref)。
**如实解读**: 偏好准确率上去了, 差值从 6.5 拉到 11.2 —— 但 chosen 自己的 log-prob 也掉了, 生成质量 (EM) 反而下降。DPO 的 loss 只看差值, "两边一起降、rejected 降得更多" 同样让 loss 变小 (likelihood displacement)。脚本对这一现象设了断言。对照 `simpo_orpo`: ORPO 的 NLL 项正是为此而设。

训练走通用 `Trainer`: `PairwiseForward.with_frozen_copy(policy)` 把 "policy 前向 + ref 前向" 包成一个 Module; 没有 DPOTrainer, 没有复制的训练循环。

## 常见误区
- **prompt mask 多盖一位**, 并辩称 "那一步转移对 chosen / rejected 相同"。上下文相同, 但**目标 token y₁ 不同**, log p(y₁|x) 恰恰是区分两者的一项。
- 以为 reward margin 变大 = 模型变好。要同时盯 `logp_chosen`。
- ref 被注册成子模块: 会进 optimizer、被 `.train()` 切模式。这里故意放在 tuple 里。
- β 越大越 "用力"? 相反: β 大 → σ 更快饱和 → 梯度更早消失 → 更贴近 ref。

## 自测题
1. 为什么第 1 步 loss 一定是 ln 2? — policy = ref ⇒ 两个 log-ratio 都是 0 ⇒ −log σ(0)。
2. DPO 每步比 SFT 多多少计算? — 序列数 ×2 (chosen + rejected), 再加一次不带梯度的 ref 前向; 常驻权重 ×2。
3. loss 在降而 chosen 的概率也在降, 可能吗? — 可能, 只要 rejected 降得更快。上表就是。
