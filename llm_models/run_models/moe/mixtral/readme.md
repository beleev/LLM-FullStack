# Mixtral (Sparse MoE)

## 直觉

LLaMA 每层只有一个 FFN, 想加容量就得整体加宽, 每个 token 的算力跟着涨。Mixtral 把 FFN 换成
8 个 "专家" FFN, router 给每个 token 挑 2 个: **参数量按 8 个算, 算力按 2 个算**
(8x7B: 总参 47B, 每 token 激活 ~13B)。attention / RMSNorm / RoPE 与 LLaMA 完全相同。

## 核心公式

```
p       = softmax(router(x))                 [N, E]
topk    = top-K(p)                           K 个专家 (不可导)
w_i     = p_i / Σ_{j∈topk} p_j               K 个权重和为 1 (梯度从这里回 router)
y       = Σ_{i∈topk} w_i · expert_i(x)

aux     = E · Σ_i f_i · P_i                  f_i = 专家 i 被选中次数 / token 数,  P_i = mean_t p_t[i]
total   = lm_loss + 0.01 · aux
```

aux 完全均衡时 = **K** (f_i = K/E, P_i = 1/E), 完全坍塌到固定 K 个专家时 = **E**。

## 运行命令

```bash
python -m llm_models.run_models.moe.mixtral.infer_mixtral
python -m llm_models.run_models.moe.mixtral.train_mixtral
```

## 运行后应该看到什么 (CPU, 各 < 5 s)

infer:
```
Mixtral Mini | 总参数 4,977,920 | 每 token 激活 2,815,232 (56.6%)
  Layer 0: 前 3 个 token 的专家 [[2, 0], [1, 2], [2, 1]]  权重 [[0.553, 0.447], [0.524, 0.476], [0.667, 0.333]]
prefill 10 + 逐 token decode 20 步 vs 一次性 forward: logits 最大差 9.54e-07
贪心生成 20 token, cache 与无 cache 完全一致
```
train:
```
aux loss 极值 (E=8, K=2): 均衡 = 2.0000 (应为 K), 坍塌 = 8.0000 (应为 E)
Step [   1/50] | lm_loss: 6.9587 | aux_loss: 2.0113        ← ln 1000 = 6.9078
Step [  50/50] | lm_loss: 4.6168 | aux_loss: 2.0084
```
修复前 (默认 N(0,1) embedding + weight tying + ·sqrt(D)) 第 1 步 lm_loss 是 **126.98**, 50 步后 total_loss 还有 117.99。
数据是固定的一个随机 batch: loss 下降只说明模型在背这个 batch, 不代表学到了语言。

## 常见误区

- "aux loss 的下界是 1": 只对 top-1 (Switch) 成立。top-K 时 f_i 每 token 计 K 次, 均衡值是 K。
- "MoE 省显存": 不省。所有专家都要在显存里, 省的是每 token 的 **算力**。
- "MoE 层里再加一个 dropout": Block 的残差 dropout 已经做了一次, 再加就是两次 (本库曾有此 bug)。
- "top-k 可导": 不可导。router 的梯度只经 routing_weights (被选中专家的归一化概率) 和 aux loss 的 P_i。

## 自测题

1. E=8, K=2, 完全均衡时 aux 等于多少? 所有 token 都只选专家 0 和 1 时呢?
   **答**: 2 (=K); 8 (=E)。train 脚本第一行输出就是这个验证。
2. 为什么 top-k 之后要再除以 Σ topk_probs?
   **答**: 让 K 个权重和为 1, 输出幅度不随 "选中专家概率总和" 波动; 否则 router 越犹豫, FFN 输出越小。
3. 本例激活比例 56.6%, 而 Mixtral 8x7B 约 28%, 为什么差这么多?
   **答**: 这里 E=4, K=2 (一半专家激活) 且 embedding/attention 这些 "始终激活" 的参数占比大; 8x7B 是 E=8, K=2 且专家占总参数的绝大部分。
