# DeepSeek-V3.2 (DSA: MLA + Lightning Indexer)

## 直觉

MLA 把 KV cache 压小了, 但每个 query 仍要和全部 L 个 key 算注意力, 算力 O(L²)。
DSA 先用一个很便宜的 **indexer** 给所有 key 粗打分, 只留 top-k, 昂贵的 MLA 只在这 k 个 key 上算 → O(L·k)。

难点: top-k 不可导, LM loss 的梯度到不了 indexer。所以 indexer 有自己的老师 —— 主注意力的分布。

## 核心公式

```
indexer:   I[t,s] = Σ_h ReLU( q_h[t]·k_h[s] / √d )            几个小 head, 无 softmax
选择:       S_t = top-k_s I[t,s]      (先做因果 mask 再 top-k)
注意力:     MLA 只在 S_t 上做 softmax

对齐 loss:  p[t,:] = mean_heads 主注意力概率   (detach)
           index_loss = KL( p[t,:] ‖ softmax(I[t,:]) )          两边限定在同一集合上
total = lm_loss + aux_w·aux + λ·index_loss
```
indexer 的输入也 detach: index_loss 只更新 indexer, LM loss 只更新主模型。
训练顺序 (论文): 稠密 checkpoint → **indexer 预热** (主模型冻结, 注意力仍稠密) → **稀疏训练**。

## 运行命令

```bash
python -m llm_models.run_models.moe.deepseek_v3_2.infer_deepseek_v3_2   # ~2 s
python -m llm_models.run_models.moe.deepseek_v3_2.train_deepseek_v3_2   # ~8 s
```

## 运行后应该看到什么

infer:
```
MLA 构造次数 2 (= 层数) | 总参数 6,004,992 = V3 + indexer 65,536
T=24: 每个 query 看到的 key 数 = [1, 2, 3, 4, 5, 6, 7, 8, 8, 8, ..., 8]  (上限 8, 稠密时是 1..24)
稀疏 vs 稠密 最后位置 logits 最大差 0.0209 (> 0)
prefill 10 + 逐 token decode 20 步 vs 一次性 forward: logits 最大差 9.54e-07
每 token 每层 cache: {'idx_k': 64, 'c_kv': 64, 'k_rope': 32} (idx_k 是 DSA 多付的代价)
```
train:
```
初始 lm_loss 6.9374 (ln V = 6.9078) | LM loss 对 indexer 的梯度: 全为 None
[A 稠密预训练 60 步]  lm_loss 6.9374 → 0.2372
[B indexer 预热 80 步] KL 0.1144 → 0.0046 | top-8 召回率 0.450 → 0.922 (随机乱选的期望 ≈ 0.45)
[C 稀疏训练 40 步]    lm_loss 0.2317 → 0.0688 | index_loss 0.0043 → 0.0035 | aux 2.001
```
未训练的 indexer 召回率 0.450 恰好等于随机乱选; 预热后 0.922。
修复前: 没有 index_loss, indexer 的梯度永远是 None (等于一直用随机 indexer 做稀疏); 初始 lm_loss 256.80;
每个模型把所有层建了两遍。数据是固定的一个随机 batch: 这些数字说明的是 "机制能跑通", 不是泛化。

## 常见误区

- "indexer 跟着 LM loss 端到端训练": top-k 切断了梯度, 必须有单独的对齐 loss。
- "先 top-k 再做因果 mask": 会选中未来 token 再被 mask 掉, 有效 key 不足 k 个, 而且 indexer 学到的是泄漏的信号。必须先 mask。
- "DSA 省 cache": 不省, 反而多缓存一份 indexer key (`idx_k`); 它省的是注意力 **算力**。
- "用 torch.topk 就行": ReLU 后约 1/4 的分数恰为 0, 并列时 topk 的取舍随行长变化, 带 cache 的 decode 会和整段 forward 选出不同的 key
  (本库实测 logits 差 8e-3)。改用 stable sort 后差 1e-6。
- "从第 0 步就能看到 KL 下降": 初始化时注意力和 indexer 都近似均匀, KL≈0; 要等注意力长出结构, KL 才有意义 —— 这也是论文先有稠密 checkpoint 再预热 indexer 的原因。

## 自测题

1. 为什么 indexer 用 ReLU 求和而不是 softmax?
   **答**: 它只需要排序, 不需要归一化的概率; ReLU 便宜、对 FP8 友好。只有算对齐 loss 时才对分数做一次 softmax。
2. 阶段 C (稀疏) 里 KL 是在哪个集合上算的? 为什么?
   **答**: 只在被选中的 top-k 集合上。此时主注意力在其余位置恒为 0, 老师只能告诉学生选中集合内部的相对重要性。
3. L=128K, k=2048, 主注意力的计算量降到原来的多少? indexer 自己是什么复杂度?
   **答**: 约 k/L ≈ 1.6%。indexer 仍是 O(L²), 但 head 少、维度小、无 softmax, 常数远小于主注意力。
