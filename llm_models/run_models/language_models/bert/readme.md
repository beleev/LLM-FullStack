# BERT (Encoder-only, MLM)

## 直觉
做完形填空: 把句子里 15% 的词遮住, 让模型根据**左右两边**的上下文猜回来。因为不需要逐词生成, 注意力可以不加因果 mask, 每个位置都看全句。

## 核心公式
```
x = tok_emb + pos_emb + seg_emb                      # 三个 embedding 相加
mask: 只有 padding mask [B, 1, T]                    # 没有下三角 ⇒ 双向
loss = CE(logits[选中位置], 原 token)                # 未选中位置 label = -100, 不计入
选中 15%: 80% → [MASK], 10% → 随机 token, 10% → 保持原样
```
10% 随机 + 10% 不变是为了缩小预训练 (到处是 [MASK]) 与微调 (没有 [MASK]) 的差距。

## 运行命令
```bash
python -m llm_models.run_models.language_models.bert.infer_bert
python -m llm_models.run_models.language_models.bert.train_bert
```

## 运行后应该看到什么 (CPU 实测)
- infer: 改末尾 token → 位置 0 的 logits 变化 0.0031 (>0, 双向); 改 pad 处 token → 有效位置变化 0.00e+00 (mask 生效)。
- train: 被选中位置 16/128 (12.5%), 且只有这些位置的 logits 有梯度 (断言);
  初始 loss 6.321 (ln 500 = 6.215) → 60 步后 0.480。修复初始化前初始 loss 是 **40.76**。
- 数据是**固定的随机 batch**, 下降 = 记忆, 不代表学到语言规律。

## 常见误区
- "BERT 的输出 logits 和 GPT 同形, 所以也能拿来生成" —— 不能。位置 t 的输出已经看过 t 之后的 token, 逐词生成时这些 token 并不存在。
- "loss 是对所有位置求平均" —— 只对被选中的 ~15% 位置; 这也是 BERT 每个样本的训练信号比 GPT 少的原因。
- "weight tying + 默认 N(0,1) embedding 没问题" —— 初始 logits std ≈ sqrt(D), 初始 CE≈40; 需 N(0, 0.02²) 初始化。

## 自测题
1. 为什么 BERT 不需要因果 mask? —— 目标是还原被遮的 token 而不是预测下一个, 答案不在输入里, 看全句不算泄漏。
2. 被选中但"保持原样"的 10% 位置算不算 loss? —— 算。模型不知道哪些位置被选中, 因此必须对每个位置都保持好的表征。
3. 句子级分类用哪个向量? —— `hidden[:, 0]` ([CLS] 位置), 它通过双向注意力汇聚了全句信息。
