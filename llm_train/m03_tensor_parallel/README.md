# M03 — 张量并行 (Megatron TP)

运行: `python -m llm_train.m03_tensor_parallel.demo`

## 直觉
一层 MLP 的 `W1 [D,H]` 按列切、`W2 [H,O]` 按行切。每张卡只算隐藏层的 1/N 宽度; relu 是逐元素的, 夹在中间不需要通信。只有两处必须把各卡结果加起来: 前向的输出, 反向的 dX。

## 核心公式
- 前向: `Y = Σ_r relu(X·W1_r)·W2_r + b2` —— **g 算子**: 前向 all-reduce, 反向恒等。
- 反向: `dX = Σ_r dZ_r · W1_rᵀ` —— **f 算子**: 前向恒等, 反向 all-reduce。

## 运行后应该看到什么
```
max output diff            = 0.00e+00
max grad W1/b1/W2/b2 diff  = 0.00e+00
max grad x diff            = 1.86e-09
单 rank 的 dX 与真值差      = 1.83e-02  (未 all-reduce → 错)
每 rank 权重               = 24 / 48 个参数
通信量 (1 层 fwd+bwd)       = 72 B/rank  [all_reduce×2=72B]
```

## 与真实系统的差距
- 只演示 MLP; attention 的切法是按头切 QKV (列) + 输出投影 (行), 通信模式相同。
- 真实 Megatron 还有 sequence parallel: 把 LayerNorm/Dropout 的激活也按序列切, all-reduce 拆成 reduce-scatter + all-gather。
- all-reduce 在关键路径上无法与计算重叠, 所以 TP 基本只在 NVLink 域内 (≤8) 使用。

## 常见误区
- "b2 每张卡都加一次" —— all-reduce 之后只加一次, 否则变成 N·b2。
- "TP 的通信量和 DDP 差不多" —— DDP 每 step 一次且可重叠; TP 每层每个 micro-batch 两次且阻塞。

## 自测题
1. 为什么 W1 列切 + W2 行切中间不用通信? **答: 列切后各卡的隐藏分片互不依赖, relu 逐元素; 行切恰好消费对应分片。**
2. 反向不做 dX 的 all-reduce 会怎样? **答: 上一层收到的只是 1/N 列的贡献, 梯度错 (demo 中差 1.8e-2)。**
3. 一个 Transformer 层 (attn + MLP) fwd+bwd 共几次 all-reduce? **答: 4 次。**
