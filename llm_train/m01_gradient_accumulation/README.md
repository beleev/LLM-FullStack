# M01 — 梯度累积

运行: `python -m llm_train.m01_gradient_accumulation.demo`

## 直觉
显存装不下 batch=8, 就分 4 次每次喂 2 个, 把梯度加起来, 攒够了再更新一次。对优化器来说, 它看到的就是一个 batch=8 的梯度。

## 核心公式
loss 对样本取 mean 时: `g_full = Σ_k (n_k / N) · g_k`。等长 micro-batch 时权重就是 `1/K`。

## 运行后应该看到什么
```
max |full - accum(2,2,2,2)|  = 2.98e-08
max |full - accum(3,3,2)|    = 5.96e-08      ← 不等长也对, 因为按样本数加权
忘记缩放: |naive| / |full|    = 4.00x         ← 恰好 K 倍
```

## 与真实系统的差距
- 真实框架靠 `loss / K` 后 `backward()` 让 autograd 自动累加; DDP 下要用 `no_sync()` 避免每个 micro-batch 都 all-reduce。
- LLM 的 loss 按 **token** 平均, micro-batch 间有效 token 数不同, 必须按 token 数加权 (HF Trainer 2024 年修过这个 bug)。
- BatchNorm 类层的统计量按 micro-batch 算, 不等价 (LLM 用 LayerNorm/RMSNorm, 无此问题)。

## 常见误区
- "累积能省时间" —— 不能, K 个 micro-batch 串行, 时间是 K 倍; 省的只是激活显存。
- "梯度直接相加就行" —— 那等于学习率 ×K。

## 自测题
1. batch=8 拆成 (3,3,2), 第三个 micro-batch 的梯度权重是多少? **答: 2/8。**
2. 梯度累积减少了哪部分显存? **答: 激活 (按 micro-batch 计); 参数、梯度、优化器状态不变。**
3. DDP + 累积 4 步, 理想情况下几次 all-reduce? **答: 1 次 (前 3 个 micro-batch 用 no_sync)。**
