# Mamba (Selective SSM)

## 直觉
Transformer 每生成一个 token 都要回看全部历史 (KV cache 越来越大)。Mamba 把历史压进一个**定长状态** `h`,
每步只做 "旧状态衰减一点 + 写入一点新信息"。衰减多少 (Δ)、写什么 (B)、读什么 (C) 都由当前 token 决定 —— 这就是 selective。

## 核心公式
```
Δ_t = softplus(W_Δ x_t) > 0        A = -exp(A_log) < 0
h_t = exp(Δ_t·A) ⊙ h_{t-1} + Δ_t·B_t·x_t        y_t = C_t·h_t + D·x_t
```
`exp(Δ·A) ∈ (0,1)`: 状态只衰减不爆炸。Δ 大 ⇒ 忘掉过去、重写状态; Δ→0 ⇒ 忽略当前 token。

## 运行命令
```bash
python -m llm_models.run_models.language_models.mamba.infer_mamba
python -m llm_models.run_models.language_models.mamba.train_mamba
```

## 运行后应该看到什么 (CPU 实测)
- infer: `递推 vs 整段 logits 最大差: 4.77e-07`; 读完 64 个 token 后每层状态仍是 `conv (2,256,3)` + `h (2,256,16)`;
  生成 300 token 递推约 0.4s vs 每步重算约 7s (16×), 600 步 logits 全部有限;
  反例 `a=+16` 在第 57 步溢出为 inf。
- train: 初始 loss 6.215 (ln 500 = 6.215) → 60 步后 0.508。数据是**固定的随机 batch**, 下降 = 记忆, 只说明梯度链路通。

## 常见误区
- "Mamba 生成出 NaN 要用 nan_to_num 兜底" —— 不需要。本库曾有这个兜底, 排查后在现行代码上无法复现 NaN
  (3 个种子 × T=300、lr=1e-2 训练 300 步、200 步采样均有限): A 恒负 + Δ 恒正已从结构上排除溢出。真正危险的写法是 `A = +exp(·)` 或 Δ 不过 softplus。
- "init_weights 对所有 Linear 一视同仁就行" —— `dt_proj.bias` 被清零后 Δ≈0.69, 所有通道同一时间尺度; 必须再调 `reset_dt()`。
- "没有 mask 会不会偷看未来" —— 不会, 递推方向 + 左填充的因果卷积天然因果 (infer 脚本有断言)。
- CPU 上 `nn.Conv1d(groups=C)` 是逐组循环, 曾占前向 90% 时间; 这里用 `unfold` 直接写出 depthwise 卷积 (数值等价, 快 ~100×)。

## 自测题
1. 解码第 10000 个 token 时, Mamba 每层要保存多少状态? —— `d_inner·N + d_inner·(d_conv-1)` 个数, 与 10000 无关。
2. 为什么参数化成 `A = -exp(A_log)` 而不是直接学 A? —— 保证 A<0 ⇒ `exp(Δ·A)<1`, 无论梯度把 A_log 推到哪都稳定。
3. Δ_t 很大时这一步发生了什么? —— `exp(Δ·A)→0` 旧状态被清空, `Δ·B·x` 很大, 状态几乎完全由当前 token 重写 ("重置记忆")。
