# M11 — 专家并行 (MoE Expert Parallel)

运行: `python -m llm_train.m11_expert_parallel.demo`

## 直觉
E 个专家分给 D 张卡。token 在卡 A, 它选中的专家在卡 B → 先把 token 寄过去 (dispatch all-to-all), 算完再寄回来 (combine all-to-all)。寄多少由 router 决定, 所以 **router 不均衡 = 通信不均衡 + 算力不均衡 + 丢 token**。

## 核心公式
- `capacity = ⌈cf · N_tok / E⌉`, 超出的 token 被丢弃 (输出 0, 只剩残差)。
- Switch aux loss: `L_aux = E · Σ_e f_e · P_e` (f 实际占比, 不可导; P 平均路由概率, 可导)。
- DeepSeek-V3 aux-loss-free: 选专家用 `s_e + b_e`, 每步 `b_e += γ·sign(mean_load − load_e)`; gate 权重仍用原始 `s_e`。

## 运行后应该看到什么 (4 卡 × 每卡 2 专家, 64 token, capacity=10)
```
max |dense - EP|              = 0.0e+00
每卡收到的 token               = [11, 18, 11, 24]  (均匀应为 16)
通信量 (dispatch + combine)    = 1632 B/rank
                     每专家 token 数                 max/mean  丢弃  最热卡
倾斜 router          [1, 10, 11, 7, 4, 7, 11, 13]    1.62x     5     24
aux loss 60 步       [7, 9, 8, 7, 8, 9, 8, 8]        1.12x     0     17
aux-loss-free bias   [8, 8, 8, 8, 8, 8, 8, 8]        1.00x     0     16
```
combine 是**真的**第二次 all-to-all: 专家卡按来源切块寄回, 源卡用自己留着的行号写回并乘 gate。

## 与真实系统的差距
- top-1 路由; 真实模型 top-2 ~ top-8, 还有 shared expert。
- router 这里**只**用 aux loss 训练, 是为了隔离它的作用; 真实训练是 `L_task + α·L_aux` (α≈0.01), 两者有拉扯 —— 这正是 aux-loss-free 方法的动机。
- 真实 all-to-all 要先交换各块大小 (变长), 再用 grouped GEMM 算专家; DeepEP 等库还做 FP8 dispatch 和机内/机间两级路由。
- 没有反向 (反向是同样的两次 all-to-all, 方向相反)。

## 常见误区
- "MoE 的通信像 DDP 一样可预测" —— all-to-all 的量取决于数据和 router, 每步都不同。
- "capacity factor 调大就没事" —— 显存和计算按最坏情况 padding, cf=2 就是两倍开销。
- "bias 会改变模型输出" —— 只改 "选谁", gate 概率不变 (demo 有断言)。

## 自测题
1. 64 token、8 专家、cf=1.25, capacity = ? **答: ⌈1.25·8⌉ = 10。**
2. 为什么 aux loss 里 f 当常数? **答: f 来自 argmax, 不可导; 梯度只经由 P 回传。**
3. EP 的一次 MoE 层前向有几次 all-to-all? 反向呢? **答: 前向 2 次, 反向 2 次。**
