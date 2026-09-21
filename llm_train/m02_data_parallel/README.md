# M02 — 数据并行 (DDP)

运行: `python -m llm_train.m02_data_parallel.demo`

## 直觉
N 张卡各持一份完整模型, 各算 batch 的 1/N, 把梯度求平均后各自做同样的更新。初值相同 + 梯度相同 ⇒ 参数永远相同。

## 核心公式
`g = (1/N) Σ_r g_r` (等分 shard 时, local mean 的平均 = global mean)。
ring all-reduce 每 rank 发送 `2(N-1)/N · |g|` 字节。

## 运行后应该看到什么
```
local grad 彼此不同            = 1.818
梯度大小                       = 56 B
通信量                         = 84 B/rank  [all_reduce×2=84B]     ← 2·(3/4)·56
max |single - ddp| after step = 3.73e-09
```
断言还检查 4 个副本逐位相同。

## 与真实系统的差距
- 真实 DDP 把梯度装进 ~25MB 的 bucket, 反向算到哪个 bucket 就立刻异步 all-reduce, 与剩余反向重叠。
- 这里每个参数单独 all-reduce 一次 (W 和 b 两次); 真实系统会把小张量拼起来发, 减少 latency 项。
- 没有模拟 straggler、网络拓扑、梯度压缩。

## 常见误区
- "DDP 省显存" —— 不省, 每卡都是完整的参数 + 梯度 + 优化器状态 (→ m05)。
- "N 卡就要 N 倍通信" —— ring 下每卡通信量几乎与 N 无关。

## 自测题
1. 7B 模型 fp16 梯度 14GB, 64 卡 ring all-reduce 每卡每步发多少? **答: 2·63/64·14 ≈ 27.6GB。**
2. 为什么副本不会漂移? **答: 初值相同, 每步应用的是同一份平均梯度, 更新是确定性的。**
3. shard 不等长时直接平均 local mean 对吗? **答: 不对, 要按样本 (token) 数加权。**
