# M12 — 上下文并行 (Context Parallel): Ring Attention + zigzag

运行: `python -m llm_train.m12_sequence_parallel.demo`

> 命名: 目录名沿用 `sequence_parallel`, 但这里做的是 **context parallel** —— 在注意力内部按序列切。
> Megatron 的 "sequence parallel" 指 TP 组内切 LayerNorm/Dropout 激活; 另一条按序列切的路线 Ulysses 见 [m16](../m16_ulysses_sequence_parallel/)。

## 直觉
序列切成 D 段, 每卡常驻自己的 Q。KV 块像击鼓传花一样沿环传 D-1 次; 每到一块, 用 online softmax 把它并进当前结果 —— 与 FlashAttention 的分块技巧完全相同, 只是块来自别的卡。

因果 mask 下, 连续切分让 rank 0 只有 1 块要算、rank D-1 有 D 块。每一轮大家要等最慢的, 所以**省了一半计算, 墙钟时间几乎没省**。zigzag: 切成 2D 块, rank r 拿第 r 和第 2D-1-r 块 (一早一晚), 每卡每轮工作量完全相同。

## 核心公式
online softmax: `m' = max(m, max_j s_j)`; `den' = den·e^{m−m'} + Σ_j e^{s_j−m'}`; `acc' = acc·e^{m−m'} + Σ_j e^{s_j−m'} v_j`; 输出 `acc/den`。

## 运行后应该看到什么 (D=4, T=32)
```
max |full - ring| 连续 / zigzag = 4.4e-16 / 4.4e-16
连续切分:  step 0 [36,36,36,36]  step 1 [0,64,64,64]  step 2 [0,0,64,64]  step 3 [0,0,0,64]
zigzag:    step 0 [36,36,36,36]  step 1..3 [32,32,32,32]
每卡总工作量 连续 = [36, 100, 164, 228];  zigzag = [132, 132, 132, 132]
墙钟 Σ_step max_rank = 连续 228 → zigzag 132 (1.73x);  不利用因果性 = 256
通信量 = 6144 B/rank = (D-1) 轮 × 一个 KV 块; 与是否 zigzag 无关
```

## 与真实系统的差距
- 单头、无 batch、float64; 真实实现每块调用 FlashAttention 核并合并 LSE。
- "墙钟" 是 Σ_step max_rank(q·k 对数) 的代理指标, 没有模拟 P2P 与计算的重叠 (真实系统靠它把通信藏起来)。
- 反向也要环传 (KV 的梯度跟着 KV 一起转), 这里只有前向。
- Llama-3 的 CP 用 all-gather KV 而非 ring (GQA 下 KV 小, 实现更简单)。

## 常见误区
- "因果 mask 让 Ring Attention 自动快一倍" —— 不 zigzag 的话只省 ~10% (228 vs 256)。
- "online softmax 是近似" —— 精确等价, 误差 4e-16 是浮点舍入。

## 自测题
1. D=4 连续切分, rank 3 共算几个 KV 块? rank 0 呢? **答: 4 块 / 1 块。**
2. zigzag 下 rank 1 (D=4) 拿哪两块? **答: 第 1 块和第 6 块 (共 8 块)。**
3. Ring 每 rank 的通信量随 D 怎么变? **答: (D-1)/D · 2·T·d, 基本不变 (对比 m16 Ulysses 随 D 下降)。**
