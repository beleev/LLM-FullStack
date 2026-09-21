# M16 — DeepSpeed-Ulysses 序列并行

运行: `python -m llm_train.m16_ulysses_sequence_parallel.demo`

## 直觉
注意力之外的层 (MLP、norm) 对序列是逐 token 的, 按序列切没有任何通信。唯一需要 "看到整条序列" 的是注意力。Ulysses 的办法: 进注意力前用 all-to-all 把 **按序列切 `[T/P, H, d]`** 换成 **按头切 `[T, H/P, d]`**; 各头独立, 每卡对完整序列算自己的 H/P 个头; 算完再换回来。

## 核心公式 (每 rank 发送, `B₀ = T/P·H·d` 个元素)
- Ulysses: `4 · (P−1)/P · B₀` (Q、K、V、输出各一次 all-to-all) → 随 P **下降**
- Ring (m12): `(P−1) · 2·B₀` → 随 P 基本**不变**
- 约束: `H % P == 0`, 即 P ≤ 头数 (GQA 下受 KV 头数限制)。

## 运行后应该看到什么 (T=32, H=8, d=8)
```
   P   Ulysses    公式   Ring (m12)   公式   max|Δ| Ulysses
   2    16384    16384     16384     16384      0.0e+00
   4    12288    12288     24576     24576      0.0e+00
   8     7168     7168     28672     28672      4.4e-16
P=8 时 Ring / Ulysses 通信量 = 4.0x
P=16 > H=8 → AssertionError: Ulysses 要求头数能被并行度整除
```
Ring 那一列是真的逐头调用 m12 的 `ring_attention` 并由 `core.comm` 计数得到的。

## 与真实系统的差距
- 这里对 Q/K/V 做了 3 次 all-to-all; 真实实现常把 QKV 拼起来发 1 次。
- all-to-all 吃对分带宽, 跨机时不如 ring 的邻居 P2P; 实战常**混用**: 机内 Ulysses × 机间 Ring (USP / 2D context parallel)。
- 每卡要物化完整序列长度 T 的 Q/K/V (只是头少了), 注意力核内的激活是 `T × H/P`; Ring 则是 `T/P × H`。
- 只有前向; 反向是方向相反的同样 4 次 all-to-all。

## 常见误区
- "Ulysses 和 Ring 二选一" —— 可以叠加, 并行度相乘。
- "Ulysses 需要改注意力核" —— 不需要, 每卡就是一次普通的因果 FlashAttention, 也不需要 zigzag。
- "头数 64 就能开 64 路" —— GQA 只有 8 个 KV 头时, KV 要么复制要么限制 P ≤ 8。

## 自测题
1. all-to-all 前后单卡张量的 shape? **答: [T/P, H, d] → [T, H/P, d]。**
2. P 从 4 加到 8, Ulysses 每卡通信怎么变? **答: 12288 → 7168 B, 下降 (∝ (P−1)/P²)。**
3. H=8 的模型最多开几路 Ulysses? 想要 32 路 CP 怎么办? **答: 8 路; 再叠 4 路 Ring。**
