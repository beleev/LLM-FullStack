# M11 — FlashAttention: 分块 + online softmax, 永不落地 (T,T) 矩阵

## 直觉
朴素 attention 要把 `S = QKᵀ` 和 `P = softmax(S)` 两张 (T,T) 矩阵写进 HBM 再读回来;
T=4096 就是 16M 元素/头, 慢的不是算, 是搬。FlashAttention 把 Q、K/V **都**切成小块,
每次只在 SRAM 里算一块 (b_q, b_k), 用 logsumexp 把各块结果精确地"接"起来。
softmax 看似需要整行的全局分母, 但分母可以增量维护 —— 这就是全部技巧。

## 核心数据结构或公式
每个 query 行只维护两个量: 已归一化的输出 `O` 和分母的 log `lse = log Σ_j exp(S_j)`。
来了一块新分数 `S_b` (b_q, b_k):
```
  lse_new = logaddexp(lse, logsumexp(S_b))
  O_new   = exp(lse - lse_new) · O  +  exp(S_b - lse_new) @ V_b
```
- 循环顺序: **外层 Q 块, 内层 K/V 块 = FlashAttention-2** (FA-1 相反)。Q 块的 (O, lse) 在内层循环里
  常驻 SRAM, 只写回一次; Q 块之间无依赖, 可并行。
- 内存: 输出 O (Tq,dv) 与 lse (Tq,) 是 O(T); 临时工作集只有 `S` 一块 = **b_q × b_k, 与 T 无关**。
  (只切 K/V 不切 Q 的话工作集是 Tq×b_k, 仍随 T 增长。)
- causal: K 块的最早 key 晚于 Q 块的最晚 query → 整块跳过, 连 matmul 都不做; 只有跨对角线的块才逐元素 mask。
- `merge_attention(O1, lse1, O2, lse2)`: 同一批 query 在两段 K/V 上的部分结果 → 全量结果。
  内层循环本身就是反复做这个 merge。**lse 必须返回**: ring attention / chunked prefill 靠它跨设备、跨 chunk 合并;
  反向传播靠它重算 `P = exp(S - lse)`, 不用存 (T,T)。

## 运行后应该看到什么
```bash
python -m llm_infer.m11_flash_attention.demo
```
```
[1] 30 组配置 (Tq/Tk/b_q/b_k/causal) 的最坏误差 = 1.78e-06      (assert < 1e-5, O 与 lse 都对拍)
[2][3]  T   full  partial  skipped  跳过比例  peak b_q×b_k          T²      省
      256      6        4        6    37.5%        4,096      65,536     16x
     1024    120       16      120    46.9%        4,096   1,048,576    256x
     4096   2016       64     2016    49.2%        4,096  16,777,216   4096x
[4] merge(O1,lse1,O2,lse2) vs 全量: max|ΔO| = 1.16e-07, max|Δlse| = 5.12e-07
    反例 (O1+O2)/2               : max|ΔO| = 2.12e-01
```
基线是 `core.dense_attention`; 跳过块数 assert 为 n(n-1)/2 (n = T/64), 峰值 assert 恒为 64×64。

## 与真实系统的差距
- 真实收益主要是**延迟/吞吐** (少读写 HBM, 2~4× 提速), 不只是显存; numpy 里块循环反而更慢, 这里只验证算法。
- 真 kernel 是 fused CUDA/Triton: 块大小按 SRAM 容量选, 多头/batch 维并行, 支持 GQA、sliding window、ALiBi、varlen。
- 反向传播 (用 lse 重算 P)、dropout、FP8 (FA-3) 没有实现。
- decode (Tq=1) 时 attention 本就是 O(T), 真实系统另有 paged/decode 专用 kernel (FlashInfer, FlashDecoding)。

## 常见误区
- "FlashAttention 是近似算法" —— 不是, 它与朴素 attention 数学上严格相等 (只差浮点舍入 ~1e-6)。
- "显存是 O(b²) 所以总显存与 T 无关" —— 工作集与 T 无关, 但 O、lse、KV 本身仍是 O(T)。
- "两段 attention 输出取平均就能合并" —— 两段的 softmax 质量 (e^lse) 不等, 必须按 lse 加权 (demo [4] 的反例)。
- "causal 省一半计算是靠 mask" —— mask 只是把结果置 -inf, 省计算靠的是整块跳过。

## 自测题
1. 为什么只切 K/V 不切 Q 时工作集不是 O(b²)?
   **答**: 每块分数矩阵是 (Tq, b_k), Tq 仍随序列增长; 必须 Q 也切块才得到 (b_q, b_k)。
2. T=4096, b=64, causal: 总共多少块? 跳过多少? 需要逐元素 mask 的有多少?
   **答**: n=64, 共 64²=4096 块; 跳过 n(n-1)/2=2016; 对角线上 64 块需要 mask; 其余 2016 块全可见。
3. 两张卡各持一半 K/V, 各自算出 (O1,lse1)、(O2,lse2), 怎样得到全量结果? 通信量多大?
   **答**: lse=logaddexp(lse1,lse2), O=e^(lse1-lse)·O1+e^(lse2-lse)·O2; 只需传 O (Tq,dv) 和 lse (Tq,), 不传 (T,T)。
