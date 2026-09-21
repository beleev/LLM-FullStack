# M22 — 稀疏 attention decode (Quest / NSA / DSA 风格)

## 直觉
decode 每生成 1 个 token 都要把**全部** KV 读一遍 —— 长上下文下这是纯访存瓶颈。但训练过的 LLM 的
attention 高度集中: 绝大部分概率质量落在少数 token 上。如果能**不读 KV 就猜出哪些 block 重要**,
只对这些 block 做 attention, 访存就按比例下降, 输出几乎不变。做法: 每个 KV block 常驻一份很小的摘要,
用 q 对摘要打分, 选 top-k block; 永远保留第 0 块 (attention sink, m16) 和最后一块 (最近的 token)。

## 核心数据结构或公式
```
KV 切块         K (T, d) → (nb, bs, d)
Quest 摘要      kmin, kmax = 逐维 min/max                        (nb, d) × 2
Quest 打分      ub_b = Σ_i max(q_i·kmin_{b,i}, q_i·kmax_{b,i})   保证 ub_b ≥ max_{k∈block b} q·k
mean 打分       s_b = q · mean(K_b)                              NSA 压缩分支 / indexer 的最简替身
选块            blocks = {0, nb−1} ∪ top-(k−2) by score
稀疏 attention  dense_attention(q, K[idx], V[idx]),  idx = 选中 block 的 token 下标
评测            相对 L2 误差 vs 全量 dense_attention; recall = 选中 block 覆盖的真实 attention 概率质量
```
上界为什么成立: 对每一维, k_i ∈ [kmin_i, kmax_i], 所以 q_i·k_i ≤ max(q_i·kmin_i, q_i·kmax_i)
(q_i 为正取 kmax, 为负取 kmin); 逐维相加即得。上界保证 "不会漏掉真正的高分块", 但可能很松。

## 运行后应该看到什么
```bash
python -m llm_infer.m22_sparse_attention.demo     # < 1 s
```
**needle 负载** (T=4096, d=64, 256 blocks; 4 个 needle block 各含 2 个与 q 对齐的 key, logit≈+10, 其余是噪声):
```
[1] min(bound − true max) = 29.969 (≥ 0);  mean(bound / true max) = 6.68x  ← 上界成立但很松
   k  KV read |  quest err  recall |   mean err  recall |  random err  recall
   4     1.6% |     0.8265   59.1% |     0.8265   59.1% |      1.0793    0.0%     ← 首尾占 2 个名额, 只够装 2 根针
   8     3.1% |     0.0235   97.7% |     0.0234   97.7% |      1.0582    0.1%
  32    12.5% |     0.0206   98.0% |     0.0201   98.0% |      1.3214    6.2%
 128    50.0% |     0.0109   98.9% |     0.0101   99.0% |      1.0404   35.5%
 256   100.0% |     0.0000  100.0% |     0.0000  100.0% |      0.0000  100.0%
```
读 3.1% 的 KV 就拿到 97.7% 的 attention 质量、误差 2.3%; 随机选块读 50% 误差仍 >100%。

**真实 TinyLM KV** (T=1024 随机 token, 4 层平均, 64 blocks) —— 如实报告, **效果弱**:
```
   k  KV read |  quest err  recall |   mean err  recall |  random err  recall
   8    12.5% |     0.6179   15.3% |     0.6359   19.2% |      0.5085   12.0%
  16    25.0% |     0.4218   30.1% |     0.3590   35.5% |      0.3859   24.4%
  32    50.0% |     0.2444   57.6% |     0.1682   63.2% |      0.2209   50.6%
```
随机权重模型的 attention 近乎均匀, recall ≈ 读取比例, 打分只比随机好几个点, Quest 的误差甚至不如随机
(k=8: 0.62 vs 0.51)。**稀疏 attention 的收益来自训练出来的注意力集中性, 不是来自算法本身。**
断言: Quest 上界 ≥ 真实 max (两种数据); k=全部 → 误差 < 1e-6; needle 上同预算 quest 误差 < 随机且
recall > 随机; quest 误差随 k 大体单调下降; TinyLM 上只断言 recall ≥ 随机。

## 与真实系统的差距
- 这里只统计 "读了多少 KV", 没有真实 kernel; 真实收益需要 block-sparse / paged kernel (FlashInfer, FlashMLA sparse)。
- DSA 的 lightning indexer 和 NSA 的压缩/选择分支是**训练出来**的 (NSA 原生稀疏训练; DSA 在 dense 模型上续训),
  比 min/max 启发式准得多; DSA 是 token 级 top-k (2048 个), 不是 block 级。
- 真实系统按 head / 按层分别选块, 前几层通常保持 dense; 这里单头。
- 打分本身也有成本: 摘要 ≈ KV 的 2/bs, 每步对所有 block 打分是 O(nb·d), 超长上下文下也需要优化。

## 常见误区
- "Quest 上界保证选得准" —— 它只保证不低估; 在各向同性噪声上界很松 (6.68x), 排序信息主要来自结构化的 key。
- "稀疏 attention 省的是计算" —— decode 阶段省的主要是访存/带宽; prefill 才是省 FLOPs。
- "和 attention sink (m16) 一样是丢 KV" —— m16 永久丢弃旧 KV; 这里 KV 全保留, 每步按 q 动态选, 不同 q 选不同块。
- "随机权重模型上也该有效" —— 见上, 没有集中性就没有可利用的稀疏性。

## 自测题
1. bs=16, d=128 时 Quest 摘要占 KV (只算 K+V) 的比例? **答**: 每 block 2 个向量 vs 32 个向量 = 1/16。
2. 为什么 k=4 时 quest 的 recall 只有 59.1%? **答**: 首块和末块强制占 2 个名额, 只剩 2 个给 4 个 needle block。
3. 若 q 的某一维为负, 该维上界用 kmin 还是 kmax? **答**: kmin (负数 × 最小值 = 该维最大的乘积)。
