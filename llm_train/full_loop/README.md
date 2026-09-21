# Full Loop — 拼起来的训练主循环

运行: `python -m llm_train.full_loop.demo`

## 直觉
前面每个模块单独成立; 这里验证它们**叠在一起仍然成立**。`train()` 是唯一的训练函数: `world=1, micro=1, amp=False` 就是单卡基线, 所以等价性断言是同一份代码的自洽检验。

## 一个 step 里发生的事
1. `lr = warmup_cosine(step)` (m10)
2. all-gather 各 rank 的 master 分片 (转成 fp16 再发) → 计算副本 (m05/m06)
3. 每 rank K 个 micro-batch: fp16 前向/反向, loss × scale, fp32 累积 (m01/m02/m06)
4. 任一 rank 梯度含 Inf/NaN → **所有 rank** 一起跳过, scale 减半 (m06/m10)
5. reduce-scatter(mean) → 每 rank 只拿自己那片梯度 (m05/m09)
6. 全局范数 = √(all-reduce(各分片平方和)) → 裁剪 (m10)
7. 每 rank 在自己的 master/m/v 分片上 Adam (m05)
8. 每 rank 写自己的 checkpoint 分片 + 一个 meta (m08)

## 运行后应该看到什么
```
[A] fp32: 分布式 vs 单卡   max |Δ master| = 6.0e-08;  NaN batch 被跳过 1 / 1 步
[B] AMP: 跳过 4 / 40 步 (3 次溢出 + 1 个 NaN batch), 最终 scale = 2^18
    val loss: 0.9270 → 0.00363 (AMP) / 0.00606 (fp32)
    max |Δ master| AMP 分布式 vs AMP 单卡 = 7.1e-05  (参数共移动 0.79)
    通信 (40 步) = 1856 B/rank [all_gather×40, reduce_scatter×36, all_reduce×36]
[C] 文件 = ['meta.pkl', 'rank0.pkl', 'rank1.pkl'];  续训 vs 不中断 max |Δ| = 0.0e+00
```
断言: fp32 下分布式 ≈ 单卡 (<1e-5); AMP 下只差 fp16 舍入; 续训后 master、m、v、scaler 状态**逐位相同**。
(AMP 与 fp32 的 val loss 不同是因为 AMP 多跳过了 3 步, 优化器步数不同。)

## 与真实系统的差距
- 被跳过的 step 照样消耗数据和 lr 调度步 (与 PyTorch 行为一致), 但 Adam 的 t 不前进。
- 没有 TP/PP/EP, 没有通信-计算重叠, 没有异步 checkpoint, 不支持换 world size 加载 (resharding)。
- NaN batch 只是跳过; 真实系统还会记录坏样本位置以便回滚后绕开。

## 常见误区
- "各 rank 各自判断要不要跳过" —— 必须全局一致, 否则副本分叉。
- "分片后各自裁剪自己的梯度" —— 范数必须是全局的。
- "loss scale 不用存 checkpoint" —— 不存就无法逐位续训 (scale 影响哪一步溢出)。

## 自测题
1. 为什么 all-gather 的是 fp16 而不是 fp32 master? **答: 计算只需要 fp16 副本, 通信量减半。**
2. 40 步里 reduce_scatter 只有 36 次, 为什么? **答: 4 个被跳过的 step 在溢出检查后直接返回。**
3. 把 world 从 2 改成 4 后能直接加载这个 checkpoint 吗? **答: 不能, 分片边界变了, 需要 resharding (先合并再重切)。**
