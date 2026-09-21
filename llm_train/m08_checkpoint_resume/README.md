# M08 — Checkpoint / Resume

运行: `python -m llm_train.m08_checkpoint_resume.demo`

## 直觉
"能恢复" 的标准不是 "loss 差不多", 而是 **逐位相同**: 只有这样, 一次 loss spike 才能靠回滚 + 重放来定位。需要保存的是 "下一步计算会读到的一切"。

## 状态清单
参数 · 优化器状态 (动量 / Adam m,v, step 计数) · 全局 step (决定 lr) · 数据流 (seed + cursor) · RNG 状态 (dropout、shuffle、采样)。
AMP 还要加 loss scale (见 full_loop)。

## 运行后应该看到什么
```
checkpoint 大小               = 3047 B (其中 RNG 状态 ~2.5KB, 比模型还大)
完整恢复: max |Δ| vs 不中断    = 0.0e+00
漏掉 RNG 状态                 = 3.1e-02
漏掉 optimizer 状态           = 5.8e-02
漏掉 data seed/cursor         = 1.5e-01
```
"新进程" 里所有对象都用**不同的种子**重建, 只有 checkpoint 能把它们拉回来。每步的输入 dropout 真实消耗 RNG。

## 与真实系统的差距
- 单文件 pickle; 真实系统每 rank 写自己的分片 (full_loop 演示了最小版), 并支持 **resharding** (换并行度加载, 如 PyTorch DCP)。
- 真实系统异步落盘 (先拷到 pinned CPU 内存, 后台写), 否则千卡等一块盘。
- 每个 rank、每个 dataloader worker、CUDA 各有自己的 RNG; 这里只有一个。
- pickle 不安全, 只能加载自己写的文件; 生产用 safetensors。

## 常见误区
- "只存模型权重就能续训" —— 能跑, 但 Adam 状态归零相当于重新 warmup, loss 会跳。
- "恢复后 loss 对得上就行" —— 漏掉 RNG/数据游标**不会报错**, 只是悄悄变成另一条轨迹。
- "写到一半被杀没关系" —— 必须 tmp + 原子 rename, 否则最新的 checkpoint 是坏的。

## 自测题
1. 为什么数据流要同时存 seed 和 cursor? **答: 第 k 个 batch 由 (seed, k) 决定, 缺一个都复现不了顺序。**
2. 恢复后 lr 不对, 最可能漏了什么? **答: 全局 step (scheduler 状态)。**
3. RNG 状态比这个模型还大, 为什么? **答: MT19937 内部状态是 624 个 32 位字。**
