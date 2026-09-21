# M05 — ZeRO-1/2/3 与 FSDP

运行: `python -m llm_train.m05_zero_fsdp.demo`

## 直觉
DDP 的 N 张卡存了 N 份一模一样的优化器状态、梯度、参数 —— 纯冗余。ZeRO 让 rank r 只保管第 r 片, 用到完整张量时临时通信。Adam 是逐元素运算, 所以 "每人更新自己那片" 与 "整体更新" 逐位相同。

## 核心公式 (混合精度 Adam, Ψ 个参数, 每 rank 字节)
| | 参数 fp16 | 梯度 fp16 | master+m+v fp32 | 合计 |
|---|---|---|---|---|
| DDP | 2Ψ | 2Ψ | 12Ψ | 16Ψ |
| ZeRO-1 | 2Ψ | 2Ψ | 12Ψ/N | 4Ψ + 12Ψ/N |
| ZeRO-2 | 2Ψ | 2Ψ/N | 12Ψ/N | 2Ψ + 14Ψ/N |
| ZeRO-3 / FSDP | 2Ψ/N | 2Ψ/N | 12Ψ/N | 16Ψ/N |

## 运行后应该看到什么 (Ψ=192, N=4)
```
               常驻 B/rank   公式   通信 B/rank/step   vs DDP
DDP                 3072   3072        576           1.00x
ZeRO-1              1344   1344        864           1.50x
ZeRO-2              1056   1056        576           1.00x
ZeRO-3/FSDP          768    768        864           1.50x  (+256 B 瞬时: 一层的完整 fp16 参数)
max |ZeRO-k - DDP| = 0.0e+00 (k=1,2,3)
max |ZeRO-3 - 单卡 fp32 Adam| = 3.0e-04  (参数本身移动了 5.0e-02)
```
"常驻" 是 rank 0 状态字典里所有数组 `nbytes` 之和, 断言与公式**精确相等**。

## 三个 stage 在代码里的差别
- stage ≥1: `master/m/v` 只存 `shard(...)`; 更新后 all-gather fp16 参数。
- stage ≥2: 梯度同步从 `all_reduce` 换成 `reduce_scatter`, 只留 1/N。
- stage 3: `p16` 也只存分片; 前向/反向逐层 `all_gather → 计算 → del` (FSDP 的 gather-compute-free)。

## 与真实系统的差距
- 本实现的 ZeRO-1 用 all-reduce + all-gather (1.5×); DeepSpeed 用 reduce-scatter, 与 DDP 持平。
- ZeRO-2 在 reduce-scatter 之前仍瞬时持有完整的本地梯度; 真实系统按 bucket 边反向边 reduce, 瞬时量只是一个 bucket。
- 没有 prefetch (提前 gather 下一层)、没有 HSDP (机内分片 × 机间复制)、没有 CPU/NVMe offload。
- 激活显存不在这张表里 (→ m07)。

## 常见误区
- "ZeRO-3 是模型并行" —— 不是。计算时每卡仍用**完整**的层权重, 只是不常驻; 它是省显存的数据并行。
- "分片后数值会有差异" —— 逐位相同。

## 自测题
1. 7B 模型 64 卡, ZeRO-1 每卡模型状态多大? **答: 4·7 + 12·7/64 ≈ 29.3GB (DDP 为 112GB)。**
2. 为什么 ZeRO-2 通信量不比 DDP 多? **答: all-reduce = reduce-scatter + all-gather; ZeRO-2 把 all-gather 的对象从梯度换成了更新后的参数。**
3. FSDP 的瞬时峰值由什么决定? **答: 最大的那个 FSDP unit (通常一个 Transformer block) 的完整参数。**
