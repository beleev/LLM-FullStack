# M13 — FP8 训练

运行: `python -m llm_train.m13_fp8_training.demo`

## 直觉
FP8 只量化 **矩阵乘的输入** (权重、激活、输出梯度), GEMM 输出、累加、优化器全留高精度。难点是 8 位浮点的动态范围很窄, 需要三件互相独立的事:
1. **scaling** —— 把张量搬进 FP8 能表示的区间 (决定能不能用);
2. **master weights** —— 小更新不被舍掉 (决定能不能收敛);
3. **scaling 粒度** —— 一个 outlier 别拖垮整个张量 (决定扛不扛得住 outlier)。

## 两套配方 (别混淆)
| | 前向 | 反向梯度 | scaling 粒度 |
|---|---|---|---|
| Transformer Engine | E4M3 | **E5M2** | per-tensor (delayed / current) |
| DeepSeek-V3 | E4M3 | **E4M3** | 激活 1×128 tile, 权重 128×128 block |

TE 用 E5M2 的**范围**兜住梯度; V3 用更细的 **scale** 兜住范围, 于是梯度也能用 E4M3 的精度。

## 运行后应该看到什么
```
[1] e4m3: 主体相对误差 0.026, 尖峰 3e4 → 448 (饱和);  e5m2: 0.053, 3e4 → 28672
[2] outlier 幅度   per-tensor   1×128 tile   per-tensor 冲零比例
    1e+02            0.025        0.026          0.0%
    1e+04            0.026        0.026          1.9%
    1e+05            0.126        0.026         17.8%
    1e+06            0.982        0.026         97.0%
[3] FP32 基线 1.02e-08 | A 无 scaling 1.31e-02 | B per-tensor(TE) 4.54e-04 | C block(V3) 4.61e-04 | D 无 master 2.69e-01
    scaling 的收益 A/B = 29x;  master 的收益 D/C = 584x;  粒度的收益 B/C = 0.99x
```
**诚实结论**: E4M3 自带 2¹⁵ 的动态范围, outlier 在 1e4× 以内 per-tensor 完全扛得住; 干净数据上 B 与 C 打平。细粒度 scaling 的价值只在重尾/outlier 出现时才体现。

## 与真实系统的差距
- 假量化: 量化后立刻反量化再用 FP32 做矩阵乘, 只模拟舍入, 不模拟 FP8 GEMM 的速度和累加位宽。
- 权重用的是一维 128 block, 不是 V3 的二维 128×128; 激活的 block=128 恰好等于 1×128 per-token tile。
- FP8 终点 (4.6e-4) 比 FP32 (1e-8) 差很多, 是因为这个玩具问题**没有噪声**, 量化噪声底裸露了出来; 真实 LLM 的 loss 噪声远大于它 (V3 报告相对误差 < 0.25%)。
- 没有 delayed scaling (用历史 amax)、没有 FP8 通信、没有 FP8 优化器状态。

## 常见误区
- "DeepSeek-V3 用 E4M3 前向 + E5M2 反向" —— 那是 Transformer Engine; V3 全程 E4M3。
- "block scaling 总是比 per-tensor 好很多" —— 没有 outlier 时几乎一样。
- "FP8 训练 = 权重存成 FP8" —— 权重的真身是高精度 master, FP8 只是每步的临时计算副本。

## 自测题
1. E4M3 最大值为什么是 448 不是 480? **答: 最高的尾数编码留给了 NaN。**
2. 梯度量级 1e-5, 不 scaling 直接转 E5M2 会怎样? **答: 最小 subnormal 1.5e-5, 基本全部冲成 0 或只剩 1 级。**
3. 为什么 per-token tile 能隔离 outlier? **答: scale 只在 128 个元素内共享, outlier 只压低自己那一块的分辨率。**
