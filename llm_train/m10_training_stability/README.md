# M10 — 训练稳定性: cosine / WSD · 梯度裁剪 · NaN guard

运行: `python -m llm_train.m10_training_stability.demo`

## 直觉
- **warmup**: Adam 的二阶矩估计开头不准, 先用小 lr。
- **cosine** 平滑降到 ~0, 但曲线形状依赖总步数 —— 想多训一点, 整条曲线都得换。
- **WSD** (warmup-stable-decay): 大部分时间 lr 恒定, 最后 ~10% 快速退火。稳定段与总步数无关 → 任何稳定段 checkpoint 都能继续训, 或分叉出一个 decay 分支来评测。
- **全局范数裁剪**: 偶发的梯度尖峰只缩长度、不改方向。
- **NaN guard**: 坏 step 直接丢弃, 参数零改动。

## 核心公式
`cosine: lr = base·(r + (1-r)·½(1+cos(π·p)))`; `WSD: lr = base (stable), base·(1-(1-r)·p) (decay)`;
`clip: g ← g·min(1, c/‖g‖₂)`, 范数取**所有参数**拼在一起的 L2。

## 运行后应该看到什么
```
step          = [0, 9, 30, 60, 89, 95, 99]
cosine ×1e-3  = [0.1, 1.0, 0.883, 0.413, 0.036, 0.008, 0.0]
WSD    ×1e-3  = [0.1, 1.0, 1.0, 1.0, 1.0, 0.5, 0.1]
总步数 100→200, 前 90 步 lr 最大变化 = cosine 5.96e-04,  WSD 0.00e+00
‖g‖ 裁剪前 → 后 = 111.83 → 5.00 (scale=0.0447);  方向 cos 相似度 = 1.000000
坏 step: 执行? / 参数被改? = False / False
```

## 与真实系统的差距
- 真实 WSD 的 decay 形状常用 `1-sqrt` 或指数; decay 段往往同时换成高质量数据。
- 分布式下范数要跨 TP/PP/ZeRO 分片聚合 (full_loop 演示了 ZeRO 版)。
- 真实的 spike 处理还包括: 回滚到上一个 checkpoint 并**跳过那段数据**、z-loss、QK-norm / QK-clip (→ m14)。

## 常见误区
- "裁剪阈值越小越稳" —— 长期处于被裁状态等于在偷偷降 lr。
- "逐参数裁剪也一样" —— 那会改变梯度方向。
- "先 clip 再查 NaN" —— NaN 的范数还是 NaN, clip 救不了它 (PyTorch 里缩放系数本身变 NaN, 污染全部参数)。

## 自测题
1. WSD 相对 cosine 最大的工程好处? **答: 稳定段与总步数解耦, 可随时续训 / 从任一 checkpoint 分叉做 decay。**
2. ‖g‖=111.83, c=5, 缩放系数? **答: 5/111.83 ≈ 0.0447。**
3. loss scale 回退和 NaN guard 的关系? **答: 同一个检测 (`has_overflow`), AMP 多做一步 "scale 减半"。**
