# DiT — Diffusion Transformer (Peebles & Xie, 2023)

## 直觉

把 VAE latent 切成 patch 当 token, 用 Transformer 代替 UNet 预测噪声。时间步 t 和类别 y 是 **全局** 条件,
不值得用 cross-attention, 而是经 adaLN 变成每层的 (shift, scale, gate) 去调制 —— 且初始全为 0, 每个 block 从恒等映射起步。

## 核心公式

- DDPM 加噪: `x_t = √ᾱ_t·x_0 + √(1−ᾱ_t)·ε`, `loss = MSE(model(x_t, t, y), ε)`
- adaLN-Zero: `x = x + α·Attn((1+γ)·LN(x) + β)`, `(β,γ,α) = Linear(c)`, Linear 零初始化
- DDIM: `x̂_0 = (x_t − √(1−ᾱ_t)·ε̂)/√ᾱ_t`, `x_{t−1} = √ᾱ_{t−1}·x̂_0 + √(1−ᾱ_{t−1})·ε̂`
- CFG: `ε̂ = ε_uncond + s·(ε_cond − ε_uncond)`
- token 数 `N = (H/p)²`

## 运行命令

```bash
python -m llm_models.run_models.generative.dit.infer_dit
python -m llm_models.run_models.generative.dit.train_dit
```

## 运行后应该看到什么

`infer_dit`: 未训练 DiT 输出 **恰好全 0**; 随机化零初始化层后, 换 t 输出变化 4.258e-02、换 y 变化 3.957e-01;
DDIM 20 步 + CFG = 40 次前向, 模型看到的 t 从 999 到 0; `|x|max`: 不截断 63434.1, `clip_x0=3` → 3.00。

`train_dit`: loss 1.0016 → 0.0190。初始 ≈ 1 不是巧合: 输出为 0 → `MSE = E[ε²] = 1`。
数据是 **固定 batch** (x_t, t, ε 全被缓存), 下降 = 背下 4 个样本的 ε, 不等于学会去噪;
真实训练每步重采 t、ε, 此时对 x_0~N(0,I) 的理论下界是 `E_t[ᾱ_t] ≈ 0.5`。

## 常见误区

- "初始 loss 1.0 说明模型坏了": 恰恰说明零初始化生效。
- DDIM 从 t=999 起步时 cosine 调度的 `ᾱ_999 ≈ 2.4e-9`, 除以 `√ᾱ_t` 把误差放大约 2 万倍 → 需要截断 x̂_0 (`clip_x0`) 或避开最后几步。
- CFG 不是推理期技巧而已: 训练时必须以 `class_dropout` 概率把 y 换成 null 类, 否则没有 ε_uncond 可用。
- t 的量纲: 模型吃的是 [0, 1000) 的步数; 把归一化的 t∈[0,1] 直接喂进 sinusoidal 嵌入几乎没有区分度 (见 mmdit readme)。

## 自测题

1. patch_size 从 4 改成 2, 注意力计算量变几倍? —— token ×4, 注意力 ×16。
2. adaLN-Zero 里哪个量保证 block 初始是恒等映射? —— gate α=0 (调制 Linear 的 weight、bias 全零)。
3. guidance_scale=1 时 CFG 退化成什么? 要几次前向? —— 纯条件预测; 1 次 (实现里直接跳过 uncond 分支)。
