# VAE — Latent Diffusion 的前置压缩器

## 直觉

在 512×512×3 像素上做扩散太贵, 先用 VAE 压到 64×64×4 (元素数 ÷48) 再扩散。
普通 AE 的 latent 分布是任意的, 下游扩散无从假设; VAE 让 encoder 输出一个高斯 `(μ, σ)`, 用 KL 把它拉向 N(0, I)。

## 核心公式

- 重参数化: `z = μ + σ·ε, ε ~ N(0, I)` —— 随机性挪到 ε, 梯度可以穿过 "采样"
- `loss = MSE(x̂, x) + kl_weight · KL`, `KL = −0.5 · Σ(1 + logσ² − μ² − σ²)`
- SD 的 `kl_weight ≈ 1e-6`: 几乎就是 AE, 只求 latent 别离 N(0, I) 太远

## 运行命令

```bash
python -m llm_models.run_models.generative.vae.infer_vae
python -m llm_models.run_models.generative.vae.train_vae
```

## 运行后应该看到什么

`infer_vae`: `x (2,3,64,64) → z (2,4,16,16)`, 12288 → 1024 维 (12×); `KL(N(1,I)‖N(0,I)) = 512.0 = 0.5 × 1024`; 未训练 encoder 的 KL = 85.03。

`train_vae` (固定 4 张低频色块图, **背 batch, 不代表泛化**): recon 0.3421 → 0.0457; KL 21.04 → 157.06。
KL **上升** 是预期的: `kl_weight=1e-4` 时, 模型用更尖的后验 (更大的 KL) 换重建质量是划算的。

## 常见误区

- "KL 越小越好": KL=0 意味着 z 与 x 无关 (posterior collapse), decoder 只能输出平均图。
- "推理时也要采样 z": 给扩散模型做压缩时通常直接取 μ (或采样后乘 scaling factor)。
- 数据用白噪声也能测: 白噪声不可压缩, recon 几乎不降 (旧脚本 1.06 → 0.97); 现在数据是低频色块且值域 [-1,1] 与 decoder 的 tanh 对齐。

## 自测题

1. 为什么不能直接对 `z ~ N(μ, σ²)` 采样后反传? —— 采样算子不可导; 写成 `μ + σ·ε` 后 z 对 μ、σ 可导。
2. μ=1, σ=1 的 1024 维后验, KL 是多少? —— 每维 0.5·μ² = 0.5, 共 512。
3. `levels=3` 时 64×64 的图压成多大的 latent? —— 8×8 (空间 ÷8)。
