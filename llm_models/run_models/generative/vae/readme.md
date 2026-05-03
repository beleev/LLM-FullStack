# VAE

- 变分自编码器: Encoder 输出 (mu, logvar), 重参数化采样, Decoder 重建.
- Loss = 重建损失 + KL 散度.
- 对应论文: Auto-Encoding Variational Bayes (Kingma & Welling, 2013).

## 运行命令

### 推理

```bash
python -m llm_models.run_models.generative.vae.infer_vae
```

### 训练

```bash
python -m llm_models.run_models.generative.vae.train_vae
```
