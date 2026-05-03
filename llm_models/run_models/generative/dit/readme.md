# DiT (Diffusion Transformer)

- 图像扩散模型, 用 Transformer 替代 UNet 作为去噪骨干.
- 条件注入: AdaLN-Zero 把 timestep / class label 注入每层.
- 对应论文: Scalable Diffusion Models with Transformers (Peebles & Xie, 2023).

## 运行命令

### 推理

```bash
python -m llm_models.run_models.generative.dit.infer_dit
```

### 训练

```bash
python -m llm_models.run_models.generative.dit.train_dit
```
