# MMDiT (Multimodal DiT)

- 多模态 Diffusion Transformer, 同时处理图像 patch 与文本 token.
- Stable Diffusion 3 的核心架构.
- 对应论文: Scaling Rectified Flow Transformers for High-Resolution Image Synthesis.

## 运行命令

### 推理

```bash
python -m llm_models.run_models.generative.mmdit.infer_mmdit
```

### 训练

```bash
python -m llm_models.run_models.generative.mmdit.train_mmdit
```
