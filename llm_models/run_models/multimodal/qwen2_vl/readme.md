# Qwen2-VL

- 视觉语言模型 (VLM): 视觉编码器 + LLM decoder, 支持原生分辨率输入.
- 动态分辨率: 不同尺寸图像生成不同长度的 visual tokens.

## 运行命令

### 推理

```bash
python -m llm_models.run_models.multimodal.qwen2_vl.infer_qwen2_vl
```

### 训练

```bash
python -m llm_models.run_models.multimodal.qwen2_vl.train_qwen2_vl
```
