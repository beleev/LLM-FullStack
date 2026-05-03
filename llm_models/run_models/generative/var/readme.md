# VAR (Visual AutoRegressive)

- Visual AutoRegressive 模型: 多尺度自回归 (next-scale prediction) 而非 next-token.
- 比像素级 AR 快, 比 diffusion 简单.
- 对应论文: Visual Autoregressive Modeling (Tian et al., 2024).

## 运行命令

### 推理

```bash
python -m llm_models.run_models.generative.var.infer_var
```

### 训练

```bash
python -m llm_models.run_models.generative.var.train_var
```
