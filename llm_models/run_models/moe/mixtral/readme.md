# Mixtral (Sparse MoE)

- Sparse Mixture-of-Experts Decoder: 每个 token 路由到 Top-K 专家.
- 总参数大, 推理时仅激活部分参数, 兼顾容量和效率.
- 对应论文: Mixtral of Experts (Jiang et al., 2024).

## 运行命令

### 推理

```bash
python -m llm_models.run_models.moe.mixtral.infer_mixtral
```

### 训练

```bash
python -m llm_models.run_models.moe.mixtral.train_mixtral
```
