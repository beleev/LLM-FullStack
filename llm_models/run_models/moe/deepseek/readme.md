# DeepSeek (MoE + MLA)

- DeepSeek 架构: Multi-head Latent Attention (MLA) + Sparse MoE.
- MLA: 用低秩压缩降低 KV cache 内存; MoE: 路由到部分专家以扩展容量.
- 对应论文: DeepSeek-V2 / DeepSeek-V3.

## 运行命令

### 推理

```bash
python -m llm_models.run_models.moe.deepseek.infer_deepseek
```

### 训练

```bash
python -m llm_models.run_models.moe.deepseek.train_deepseek
```
