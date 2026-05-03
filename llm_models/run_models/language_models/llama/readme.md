# LLaMA

- Decoder-only Transformer 现代版: RMSNorm + RoPE + SwiGLU FFN.
- 对 GPT 的若干结构改进, 训练更稳定, 推理更快.
- 对应论文: LLaMA (Touvron et al., 2023).

## 运行命令

### 推理

```bash
python -m llm_models.run_models.language_models.llama.infer_llama
```

### 训练

```bash
python -m llm_models.run_models.language_models.llama.train_llama
```
