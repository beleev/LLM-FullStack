# GPT-3 (Decoder-only)

- Decoder-only Transformer, 自回归语言建模 (next-token prediction).
- 因果掩码保证位置 t 看不到 t+1..T-1.
- 对应论文: Language Models are Few-Shot Learners (Brown et al., 2020).

## 运行命令

### 推理

```bash
python -m llm_models.run_models.language_models.gpt3.infer_gpt3
```

### 训练

```bash
python -m llm_models.run_models.language_models.gpt3.train_gpt3
```
