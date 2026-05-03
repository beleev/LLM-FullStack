# Transformer (Encoder-Decoder)

- 原始 Encoder-Decoder Transformer, 用于翻译等 seq2seq 任务.
- Encoder 双向注意力 + Decoder 因果注意力 + Cross-Attention.
- 对应论文: Attention Is All You Need (Vaswani et al., 2017).

## 运行命令

### 推理

```bash
python -m llm_models.run_models.language_models.transformer.infer_transformer
```

### 训练

```bash
python -m llm_models.run_models.language_models.transformer.train_transformer
```
