# BERT

- Encoder-only Transformer, 双向自注意力.
- 预训练目标: Masked Language Modeling (MLM) + Next Sentence Prediction.
- 对应论文: BERT (Devlin et al., 2018).

## 运行命令

### 推理

```bash
python -m llm_models.run_models.language_models.bert.infer_bert
```

### 训练

```bash
python -m llm_models.run_models.language_models.bert.train_bert
```
