# MTP (Multi-Token Prediction)

- LLaMA 主干 + 串行 MTP 级联: 位置 i 同时预测 t+1, t+2, ..., t+1+K.
- 训练信号更密 (K+1 倍监督) + 表征向前规划 + 推理免费拿投机解码草稿.
- Embedding 与 lm_head 与主干共享, 每级只新增一个拼接投影 + 一个 Block.
- 对应论文: Gloeckle et al. 2024 (并行版) / DeepSeek-V3 2024 (串行版, 本实现).

## 运行命令

### 推理

```bash
python -m llm_models.run_models.language_models.mtp.infer_mtp
```

### 训练

```bash
python -m llm_models.run_models.language_models.mtp.train_mtp
```
