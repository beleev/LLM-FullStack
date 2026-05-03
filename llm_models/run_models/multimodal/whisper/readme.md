# Whisper (ASR)

- Encoder-Decoder Transformer 用于语音识别.
- Encoder 处理 mel spectrogram, Decoder 自回归生成文字.
- 对应论文: Robust Speech Recognition via Large-Scale Weak Supervision (Radford et al., 2022).

## 运行命令

### 推理

```bash
python -m llm_models.run_models.multimodal.whisper.infer_whisper
```

### 训练

```bash
python -m llm_models.run_models.multimodal.whisper.train_whisper
```
