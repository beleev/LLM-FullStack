# CLIP

- 图文对比学习模型: 图像编码器 + 文本编码器, 共享对比损失.
- 训练: InfoNCE / 对称交叉熵在图文对上做对比.
- 对应论文: Learning Transferable Visual Models From Natural Language Supervision (Radford et al., 2021).

## 运行命令

### 推理

```bash
python -m llm_models.run_models.multimodal.clip.infer_clip
```

### 训练

```bash
python -m llm_models.run_models.multimodal.clip.train_clip
```
