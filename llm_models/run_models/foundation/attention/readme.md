# Multi-Head Attention

- 演示 MultiHeadAttention 的最小自注意力前向。
- 对应论文: Attention Is All You Need (Vaswani et al., 2017).
- 性质: 输出形状与输入完全一致, 方便堆叠多层 + 残差连接.

## 运行命令

### 推理

```bash
python -m llm_models.run_models.foundation.attention.infer_attention
```

### 训练

```bash
python -m llm_models.run_models.foundation.attention.train_attention
```
