# Qwen3-Next (混合线性注意力)

- Gated DeltaNet (75% 层) + 全注意力 (25% 层) 的混合架构.
- DeltaNet: 固定大小状态矩阵替代 KV cache, delta rule 精准覆写 + α 门整体衰减.
- 线性层管流畅局部建模 (O(1) 状态), 全注意力层兜底长程精准检索.
- 对应: Gated Delta Networks (Yang et al., 2024) / Qwen3-Next (2025).
- 同路线: MiniMax Lightning Attention (7:1), Jamba (Mamba+Attn).

## 运行命令

### 推理

```bash
python -m llm_models.run_models.language_models.qwen3_next.infer_qwen3_next
```

### 训练

```bash
python -m llm_models.run_models.language_models.qwen3_next.train_qwen3_next
```
