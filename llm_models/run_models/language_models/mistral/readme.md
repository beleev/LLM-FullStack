# Mistral

- LLaMA 骨架 + Sliding Window Attention (SWA): 每个位置只看最近 W 个 token.
- 注意力 O(T^2) → O(T·W), 推理 KV cache O(T) → O(W) (rolling buffer).
- 感受野不被掐断: 信息跨层接力, L 层理论感受野 ≈ L·W.
- 对应论文: Mistral 7B (Jiang et al., 2023). 后继者: Gemma 2/3、GPT-OSS.

## 运行命令

### 推理

```bash
python -m llm_models.run_models.language_models.mistral.infer_mistral
```

### 训练

```bash
python -m llm_models.run_models.language_models.mistral.train_mistral
```
