# LoRA — Low-Rank Adaptation (PEFT)

```bash
python -m llm_finetune.run_finetune.lora.train_lora
```

**核心点 (代码中可对照):**
- `apply_lora` 把 `w_q / w_k / w_v / w_o` 替换为 `LoRALinear (r=8)`
- `mark_only_lora_as_trainable` 把基座全部 freeze, 仅 ~1.6% 参数可训
- 训练后 `merge_lora_weights` 把 BA 合并回 W → 推理零开销
- `get_lora_state_dict` 抽出 ~112 KB adapter, 部署时只需带这一份

**预期输出:**
- trainable 比例: 100% → 1.63%
- loss 极快收敛 (lr 3e-3 比 SFT 高 10x), 50 步从 ~250 → ~27
- 末尾打印 adapter 大小 + 合并 + 推理通过
