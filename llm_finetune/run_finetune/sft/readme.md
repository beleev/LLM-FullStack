# SFT — 全参监督微调

```bash
python -m llm_finetune.run_finetune.sft.train_sft
```

**核心点 (代码中可对照):**
- `InstructionDataGenerator` 把 prompt 段 label 设为 -100
- `SFTLoss` 用 `cross_entropy(ignore_index=-100)`, 等价于"loss 只在 response 上算"
- `Trainer` 完全复用 `llm_models.training`, 唯一变量是 LossComputer + DataGenerator

**预期输出:**
loss 从 ~250 下降到 ~210 左右 (50 步, lr=3e-4, 全参可训)。脚本末尾的
`assert last < first` 保证收敛。
