# DPO — Direct Preference Optimization

```bash
python -m llm_finetune.run_finetune.dpo.train_dpo
```

**核心点 (代码中可对照):**
- `PreferenceDataGenerator` 产 (chosen, rejected) 共享 prompt 的偏好对
- `DPOTrainer.__init__` 自动 deepcopy + freeze + eval() 出 reference 模型
- `train_step` 做 4 次前向 (policy×2, ref×2)
- `DPOLoss` 形式: `-log σ(β · (Δ logratio_chosen - Δ logratio_rejected))`
- 监控 `reward_chosen / reward_rejected / reward_margin / accuracy`

**预期输出:**
- DPO loss 从 0.6931 (= log 2, 起点 σ=0.5) 单调下降到 ~0.002
- reward_margin 从 0 扩大到 ~+6.3
- accuracy = 1.00 (chosen 全部高于 rejected)
