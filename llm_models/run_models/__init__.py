"""
可运行示例 — 布局: run_models/<category>/<model>/{infer_X.py, train_X.py, readme.md}

- infer_X: 最小前向/生成 + 断言 (KV cache 一致性、形状、数值性质)
- train_X: 在固定的合成 batch 上训几十步, 断言 "初始 loss ≈ ln V" 且 loss 下降 (= 能记住这批数据)
- readme : 直觉 / 核心公式 / 运行后应该看到什么 / 常见误区 / 自测题

运行: python -m llm_models.run_models.<category>.<model>.train_X   (CPU, < 30s)
"""
