# GPT-3 — decoder-only Transformer

## 直觉
把一切 NLP 任务都写成 "续写": 模型只做一件事 —— 看着前面的 token 猜下一个。结构是 N 个相同的积木
`x = x + Attn(LN(x)); x = x + FFN(LN(x))` (Pre-LN), 因果 mask 保证位置 t 看不到 t 之后。

## 核心公式
- 目标: `L = -Σ_t log P(x_{t+1} | x_{≤t})`; 标签 = 输入右移一位。
- 初始 loss 应 ≈ `ln V` (均匀瞎猜)。V=1000 → 6.908。
- KV cache: 过去 token 的 K/V 不再变化, 缓存后每步只算 1 个新 token, 生成 C 个 token 从 O(C·T²) 降到 O(C·T)。

## 运行命令
```bash
python -m llm_models.run_models.language_models.gpt3.train_gpt3
python -m llm_models.run_models.language_models.gpt3.infer_gpt3
```

## 运行后应该看到什么 (实测, CPU)
- train: 参数量 1,836,032; `初始 loss 6.947 vs ln V = 6.908 | 最终 loss 0.071` (60 步)。
  两个 assert: `|初始 − ln V| < 0.5`, `最终 < 0.5 × 初始`。
- infer: Sin-PE 与 RoPE 各跑一遍: 改动后半段 token, 前半段 logits 逐位不变 (因果性);
  生成 200 token 有/无 cache 输出完全一致, 加速约 3.4x / 3.9x (数值随机器波动)。

## 常见误区
- "loss 从 6.9 降到 0.07 = 学会了语言": 不是。数据是**固定的一个随机 batch**, 下降只说明模型背下了它,
  验证的是 forward/backward/优化器通路。每步换新随机 batch 时 loss 会停在 ln V。
- "初始 loss 多大无所谓": 本库修复前是 ~255 (N(0,1) embedding + weight tying → 初始 logits 标准差 ≈ sqrt(D))。
  现在统一 `init_weights` N(0, 0.02²)。
- "KV cache 是近似": 不是, 是精确等价; 本脚本用 `torch.equal` 断言。Sin-PE 下要记得给位置编码加 offset。

## 自测题
1. 为什么 weight tying 会放大坏初始化的后果? —— logits = h·Eᵀ, E 既是输入又是输出; E~N(0,1) 时 h 的范数 ≈ sqrt(D), logits 标准差 ≈ sqrt(D), softmax 极尖。
2. 带 cache 解码第 t 步时, 因果 mask 取哪一块? —— 行 `[past:past+1]`、列 `[:past+1]`, 即 "新 query 看全部历史"。
3. Pre-LN 末尾为什么必须有 `ln_f`? —— 残差主路从不过 norm, 范数随层数累积, 不归一化直接进 lm_head 会让 logits 尺度失控。
