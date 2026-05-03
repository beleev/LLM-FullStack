# M10 — Sampling Strategies: 把 logits 变成 token

decode 的最后一步: `logits (V,) → token id`。看似简单, 但策略选错会让模型"复读机"或"胡言乱语"。

## 算法清单
| 名字 | 公式 | 行为 |
|---|---|---|
| **greedy** | `argmax(logits)` | 确定性, 最高概率, 容易复读 |
| **temperature** | `softmax(logits / T)` | T<1 更尖, T>1 更平; T=0 退化 greedy |
| **top-k** | 只在 top-k 里采样, 其余置 -inf | 截断长尾噪声 |
| **top-p (nucleus)** | 累计概率到 p 为止, 余下置 -inf | 长尾自适应 |
| **min-p** | 概率 < min_p × max_prob 的全砍 | top-p 的简化, 对低熵分布更稳 |
| **repetition penalty** | 对已出现 token 的 logits × (1/penalty) | 缓解复读 |

## 组合顺序 (vLLM/SGLang 标准)
```
  raw logits
      │
      ▼  apply repetition penalty
      │
      ▼  divide by temperature
      │
      ▼  top_k 截断
      │
      ▼  top_p 截断
      │
      ▼  min_p 截断
      │
      ▼  softmax
      │
      ▼  multinomial 采样
```

## 与 nano-vllm 的对比
nano-vllm 的 `sampler.py` 用了 **Gumbel-Max trick**:
```python
probs / Gumbel(0,1) → argmax
```
等价于 `multinomial(probs)` 但全程 element-wise + argmax, 一次 kernel 调用搞定 GPU 友好。本模块两种都实现, 数值与统计验证。

## 运行
```bash
python -m llm_infer.m10_sampling.demo
```
