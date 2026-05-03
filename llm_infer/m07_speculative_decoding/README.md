# M07 — Speculative Decoding: 一步出多 token

## 痛点
decode 阶段每步只能产 1 个 token, 单步显存带宽利用率 < 5% (大模型权重读了一遍只用一次)。

## 思路 (Leviathan et al. 2023)
让一个**便宜的 draft model** 一口气 (autoregressive) 猜 K 个 token, 然后用
**昂贵的 target model** 一次 forward 校验这 K+1 个位置, 接受能匹配的前缀。

```
  step:  draft 提议             target 校验                  接受
  ─────  ──────────────────     ────────────────────────     ───────
   1     d_1, d_2, d_3, d_4     在位置 0..4 算 5 个 logits   若 logits 与 d_i 一致就接受
   2     从接受点起继续 draft   ...                          重复
```

理想情况: 一次 target forward → 接受 K 个 token, 加速比 ≈ K。
实际: 接受率 α (典型 0.6~0.8), 加速比 ≈ (1 + α + α² + ... + α^K) / 1 ≈ 2~3x。

## 接受规则
**严格匹配 greedy 等价**:
    target.argmax(logits[i]) == d[i]  → 接受 d[i]
    第一个不匹配处, 用 target.argmax 替换并停止接受
**采样等价 (rejection sampling)**:
    accept with prob min(1, p_target(d_i) / p_draft(d_i))

本模块实现 greedy 版本, 数值上**与 target 单独 greedy decode 完全一致**。

## 关键: 为什么 target forward 1 次能算 K+1 个 logits?
target 接受 (prompt + draft tokens) 作为输入做 prefill 风格的 forward,
因果 mask 保证位置 i 的 logits 只看到位置 ≤ i 的 token, 等价于"如果 i 是
最后一个 token, target 会输出什么 logits"。一次 forward 完成 K+1 个位置的
独立预测, 显存带宽几乎与单 token forward 一样 (decode 限于 weight load)。

## 运行
```bash
python -m llm_infer.m07_speculative_decoding.demo
```

输出:
- 与 baseline greedy 的等价性
- 平均接受 token 数 / step
- target 调用次数对比 (省了多少)
