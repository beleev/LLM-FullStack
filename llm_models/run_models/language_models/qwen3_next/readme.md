# Qwen3-Next — 混合线性注意力

## 直觉
不是每一层都需要完整注意力。75% 的层用 Gated DeltaNet: 一个固定大小的状态矩阵 S 当 "记忆", 每来一个 token 就
"先擦掉 k 方向的旧值, 再写入新值"; 25% 的层保留全注意力兜底精准检索。层排布 `[Δ Δ Δ A Δ Δ Δ A …]`。

## 核心公式
- delta rule: `S ← α·(S − β·k·(kᵀS)) + β·k·vᵀ`, 读出 `o = Sᵀq`; α 是遗忘门, β 是写入强度, q/k 做 L2 归一化保证收缩。
- 缓存: attn 层 `T × 2·Hkv·Dh` (随 T 增长); delta 层 `H × Dh × Dh` (与 T 无关)。
- 同一个 cache dict 协议: attn 层往里放 `k/v`, delta 层往里放 `state`, 主干循环不区分层类型。

## 运行命令
```bash
python -m llm_models.run_models.language_models.qwen3_next.train_qwen3_next
python -m llm_models.run_models.language_models.qwen3_next.infer_qwen3_next
```

## 运行后应该看到什么 (实测, CPU)
- train (4 层 = 3Δ + 1A): `初始 loss 7.058 vs ln V = 6.908 | 最终 loss 0.049`。
- infer (8 层 = 6Δ + 2A, 参数量 1,905,840):
  - `[2]` 因果性: `|logits(全长)[:8] − logits(截断到 8)| = 4.8e-07`。
  - `[3]` 缓存元素数 T=8 → T=32: attn 层 4096 → 16384 (×4), delta 层 49152 → 49152 (不变)。
  - `[4]` 生成 60 token 有/无 cache 一致, 加速约 7.9x。

## 常见误区
- "线性注意力的缓存一定更小": 看 T。本例 T=32 时 delta 状态 (49152) 反而比 attn cache (16384) 大; 优势在 T 很大时才出现 —— O(1) vs O(T)。
- "DeltaNet 没有 mask 所以会偷看未来": 递推只从过去流向未来, 天然因果 (脚本第 2 项断言)。
- "init_weights 可以无脑套": 它会把 α 门的 bias 清零; 本模型在其后把 bias 重新设回 +2 (sigmoid ≈ 0.88, 初期偏向记住)。
- 教学版 DeltaNet 是逐 token 的 Python 循环, 训练很慢; 真实实现用 chunk 并行 kernel。loss 下降仍然只是背诵固定 batch。

## 自测题
1. T 多大时本例 attn 层总缓存超过 delta 层? —— attn: 2 层 × T × 2·2·32 = 256T; delta: 49152 → T > 192。
2. 为什么无 cache 时本模型的生成比纯注意力模型更吃亏? —— DeltaNet 每步都要从 token 0 重新递推整个前缀 (Python 循环 O(T)), cache 后每步只推 1 步。
3. 为什么 q/k 要 L2 归一化? —— `I − β·k·kᵀ` 在 |k|=1、β∈(0,1) 时是收缩映射, 状态不会数值爆炸。
