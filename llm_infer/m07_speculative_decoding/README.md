# M07 — Speculative Decoding: 一次 target forward 出多个 token

## 直觉
decode 每步只产 1 个 token, 却要把整份权重读一遍 (带宽受限), 算力大量闲置。
让便宜的 **draft** 连猜 K 个 token, **target** 一次 forward 并行检查 K+1 个槽位:
猜对的前缀全收, 第一个猜错的位置由 target 纠正, 全对再白送 1 个 bonus。
每次 target 调用至少产出 1 个 token (不会比 baseline 差), 最多 K+1 个。

## 核心数据结构或公式
- 不变量: target 的 `kv` 恰好覆盖 `out[:-1]`; `out[-1]` 已确定但还没喂进去。
- 验证: `target.forward([out[-1], d_0..d_{K-1}], kv)` → logits (K+1, V), 因果 mask 让第 i 行只看到 `d_{<i}`。
- 回滚: `truncate_kv(kv, n_ctx + 1 + n)`, 被拒 draft 的 KV 直接截掉, **从不重新 prefill**。
  draft 侧同理 (`ModelDrafter` 用"已喂 token 与 out 的公共前缀"决定保留多少 KV)。
- greedy 规则 `accept_greedy`: `d_i == argmax(t_logits[i])` 则接受, 否则用 target 的 argmax 纠错并停。
- 采样规则 `accept_sampling`: 以 `min(1, p_t(d_i)/p_d(d_i))` 接受; 拒绝则从 `max(0, p_t − p_d)/Z` 重采样。
  无损性: P(输出 x) = p_d(x)·min(1, p_t/p_d) + P(拒绝)·残差(x) = min(p_t, p_d) + max(0, p_t − p_d) = p_t(x)。
- 期望产出: 每 token 接受率 α → 每次 target 调用 (1 − α^(K+1)) / (1 − α) 个 token。

## 运行后应该看到什么
`python -m llm_infer.m07_speculative_decoding.demo` (约 8 s)
```
[1][2] greedy, K=4, 生成 48 token (baseline target 调用 = 48)
  draft == target (上限)          target 调用 11 (4.36x), 每轮接受 4.00/4, draft 调用 40
  权重加噪 10% (模拟蒸馏 draft)    target 调用 26 (1.85x), 每轮接受 0.88/4, draft 调用 100
  只用前 2 层 (LayerSkip 式)      target 调用 36 (1.33x), 每轮接受 0.34/4, draft 调用 140
  独立随机 1 层小模型              target 调用 43 (1.12x), 每轮接受 0.14/4, draft 调用 168
[3] V=16, T=1.0, K=2, N=2500 条 × 4 token, 每轮接受 1.13/2
  plain target 采样       TV 0.035 0.031 0.038 0.032 | χ²  20.3  15.1  25.3  19.2
  投机采样                TV 0.021 0.027 0.025 0.027 | χ²   8.5  13.6   8.7  12.3
  错误规则: 全收 draft     TV 0.036 0.188 0.198 0.037 | χ²  21.7 584.7 585.6  19.8
```
assert: 四种 draft 的输出都与 `target.generate_greedy` 逐 token 相同; target 调用 = 1 次 prefill + 轮数;
投机采样在 4 个位置的边缘分布上 χ² < 37.7 (df=15 的 99.9% 分位) 且 TV 与 plain 采样噪声同量级;
"全收 draft"的错误规则在同一检验下 χ² > 377。精确分布靠枚举 1+16+256+4096 个前缀得到。
(第 1 个 token 来自 prefill、第 4 个多为 bonus, 都直接采自 target, 所以错误规则只在第 2、3 位露馅。)

## 与真实系统的差距
- 加速只按 target 调用数算, **没计 draft 开销**; 真实加速 ≈ 产出 / (1 + K·c), c = draft/target 单步耗时比。
- 无 batch: vLLM 里每条序列接受数不同, 需要 ragged 的 KV 回滚与 slot 回收。
- "权重加噪"的 draft 并不更便宜, 只用来模拟"蒸馏得不错"的接受率; 真 draft 是蒸馏小模型 / EAGLE 头 (m17)。
- 真实验证用 fused kernel 一次算 K+1 个位置; 这里是普通的 (K+1, ctx+K+1) dense attention。
- 树形 draft 见 m19。

## 常见误区
- "draft 差会让输出变差" —— 不会。输出 (greedy) 或输出分布 (sampling) 只由 target 决定, draft 只影响速度。
- "greedy 比对在采样时也能用" —— 不能。采样必须用 min(1, p_t/p_d) + 残差, 否则分布被 draft 污染 (demo [3] 的错误规则)。
- "拒绝后要重新 prefill" —— 不用, 截断 KV 即可; 前缀的 KV 与后面的 token 无关 (因果)。
- "每轮接受 0.88/4 = 接受率 22%" —— 链式接受一断全断, 0.88 对应的每 token 接受率 α≈0.5。

## 自测题
1. 为什么残差分布是 max(0, p_t − p_d) 而不是直接从 p_t 重采样?
   答: 直接采 p_t 会让 p_d 偏高的 token 被重复计入 (接受一次 + 重采一次), 总分布 ≠ p_t; 残差恰好补上 p_d 低估的那部分质量。
2. K=4、draft 完全猜不中时, target 调用数与 baseline 相比如何?
   答: 每轮仍产出 1 个纠错 token, 调用数 ≈ baseline (本 demo 43 vs 48), 只亏 draft 的计算。
3. 全接受时为什么能白送 bonus?
   答: 验证 forward 的第 K 行 logits 就是"看完全部 K 个 draft 后的下一个 token 分布", 已经算出来了, 不用白不用。
