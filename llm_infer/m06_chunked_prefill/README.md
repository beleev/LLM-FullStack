# M06 — Chunked Prefill: 把长 prompt 切块, 和 decode 混在一个 batch 里

## 直觉
prefill 是算力瓶颈 (一次吃几千 token), decode 是访存瓶颈 (每步每条 1 token)。prefill 优先的调度器来一条长 prompt,
所有正在 decode 的用户就得停下来等它整段算完 —— 表现为输出"卡一下"(TBT 尖刺)。
Sarathi 的做法: 每步固定 token 预算 B, 先给每条 decode 1 个 token, 剩下的预算切一块 prefill 塞进同一个 batch。

## 核心数据结构或公式
```
每步 batch = [decode × k 条 (各 1 token)] + [prefill chunk (≤ B - k token)]
step_ms ≈ fixed_ms + per_token_ms × batch_tokens        → TBT 上限 = step_ms(B)
分块等价性: 第 j 块的 Q (chunk, D) 对 K[0 : 已缓存+chunk] 做因果 attention, 前面 token 的 KV 不依赖后面 → 结果与整段相同
attention 分数矩阵峰值: T×T → chunk×T
```
实现: `chunked_prefill()` 就是对 `TinyLM.forward(ids_chunk, kv)` 的循环; 混批调度是 m03 `Scheduler(chunked_prefill=True)` 本身。

## 运行后应该看到什么
`python -m llm_infer.m06_chunked_prefill.demo`
- chunk ∈ {8,16,32,64}: logits / KV 与整段 prefill 的 max|Δ| ≤ 2.86e-06 (chunk=8; 其余为 0, 差异只来自 fp32 matmul 分块求和顺序); 分数矩阵峰值 512 / 1024 / 2048 / 4096
- 负载: 4 条在 decode, 第 10 步来一条 1024-token prompt; **代价模型** `step_ms = 20 + 0.25 × tokens` (非实测)

| 策略 | 每步最大 token | decode TBT p50 / max | 长请求 TTFT |
|---|---|---|---|
| prefill 优先 | 1024 | 21.0 / **297.2 ms** | 276 ms |
| 分块 B=512 | 512 | 21.0 / 148.0 ms | 319 ms |
| 分块 B=128 | 128 | 21.0 / **52.0 ms** | 445 ms |

## 与真实系统的差距
- 延迟来自线性代价模型; 真实 prefill 还有 attention 的 O(T²) 项, 且每个 chunk 要重读前面所有 KV (chunk 越小总开销越大)
- vLLM V1 默认开启, 旋钮是 `max_num_batched_tokens`; SGLang 是 `chunked_prefill_size`; 混批依赖 varlen attention kernel
- 分块与 P/D 分离 (m15) 是同一问题的两种解法: 一个在时间上错开, 一个在空间上隔离

## 常见误区
- "分块是为了省显存" —— 激活峰值确实降了, 但主要目的是**压 TBT 尖刺**
- "分块白赚" —— 长请求自己的 TTFT 变长 (276 → 445 ms), B 是 TBT 与 TTFT 之间的旋钮
- "chunk 之间要重算前面的 token" —— 不用, 前面 chunk 的 KV 已在 cache 里, 只是 Q 分批算

## 自测题
1. B=256, 当前 10 条在 decode, 这一步 prefill chunk 最多几个 token? **答**: 246。
2. 为什么分块 prefill 的结果与整段完全相同? **答**: 因果 mask 下 token i 的输出只依赖 0..i; 分块只是把 Q 的行分批算, K/V 仍是完整前缀。
3. 总耗时为什么分块反而略短 (1113 vs 1133 ms)? **答**: 代价模型里每步有固定开销, 混批把 prefill 搭进本来就要跑的 decode 步, 少了独立的 prefill 步; 真实系统里小 chunk 重读 KV 的开销会抵消一部分。
