# Full Engine — 把模块接成 mini-vLLM

## 直觉
前面每个模块单独解决一个瓶颈; 引擎只做一件事: 每个 step 把它们按顺序串起来 ——
**调度** (谁、算几个 token) → **前向** (KV 读写走分页 pool) → **采样** → **后处理** (登记前缀缓存、结束的还 block)。

## 核心数据结构或公式
| 文件 | 职责 | 复用的模块 |
|---|---|---|
| `engine.py` | `add_request / step / generate`, 50 行胶水 | m03 `Scheduler` (同一个类, 非拷贝), m10 `sample` |
| `model_runner.py` | 物理 KV pool `(L, num_blocks, bs, D)` + 分页前向 | m02 `write_kv / paged_attention` |
| (m02) `BlockManager` | block 记账、ref_count、LRU free_list | |
| (m04) `PrefixCache` | 链式 hash 索引, 随 block 被覆盖而失效 | |
| (m03+m06) `Scheduler` | 连续批、抢占、`chunked_prefill` 混批 | |

```
step():  batch = scheduler.schedule()                       # [(seq, n)]
         logits = runner.run(ids[:computed+n], block_table, start_pos=computed)   # 只算 n 个, 其余 KV 从 pool 读
         token  = sample(logits) if computed+n == num_tokens else None            # prefill 中途的 chunk 不采样
         scheduler.postprocess(batch, tokens)
对账:    Σ(每条序列 num_tokens-1) = tokens_computed + prefix_hit_tokens      (无抢占时)
```

## 运行后应该看到什么
`python -m llm_infer.full_engine.demo` (~1.5 s)
- [1] `step 1 [s0:P45 s1:P3]`, `step 2 [s0:D s1:P39 s2:P8]` —— prefill chunk 与 decode 同步混跑, 每步 ≤ 48 token;
  需要 KV 的 token 305 = 实算 233 + 前缀命中 72
- [2] 同样 5 条再来一遍: 实算 105 (第一轮 233) —— 已释放的 block 仍可命中, 且不触发 stale-entry 崩溃
- [3] 9 个 block 的 pool: 82 / 81 步, 抢占 4 次, 无活锁 (调度写错时这里会死循环)
- [4] 同 batch 三种采样参数; [5] 30 组随机配置 fuzz: 214 条请求、34 次抢占
- 全部断言: greedy 输出与 `TinyLM.generate_greedy` **逐 token 相同**, 无 block 泄漏

## 与真实系统的差距
- 逐序列循环前向; vLLM 把 batch 摊平成一个张量 + varlen FlashAttention + CUDA Graph (m11 / m12)
- 离线同步循环; 真实引擎有异步 API server、流式输出、detokenize 线程、多进程 worker (TP, m09)
- 未集成: m05 radix cache (与 m04 二选一, 接口不同: radix 按 token 粒度匹配并自己管 LRU)、m07/m17/m19 投机解码、m08 量化
- 抢占只有 recompute, 没有 swap

## 常见误区
- "分页 / 前缀缓存只是记账" —— 如果 KV 挂在 Sequence 上、命中了照样整段 prefill, 那确实只是记账。这里 KV 只存在 pool 里, 命中的 token 不做前向
- "引擎很复杂" —— 复杂度都在调度策略与 kernel 里, 主循环就四行
- "优化会改变输出" —— 这里所有优化都是**精确**的, 所以能用逐 token 相等做回归测试 (量化 / 稀疏注意力才是有损的)

## 自测题
1. 一条序列前缀命中 24 token, prompt 共 45 token, 第一步 `runner.run` 的 `start_pos` 和 Q 的行数是多少 (预算充足)? **答**: start_pos=24, Q 有 21 行; K/V 经页表读回 45 行。
2. 为什么 prefill 中途的 chunk 返回 `None` 而不采样? **答**: 它最后一个位置的 logits 预测的是 prompt 里已知的下一个 token, 不是新 token。
3. block 不够时, 什么样的调度会让引擎死循环? **答**: `waiting` 非空就只走 prefill 分支, 队首拿不到 block 时什么都没选中就返回, running 从不 decode → block 永不释放。
