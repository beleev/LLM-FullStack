# M20 — 分层 KV cache 卸载 (GPU → CPU → 磁盘)

## 直觉
多轮对话每一轮的 prompt 都是 "整段历史 + 新消息"。prefix cache (m04/m05) 能让历史部分免重算 ——
前提是它的 KV 还在 GPU 上。用户一多, GPU 装不下所有人的历史, LRU 把别人的 block 挤掉, 下一轮又得
从头 prefill。分层卸载的想法: **被挤掉的 block 不丢, 降级到更大更慢的一层; 再命中时搬回 GPU**。
搬运 1 token 的 KV 通常比重算它快一个数量级。但并非总是如此 —— 所以还要一个 "加载 vs 重算" 的判断。

> 本模块没有真的搬数据: **所有时间来自代价模型 (`CostModel` + `Tier` 的可见参数), 不是实测**。

## 核心数据结构或公式
```
block hash   h_i = sha256(h_{i-1} ‖ block_i.tobytes())        链式: h_i 标识 "到第 i 块为止的整个前缀"
tiers        [OrderedDict(hash → True)] × 层数                 每层一个 LRU, 容量 = capacity_blocks
驱逐 = 降级   GPU 满 → 最旧块进 CPU → CPU 满 → 进 disk → disk 满 → 丢弃     (_insert 递归)
命中 = 提升   从所在层删除, 放回 GPU 的 MRU 端                              (store)
查找          沿链逐块找所在层, 第一个全层 miss 处停 (KV 依赖完整前缀)       (lookup)

load_ms(tier, n)  = latency + n·block_size·kv_bytes_per_token / bandwidth
recompute_ms(n)   = n·block_size / prefill_tok_per_s
TTFT = recompute(miss tokens) + Σ_tier min(load, recompute)(该层命中的块)
交叉点 n* = latency / (每块重算时间 − 每块加载时间);  若每块加载 ≥ 每块重算, 永远重算
```
倒序 touch: 写入一条链时从尾到头 touch, 让链尾比链头先被驱逐 (近似 radix tree 的 "先驱逐叶子");
否则 LRU 会先踢掉 block 0, 整条链立刻全部失效。

## 运行后应该看到什么
```bash
python -m llm_infer.m20_kv_offload.demo     # < 1 s
```
代价模型参数: block=16, KV=131,072 B/token (LLaMA-3-8B fp16), 重算 8000 tok/s = 2.00 ms/block;
GPU 64 blocks; CPU 192 blocks / 25 GB/s / 0.2 ms (0.084 ms/block); disk 4096 / 3 GB/s / 2 ms (0.699 ms/block)。
```
[1] 8 users × 6 turns 轮流发言, 最终历史 ≈442 blocks (GPU 只放得下 64)
config            hit GPU  hit CPU hit disk    miss     mean TTFT(模拟)
GPU only            13.3%     0.0%     0.0%   86.7%          51.50 ms
GPU+CPU             13.3%    29.5%     0.0%   57.2%          34.81 ms
GPU+CPU+disk        13.3%    29.5%    41.4%   15.7%          19.50 ms
no cache (全量重算)                                           59.42 ms

[2] disk: n=1 load 2.70 vs recompute 2.00 ms → recompute;  n=2 load 3.40 vs 4.00 → load;  n* = 1.54 blocks

[3] 慢链路 remote (0.5 GB/s, 20 ms → 4.194 ms/block, 比重算 2.00 ms/block 还慢)
GPU+CPU (无慢层)            34.81 ms
+remote, 命中就加载          68.91 ms   ← 加了一层缓存反而更慢
+remote, 加载/重算取小        34.81 ms   (命中但选择重算 9456 tokens)
```
断言: TTFT 随层数单调改善; hit + miss == total (token 守恒); 每层不超容量; 盲目加载劣于无慢层;
带判断的策略逐请求 ≤ 全量重算。剩余 15.7% miss 是每轮的新消息 + 第一轮, 任何缓存都救不了。

## 与真实系统的差距
- 真实重算时间不是线性的 (attention O(T²)), 长前缀重算更贵 → 真实交叉点更偏向加载。
- 真实系统按层流水加载 (layer-wise, 边传边算), 加载与 miss 部分的计算可重叠; 这里是串行相加。
- 写入也有成本: GPU→CPU 的异步拷贝占带宽, 磁盘有写放大; 这里降级是免费的。
- 真实系统用 radix tree + 引用计数防止正在使用的 block 被驱逐 (m05); 这里用倒序 touch 近似。
- Mooncake 还有跨节点的分布式 KV 池 (RDMA), 以及按命中位置调度请求 (cache-aware routing)。

## 常见误区
- "缓存层越多越好" —— 慢层若每 token 加载时间 > 重算时间, 盲目加载会让 TTFT 变差 ([3])。
- "命中率高 = TTFT 低" —— 要看命中在哪一层、每次加载的固定延迟能否被摊薄 ([2])。
- "中间的 block 命中了也能用" —— 不能, KV 依赖完整前缀, 链断了后面全部作废。
- "hash 用 `bytes(list)` 就行" —— token id > 255 会抛错; 用 `np.asarray(ids, np.int64).tobytes()`。

## 自测题
1. 把 disk 的 latency 从 2 ms 提到 10 ms, 交叉点变成多少? **答**: 10 / (2.00 − 0.699) ≈ 7.7 blocks (≈123 tokens)。
2. 为什么三种配置的 GPU 命中率都是 13.3%? **答**: 下层只接收 GPU 驱逐出来的块, 不改变 GPU 层自身的 LRU 内容。
3. KV 换成 MLA (m18, 约 70 KiB/token → 此处为 1/1.9) 对卸载意味着什么? **答**: 每块加载时间同比缩小,
   交叉点更小、慢链路也可能变得划算 —— 这是 DeepSeek 能做磁盘 KV 缓存的原因之一。
