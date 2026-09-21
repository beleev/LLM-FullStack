# M05 — Radix Cache (SGLang RadixAttention): 任意长度前缀共享

## 直觉
多轮对话、few-shot、共享 system prompt 的请求, 前缀大量重复。把所有见过的 token 序列存进一棵
**基数树** (边上是一段 token, 不是单个 token), 新请求从 root 沿边走到走不动为止 —
走过的部分就是最长公共前缀, 它的 KV 直接复用, 只 prefill 剩下的后缀。

## 核心数据结构或公式
```
RadixNode: edge_tokens [t0..tk]   父→本节点这条边上的 token
           slots       [s0..sk]   等长; 每个 token 的 KV 在 pool 里的槽位 (SGLang 的 TreeNode.value)
           children {首token: node}, ref_count, last_used
```
- `match(tokens) → (node, slots)`: 最长前缀匹配; 停在边中间就 **split**, 让 node 恰好代表命中前缀。
- `insert(tokens, slots) → n`: 返回树里已有的前缀长度 n; `slots[:n]` 是重复 KV, 调用方释放。
- `_split(child, n)`: `(edge_tokens, slots)` 同步切两半; 新中间节点**继承 ref_count**。
- `lock_path / unlock_path`: node→root 整条路径 ref_count ±1。
- `evict(n)`: 只驱逐 `ref_count==0` 的**叶子**, 按 last_used 最旧优先; 父节点变成叶子后才轮到它。
- 不变量: `树内 token 数 + 空闲槽位数 == pool 大小`。

**与 m04 (block hash) 的粒度差别**: hash 以整 block 为键, 公共前缀 n 只能命中 `⌊n/B⌋·B` 个 token;
radix 命中全部 n 个。两者数据结构和接口不同 (m04 是 hash→block 的 dict, 这里是 token→slot 的树),
本模块**不是** m04 的即插即用替换, demo 里只按 m04 的粒度规则算了一个对照数字。

## 运行后应该看到什么
```bash
python -m llm_infer.m05_radix_cache.demo      # ~0.4 s
```
```
[1] B=[1,2,3,4,9,9] 插入后 [1..8] 被切成 [1,2,3,4] + [5,6,7,8], 新分支 [9,9] 只占 slots 204,205
    B 与树重复的前缀长度 = 4; match [1,2,3,9] → 命中 3 个 slots=[100,101,102]
[2] 随机 300 条请求: match 长度 == 暴力 LCP (300/300); 树内 1759 / 空闲 2337 / pool 4096
[3] 总 token 3189: radix 命中 1430 (44.8%)   block-hash 粒度(B=4) 命中 1068 (33.5%)
[4] 锁 5 条路径后 evict: 1759 → 49 (= 锁定路径并集); 解锁后全部驱逐: 0 / 4096
```
断言: match 长度 == 对全部历史序列暴力求 LCP; 命中 slots 里存的 token == 查询前缀; split 后解锁
ref_count 全归零; 驱逐释放的槽位与锁定槽位不相交; 剩余节点全部 ref_count>0; 每一步槽位守恒且无重复释放。

## 与真实系统的差距
- slots 在这里只是整数; SGLang 里是指向 `token_to_kv_pool` 的索引张量, 命中后直接拼进请求的 KV 索引表。
- SGLang `page_size>1` 时 match 也会对齐到页边界, 此时粒度优势缩小到"页内"; vLLM 的 APC 是 block hash。
- evict 每次 O(N) 扫全树 (代码里 `ponytail:` 注释); SGLang 用按访问时间的小顶堆。
- 没有 cache-aware 调度: SGLang 会优先调度命中前缀最长的请求 (longest-prefix-first), 让命中率更高。
- 真实负载的命中率取决于流量形态; [3] 的 44.8% / 33.5% 只对这份合成负载成立。

## 常见误区
- "match 是只读操作": 不是, 停在边中间会 split, 树结构会变 (token 总数不变)。
- "split 出来的中间节点 ref_count 从 0 开始": 错。锁着下游节点的请求也经过它, 不继承的话它会被误驱逐,
  而且 unlock 时减成负数。
- "驱逐可以挑任意 ref==0 的节点": 只能挑叶子。中间节点的 KV 没了, 子孙的 KV 就失去了前缀, 留着也不能用。
- "insert 传进去的 slots 都进了树": 只有未命中的后缀进树, 前 n 个是重复的, 不释放就泄漏。

## 自测题
1. **两条序列公共前缀 6 个 token, block_size=4。block hash 与 radix 各能复用几个?**
   hash: ⌊6/4⌋·4 = 4; radix: 6。
2. **请求 R 锁住路径 root→[1,2,3,4]→[9,9]。此时另一请求 match [1,2,7], 树怎么变, ref_count 怎么变?**
   [1,2,3,4] 被切成 [1,2] (新, 继承 ref=1) + [3,4] (ref=1); R 解锁时沿 parent 链三个节点各减 1, 全部归零。
3. **为什么 LRU 时间戳要沿 match 路径全部刷新, 而不是只刷新终点?**
   保证父节点不比子节点旧; 否则子节点被驱逐后, 刚变成叶子的父节点会因为时间戳旧而被立刻驱逐, 丢掉热前缀。
