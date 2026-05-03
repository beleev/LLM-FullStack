# M05 — Radix Cache (SGLang 风格): 任意长度前缀共享

## m04 的局限
hash 缓存只能命中**完整 block**:
```
  USER_A: [10, 11, 12, 13, 20, 21]    block_size=4 → block 0 完整, block 1 残缺
  USER_C: [10, 11, 12, 13, 99, 99]    USER_C 与 USER_A 共享 4 token = 1 整 block
                                      但若是 5 token 重叠? hash 完全失效, 0 命中
```

## Radix Tree 思路
**用基数树 (radix / patricia trie) 存所有出现过的 token 序列, 边上是 token
子串 (而非单个 token), 节点上挂 KV cache 引用。**

```
                                   root
                                    │
                          ┌─────────┴─────────┐
                          │                   │
                       [10,11,12,13]      [50,51,52,53,60,61]
                          │                   │
              ┌───────────┴────────┐         (USER_X)
              │                    │
         [20,21,22,23,...]      [99,99,99,99]
         (USER_A 的剩余)         (USER_C 的剩余)
```

新请求来 → 从 root 出发, 沿边匹配 token, 在分叉点 split 节点。
能精确利用**最长公共前缀**, 而不仅是 block 边界。

## 核心操作
- `match(tokens) → (matched_node, matched_len)`: 走树, 在最长公共前缀处停
- `insert(tokens, kv_handle)`: 把序列插入, 必要时 split
- `evict_lru(k)`: 按 LRU 时间戳驱逐 k 个不被引用的叶节点
- `lock(node)` / `unlock(node)`: 引用计数, 防止正在用的被驱逐

## 与 m04 对比
| 维度 | m04 hash | m05 radix |
|---|---|---|
| 命中粒度 | 完整 block | 任意长度 token |
| 数据结构 | dict | tree |
| 实现复杂度 | 低 | 中 |
| 命中率 (实测对话场景) | ~30% | ~70% |
| 用于 | vLLM | SGLang |

## 运行
```bash
python -m llm_infer.m05_radix_cache.demo
```
