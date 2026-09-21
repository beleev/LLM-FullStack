# M19 — Tree Speculation: 一次 target forward 验一整棵 token 树

## 直觉
链式 draft (m07) 第一个 token 猜错, 后面 K−1 个全废。可 draft 的第 2、第 3 候选经常是对的 ——
那就别只押 top-1: 每个节点保留 top-k 个孩子, 长成一棵树, 让 target **一次 forward 验完所有分支**,
接受与 target greedy 一致的最长路径。decode 是带宽受限的, 多验十几个 token 几乎不加延迟。
Medusa、SpecInfer、EAGLE-2 都是这个套路, 区别只在树怎么长。

## 核心数据结构或公式
- 树: BFS 编号, `parents (n,)`, `depth (n,)`; 节点 0 = 根 = `out[-1]`。`widths=[3,2,1]` → 1+3+6+6 = 16 节点。
- 祖先矩阵 `anc (n, n)`: `anc[i] = anc[parent[i]] | onehot(i)`。
- tree mask `(n, n_ctx + n)`: 上下文列全 0 (可见); 树内列 `anc ? 0 : −inf` —— 兄弟/堂兄弟互相看不见。
- RoPE 位置 `n_ctx + depth`: 同深度节点**共享位置**, 因为它们是同一个位置的不同候选。
- 验证: `target.forward(tree_tokens, kv, positions=n_ctx+depth, mask=tree_mask)` → logits (n, V),
  第 i 行 = "上下文 + i 的祖先 + i" 之后的预测, 与顺序 forward 该路径完全相同。
- 接受 `accept_tree`: 从根出发, 当前节点的 target argmax 命中哪个孩子就走进去; 走不动时该 argmax 就是纠错/bonus token。
- `gather_kv`: forward 后 KV 多了 n 行, 只保留 `n_ctx + path` 这几行。K 是 RoPE 之后存的, 路径上 depth 连续, 挑行即等价于顺序 decode。

## 运行后应该看到什么
`python -m llm_infer.m19_tree_speculation.demo` (约 3 s)
```
[1] widths=[2,1], 上下文 3 个 token; parents = [-1, 0, 0, 1, 2]; RoPE 位置 = [3, 4, 4, 5, 5]
    节点0  ■ ■ ■ | ■ · · · ·
    节点1  ■ ■ ■ | ■ ■ · · ·
    节点2  ■ ■ ■ | ■ · ■ · ·
    节点3  ■ ■ ■ | ■ ■ · ■ ·
    节点4  ■ ■ ■ | ■ · ■ · ■
  路径 [0, 2, 4]: 树形 vs 顺序 max|Δlogits| = 9.5e-07;  gather 出的 KV vs 顺序 KV max|Δ| = 7.2e-07
[2] 12 个 prompt × 32 token, draft = target 权重加噪 10% (baseline target 调用 = 384)
                              target调用  每轮接受  token/target调用  验证token/轮  draft调用/轮
  chain K=3 (同 draft 调用数)        206      0.98       1.86            4          3.0
  chain K=15 (同验证 token 数)       180      1.40       2.13           16         15.0
  tree [3,2,1] (16 节点)            149      1.80       2.58           16          3.0
  同一状态下: 树 vs 它自己的 top-1 脊 = 1.80 vs 1.09 (每轮接受)
```
之后打印一棵真实的树, ✓ 标出被接受路径 (走的是第 2 个孩子那一支, top-1 脊在第一层就断了)。
assert: 树形/链式输出都与 `target.generate_greedy` 逐 token 相同; 树形 logits 与 gather 后的 KV 和顺序 forward 差 < 1e-4;
逐轮 树接受数 ≥ 同一棵树的 top-1 脊; 总体 tree 每轮接受 ≥ chain K=3 且 target 调用 ≤ chain。

## 与真实系统的差距
- 固定形状的树; EAGLE-2 按 draft 置信度动态展开并重排, 只把最可能的 ~60 个节点送去验证。
- draft 侧没做 KV gather, 每轮丢掉树的 KV、下轮重喂被接受的 token (不多花调用, 多花一点计算)。
- 只有 greedy 接受; 采样版需要多候选 rejection sampling (SpecInfer 的 multi-round / 无放回采样), 比链式复杂。
- 验证 16 个 token 在带宽受限的小 batch 下近乎免费; 大 batch 算力受限时树要缩小 —— 真实系统按负载调节树大小。
- 真实 kernel 直接吃稀疏的 tree mask (FlashInfer), 这里是 dense 加性 mask。

## 常见误区
- "兄弟节点位置应该依次 +1" —— 错。它们是同一位置的不同候选, 位置 = n_ctx + depth; 否则被接受路径的 KV 与顺序 decode 对不上。
- "接受后要重算 KV" —— 不用, gather 出路径那几行即可, demo [1] 验证了与顺序 forward 逐元素相等。
- "树总是赚的" —— 只赚 target 调用数; 验证 token 数是链的 4 倍, 算力受限时可能更慢。
- "chain K=15 验证量相同所以应该打平" —— 链越长越难全中 (α^k 衰减), 同样 16 个验证 token, 花在"宽"上比花在"深"上值, 且链要 15 次串行 draft。

## 自测题
1. widths=[3,2,1] 的树里, 节点 5 (父 1) 能看到节点 2 吗? 为什么必须看不到?
   答: 不能。节点 2 是节点 1 的兄弟, 属于另一条候选序列; 看到它就等于在一个不存在的上下文上算 logits, 验证结果无意义。
2. 为什么 gather KV 的行就够了, 不需要对 K 重新做 RoPE?
   答: K 存的是 RoPE 之后的值, 位置 n_ctx+depth 已烙进去; 被接受路径的 depth 是 0,1,2,… 连续的, 正好等于顺序 decode 的位置。
3. 同样验证 16 个 token, 为什么树 (每轮 1.80) 比 chain K=15 (1.40) 好?
   答: 链的第 k 个 token 被接受的概率 ≈ α^k, 后面的槽位几乎都浪费; 树把预算花在前几层的备选上, 把首槽命中从 top-1 提到 top-3。
