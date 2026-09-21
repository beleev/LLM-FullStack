# M02 — Paged Attention: KV cache 的"虚拟内存"

## 直觉
m01 给每条序列一块连续的 KV 数组, 只能按 `max_seq_len` 预留: 实际用 100 token 却占 4096 的位置 (内碎片),
序列长短不一、结束时间不同又留下填不上的空洞 (外碎片)。vLLM 论文测得传统方案只有 20~40% 的 KV 显存装着真数据。
PagedAttention 照搬操作系统分页: KV 切成定长 block, 每条序列一张页表, 物理 block 在全局 pool 里随便放、按需要、随时还。

## 核心数据结构或公式
```
block_table[seq] = [7, 2, 5]                      逻辑第 i 页 → 物理 block id
token pos 的 KV  = pool[block_table[pos // bs], pos % bs]      pool: (num_blocks, bs, D)
浪费上界         = 每条序列 < bs 个 slot (只有末页有空槽)
ref_count[blk]   > 1 ⇒ 多条序列共享同一物理 block (前缀共享, m04)
free_list        归还顺序 = 被复用顺序 = prefix cache 的 LRU 淘汰顺序; 复用前调 on_evict(blk)
```
| 文件 | 内容 |
|---|---|
| `block_manager.py` | 只记账: `allocate(seq, n, shared=[])` / `ensure_capacity` / `free` / `share_block` / `can_allocate` |
| `paged_attention.py` | 碰张量: `write_kv` (slot mapping) / `gather_kv` / `paged_attention` |

## 运行后应该看到什么
`python -m llm_infer.m02_paged_attention.demo`
- 三条序列 (10/6/3 token, bs=4) 拿到 `[0,1,2] [3,4] [5]`, pool 利用率 75%; seq 102 结束后 block 3,4 回到 free_list 队尾
- seq 101 从 12 长到 13 token → 新分配 1 个 block; 申请 100 token → `MemoryError` (m03 抢占的触发信号)
- `max |paged - dense| = 0.00e+00` (decode, T_q=1) 和 `T_q=5 (prefill chunk) = 0.00e+00`; 内碎片 1/12 slot, 上界 3

## 与真实系统的差距
- 这里 `gather_kv` 先把散落的 block 拷成连续数组再算; vLLM / FlashInfer 的 kernel 直接按页表跳着读, 零拷贝
- 真实 pool 形状是 `(num_blocks, bs, n_kv_head, head_dim)`, 每层一个, 启动时按"显存余量 × gpu_memory_utilization"一次分配
- `share_block` 从 free_list 捞 block 用的是 O(n) 的 `deque.remove`; vLLM 用双向链表 O(1)
- 没有 copy-on-write: 本库只共享写满的 block (不可变), beam search 那种共享未满 block 才需要 CoW

## 常见误区
- "分页让 attention 变快" —— 不, 单次 attention 反而因间接寻址略慢; 赚的是**显存利用率 → 更大 batch → 吞吐**
- "block 越小越好" —— 碎片少了, 但页表变长、kernel 间接访问变多; vLLM 默认 16
- "free 了 block 内容就没了" —— 只是 ref_count 归 0 进队列, 内容留到被覆盖为止, m04 正是靠这一点

## 自测题
1. bs=16, 100 条序列平均长 1000 token, 最多浪费多少 slot? **答**: 每条 < 16, 总共 < 1600 slot, 约 1.6%; 连续预留 4096 的方案浪费约 75%。
2. 为什么 `allocate` 要先 `share_block` 命中的 block, 再 pop 新 block? **答**: 命中的 block 可能 ref=0 正躺在 free_list 里, 先 pop 可能恰好把它当新 block 拿走并覆盖。
3. token 位置 37, bs=8, 页表 `[9,4,6,1,3]`, KV 在哪? **答**: 37//8=4 → 物理 block 3, slot 37%8=5。
