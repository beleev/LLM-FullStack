# M02 — Paged Attention: 显存的"虚拟内存"

## 痛点
m01 的 KV cache 是**连续 numpy 数组**, 每个序列预先按 `max_seq_len` 分配。
这带来两个问题:
1. **内部碎片**: 实际只用 100 token 却占 4096 的空间, 显存利用率 < 5%
2. **外部碎片**: 不同序列长短不一, 中途有序列结束时, 难以复用空洞

## vLLM 的方案: PagedAttention
**把 KV cache 切成定长 block (典型 16~256 token), 每个序列维护自己的"页表"
`block_table`, 物理 block 散落在大池子里, 按需分配/归还。**

```
  逻辑视图 (序列 A 视角):           物理视图 (block pool):
  ┌──────────────────────────┐     ┌────────────────────────────┐
  │ token 0..15  │ block_id=7│     │ block 0  block 1  block 2 ...
  │ token 16..31 │ block_id=2│     │  [   ]    [seqB]    [seqA] │
  │ token 32..47 │ block_id=5│     │ block 3  block 4  block 5  │
  └──────────────────────────┘     │  [seqB]    [   ]    [seqA] │
            │                       │ block 6  block 7  ...      │
            └─→ block_table=[7,2,5] │  [seqC]    [seqA]          │
                                    └────────────────────────────┘
```

## 收益
- 显存利用率从 < 50% → > 90% (vLLM 论文核心数字)
- 多序列共享池子, 短序列释放的 block 立即给长序列
- 跨序列共享前缀 (m04/m05) 直接靠 ref_count 在 block 层面实现, 无需复制

## 接口
- `BlockManager.allocate(seq_id, n_tokens) → block_table`
- `BlockManager.append(seq_id) → maybe new block`
- `BlockManager.free(seq_id)`
- `paged_attention(q, k_cache_pool, v_cache_pool, block_table, ctx_len)`

## 运行
```bash
python -m llm_infer.m02_paged_attention.demo
```
