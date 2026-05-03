# M04 — Prefix Cache (hash 链): vLLM 风格的前缀复用

## 场景
多用户共享系统提示 (system prompt), 多轮对话共享前几轮历史, beam search
多分支共享 prompt → **同一段 KV 被算了 N 次**, 浪费惊人。

## 思路
**对每个完整 block 算 hash, 用 hash 作 key 去查"这个 block 之前算过没"。
hash 链式依赖前一 block, 保证语义正确。**

```
  block 0:  ids = [1,2,3,4]            hash_0 = H(parent=None, ids)
  block 1:  ids = [5,6,7,8]            hash_1 = H(parent=hash_0, ids)
  block 2:  ids = [9,10,11,12]         hash_2 = H(parent=hash_1, ids)
                                                    ↑
                          只有前面的 block 完全相同, 当前 block 才命中
```

新请求来了:
1. 把 prompt 切成 block_size 大小的 chunk
2. 顺序对每个 chunk 算 hash
3. 查 hash → 物理 block 命中: ref_count++, 直接复用
4. 一旦不命中, 后面所有 block 都视为 miss, 必须重新算

命中的 block **不需要重做 attention** — 它们的 K/V 已经在池子里了, 直接拿来用。

## 与 m02 的关系
本模块**复用** m02 的 `BlockManager`, 在外层加一个 `PrefixCache` 索引:
```
  PrefixCache:
      hash → physical_block_id     (查询)
      maintain ref_count via BlockManager.share_block
```

## 局限
- 只能命中**完整 block**, "前 17 个 token 相同" 的情况下只命中第 0 块
- 这正是 m05 Radix Cache 要解决的问题

## 运行
```bash
python -m llm_infer.m04_prefix_cache.demo
```
