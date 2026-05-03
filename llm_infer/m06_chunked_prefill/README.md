# M06 — Chunked Prefill: 长 prompt 分片

## 痛点
prefill 一次吃整段 prompt → 显存峰值与 prompt 长度成正比。
prompt = 32k token 时, 单步 attention 矩阵 32k × 32k, 一张 H100 都吃不下。

## 思路
**把 prompt 切成 chunk_size (例 512) 一段, 顺序 prefill。**
每段计算 attention 时, K/V 范围是 "已经 prefill 过的所有 token", 但
**Q 只覆盖本 chunk** → 显存 ~ chunk_size × ctx_len, 而不是 ctx_len²。

```
  full prefill (T=8000):
    Q: 8000 × D    K: 8000 × D    attn: 8000 × 8000  ← 64 M 元素
    显存峰值 ~ T²

  chunked prefill (chunk=1000, 8 chunks):
    chunk 0: Q 1000×D  K 1000×D    attn 1000×1000  =  1M
    chunk 1: Q 1000×D  K 2000×D    attn 1000×2000  =  2M
    ...
    chunk 7: Q 1000×D  K 8000×D    attn 1000×8000  =  8M     ← 峰值
    总 FLOPs 不变, 但峰值显存只有 1/8
```

## 额外好处
- **与 decode 混批**: prefill chunk 与正在 decode 的 batch 一起跑, GPU 利用率更稳
- **更早响应 TTFT**: 第一段 prefill 完就能开始 decode, 不必等整段

## 接口
- `chunked_prefill(model, prompt_ids, chunk_size) → kv_cache`
- 输出 KV 与一次性 prefill 等价 (数值验证)

## 运行
```bash
python -m llm_infer.m06_chunked_prefill.demo
```
