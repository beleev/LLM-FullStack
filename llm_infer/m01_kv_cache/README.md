# M01 — KV Cache: 一切推理优化的起点

## 现象
Decoder-only LM 自回归生成时, 第 t 步计算 token t 的 attention 需要
所有过去 token 的 K, V。如果**每一步都重新算 K, V**, 复杂度是
**O(T²)**, 生成 1024 个 token 时计算量比 prefill 多两个数量级。

## 优化
**把 K, V 缓存下来, decode 时只算"新 token"的 K, V, 拼到尾部即可。**

```
  无 cache:                    有 cache:
                               ┌─[K_prev, V_prev]─┐ (T-1 个, 缓存)
  step t:                      │                  │
    Q,K,V = X · W              │                  │
    scores = Q @ K.T           Q_new = x_new · Wq
    out    = softmax · V       K_new = x_new · Wk → append
                               V_new = x_new · Wv → append
    O(t² · d)                  O(t · d)
```

复杂度从 **O(T²·d)** 降到 **O(T·d)** (单 step 看是 O(t·d), 累计还是 O(T²·d)
但已无重复计算; FLOPs 砍掉 一半到 90%, 取决于 prompt/output 比例)。

## 代价
**显存爆炸**: KV cache 占用 = `2 · n_layer · n_head · head_dim · T · sizeof(dtype)`
LLaMA-7B, T=4096, fp16 → 2 GB, 远超模型权重之外的可用显存。
后续模块 (PagedAttention / 量化 / 共享前缀 / offload) 全部围绕"如何
高效管理这块越来越大的 KV cache"展开。

## 运行
```bash
python -m llm_infer.m01_kv_cache.demo
```

输出会对比"无 cache 的暴力解码"与"带 cache 的增量解码"在
- 输出一致性 (logits max-abs-diff)
- 单步耗时
- 累计耗时
三个维度的差异。
