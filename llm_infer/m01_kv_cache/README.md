# M01 — KV Cache: 一切推理优化的起点

## 直觉
自回归生成第 t 步只新增 1 个 token, 但 attention 要用到全部历史 token 的 K/V。
因果 mask 下历史 token 的 K/V **不会因为后面来了新 token 而改变**, 所以没必要每步重算:
存下来, 每步只算新 token 的 q/k/v, 把 k/v 追加到尾部。用显存换计算。

## 核心数据结构或公式
```
kv_cache = [(K, V)] × n_layer          K, V: (T, D), K 存的是 RoPE 之后的值
decode:  q,k,v = x_new·W   (1, D)
         K = concat(K, k)  (T+1, D)     ← 只有这一步在增长
         out = softmax(q·Kᵀ/√d)·V       (1, T+1) 的分数, 而不是 (T+1, T+1)
```
- 过模型的 token 数: 无 cache `Σ_t t = O(T²)`, 有 cache `T`。
- 显存: `KV bytes = 2 · n_layer · T · D · sizeof(dtype)` (多头时 D = n_kv_head · head_dim)。
  LLaMA-7B fp16: 2·32·4096·2 B = 0.5 MiB/token, T=4096 → 2 GiB/请求。

## 运行后应该看到什么
```bash
python -m llm_infer.m01_kv_cache.demo      # ~0.2 s
```
```
[1] token 逐个一致 = True, logits max-abs-diff (32 步) = 2.80e-06
[2] cache 内 token 数 T = 37; 2·L·T·D·4 = 75776 B = 实际 nbytes 75776 B
    LLaMA-7B fp16: 0.50 MiB/token, T=4096 → 2.00 GiB
[3] 累计过模型的 token 数: 无 cache 688, 有 cache 37 → 18.6x
[4] 实测: 无 cache 单步 0.37 → 0.53 ms 随 t 上涨; 有 cache 恒定 ~0.25 ms; 总耗时 15.8 / 8.4 ms → 1.9x
```
断言: token 逐个相等; logits 差 < 1e-4; 公式字节数 == numpy nbytes; token 计数 688 / 37。
[4] 的毫秒数每台机器不同, 只看趋势 (一条线性上升, 一条水平)。

## 与真实系统的差距
- 这里每步 `np.concatenate` 会**整块拷贝** KV (O(T) 拷贝); 真实系统预分配显存, 原地写入 —
  vLLM / SGLang 用分页 block pool (m02), TensorRT-LLM 也是 paged KV。
- TinyLM 单头、无 batch、float32 (所以 demo 里 sizeof=4); 真实模型是 fp16/bf16 + GQA
  (KV 头数 < Q 头数, KV 直接缩小 4–8×) 甚至 fp8 KV (m08)。
- 有 cache 后 decode 变成**访存瓶颈**: 每步要把全部权重 + 全部 KV 读一遍只为算 1 个 token,
  这才引出 continuous batching (m03)、投机解码 (m07)。

## 常见误区
- "有 cache 后 decode 是 O(1)": 不是。q 仍要和 T 个 key 做点积, 单步 O(T·d); 省掉的是对历史 token 重跑所有层。
- "cache 存 Q 也有用": 没用。历史 token 的 q 只用于它自己那一步的输出, 之后再也不会被读。
- "K 存 RoPE 之前还是之后无所谓": 本库存 RoPE 之后的 K, 所以**截断 / 重排 cache 时位置就烙死了**
  (m16 attention sink 要重新编号就得特殊处理)。
- 实测只快 ~1.9× 而 token 数差 18.6×: T=38 时每次 numpy 调用的固定开销占主导, 不是 cache 没用。

## 自测题
1. **为什么历史 token 的 K/V 可以复用, 双向 (BERT 式) attention 行不行?**
   因果 mask 保证第 i 个 token 的各层 hidden 只依赖 ≤ i 的 token, 新 token 不会改变它; 双向 attention 下每个 token 的 hidden 依赖全序列, 新 token 一来全部失效, 不能缓存。
2. **LLaMA-7B fp16, 80 GB 卡上权重占 14 GB, 余下显存全给 KV, 最多同时放多少个 T=4096 的请求?**
   每请求 2 GiB, (80−14)/2 ≈ 33 个 — 这就是为什么 KV 显存直接决定吞吐上限。
3. **demo 里 cache 内 token 数为什么是 37 而不是 38?**
   最后一个生成的 token 只被采样出来, 还没作为输入喂回模型, 它的 K/V 尚未计算。
