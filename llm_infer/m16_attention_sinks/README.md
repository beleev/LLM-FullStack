# M16 — Attention Sinks / StreamingLLM: 无限流下的有界 KV cache

## 直觉
KV cache 随上下文线性增长, 流式场景 (长对话 / 实时字幕) 迟早爆显存, 而且位置会超过模型的
max_seq_len。只留最近 W 个 token 的滑动窗口在真实 LLM 上一滑就崩: softmax 权重和必须为 1,
训练后的模型习惯把"无处可去"的注意力倒在**开头几个 token** 上 (attention sink)。它们被逐出后,
softmax 分母失去最大的一项, 剩余权重被整体放大重排, 输出畸变。StreamingLLM 的解法:
`cache = [开头 S 个 sink] + [最近 W 个]`, 位置按 cache 槽位重新编号。

## 核心数据结构或公式
- `SinkCache` (`sink_cache.py`): 每层 `(K_raw, V)`, 形状 (L,D), L ≤ S+W; 超预算时逐出槽位 S
  (窗口里最老的), sink 永远不动。`n_sink=0` 即纯窗口, `window=∞` 即完整 cache。
- **K 存未旋转 (pre-RoPE)**: 每步 `K = apply_rope(K_raw, positions=arange(L))`, query 用槽位 L-1。
  位置永远 < S+W, 所以 RoPE 表只要 S+W 行; 普通 cache 存 post-RoPE 的 K, 位置钉死, 无法重编号。
- 分母论证: `out = Σ_j softmax(s)_j v_j`; 若 `exp(s_0)` 占分母 90%, 删掉它 → 其余权重 ×10。
- `stream_step`: 自写逐层循环 (rms_norm → 投影 → append → 重转 → `dense_attention` → mlp)。

## 运行后应该看到什么
```bash
python -m llm_infer.m16_attention_sinks.demo      # ≈ 4 s
```
```
[A] TinyLM (随机权重) T=1024, 预算 64 = sink 4 + 窗口 60; 纯窗口同预算 W=64
  SinkCache(window=∞) vs core 标准 cache, max|Δlogit| = 0.0e+00
  RoPE 表仅 64 行: 绝对位置 1023 → IndexError; 槽位重编号 → 有界策略全程正常
  开头 4 token 注意力占比 (逐层) T=1024: 0.0008 0.0022 0.0040 0.0062 | 均匀 = 0.0039
  seed | 策略       |    PPL | KL→完整 | mean|Δlogit|
    42 | 完整 cache |  81.02 | 0.0000  | 0.0000
    42 | 纯窗口     | 169.06 | 0.7289  | 0.9799
    42 | sink+窗口  | 157.57 | 0.6608  | 0.9220
     1 | 纯窗口 112.00 / 0.3811      sink+窗口 118.93 / 0.4118
     2 | 纯窗口 155.29 / 0.5798      sink+窗口 152.15 / 0.5624
  sink+窗口 的 KL 优于纯窗口: 2/3 个权重 seed  ← 如实报告, 不做断言
[B] 人工植入 sink: 开头 4 个位置注意力占比 T=256 → 93.0%
  T=256 输出相对误差: sink+窗口 19.40%  vs  纯窗口 103.06%
```
**诚实说明**: TinyLM 随机权重、没训练过, **不存在 attention sink** (开头 token 的注意力就是均匀水平),
注意力也没有局部性, 所以两种有界策略都明显偏离完整 cache, 且谁好谁坏随 seed 翻转。[A] 只断言机制
性质: cache ≤ 64 条; 前 64 步与完整 cache 一致 (<1e-4); `window=∞` 精确复现标准 cache;
重编号后 64 行 RoPE 表跑完 1024 步而绝对位置越界。"sink 值得留"只在 [B] (人工植入) 上断言。

## 与真实系统的差距
- 真实 LLM 的 sink 是预训练产物: StreamingLLM 实测 Llama-2 纯窗口 PPL 爆到 10³ 量级, 留 4 个 sink
  即恢复到接近完整 cache。本模块无法复现这一数字, 只能复现机制。
- 每步重转整个 cache 是 O(L·D); 真实实现 (HF SinkCache) 只对被平移的 key 乘一个增量旋转。
- StreamingLLM **不扩展**上下文: 被逐出的 token 彻底遗忘, 只保证"流不崩", 不保证记得 1 万 token 前的内容。
- GPT-OSS 把 sink 做成每个 head 一个可学习 logit (进分母、不出 value), 不再占用真实 token。

## 常见误区
- "sink token 语义重要": 不, 换成换行符也行; 重要的是它们的位置 (对所有后续 token 可见) 和它们吸走的注意力质量。
- "位置用原始绝对位置就行": sink 与窗口的距离会无限增大, 超出训练长度 / RoPE 表; 必须按槽位重编号。
- "在随机权重模型上也能看到 sink+窗口 完胜": 不能, 见 [A]; 把教学 demo 的数字调到好看是作弊。
- "纯窗口只是少看了几个 token": RoPE 只依赖相对位置, 窗口内部没错; 错的是分母里少了最大的一项。

## 自测题
1. 为什么 SinkCache 要存未旋转的 K?
   答: 逐出后留下的 token 槽位前移, 位置要重新分配; post-RoPE 的 K 已把旧位置烤进去, 只能反旋转再重转。
2. S=4, W=2044, 流到第 100 万个 token 时, query 的 RoPE 位置是多少? 普通 cache 呢?
   答: 2047 (槽位); 普通 cache 是 999999, 早已超出训练长度。
3. 为什么 [A] 里 sink+窗口 没有稳定赢过纯窗口?
   答: 随机权重下注意力近似均匀, 开头 4 个 token 占比 ≈ 4/T, 分母里没有"最大项"可丢, 留它们与多留 4 个最近 token 没有本质区别。
