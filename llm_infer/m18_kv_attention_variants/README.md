# M18 — KV cache 账本: MHA / MQA / GQA / MLA

## 直觉
decode 的并发上限 = 显存 ÷ 每 token 的 KV 字节数。四种 attention 变体的数学几乎一样, 区别只在
**cache 里存什么**:
- **MHA**: 每个 query 头都有自己的 K/V 头, 全存。
- **MQA / GQA**: 多个 query 头共享 1 个 (MQA) 或 1 组 (GQA) KV 头, cache 缩小 `n_head / n_kv` 倍。
- **MLA** (DeepSeek-V2/V3): 不存 K/V, 只存一个低秩 latent `c_kv` (d_c 维) 和一份所有头共享的 RoPE key
  (d_rope 维); 每头的 K/V 在 attention 时由 `c_kv` 上投影还原 —— 用一点计算换大量显存。

## 核心数据结构或公式
```
MHA/GQA/MQA  bytes/token = 2 · n_kv · d_head · n_layer · bytes      cache: K,V 各 (T, n_kv, d_head)
MLA          bytes/token = (d_c + d_rope) · n_layer · bytes         cache: C (T, d_c), k_rope (T, d_rope)
```
MLA 一层的前向 (`MLALayer.forward`):
```
c      = x W_DKV                    (T, d_c)      ← 进 cache, 不加 RoPE
k_rope = RoPE(x W_KR)               (T, d_rope)   ← 进 cache, 所有头共享
K_h    = [c W_UK_h ; k_rope]        attention 时现场还原, 不进 cache
V_h    = c W_UV_h
```
**为什么 RoPE 要解耦**: RoPE 是依赖位置的旋转 R_t。若加在 latent 还原出的 K 上, 得到
`q R (c W_UK)ᵀ`, R 夹在中间, W_UK 无法预先合并; 把位置信息单独放进 k_rope, latent 部分就保持
"纯线性", 于是可以 **absorb**: `q·(c W_UK)ᵀ = (q W_UKᵀ)·cᵀ` —— 把 W_UK 乘到 q 上, K 直接就是
cache 本身。此时 MLA decode ≡ 一个 head_dim = d_c + d_rope 的 **MQA** (`absorb=True` 分支, FlashMLA 的做法)。

GQA 代码路径 (`GQALayer.forward`): q reshape 成 `(n_kv, n_head/n_kv, T, d)`, K 为 `(n_kv, 1, Tk, d)`,
靠 `core.dense_attention` 的前导维广播共享 KV, 不 repeat。

## 运行后应该看到什么
```bash
python -m llm_infer.m18_kv_attention_variants.demo     # < 1 s
```
```
config                             KiB/token  max tokens  batch@8192      (fp16, 40 GiB KV 预算)
LLaMA-2-7B   MHA  32kv×128, 32L        512.0      81,920          10
LLaMA-3-8B   GQA   8kv×128, 32L        128.0     327,680          40
(假想) 7B    MQA   1kv×128, 32L         16.0   2,621,440         320
(假想) DS-V3 MHA 128kv×128, 61L       3904.0      10,743           1
DeepSeek-V3  MLA  512+64,   61L         68.6     611,191          74      → MLA 比同尺寸 MHA 省 56.9x

variant                     max|Δ|  cache bytes   formula      (增量 decode vs 全量重算, 20 tokens, fp32)
MHA (n_kv=8)              1.07e-06       61,440    61,440
GQA (n_kv=2)              1.55e-06       15,360    15,360
MQA (n_kv=1)              1.43e-06        7,680     7,680
MLA (d_c=24,d_rope=8)     1.88e-06        7,680     7,680
MLA absorb                1.07e-06        7,680     7,680
```
另有: GQA 广播路径 vs 逐头 MHA 基线 max|Δ| = 0 (n_kv=8/2/1); absorb 的 score max|Δ| = 2.86e-06,
整网输出 naive vs absorb max|Δ| = 1.43e-06。全部 `assert` 通过。

## 与真实系统的差距
- 真实 MLA 还对 q 做低秩压缩 (d_c'=1536)、对 c_kv 做 RMSNorm, 这里省略 (不影响 cache 大小)。
- "40 GiB" 是纯 KV 预算, 真实部署要先扣掉权重和激活; 还有 paged 碎片 (m02)。
- 真实 GQA kernel (FlashAttention / FlashInfer) 用 stride 实现广播; 真实 MLA kernel (FlashMLA) 直接在
  576 维 latent 上算, prefill 阶段反而常用 naive 形式 (计算密集时物化 K/V 更划算)。
- TP 下 GQA 的 n_kv 要能被 TP 度整除 (否则复制 KV 头); MLA 的 latent 无法按头切, 通常每卡存全量。

## 常见误区
- "GQA 降低了计算量" —— 主要降的是 KV 显存和 decode 访存; Q 的头数与 attention FLOPs 基本不变。
- "MLA 就是对 KV cache 做 SVD 压缩" —— 不是事后压缩, 低秩投影是训练出来的模型结构。
- "MLA 每步都要还原完整 K/V, 所以很慢" —— absorb 后根本不还原, 等价于 576 维的 MQA。
- "MLA cache 比 MQA 还小" —— 不一定: DS-V3 每层 576 维 vs MQA 的 2×128=256 维; MLA 赢在质量接近 MHA。

## 自测题
1. LLaMA-3-70B (80 层, 8 KV 头, d_head=128, fp16) 每 token KV 多少? **答**: 2·8·128·80·2 = 327,680 B = 320 KiB。
2. 为什么 MLA 不能直接在 latent 上加 RoPE? **答**: RoPE 的位置相关旋转会夹在 q 与 W_UK 之间, 使 W_UK
   无法吸收进 W_Q, decode 每步都得把全部历史 latent 还原成 K 再旋转, 失去意义; 所以另设共享的 k_rope。
3. demo 里 GQA(n_kv=2) 的 cache 为什么正好是 MHA 的 1/4? **答**: 字节数 ∝ n_kv, 8 → 2。
