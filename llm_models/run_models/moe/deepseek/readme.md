# DeepSeek-V3 (MLA + MoE + aux-loss-free 均衡)

## 直觉

两个独立的省钱办法叠在一起:

- **MLA** 省 KV cache: 不缓存每个 head 的 K/V, 只缓存一个低秩 latent `c_kv` 和一份所有 head 共享的 `k_rope`,
  用的时候再现场升维成 K/V。
- **DeepSeekMoE** 省算力: 很多小专家 + 始终激活的共享专家; 用 sigmoid 给每个专家独立打分。
- **aux-loss-free 均衡**: aux loss 会和 LM loss 抢梯度。V3 改成给每个专家一个不走梯度的 bias,
  只在 "选谁" 时加上, 算权重时不用; 谁过载就降谁。

## 核心公式

```
MLA:   c_kv = W_DKV x  [r]      k_rope = RoPE(W_KR x)  [rope]       ← 只缓存这两个
       K = [W_UK c_kv | k_rope]   V = W_UV c_kv                      ← 每步现场算
       每 token 每层 cache:  MHA 2·H·Dh  |  GQA 2·Hkv·Dh  |  MLA r + rope

MoE:   s = sigmoid(router(x));   topk = top-K(s + bias);   w_i = s_i / Σ_{j∈topk} s_j
       y = Σ w_i·routed_i(x) + Σ shared_j(x)

均衡:  每个训练 step 之后 (no_grad):  bias_i += γ · sign(mean_load − load_i)
```

## 运行命令

```bash
python -m llm_models.run_models.moe.deepseek.infer_deepseek    # ~3 s
python -m llm_models.run_models.moe.deepseek.train_deepseek    # ~17 s (同一种子训两遍做对照)
```

## 运行后应该看到什么

infer (每个数字都有 assert 把关):
```
总参数 39,802,368 | 每 token 激活 14,642,688 (36.8%)
prefill 10 + 逐 token decode 20 步 vs 一次性 forward: logits 最大差 1.91e-06
每 token 每层 cache 浮点数 (d_model=512, H=8, Dh=64):
  MHA  2·H·Dh   = 1024
  GQA  2·Hkv·Dh = 256  (Hkv=2, MHA 的 25.0%)
  MLA  r + rope = 96  (r=64, rope=32, MHA 的 9.4%, GQA 的 37.5%)
```
train (router 被故意初始化得偏心: 专家 0/1 的权重 ×4; aux_loss_weight=0):
```
--- 不更新 bias ---           step 1: lm 6.9617, load CV 0.349, layer0 load [45, 53, 22, 17, 29, 29, 29, 32]
                              step 100: lm 2.9793, load CV 0.269, layer0 load [37, 35, 33, 22, 27, 27, 40, 35]
--- bias 更新 (γ=1e-3) ---    step 100: lm 2.9816, load CV 0.097, layer0 load [31, 32, 36, 37, 27, 31, 29, 33]
最后 20 步平均 load CV: 无 bias 0.273  vs  有 bias 0.124
layer0 routing_bias: [-0.048, -0.015, -0.001, 0.036, 0.037, 0.027, -0.007, 0.008]
```
负载不均衡减半, LM loss 几乎不变 (2.979 vs 2.982) —— 这就是 "aux-loss-free" 的含义。
修复前: 初始 lm_loss **259.30** (应为 ln 1000 = 6.91), 且 `routing_bias` 是一个永远为 0 的 buffer。
数据是固定的一个随机 batch: loss 下降只说明模型在背它。

## 常见误区

- "MLA 的 cache 里存的是压缩后的 K 和 V": 存的是 **一个** latent, K 和 V 都从它升维; 另加一份共享的 k_rope。
- "RoPE 直接作用在 latent 上就行": 不行。RoPE 是随位置变化的旋转, 会挡在 W_UK 和 Q 之间, 生产推理的 "权重吸收" 就做不了, 所以才拆出 rope 段。
- "bias 会改变专家输出的权重": 不会。权重用不带 bias 的 s_i; bias 只影响 top-k 选谁。
- "γ 越大均衡越快越好": 更新只看 sign, 步长恒为 γ。γ 大于专家分数之间的典型差距就会来回振荡 (本例实测 γ=1e-2 的均衡效果不如 1e-3)。
- "routing_bias 是可学习参数": 它是 buffer, 不走梯度、不进 optimizer, 但要进 state_dict。

## 自测题

1. d_model=512, H=8, r=64, rope=32, 序列 1000 token、2 层, MLA 比 MHA 少缓存多少个浮点数?
   **答**: (1024 − 96) × 1000 × 2 = 1,856,000。
2. 某专家本 batch 的 load 是平均值的 3 倍, 另一个是 1.1 倍, 它们的 bias 各变化多少?
   **答**: 都是 −γ。只用 sign, 与超载多少无关 (所以严重超载的专家需要更多步才能压下去)。
3. 为什么两次训练的 lm_loss 几乎相同, 负载却差了一倍?
   **答**: bias 不参与梯度也不参与加权, 只是把边缘 token 换到相邻分数的专家; 对一个小模型背固定 batch 来说哪个专家处理差别不大, 但在真实的专家并行里负载直接决定吞吐。
