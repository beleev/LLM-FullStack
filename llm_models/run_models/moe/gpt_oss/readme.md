# GPT-OSS — 交替 SWA/全注意力 + attention sink + MoE

模型: `llm_models/models/moe/gpt_oss.py::GPTOSSMini` (= Mixtral 骨架, 只改了 mask 的分配和一个 sink 参数)。

## 直觉

**为什么交替。** 全注意力层每个 token 的算力和 KV cache 都是 O(T); SWA 层是 O(W)。全用 SWA (Mistral) 最省,
但远处的信息只能一层一层接力 (L 层 ≈ L·W), 每接力一次都被压缩一遍。全用 full 最准但最贵。
GPT-OSS 一层 SWA 一层 full: 一半的层 cache 封顶 W, 另一半保留 "任意远处一步直达" 的通路
(Gemma 2 是 1:1, Gemma 3 是 5:1, 思路相同)。

**为什么要 sink。** softmax 的行和必须是 1 —— 一个 head 即使 "这一步没什么可看的", 也得把 1 分完。
训练出来的模型会自己找一个垃圾桶: 开头几个 token (StreamingLLM 的观察: 首 token 拿走大量注意力,
把它滑出窗口模型就崩)。StreamingLLM 的补救是 **永远保留前几个 token 的 K/V**;
GPT-OSS 直接给每个 head 一个 **可学的 sink logit**: 它参与 softmax 的分母, 但没有 value, 算完就丢。
这样 SWA 层可以放心丢掉窗口外的一切, 不用特殊照顾首 token。

## 核心公式

```
偶数层 mask:  j ≤ i 且 j > i − W        (带状)          cache 长度 min(T, W)
奇数层 mask:  j ≤ i                     (下三角)        cache 长度 T

attn_i = softmax([s_i1, …, s_iT, sink_h])[:T]           Σ_j attn_ij = 1 − p_sink < 1
       = e^{s_ij} / (Σ_k e^{s_ik} + e^{sink_h})         sink_h → −∞ 时退化为普通 softmax

路由 (官方): w = softmax(top-K(router_logits))
           ≡ MixtralMoE 的 p_i / Σ_{j∈topk} p_j        (全局 softmax 的分母上下约掉) → 直接复用 MixtralMoE
```

带 cache 解码时每层拿到的 mask 不同: full 层取列 `[:past+T]`, SWA 层只取最后 `min(past, W) + T` 列
(与被滚动裁剪的 cache 对齐)。

## 运行命令

```bash
python -m llm_models.run_models.moe.gpt_oss.infer_gpt_oss
python -m llm_models.run_models.moe.gpt_oss.train_gpt_oss
```

## 运行后应该看到什么 (实测, CPU, infer ~3 s / train ~7 s)

infer (4 层, W=8, 4 专家 top-2):
```
[1] 总参数 2,687,120 | 每 token 激活 1,507,472 (56.1%)
    路由: softmax(top-k logits) == MixtralMoE 的 routing_weights (逐项相等)
[2] 每行注意力质量 (对 head 取平均): 第 0 行 0.506, 第 39 行 0.976  (< 1, 差额进了 sink)
    sink = −inf: 行和回到 1, 输出与无 sink 的 GQA 相同; sink = 0 时输出不同
[3] 改位置 0 → 第 0 层 (SWA) 位置 ≥ 8 最大变化 0.0e+00; 最终 logits 在位置 39 变化 1.06e-02
[4] 已读 40 个 token, 每层 cache 长度 [8, 40, 8, 40] (偶数层 SWA ≤ W=8, 奇数层 full = 40); 比全 full 的 160 省 40%
    贪心生成 100 token: 有/无 cache 输出完全一致, 加速 3.1x
```
- `[2]` 的 0.506 / 0.976 可以心算: 初始化时 score ≈ 0、sink = 0, 第 t 行有 t+1 个真实 key + 1 个 sink → 质量 ≈ (t+1)/(t+2) = 1/2, 40/41。
- `[4]` 的节省随 T 增大趋近 50% (SWA 层的 W 相对 T 可忽略); gpt-oss 真实配置 W=128, T=131K 时 SWA 层的 cache 是 full 层的 1/1024。

train (60 步, 固定的一个随机 batch —— loss 下降只说明模型能背下它):
```
lm_loss 6.9952 (ln V = 6.9078) → 1.1664 | aux_loss 2.0064 → 2.0092 (均衡值 K = 2, 坍塌值 E = 4)
  Layer 0 (SWA ) sink logits: [0.0039, 0.0038, 0.0106, 0.0009]      ← 初始全 0, 已被更新
  Layer 3 (full) sink logits: [-0.006, -0.0005, -0.0084, -0.0065]
  sink 梯度 |g| 最大 7.21e-05, 最小 9.20e-06 (全部非零)
  Layer 0 专家负载: [0.38, 0.39, 0.7, 0.53]  (均衡 = 0.50)
  Layer 3 专家负载: [0.5, 0.45, 0.52, 0.53]
```
sink 只动了 ~0.01: 随机 token 没有 "该不该看" 的结构, 模型没有理由用它; 这里只证明梯度通路是通的。
真实模型里 sink logit 会学到明显非零的值, 且各 head 不同。

## 常见误区

- "sink 是一个特殊 token / 多占一格 KV cache": 不是。它只是每 head 一个标量, 进 softmax 分母, 没有 K 也没有 V; 参数量 = 层数 × head 数。
- "sink 让注意力输出变小, 是一种损失": 这正是目的 —— head 可以输出接近 0 的向量 (等价于 "本层我不发言"), 而不是被迫搬运首 token 的 value。
- "SWA 层看不到的 token 对模型就不可见": `[3]` 显示第 0 层确实看不到位置 0, 但下一层 (full) 直接看得到, 最终 logits 变了。
- "所有层共用一个 cache 长度": 不行。full 层的 cache 如果也裁到 W, 它的 mask (past+T 列) 和 K (W+T 个) 形状就对不上; 连 mask 一起裁, 算的就是另一个 (全 SWA) 模型, 与无 cache 的输出不一致。
- "top-k 后 softmax 是一种新路由": 与 Mixtral 的重归一在数学上完全相同, 只是省掉了对全部 E 个专家做 softmax。

## 自测题

1. gpt-oss-20b 有 24 层、W=128。上下文 T=131072 时, KV cache 比 24 层全 full 省多少?
   —— 12 层 full + 12 层 ×128: (12·131072 + 12·128) / (24·131072) ≈ 50.05%, 省约 50%。想再省就得提高 SWA:full 的比例 (Gemma 3 的 5:1)。
2. 某个 head 某一行的真实 score 全是 0、共 3 个可见 key, sink logit = ln 3。分给真实 token 的总质量是多少?
   —— 3·e⁰ / (3·e⁰ + e^{ln 3}) = 3/6 = 0.5。
3. 为什么 SWA 层裁掉旧 K/V 后, 新 token 的 RoPE 位置不会错?
   —— cache 里存的是 RoPE **之后**的 K (绝对位置已经转进向量里), 新 query 用 `position_ids = past…` 旋转; 相对位置由两者的旋转角之差决定, 与 K 在 cache 里排第几个无关。
