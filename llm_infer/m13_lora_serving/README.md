# M13 — Multi-LoRA Serving: 一份底模, N 个 adapter 同 batch

## 直觉
一个底模要同时服务很多客户, 每个客户有自己的 LoRA 微调。两条朴素路线都不行:
**合并权重** (W' = W + ΔW) → 每个客户一份完整 W', 显存 ×N, 且不同客户的请求没法同 batch;
**逐客户串行** → batch 小, GPU 吃不饱。解法: **不合并**。底模部分全 batch 共享一次 gemm,
LoRA 部分是两个很瘦的矩阵, 按每个 token 的 adapter id 去取各自的 A/B 来算。

## 核心数据结构或公式
命名遵循 LoRA 论文 / HF PEFT (W 按 `nn.Linear` 布局 `(d_out, d_in)`, 行向量前向 `x @ W.T`):
```
A: (r, d_in)    down-projection (降维), 随机初始化
B: (d_out, r)   up-projection  (升维), 零初始化   → 训练开始时 ΔW = 0
ΔW = (alpha/r) · B · A                  (d_out, d_in)
y  = x·Wᵀ + (alpha/r) · (x·Aᵀ)·Bᵀ       (N,d_in) → (N,r) → (N,d_out)
```
- `tok_adapter (N_tok,)`: 把所有请求的 token 拼平后, 每个 token 的 adapter id。
- **BGMV**: `A_all[tok_adapter]` gather 出每个 token 的 A → 批量 mat-vec (shrink), B 同理 (expand)。适合 decode (每请求 1 token)。
- **SGMV**: 同 adapter 的 token 归为一段, 每段一次 gemm。适合 prefill (每请求很多 token)。
- 参数量: adapter `r·(d_in+d_out)` vs 底模 `d_in·d_out`。

## 运行后应该看到什么
```bash
python -m llm_infer.m13_lora_serving.demo      # ~0.2 s
```
```
[1] A.shape / B.shape / ΔW.shape = (4, 64) / (96, 4) / (96, 64);  B=0 时 |LoRA 输出 - 底模输出| = 0.0
[2] loop / BGMV / SGMV vs 合并权重 = 7.15e-07 (三者相同)
    (对照) LoRA 对输出的改变量 = 3.97e+00;  全部错用 adapter 0 的误差 = 3.89e+00
[3] 底模 W 24576 B, 单个 adapter 2560 B = r·(d_in+d_out)·4 (10.4% of W)
    1000 个 adapter: 合并 24.58 MB / 不合并 2.58 MB (9.5x);  d=4096, r=16 时 adapter/W = 0.78%
[4] 500 条 decode 请求: 底模 gemm 500 次 vs 1 次; loop 2.01 ms, BGMV 0.81 ms, SGMV 0.58 ms
```
断言: 零初始化 B 时输出与底模逐位相同; BGMV / SGMV / loop 与合并权重参考的 max-abs-diff < 1e-5,
同时 LoRA 的效果 > 1e-2、路由错 adapter 的误差 > 1e-2 (证明等价性检查不是空的);
adapter 字节数 == `r·(d_in+d_out)·4`; SGMV 比逐请求循环快。[4] 的毫秒数是 numpy/CPU, 只看方向。

## 与真实系统的差距
- 这里的 BGMV 用 numpy fancy-index, 会真的拷贝出 `(N_tok, r, d_in)`; Punica 的 CUDA kernel 不拷贝,
  每个线程直接用 adapter id 去显存里寻址。SGMV 在这里是 Python 循环, 真实是一次 grouped-GEMM launch。
- 只演示了一个线性层、所有 adapter 同 rank。真实系统对 q/k/v/o/gate/up/down 每层都加, rank 各不相同
  (vLLM 按 max_lora_rank padding; S-LoRA 用 unified paging 把 adapter 权重和 KV cache 放进同一个分页池)。
- 没有 adapter 的换入换出: 真实系统 adapter 常驻 CPU, 按请求 LRU 换进 GPU (vLLM `max_loras` / `max_cpu_loras`)。
- 与前缀缓存的交互: 不同 adapter 的 KV 不同, 前缀缓存的 key 必须带上 adapter id。

## 常见误区
- **A/B 记反**: A 是降维 `(r, d_in)` 且随机, B 是升维 `(d_out, r)` 且为零。本模块旧版写反了 (A 当升维、
  两个都随机); d_in == d_out 时形状恰好对得上, 错误不会报出来 — 所以 demo 特意用 d_in=64 ≠ d_out=96。
- "两个都随机初始化也行": 那样 ΔW ≠ 0, 微调一开始就把底模行为改掉了; 反过来两个都为 0 则梯度恒为 0, 学不动。
- "serving 时合并权重最快": 单 adapter 时是对的 (零额外开销); 多 adapter 同 batch 时合并反而不可行。
- "LoRA 的额外计算可以忽略": 参数量小, 但 shrink/expand 是两次额外的 kernel, 且是访存密集的 gather,
  decode 时延迟开销并不小 — 这正是 Punica 要写专用 kernel 的原因。

## 自测题
1. **d_in=d_out=4096, r=16, 一个线性层的 adapter 占底模该层参数的多少?**
   16·(4096+4096) / 4096² = 0.78%。
2. **为什么 B 零初始化而 A 随机, 不能反过来都为零?**
   ΔW = B·A 要在起点为 0 (不破坏底模); 若 A、B 都为 0, 则 ∂L/∂A ∝ Bᵀ = 0 且 ∂L/∂B ∝ A = 0, 梯度全零无法训练。一个为零一个随机, 零的那个能拿到非零梯度。
3. **一个 batch 里 8 条 decode 请求用 8 个不同 adapter, 与 8 条 prefill (各 512 token) 用 2 个 adapter, 各适合 BGMV 还是 SGMV?**
   前者 BGMV (每 token 各取各的, 分段没有意义); 后者 SGMV (只有 2 段, 每段是一次大 gemm, 算术强度高)。
