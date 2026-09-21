# LLM Infer — 推理与部署优化教具库

> 用几千行 numpy（零 GPU 依赖）讲清楚业界主流的 LLM 推理优化技术：22 个模块 + 1 个 mini-engine。
> 每个模块独立可跑，也能组合成一个完整的 mini-engine。

---

## 设计原则

- **零依赖**：只用 `numpy`，与 `llm_basic` 风格一致；让你看见算法骨架而不是 PyTorch 的糖衣
- **模块自治**：每个 `mXX_*/` 都能在仓库根目录用 `python -m llm_infer.mXX_name.demo` 独立运行（包内绝对 import，直接 `python demo.py` 不行）
- **demo 即测试**：每个 demo 以 `assert` 收尾（与朴素基线逐 token / max-abs-diff 对拍）；凡是代价模型或 `sleep` 模拟出来的数字都在输出里明说
- **可组合**：`full_engine/` 把核心模块串成一个 mini-vLLM
- **重原理、轻性能**：CPU、小张量、慢但清晰；公共件在 `core/`（TinyLM、`dense_attention` 基线、`Sequence`），模块不重复造

---

## 学习路径（按依赖顺序）

```
                    ┌─ m01 KV Cache ──────────┐
                    │  (一切优化的起点)        │
                    │                          │
                    ├─ m02 Paged Attention ───┤
                    │  (vLLM 核心)             │
                    │                          │
                    ├─ m03 Continuous Batching ┤   <-- 调度层
                    │                          │
            ┌───────┼─ m04 Prefix Cache (hash)─┤   <-- 共享缓存
            │       │                          │
            │       ├─ m05 Radix Cache ───────┤   (SGLang 招牌)
            │       │                          │
            │       ├─ m06 Chunked Prefill ───┤
            │       │                          │
            │       ├─ m11 FlashAttention ────┤   <-- kernel 层
            │       │                          │
            │       ├─ m12 CUDA Graph ────────┤
            │       │                          │
            │       └─ m09 Tensor Parallel ───┘
            │
            │       ┌─ m07 Speculative Decode ┐   <-- 解码加速
            │       │                          │
            │       ├─ m08 Quantization ──────┤
            │       │                          │
            │       ├─ m10 Sampling ──────────┤
            │       │                          │
            │       ├─ m13 Multi-LoRA ────────┤   <-- serving 增强
            │       │                          │
            │       ├─ m14 Structured Output ─┤
            │       │                          │
            │       ├─ m15 P/D Disaggregation ┤
            │       │                          │
            │       ├─ m16 Attention Sinks ───┤   <-- 流式长上下文
            │       │                          │
            │       ├─ m17 EAGLE Speculative ─┤   <-- 特征级 draft
            │       │                          │
            │       ├─ m19 Tree Speculation ──┤   <-- 树形 draft + tree mask
            │       │                          │
            │       ├─ m18 MHA/GQA/MQA/MLA ───┤   <-- KV 体积
            │       │                          │
            │       ├─ m20 KV Offload ────────┤   <-- GPU→CPU→disk 分层
            │       │                          │
            │       ├─ m21 MoE Serving ───────┤   <-- EP + EPLB
            │       │                          │
            │       └─ m22 Sparse Attention ──┘   <-- 只读 top-k KV block
            │
            ▼
    full_engine/   把以上模块组装成 mini-vLLM
```

---

## 模块清单

| # | 模块 | 核心文件 | 关键概念 |
|---|------|---------|---------|
| 01 | [KV Cache](m01_kv_cache/) | `demo.py` | 重复计算 → 增量计算，每步 O(T²)→O(T) |
| 02 | [Paged Attention](m02_paged_attention/) | `block_manager.py`, `paged_attention.py` | block table、ref_count、物理 KV pool 读写 |
| 03 | [Continuous Batching](m03_continuous_batching/) | `scheduler.py` | iteration-level 调度、抢占、无活锁；full_engine 直接复用 |
| 04 | [Prefix Cache (hash)](m04_prefix_cache/) | `prefix_cache.py` | 链式 hash、缓存项寿命 = block 内容寿命（惰性 LRU 失效） |
| 05 | [Radix Cache](m05_radix_cache/) | `radix_tree.py` | 任意长度前缀共享 + LRU |
| 06 | [Chunked Prefill](m06_chunked_prefill/) | `chunked_prefill.py` | Sarathi：prefill chunk 与 decode 混批，token 预算封顶 TBT |
| 07 | [Speculative Decoding](m07_speculative_decoding/) | `speculative.py` | draft + 一次验证、KV 回滚、分布保持的经验检验 |
| 08 | [Quantization](m08_quantization/) | `int8_weight.py`, `int4_awq.py`, `kv_quant.py` | RTN INT8/INT4、group-wise、AWQ 缩放、KIVI (K per-channel / V per-token) |
| 09 | [Tensor Parallel](m09_tensor_parallel/) | `parallel_linear.py` | 多头 attention + SwiGLU MLP 的 column→row 切分，每层 2 次 all-reduce |
| 10 | [Sampling](m10_sampling/) | `samplers.py` | greedy/temp/top-k/top-p/min-p/重复惩罚 |
| 11 | [FlashAttention](m11_flash_attention/) | `flash_attention.py` | Q/K 双向分块、online softmax、LSE 输出 |
| 12 | [CUDA Graph](m12_cuda_graph/) | `graph.py` | capture/replay（launch 开销为显式模拟参数） |
| 13 | [Multi-LoRA Serving](m13_lora_serving/) | `lora.py` | batched LoRA（ΔW = B·A）、SGMV 思路 |
| 14 | [Structured Output](m14_structured_output/) | `grammar.py` | FSM × 多字符词表的 token mask 预编译（xgrammar/outlines），驱动真实 logits |
| 15 | [P/D Disaggregation](m15_pd_disaggregation/) | `pd.py` | prefill/decode 跨节点、KV 传输带宽账 |
| 16 | [Attention Sinks](m16_attention_sinks/) | `sink_cache.py` | StreamingLLM：sink + 滚动窗口、cache 内 RoPE 重编号 |
| 17 | [EAGLE Speculative](m17_eagle_speculative/) | `eagle.py` | 特征级 draft、共享 lm_head |
| 18 | [KV 体积: MHA/MQA/GQA/MLA](m18_kv_attention_variants/) | `attention_variants.py` | bytes/token 公式、只缓存 latent 的 MLA decode |
| 19 | [Tree Speculation](m19_tree_speculation/) | `tree_spec.py` | Medusa/EAGLE-2：token 树 + tree attention mask 一次验证 |
| 20 | [KV Offload](m20_kv_offload/) | `tiered_cache.py` | GPU→CPU→disk 分层 LRU、命中率与 TTFT 代价模型（LMCache/Mooncake） |
| 21 | [MoE Serving](m21_moe_serving/) | `moe.py` | EP dispatch/combine、热点专家、EPLB 冗余专家放置 |
| 22 | [Sparse Attention](m22_sparse_attention/) | `sparse_attention.py` | DSA/NSA/Quest：block 打分选 top-k KV，误差 vs 读取比例 |
| ★  | [Full Engine](full_engine/) | `engine.py`, `model_runner.py` | m02 真分页 + m03 调度 + m04 前缀复用 + m06 分块 + m10 采样；输出与朴素 greedy 逐 token 相同 |

---

## 业界覆盖度自评

| 业界主流技术 | 本仓库 | 备注 |
|---|:---:|---|
| PagedAttention | ✅ m02 | block 分配/释放/引用计数 |
| Continuous batching | ✅ m03 | + preempt |
| Prefix caching (hash) | ✅ m04 | vLLM style |
| Radix prefix cache | ✅ m05 | SGLang style |
| Chunked prefill | ✅ m06 | 与 decode 混批 + TBT 对比 |
| Speculative decoding | ✅ m07 | draft model + KV 回滚 + 分布保持检验 |
| Weight quantization | ✅ m08 | INT8 / group-wise INT4 / AWQ |
| KV cache quantization | ✅ m08 | KIVI: K per-channel, V per-token |
| Tensor parallelism | ✅ m09 | column/row + all-reduce |
| FlashAttention | ✅ m11 | online softmax |
| CUDA Graph | ✅ m12 | 用 Python 模拟 capture/replay |
| Multi-LoRA serving | ✅ m13 | batched LoRA |
| Structured output | ✅ m14 | logits mask |
| P/D disaggregation | ✅ m15 | KV transfer 概念 |
| Attention sinks / StreamingLLM | ✅ m16 | sink + 滑动窗口, 有界 KV |
| EAGLE / 特征级投机解码 | ✅ m17 | draft 吃 target hidden state, 共享输出头 |
| Pipeline parallelism | ❌ | 训练为主，推理少用 |
| Expert parallelism / EPLB | ✅ m21 | dispatch/combine + 冗余专家负载均衡 |
| MLA / GQA / MQA KV 压缩 | ✅ m18 | latent cache decode |
| Medusa / EAGLE-2 树形投机 | ✅ m19 | tree attention mask |
| 分层 KV offload | ✅ m20 | LMCache / Mooncake / HiCache 思路 |
| 稀疏注意力 decode | ✅ m22 | DSA / NSA / Quest 思路 |
| token 级语法约束 | ✅ m14 | xgrammar / outlines 的 mask 预编译 |

---

## 运行

```bash
# 单模块
python -m llm_infer.m01_kv_cache.demo
python -m llm_infer.m02_paged_attention.demo
# ...

# 集成 mini-engine
python -m llm_infer.full_engine.demo

# 全部 (23 个 demo, 每个都以 assert 收尾)
python -m llm_infer.run_all
```
均需在仓库根目录执行 (`-m` 方式)。

每个 demo 都会 print 出"现象 → 数字 → 结论"三段式输出，便于直观对比。

---

## 与参考项目的对应

| 本仓库模块 | nano-vllm 对应 | mini-sglang 对应 |
|---|---|---|
| m01 KV Cache | `layers/attention.py` 的 cache 写入 | 同 |
| m02 Paged Attention | `engine/block_manager.py` | `kvcache/mha_pool.py` |
| m03 Continuous Batching | `engine/scheduler.py` | `scheduler/scheduler.py` |
| m04 Prefix Cache | `block_manager.py` 的 hash 链 | — |
| m05 Radix Cache | — | `kvcache/radix_cache.py` |
| m06 Chunked Prefill | — | `scheduler/prefill.py` |
| m09 Tensor Parallel | `layers/linear.py` | `distributed/impl.py` |
| m11 FlashAttention | 调 flash-attn 库 | 调 flash-attn 库 |
| m12 CUDA Graph | `engine/model_runner.py` 的 `capture_cudagraph` | `engine/graph.py` |

读完本目录后再去读上游源码，会有"原来如此"的恍然大悟感：
- nano-vllm: https://github.com/GeeeekExplorer/nano-vllm
- mini-sglang: https://github.com/sgl-project/mini-sglang
- vLLM: https://github.com/vllm-project/vllm · SGLang: https://github.com/sgl-project/sglang
