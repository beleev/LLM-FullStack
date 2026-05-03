# LLM Infer — 推理与部署优化教具库

> 用 ~3000 行 numpy（零 GPU 依赖）讲清楚业界 80% 的 LLM 推理优化技术。
> 每个模块独立可跑，也能组合成一个完整的 mini-engine。

---

## 设计原则

- **零依赖**：只用 `numpy`，与 `llm_basic` 风格一致；让你看见算法骨架而不是 PyTorch 的糖衣
- **模块自治**：每个 `mXX_*/` 子目录都能 `python demo.py` 独立运行
- **可组合**：`full_engine/` 把核心模块串成一个 mini-vLLM
- **重原理、轻性能**：CPU、小张量、慢但清晰；带详细 ASCII 时序图与中文注释

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
            │       └─ m15 P/D Disaggregation ┘
            │
            ▼
    full_engine/   把以上模块组装成 mini-vLLM
```

---

## 模块清单

| # | 模块 | 核心文件 | 关键概念 |
|---|------|---------|---------|
| 01 | [KV Cache](m01_kv_cache/)              | `demo.py` | 重复计算 → 增量计算，O(T²)→O(T) |
| 02 | [Paged Attention](m02_paged_attention/) | `block_manager.py` | 显存"虚拟内存"、block table |
| 03 | [Continuous Batching](m03_continuous_batching/) | `scheduler.py` | prefill/decode 分离、preempt |
| 04 | [Prefix Cache (hash)](m04_prefix_cache/) | `demo.py` | 链式 hash 跨请求复用 KV |
| 05 | [Radix Cache](m05_radix_cache/) | `radix_tree.py` | 任意长度前缀共享 + LRU |
| 06 | [Chunked Prefill](m06_chunked_prefill/) | `demo.py` | 长 prompt 切片 prefill |
| 07 | [Speculative Decoding](m07_speculative_decoding/) | `demo.py` | draft + verify，1 步出多 token |
| 08 | [Quantization](m08_quantization/) | `int8_weight.py`, `kv_quant.py` | 权重 INT8、KV cache INT8 |
| 09 | [Tensor Parallel](m09_tensor_parallel/) | `demo.py` | column/row parallel + all-reduce |
| 10 | [Sampling](m10_sampling/) | `samplers.py` | greedy/temp/top-k/top-p/min-p |
| 11 | [FlashAttention](m11_flash_attention/) | `demo.py` | 分块 softmax、O(N) 显存 |
| 12 | [CUDA Graph](m12_cuda_graph/) | `demo.py` | trace + replay，去 launch 开销 |
| 13 | [Multi-LoRA Serving](m13_lora_serving/) | `demo.py` | batched LoRA、SGMV 思路 |
| 14 | [Structured Output](m14_structured_output/) | `demo.py` | JSON/grammar 约束 logits mask |
| 15 | [P/D Disaggregation](m15_pd_disaggregation/) | `demo.py` | prefill/decode 跨节点解耦 |
| ★  | [Full Engine](full_engine/) | `engine.py` | 集成 m01+m02+m03+m04+m10 的 mini-vLLM |

---

## 业界覆盖度自评

| 业界主流技术 | 本仓库 | 备注 |
|---|:---:|---|
| PagedAttention | ✅ m02 | block 分配/释放/引用计数 |
| Continuous batching | ✅ m03 | + preempt |
| Prefix caching (hash) | ✅ m04 | vLLM style |
| Radix prefix cache | ✅ m05 | SGLang style |
| Chunked prefill | ✅ m06 | |
| Speculative decoding | ✅ m07 | draft model 思路 |
| Weight quantization | ✅ m08 | INT8 对称量化 |
| KV cache quantization | ✅ m08 | per-token INT8 |
| Tensor parallelism | ✅ m09 | column/row + all-reduce |
| FlashAttention | ✅ m11 | online softmax |
| CUDA Graph | ✅ m12 | 用 Python 模拟 capture/replay |
| Multi-LoRA serving | ✅ m13 | batched LoRA |
| Structured output | ✅ m14 | logits mask |
| P/D disaggregation | ✅ m15 | KV transfer 概念 |
| Pipeline parallelism | ❌ | 训练为主，推理少用 |
| Expert parallelism | ❌ | 见 `llm_models/layers/sparse/` |
| MLA / KV 压缩 | ❌ | 见 `llm_models/` DeepSeek-V3 |
| Lookahead/Medusa | 部分 ✅ | m07 投机思想可外推 |

---

## 运行

```bash
# 单模块
python -m llm_infer.m01_kv_cache.demo
python -m llm_infer.m02_paged_attention.demo
# ...

# 集成 mini-engine
python -m llm_infer.full_engine.demo
```

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
| m12 CUDA Graph | `model_runner.py:223` | `engine/graph.py` |

读完本目录后再去读 `ref/nano-vllm/` 和 `ref/mini-sglang/`，会有"原来如此"的恍然大悟感。
