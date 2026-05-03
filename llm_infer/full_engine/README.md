# Full Engine — 把核心模块串成 mini-vLLM

## 集成模块
本目录是**前面 15 个模块的拼装演示**, 选取最关键的几个组合成可跑的引擎:

```
  ┌─────────────────────────────────────────────────────┐
  │  Engine                                             │
  │                                                     │
  │   ├─ TinyLM           (core)                        │
  │   ├─ CharTokenizer    (core)                        │
  │   ├─ BlockManager     (m02 paged attention)         │
  │   ├─ PrefixCache      (m04 hash 前缀缓存)            │
  │   ├─ Scheduler        (m03 continuous batching)     │
  │   └─ SamplingParams   (m10 sampling)                │
  │                                                     │
  └─────────────────────────────────────────────────────┘
```

## 设计原则
- 不展示所有 15 个模块的并联 (会糊在一起)
- 选**最常用的核心组合**: paged + cont batch + prefix + sampling
- 接口模仿 vLLM `LLM.generate(prompts, params)`
- 数值与单序列朴素 generate 一致

## 怎么把其他模块也接进来?
- **m05 Radix Cache**: 替换 `PrefixCache` 即可, 接口同
- **m06 Chunked Prefill**: 在 `engine.run_prefill` 里把 prompt 切片
- **m07 Speculative Decoding**: 加一个 `draft_lm`, 改 `decode_step` → spec_loop
- **m08 Quantization**: 用 `quantize_int8` 把权重压缩, 替换 `lm.w.layers[i].wq` 为 `QInt8Tensor`
- **m11 FlashAttention**: 用 `flash_attention` 替换 `attn_forward` 中的 softmax(QK)V
- **m14 Structured Output**: 在 sample 前应用 FSM mask

## 运行
```bash
python -m llm_infer.full_engine.demo
```
