# llm_train — 大模型规模化训练教学章节

> 用纯 `numpy` 单进程模拟大模型训练里最常见的分布式、显存、数值和容错技术。
> 代码不依赖 GPU, 每个模块都能独立运行, 也能通过 `full_loop/` 组合成一个最小训练闭环。

## 设计目标

- **原理优先**: 用小张量和手写梯度展示数据流, 不把关键逻辑藏进框架。
- **模块自治**: 每个 `mXX_*/demo.py` 都可以单独运行。
- **可组合**: `full_loop/demo.py` 把 DDP、梯度累积、混合精度、裁剪、ZeRO-style update 和 checkpoint 串起来。
- **CPU 可跑**: 只需要 `numpy`, 没有 GPU/NCCL/torch.distributed 依赖。

## 学习路径

```
m01 Gradient Accumulation  显存不够时拆 micro-batch
        ↓
m02 Data Parallel / DDP    batch 维并行 + 梯度 all-reduce
        ↓
m03 Tensor Parallel        层内矩阵切分 + 层内通信
        ↓
m04 Pipeline Parallel      层间切分 + micro-batch 流水线
        ↓
m05 ZeRO / FSDP            参数/梯度/优化器状态分片
        ↓
m06 Mixed Precision        FP16/BF16 思路 + loss scaling
        ↓
m07 Activation Checkpoint  重算换显存
        ↓
m08 Checkpoint / Resume    参数、优化器、数据游标、随机状态恢复
        ↓
m09 Collectives            all-reduce / reduce-scatter / all-gather / all-to-all
        ↓
m10 Stability              warmup、cosine、grad clip、NaN 检测
        ↓
full_loop                  组合成一个最小分布式训练闭环
```

## 模块清单

| # | 模块 | 覆盖的业界技术 |
|---|------|----------------|
| 01 | [Gradient Accumulation](m01_gradient_accumulation/) | micro-batch、大 batch 等效 |
| 02 | [Data Parallel / DDP](m02_data_parallel/) | replica、local grad、gradient all-reduce |
| 03 | [Tensor Parallel](m03_tensor_parallel/) | Megatron column/row parallel |
| 04 | [Pipeline Parallel](m04_pipeline_parallel/) | stage、micro-batch、pipeline bubble |
| 05 | [ZeRO / FSDP](m05_zero_fsdp/) | optimizer/grad/param sharding |
| 06 | [Mixed Precision](m06_mixed_precision/) | FP16 下溢、loss scaling、FP32 master weight |
| 07 | [Activation Checkpointing](m07_activation_checkpointing/) | save activation vs recompute |
| 08 | [Checkpoint / Resume](m08_checkpoint_resume/) | optimizer state、data cursor、RNG state |
| 09 | [Collectives](m09_collectives/) | all-reduce、reduce-scatter、all-gather、all-to-all |
| 10 | [Training Stability](m10_training_stability/) | warmup cosine、global grad clip、NaN/Inf 检测 |
| ★ | [Full Loop](full_loop/) | 多技术组合的训练主循环 |

## 运行

```bash
# 单模块
python -m llm_train.m01_gradient_accumulation.demo
python -m llm_train.m02_data_parallel.demo
python -m llm_train.m05_zero_fsdp.demo

# 整体跑一遍
python -m llm_train.run_all

# 组合闭环
python -m llm_train.full_loop.demo
```

## 业界覆盖度自评

| 训练技术 | 本目录 | 说明 |
|---|:---:|---|
| Gradient accumulation | yes | `m01`, `full_loop` |
| Data parallel / DDP | yes | `m02`, `full_loop` |
| Tensor parallel | yes | `m03` |
| Pipeline parallel | yes | `m04` |
| ZeRO / FSDP | yes | `m05`, `full_loop` |
| Mixed precision / loss scaling | yes | `m06`, `full_loop` |
| Activation checkpointing | yes | `m07` |
| Distributed checkpoint | partial | `m08` 演示状态完整性, 未做多文件 shard |
| Communication collectives | yes | `m09` |
| Warmup / cosine / clipping / NaN guard | yes | `m10` |
| Expert parallel / MoE all-to-all | partial | `m09` 展示 all-to-all, MoE 模型见 `llm_models` |
| Real NCCL / multi-process launch | no | 教学模拟, 不依赖 GPU |

读完本目录后, 再看真实框架时可以把名词映射成几条主线:
**batch 怎么切、层怎么切、状态怎么切、通信怎么走、坏 step 怎么恢复。**

