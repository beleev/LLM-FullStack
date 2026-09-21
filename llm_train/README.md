# llm_train — 大模型规模化训练教学章节

> 用纯 `numpy` 单进程模拟大模型训练里的分布式、显存、数值和容错技术。一个 "rank" 就是 Python list 里的一个元素。
> 不依赖 GPU; 每个模块独立可跑 (< 1 秒), 全部跑完约 2 秒。

## 设计约定

- **原理优先**: 小张量 + 手写梯度, 数据流不藏在框架里。
- **每个 demo 以真实的 `assert` 结尾**: 凡数学上应与单卡基线等价的, 都断言等价 (多数是逐位相同)。
- **通信有账本**: `core.comm` 按 ring 算法记录每 rank 发送的字节, 各模块打印自己的通信成本。
- **共享件进 `core/`**: 通信原语 (含真实逐步传 chunk 的 ring all-reduce)、低精度浮点假量化 (BF16/FP8/FP4 共用)、Adam、checkpoint 读写、溢出检测。
- **每个模块目录有 README**: 直觉 / 核心公式 / 运行后应该看到什么 / 与真实系统的差距 / 常见误区 / 3 道自测题。

## 学习路径

```
切 batch      m01 梯度累积 → m02 DDP
切模型        m03 张量并行 → m04 流水线 (GPipe / 1F1B / Interleaved) → m11 专家并行
切状态        m05 ZeRO-1/2/3 · FSDP
切序列        m12 Context Parallel (Ring + zigzag) → m16 Ulysses
省显存        m07 激活重算
数值格式      m06 FP16/BF16 混合精度 → m13 FP8 → m15 FP4 (MXFP4 / NVFP4)
优化与稳定    m10 cosine / WSD · clip · NaN guard → m14 Muon
通信与容错    m09 通信原语 → m08 checkpoint / resume
合起来        full_loop
```

## 模块清单

| # | 模块 | 内容 | 关键断言 |
|---|------|------|----------|
| 01 | [Gradient Accumulation](m01_gradient_accumulation/) | micro-batch 加权累积 | == 大 batch; 不缩放恰好 ×K |
| 02 | [Data Parallel](m02_data_parallel/) | 梯度 all-reduce | == 单卡; 副本逐位一致; 通信 = 2(N-1)/N·\|g\| |
| 03 | [Tensor Parallel](m03_tensor_parallel/) | 列切 + 行切, f / g 算子 | 输出、全部梯度、**dX** == dense |
| 04 | [Pipeline Parallel](m04_pipeline_parallel/) | GPipe / 1F1B / Interleaved 1F1B, B 占 2 格 | 1F1B 峰值 == [PP-s]; 气泡 == (PP-1)/(vM+PP-1) |
| 05 | [ZeRO / FSDP](m05_zero_fsdp/) | 真 Adam, stage 1/2/3 三条代码路径, gather→compute→free | 各 stage 与 DDP 逐位相同; 字节 == 2+2+12 公式 |
| 06 | [Mixed Precision](m06_mixed_precision/) | FP16 vs BF16 (模拟), 真实 fp16 溢出, 动态 scaler, master | 下溢/上溢/舍入各一条断言 |
| 07 | [Activation Checkpointing](m07_activation_checkpointing/) | 真实分段重算反向 | 梯度逐位相同; 峰值 == L/k + k |
| 08 | [Checkpoint / Resume](m08_checkpoint_resume/) | 参数 + 优化器 + 数据 seed/cursor + RNG | 续训逐位相同; 漏任一项都偏 |
| 09 | [Collectives](m09_collectives/) | 四个原语 + 逐步 ring all-reduce + 字节计数 | ring 实测字节 == 公式 |
| 10 | [Training Stability](m10_training_stability/) | cosine / **WSD**, 全局裁剪, NaN guard | WSD 稳定段与总步数无关 |
| 11 | [Expert Parallel](m11_expert_parallel/) | 两次**真实** all-to-all, capacity, aux loss, aux-loss-free bias | EP == 单卡; 两种均衡都降低丢弃 |
| 12 | [Context Parallel](m12_sequence_parallel/) | Ring Attention, online softmax, **zigzag** | == 完整注意力; zigzag 每卡工作量相同 |
| 13 | [FP8 Training](m13_fp8_training/) | E4M3/E5M2, TE vs DeepSeek-V3 配方, 三臂消融 | scaling 29×, master 584×, 粒度仅在 outlier 下有效 |
| 14 | [Muon](m14_muon_optimizer/) | Newton–Schulz 正交化动量, QK-clip | 非轴对齐病态问题上胜过调好的 AdamW |
| 15 | [FP4 Microscaling](m15_fp4_microscaling/) | E2M1, MXFP4, NVFP4, 对照 FP8 / INT4 | 误差 FP8 < NVFP4 < MXFP4 |
| 16 | [Ulysses](m16_ulysses_sequence_parallel/) | all-to-all 序列切 ↔ 头切 | == 完整注意力; 通信随 P 下降; P ≤ 头数 |
| ★ | [Full Loop](full_loop/) | DP × 累积 × AMP × 分片 Adam × clip × NaN guard × 分片 checkpoint | ≈ 单卡; 续训逐位相同 |

## 运行

```bash
python -m llm_train.m05_zero_fsdp.demo      # 单模块
python -m llm_train.run_all                 # 全部 (任一 assert 失败即失败)
```

## 业界覆盖度自评

| 训练技术 | 覆盖 | 说明 |
|---|:---:|---|
| Gradient accumulation | yes | `m01`, `full_loop` |
| Data parallel / DDP | partial | `m02`; 无 bucket、无通信-计算重叠 |
| Tensor parallel | partial | `m03` MLP 的 fwd+bwd; 无 attention 切分、无 Megatron sequence parallel |
| Pipeline parallel | partial | `m04` 三种调度的时间表仿真; 不真的算梯度; 无 zero-bubble / DualPipe |
| ZeRO-1/2/3, FSDP | yes | `m05`; 无 prefetch / HSDP / offload |
| Mixed precision FP16 / BF16 | yes | `m06`, `full_loop`; BF16 为模拟 |
| Activation checkpointing | partial | `m07` 整段重算; 无选择性重算 |
| Checkpoint / resume | partial | `m08` 单文件, `full_loop` 每 rank 分片; 无 resharding、无异步落盘 |
| Collectives | partial | `m09`; 只计带宽项, 无延迟项 / 拓扑 / tree 算法 |
| LR schedule / clipping / NaN guard | yes | `m10` cosine + WSD |
| Expert parallel (MoE) | partial | `m11` 前向, top-1; 无反向、无 top-k / shared expert |
| Context parallel (Ring) | partial | `m12` 前向 + zigzag; 无反向 |
| Sequence parallel (Ulysses) | partial | `m16` 前向 |
| FP8 training | partial | `m13` 假量化 + 线性回归消融; 无真实 FP8 GEMM、无 delayed scaling |
| FP4 (MXFP4 / NVFP4) | partial | `m15` 仅量化误差, 未做 FP4 训练 |
| Muon / MuonClip | partial | `m14` 玩具问题; 无分布式 Muon |
| 多维并行组合 (3D/4D/5D) | no | 各维度单独演示, 未组合 |
| 通信-计算重叠 | no | 单进程顺序执行 |
| Real NCCL / 多进程 | no | 教学模拟 |

读完后再看真实框架, 可以把名词映射到几条主线:
**batch 怎么切、层怎么切、状态怎么切、序列怎么切、数用几位存、通信怎么走、坏 step 怎么恢复。**
