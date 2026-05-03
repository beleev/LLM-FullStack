# M09 — Tensor Parallelism: 把矩阵切给多卡

## 痛点
LLaMA-70B 权重 fp16 = 140 GB, 单卡 H100 80 GB 装不下。

## 思路
**把每层的权重矩阵在某个维度切成 N 份, 分给 N 张卡, 每张卡只算自己负责的那块。**
两种基本切法 (Megatron-LM 命名):

### Column-parallel (按输出维切)
```
  Y = X · W              shape: (B,T,D_in) · (D_in, D_out) → (B,T,D_out)

  W 切成 [W_1 | W_2 ... | W_N], 每块 (D_in, D_out/N)
  rank i: Y_i = X · W_i              (B,T,D_out/N)
  最后:   Y = concat([Y_1, ..., Y_N], axis=-1)
  
  通信: 无 (前向); 后续算子如果按 D_out 切就直接接上, 不必 gather
```

### Row-parallel (按输入维切)
```
  W 切成 [W_1; W_2; ...; W_N], 每块 (D_in/N, D_out)
  X 也按列切: X = [X_1 | X_2 | ... | X_N]      (B,T,D_in/N)
  rank i: Y_i = X_i · W_i              (B,T,D_out)  ← 都是部分和
  最后:   Y = sum(Y_1, ..., Y_N) = ALL_REDUCE
  
  通信: 一次 all-reduce
```

### Megatron 标准模式 (TransformerBlock)
```
  attention:  QKV proj   = column-parallel  (输出按头切, 每卡几个头)
              Out proj   = row-parallel    (head 维 → all-reduce)
  MLP:        Up/Gate    = column-parallel
              Down       = row-parallel    (一次 all-reduce)

  → 每个 sub-layer 一次 all-reduce, 总通信 2 次/Block
```

## 本模块演示
单进程模拟 N "rank", 用 numpy 数组列表代替分布式张量。
关键代码:
- `column_parallel_linear(x, W_shards) → [Y_1, ..., Y_N]`
- `row_parallel_linear(x_shards, W_shards) → Y_full   (all_reduce)`
- 数值验证: column + row = 单卡 dense matmul

## 运行
```bash
python -m llm_infer.m09_tensor_parallel.demo
```
