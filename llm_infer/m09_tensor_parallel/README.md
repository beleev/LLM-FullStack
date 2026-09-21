# M09 — Tensor Parallelism: 把一层的矩阵切给多张卡

## 直觉
LLaMA-70B 的 fp16 权重 140 GB, 一张 80 GB 的卡装不下。Tensor Parallel (Megatron-LM) 把**每一层**的
权重矩阵切成 tp 份, 每张卡只存、只算自己那份; 激活 x 在每张卡上都有完整副本。
巧妙之处在于"先列切、后行切"配对: 中间结果天然是切开的, 整个子层只需要在末尾做 **1 次 all-reduce**。

## 核心数据结构或公式
```
列切 (按输出维): W=[W_1|…|W_tp]   X@W = concat_r(X@W_r)     无通信, 输出天然按最后一维切开
行切 (按输入维): W=[W_1;…;W_tp]   X@W = Σ_r X_r@W_r         求和 = 1 次 all-reduce
```
一个 block (`parallel_linear.py: tp_block`):
```
attention:  Q/K/V 列切 → 每卡 n_head/tp 个完整的头, 各自 softmax → O 行切 → all-reduce #1
MLP:        gate/up 列切 → silu(gate)*up 逐元素, 各卡独立        → down 行切 → all-reduce #2
RMSNorm gamma: 很小, 每卡复制
```
- 列切边界必须落在 **head 边界**: softmax 在头内归一化, 头是 attention 的最小切分单位 (所以要求 n_head % tp == 0)。
- rank 本地计算就是同一个 `mha` / `swiglu`, 只是传入更窄的权重; 每卡输出的 (T,D) 是"本卡那部分的贡献", 求和即全量。
- 每次 all-reduce 载荷 = T×D 个激活, 与 tp 无关; ring all-reduce 每卡实际发送 2(tp-1)/tp × 载荷。

## 运行后应该看到什么
```bash
python -m llm_infer.m09_tensor_parallel.demo
```
配置 T=16 D=64 n_head=8 d_mlp=192, fp32:
```
[0] 列切 concat vs X@W = 0.00e+00      行切 Σ vs X@W = 1.79e-07
[1][2]
 tp     max|Δ|  all-reduce/block  载荷 bytes  ring 每卡发送  每卡权重 bytes  每卡头数
  1   0.00e+00                 0           0              0         213,504         8
  2   9.54e-07                 2       8,192          8,192         107,008         4
  4   7.15e-07                 2       8,192         12,288          53,760         2
[3] max|单头 - 硬切 4 份| = 5.63e-01   ← 不按 head 边界切, 结果就错
```
assert: max|Δ| < 1e-5 (基线是不切分的 `dense_block`, attention 用 `core.dense_attention`);
all-reduce 次数 == 2; 每卡权重 == 矩阵/tp + 2 个 gamma。~1e-7 的差异只来自浮点求和顺序。

## 与真实系统的差距
- 这里的 rank 是 Python 列表, all-reduce 是顺序求和, **不含通信耗时**。真实系统里 all-reduce 在关键路径上,
  TP 通常只在单机 NVLink 内用 (tp ≤ 8), 跨机改用 pipeline parallel。
- vLLM 把 Q/K/V 合成一个 `QKVParallelLinear`, gate/up 合成 `MergedColumnParallelLinear`, 各一次 GEMM。
- GQA/MQA 下 KV 头数可能 < tp, 需要复制 KV 头; embedding / lm_head 按词表切 (VocabParallelEmbedding)。
- 没演示: KV cache 也随头切成 1/tp (每卡只存自己头的 KV); sequence parallel (把 norm/dropout 的激活也切开)。

## 常见误区
- "TP 输出只是近似一致" —— 数学上严格相等; 若基线用单头、TP 侧却按 4 个头各自 softmax, 才会看到差异 (见 [3])。
- "QKV 列切后要先 all-gather 再算 attention" —— 不用, 头之间本来就独立, 直到 O 投影后才需要求和。
- "tp 越大通信载荷越小" —— 每次 all-reduce 的载荷恒为 T×D; tp 变大只让每卡的计算变少, 通信占比反而上升。
- "激活也被切了" —— 标准 TP 里子层的输入/输出 x 在每卡上是完整副本, 切的是权重和子层内部的中间激活。

## 自测题
1. 为什么 MLP 是"先列切再行切", 反过来 (先行切再列切) 行不行?
   **答**: 先行切的输出要 all-reduce 后才能进激活函数 (silu(a+b) ≠ silu(a)+silu(b)), 会多一次通信; 列切的输出按元素独立, 可以直接过 silu。
2. 32 层模型, tp=4, decode 一步 (T=1, D=4096, fp16), 一共几次 all-reduce? 每次载荷多大?
   **答**: 32×2 = 64 次; 每次 1×4096×2 = 8 KB。载荷很小 → decode 的 TP 开销由延迟 (次数) 主导, 而非带宽。
3. n_head=8, tp=3 能切吗? n_head=8, n_kv_head=2, tp=4 呢?
   **答**: tp=3 不能 (8 % 3 ≠ 0, 头不可再分)。第二种 Q 头可切 (每卡 2 个), 但 KV 头只有 2 个 < 4, 需要把每个 KV 头复制到 2 张卡上。
