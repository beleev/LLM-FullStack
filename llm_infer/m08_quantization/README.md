# M08 — Quantization: 同样的 bit 数, 差别全在"组"怎么划

## 直觉
量化 = 把浮点数吸附到 2^bits 个等间距格点上, 每"组"数共用一个 scale。格距 = 组内 (max−min)/(2^bits−1),
所以**组里混进一个离群值, 整组的精度一起陪葬**。LLM 的离群值有结构: 激活和 K 的少数固定通道在所有 token 上都很大。
- KV cache (KIVI): K 按通道分组, 把离群通道关进自己的组; V 没有固定离群通道, 按 token 分组。
- 权重 (AWQ): 误差要在输出上看, `Σ_i x_i·ΔW_i` —— 激活大的输入通道对应的权重行最该保护, 量化前先把它们放大。

## 核心数据结构或公式
```
非对称 RTN (int8_weight.quantize_affine):  scale=(max−min)/(2^b−1)   q=round((x−min)/scale)   x̂=q·scale+min   |x−x̂| ≤ scale/2
对称 INT8  (quantize_int8):                scale=max|x|/127          q=round(x/scale)
```
| 方案 | 统计 min/max 的范围 | scale 形状 |
|---|---|---|
| per-tensor | 整个张量 | 标量 |
| per-token (K/V) | 一行 (一个 token 的 D 维) | (T,1) |
| per-channel (KIVI 的 K) | 每 32 个 token 内的一列 | (T/32,1,D) |
| group-wise INT4 权重 | 每个输出通道内相邻 g=32 个输入通道 | (D_in/g,1,D_out) |

AWQ (`int4_awq.awq_quantize`): `s_i = mean|x_i|^α` (归一化), `Ŵ = Q(s⊙W)/s`, 因为 `XW = (X/s)(s⊙W)` 数学不变;
α 在校准集上网格搜索 {0,0.1,…,1}, α=0 即 RTN。scale/lo 用 fp16 存, bytes 按 bit-pack 后计。

## 运行后应该看到什么
```bash
python -m llm_infer.m08_quantization.demo
```
```
[A] W (256,256), 激活离群输入通道 [7,50,131,200] ×30, 误差在 held-out 激活上
  方案                          输出误差    bytes  bit/权重  vs FP16
  RTN INT8 per-channel (对称)     0.0068   66,560     8.12    1.97x
  RTN INT4 per-channel (非对称)   0.1058   33,792     4.12    3.88x
  RTN INT4 group=32 (非对称)      0.0759   40,960     5.00    3.20x
  AWQ INT4 group=32 (α=0.4)       0.0273   40,960     5.00    3.20x
  α 网格 (calib): 0:0.0757  0.2:0.0380  0.4:0.0279  0.6:0.0362  0.8:0.0592  1:0.1173
[B] K (128,64), 离群通道 [3,17,40] ≈ ±8; FP32 65,536 B / FP16 32,768 B
  bits scheme      K 相对误差  V 相对误差  attn max|Δ|   bytes  vs FP32  vs FP16
     8 per-token      0.0105     0.0053      0.0036   17,408    3.76x    1.88x
     8 kivi           0.0015     0.0053      0.0021   17,920    3.66x    1.83x
     4 per-token      0.1800     0.0894      0.0911    9,216    7.11x    3.56x
     4 kivi           0.0264     0.0894      0.0216    9,728    6.74x    3.37x
     2 per-token      1.0241     0.4470      1.1913    5,120   12.80x    6.40x
     2 kivi           0.1337     0.4470      0.1097    5,632   11.64x    5.82x
```
assert: 输出误差 INT8 < AWQ INT4 < RTN group < RTN per-channel, 且 AWQ < 0.7×RTN group, 0<α<1;
KIVI 的 K 误差与 attention 误差在 8/4/2 bit 都小于 per-token; KIVI attention max|Δ| (vs `core.dense_attention`) INT8 < 0.01, INT4 < 0.05。
数据是人工合成的, 离群通道被刻意放大; INT8 下各方案都够用, 差距在 INT4/INT2 才拉开。

## 与真实系统的差距
- 没有 bit-packing 和 INT4 GEMM kernel (Marlin / AWQ kernel): 这里反量化回 fp32 再算, 只验证数值。
- 真 AWQ 逐层用真实校准数据, 还搜索 clipping 阈值; 1/s 离线折进上一层的 RMSNorm gamma / Linear, 推理零开销。GPTQ 走另一条路 (用 Hessian 逐列补偿误差)。
- 真 KIVI 把最近不满一组的 token 留在 fp16 residual 里 (per-channel 需要凑满一组才有 min/max), V 也再按 32 通道分组; 本模块要求 T 是 32 的倍数。
- vLLM 生产上常用的是 FP8 KV (per-tensor scale), 硬件原生支持, 比 INT4 KV 省心。

## 常见误区
- "INT8 KV 显存砍半" —— 相对 FP16 约 1/2, 相对 FP32 约 1/4; 还要扣 scale 开销 (本例 3.66× / 1.83×)。
- "INT4 就是 4 bit/权重" —— group=32 + fp16 scale + fp16 zero → 5.00 bit; group 越小越准也越贵。
- "per-token 对 KV 总是够用" —— K 有固定离群通道时, 每一行都被它撑大 scale; INT2 下 per-token 的 K 误差 >100%。
- "AWQ 的 α 越大越好" —— 放大的行会撑大所在 group 的 range, 同组其它行变粗; α=1 (0.1173) 比 RTN (0.0757) 还差。
- "weight-only 量化省算力" —— 激活仍是浮点, 省的是显存与带宽; decode 是 memory-bound 所以照样变快。

## 自测题
1. 为什么 per-channel 的 K 量化不能像 per-token 那样"来一个 token 量化一个"?
   **答**: 通道的 min/max 要跨 token 统计, 新 token 可能超出已有范围; 所以 KIVI 攒满一组 (如 32 个) 再量化, 最近的 token 留在 fp16 residual。
2. W 是 (4096,4096), INT4, group=128, fp16 scale + fp16 zero, 等效多少 bit/权重?
   **答**: 4 + (16+16)/128 = 4.25 bit。
3. AWQ 为什么不直接把 salient 通道保留成 FP16 (混合精度), 而要用缩放?
   **答**: 混合精度破坏了规整的内存布局, kernel 难写且慢; 缩放后所有权重仍是同一种 INT4 格式, 1/s 折进上一层, 硬件友好。
