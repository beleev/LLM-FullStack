# M11 — FlashAttention: 不存 attention 矩阵

## 痛点
标准 attention 公式:
```
  S = Q @ K.T              (T, T)        ← 显存峰值, T=4096 已 16M 元素
  P = softmax(S)           (T, T)
  O = P @ V                (T, D)
```
**S 与 P 这两块 (T,T) 矩阵会写到 HBM 再读回**, T 大时是显存带宽瓶颈
(GPT-3 forward 一半时间花在读写 attention 矩阵)。

## 思路 (Dao et al. 2022 FlashAttention)
**永不 materialize 完整的 (T,T) 矩阵。** 把 Q/K/V 切成 block, 流式
"online softmax" 累计输出, 只在 SRAM 内做一小块的乘加。

### Online softmax 的关键技巧
传统 softmax 需要先扫一遍求 max 与 sum, 再扫一遍归一化:
```
  m = max(S)              ← 全局信息
  l = sum(exp(S - m))     ← 全局信息
  P = exp(S - m) / l
```
online 版本: 一次过, 边算边维护 (m_t, l_t):
```
  当前块 S_b 的局部 max m_b
  m_new = max(m_t, m_b)
  scale = exp(m_t - m_new)              ← 之前累计的要重新归一
  l_new = scale * l_t + sum(exp(S_b - m_new))
  O_new = scale * O_t + (exp(S_b - m_new) @ V_b)
  m_t, l_t, O_t = m_new, l_new, O_new
最后: O = O_t / l_t
```
这就是 "log-sum-exp 增量更新", 数值上等价。

## 收益
- 显存: O(T²) → O(T)  (只存 m, l, O 三个 T 长向量)
- 速度: HBM 读写大量减少, 长序列下 2~10x 加速
- FA-2 / FA-3: 更细的 tile 调度 + FP8

## 本模块
- `flash_attention(Q, K, V, block_size)` — 单头, numpy 实现 online softmax
- 与朴素 `softmax(QK)V` 对比: max abs diff < 1e-5

## 运行
```bash
python -m llm_infer.m11_flash_attention.demo
```
