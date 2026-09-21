# VAR — next-scale prediction (Tian et al., 2024)

## 直觉

光栅序 next-token (VQGAN / LlamaGen) 把二维图像硬拉成一维, N 个 token 要 N 步。
VAR 像画家: 先定整体色调 (1×1), 再画大块 (2×2), 最后补细节 (4×4)。自回归的 "一步" 是一整张更细的 token map, 同一级内的 token 并行生成。

## 核心公式

- 多尺度残差量化 (`layers/diffusion/vq.py::MultiScaleVQ`, 共享码本):
  `r_k = nearest_code(down(f − f̂, s_k))`, `f̂ ← f̂ + up(e[r_k])` —— 每一级只量化 "前面还没解释掉的残差"
- 似然分解: `p(r_1..r_K) = Π_k p(r_k | r_<k)`, 级内各位置条件独立
- Transformer 输入: 第 k 级 = `down(f̂_{<k}, s_k)` (第 1 级是可学习 start token); 注意力 mask: `level(j) ≤ level(i)` 才可见
- tokenizer loss: `MSE(x̂, x) + mean_k[ ‖f̂_k − sg(f)‖² + β‖f − sg(f̂_k)‖² ]`, straight-through 传梯度

## 运行命令

```bash
python -m llm_models.run_models.generative.var.infer_var    # 结构自检
python -m llm_models.run_models.generative.var.train_var    # tokenizer → transformer 两阶段
```

## 运行后应该看到什么

`infer_var`: token 金字塔 `[(2,1,1), (2,2,2), (2,4,4)]`, L=21; mask 前 6 行是分块下三角;
"改级 2 的 token → 级 1/2 logits 变化 0.00e+00, 级 3 变化 4.95e-01"。

`train_var` (固定 8 张低频色块图, **loss 下降 = 背下这个 batch, 不代表泛化**):

| 阶段 | 指标 | 数值 |
|---|---|---|
| tokenizer 150 步 | recon MSE | 0.3700 → 0.0114 |
| | 码本 perplexity / 最大位移 | 19.4 → 45.6 / 0.336 (码本确实被训练) |
| transformer 100 步 | CE | 4.1927 → 0.2747 (ln 64 = 4.1589) |

CE 停在 0.27 而不是 0: 这个 batch 的理论下界就是 0.2476 —— 8 张图的 1×1 token 只有 3 种取值 (45 出现 4 次), 同一个粗 token 后面跟着不同的 2×2 map, 无条件模型只能学到分布。

## 常见误区

- "VAR 就是换了顺序的 next-token": 不是。序列里没有 shift-by-one, 位置 i 的输入不是 token i−1, 而是更粗尺度累计重建在该位置的特征。
- "需要 BOS token": 第 1 级的输入是一个可学习向量, 不占词表; 词表 = 码本, 采样结果永远是合法码字 (旧实现采到 BOS 后 clamp 成 0 是错的)。
- "各级 token 是同一张图的不同分辨率版本": 不是, 第 k 级编码的是 **残差**, 解码要把各级上采样后求和。
- "tokenizer 随便冻结一个就行": 随机码本下 token 与图像内容无关, AR 学到的只是噪声。

## 自测题

1. scales=(1,2,4) 时生成一张图要几次 Transformer 前向? 光栅序 4×4 要几次? —— 3 次 vs 16 次。
2. 最细一级的 token 会被用作 Transformer 的输入吗? —— 不会, 它只当 label (`infer_var` 第 4 个 assert: 改它, 所有 logits 不变)。
3. 同一级内并行采样隐含了什么假设? 代价是什么? —— 给定更粗尺度后级内 token 条件独立; 代价是 CE 下界 ≥ 真实联合熵 (本例 0.2476 > ln8/21 = 0.099)。
