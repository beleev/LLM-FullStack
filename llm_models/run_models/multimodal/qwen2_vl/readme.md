# Qwen2-VL (视觉前缀 + LLM)

## 直觉
把图片"翻译"成一串 LLM 能读的 token, 放在文本前面当前缀。LLM 本身一行不改: 因果注意力让后面的文本自然看得到前面的图。

## 核心公式
```
v = Projector(Resampler(ViT(image)))          # [B, N_v, D], Resampler 可关
x = concat([v ; emb(text)·√D])                 # [B, N_v + T, D]
logits = LLM(x);   loss = CE 只在文本位置 (视觉位置 label = -100)
M-RoPE: head 维切 3 段, 分别用 (时间, 行, 列) 位置旋转; patch = (0, r, c), 文本 = (p, p, p)
```

## 运行命令
```bash
python -m llm_models.run_models.multimodal.qwen2_vl.infer_qwen2_vl
python -m llm_models.run_models.multimodal.qwen2_vl.train_qwen2_vl
```

## 运行后应该看到什么 (CPU 实测, 两个脚本合计约 10 秒)
- infer A: `换图→文本 logits 变化 0.0034 | 换文本→视觉位置变化 0.0e+00 | 改 pad token→其余位置变化 0.0e+00`
- infer B: `对调行列坐标→输出变化 5.3e-05 | 文本位置整体 +9 → logits 变化 2.4e-07 | 与 1-D RoPE 的数值差 0.0e+00`
  —— 三轴共用**同一条频率轴** (按段分给 T/H/W), 所以纯文本时与 1-D RoPE 逐位相等 (已断言);
  H/W 分到的是较低频率, 4×4 小网格上未训练时影响只有 1e-5~1e-4 量级, 但非零。
- train: 2 条样本文本相同、图不同、标签不同。初始 loss 6.194 (ln 500 = 6.215) → 100 步后 0.198 < ln 2 = 0.693 ⇒ 文本位置读到了视觉前缀。
  若换成默认 N(0,1) 初始化 + 绑权重: 初始 loss **130.29**, 30 步后仍是 128.51 (几乎不动)。
- 数据是固定随机 batch: 记忆, 不是看图说话。

## 常见误区
- "视觉 token 也要预测下一个 token" —— 不用, 视觉位置 label = -100; 它们只是条件。
- "图文之间是双向注意力" —— 本实现 (与多数 VLM) 整条序列都是因果的: 视觉 token 看不到文本 (infer A 第 2 条断言)。
- "Resampler 和 M-RoPE 可以一起用" —— Resampler 输出的 latent 不再对应网格位置, 行列坐标失去意义; 原版 Qwen2-VL 不用 Resampler, 用 2×2 patch 合并。

## 自测题
1. 224×224 图、patch 14, 不压缩时占多少 LLM 上下文? —— 16×16 = 256 个 token。
2. 为什么文本 token 的 M-RoPE 三轴取同一个值? —— 三段旋转角都只依赖同一个位置差 i−j, 整体仍是 1-D 相对位置编码, 纯文本能力不受影响。
3. 训练脚本为什么要让两条样本文本相同? —— 否则模型光靠文本就能背下标签, loss 下降证明不了它用了图。
