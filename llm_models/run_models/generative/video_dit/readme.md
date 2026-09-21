# Video DiT — 把 DiT 搬到视频 (Sora-lite)

## 直觉

视频 latent 是 5D `[B, C, T, H, W]`。和图像 DiT 相比只改两处: 用 Conv3d 把时空体切成 tubelet 当 token; 位置嵌入要同时编码 "第几帧" 和 "哪个空间位置"。
adaLN-Zero、训练目标、DDIM、CFG 全部照搬。输入 latent 来自 Causal 3D VAE (见 `../vae3d`)。

## 核心公式

- token 数 `N = (T/p_t)·(H/p)·(W/p)`, 注意力 O(N²) —— 视频生成的算力瓶颈
- 分解式位置嵌入: `pos[t, s] = time_pos[t] + space_pos[s]`, 参数量 `T' + H'W'` 而非 `T'·H'W'`
- loss 同 DiT: `MSE(model(x_t, t, y), ε)`

## 运行命令

```bash
python -m llm_models.run_models.generative.video_dit.infer_video_dit
python -m llm_models.run_models.generative.video_dit.train_video_dit
```

## 运行后应该看到什么

`infer_video_dit`: latent (4,8,8)、tubelet (2,2,2) → 32 个 token; 位置嵌入参数 2304 (分解式) vs 4096;
交换两个 tubelet 后的输出差: 有 time_pos 2.816e-02, time_pos 清零 2.384e-07 (只剩浮点误差); DDIM 10 步输出 `|x|max = 3.000` (clip_x0=3)。

`train_video_dit`: loss 1.0163 → 0.0119 (初始 ≈ E[ε²] = 1; **固定 batch, 下降 = 背下 2 段 latent 的 ε**)。

## 常见误区

- "Transformer 天然知道帧顺序": 自注意力对 token 置换等变, 顺序信息全靠位置嵌入 —— 上面 time_pos 清零的实验就是证据。
- "视频模型必须拆成 spatial + temporal 两种注意力": 那是省算力的工程选择; 本教学版对全部时空 token 做一次完整注意力, 概念更简单。
- `p_t > 1` 时一个 token 跨多帧, 帧内的先后由 Conv3d 的权重区分, 不是位置嵌入。

## 自测题

1. latent (16, 32, 32)、tubelet (1, 2, 2) 有多少 token? 注意力矩阵多少元素? —— 16·16·16 = 4096; 4096² ≈ 1.68e7 (每头每层)。
2. 分解式位置嵌入表达不了什么? —— 时间与空间的交互项 (某位置只在某帧特殊), 需靠网络后续层弥补。
3. unpatchify 的 8 维 permute 要把什么对齐? —— 把 (T', p_t)、(H', p)、(W', p) 两两相邻, 再 view 合并成 T、H、W。
