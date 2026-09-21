# MM-DiT — SD3 / FLUX 的双流扩散 Transformer + Flow Matching

## 直觉

DiT 只能经 adaLN 注入一个全局向量, 整句文本被压成一个点。MM-DiT 把文本 token 和图像 patch token 拼成一条序列做联合注意力,
但两种模态分布差异大, 所以 QKV / FFN / adaLN 各用一套参数: **参数分, 注意力合**。
训练目标换成 Rectified Flow: 噪声与数据之间走直线, 学这条直线的速度。

## 核心公式

- `x_t = (1−t)·x_0 + t·ε`, `v = dx_t/dt = ε − x_0`, `loss = MSE(model(x_t, t·1000, text), v)`
- Euler 采样: `x_{t−Δt} = x_t − Δt·v̂`, 从 t=1 走到 0
- 联合注意力: `Attn(cat[q_img,q_txt], cat[k_img,k_txt], cat[v_img,v_txt])`, 再切回两条流

## 运行命令

```bash
python -m llm_models.run_models.generative.mmdit.infer_mmdit
python -m llm_models.run_models.generative.mmdit.train_mmdit
```

## 运行后应该看到什么

`infer_mmdit`:
- `cos(emb(t=0.1), emb(t=0.9))`: 裸 t = **0.981**, scheduler 输出的 t_norm (×1000) = **0.167**
- 换文本 token → 图像流输出变化 1.617e-02 (联合注意力通了); 换 t → 4.726e-02
- Euler 20 步, 模型看到的 t: 1000 → 50

`train_mmdit`: 喂给模型的 t = `[161.28, 995.03]`; loss 2.1937 → 0.0005。
初始 ≈ 2: 零初始化输出 0 → `MSE = E[(ε−x_0)²] = Var(ε)+Var(x_0) = 2` (只有 512 个元素, 估计标准差 ≈ 0.125)。
**固定 batch, 下降 = 背下 2 个样本**。

## 常见误区

- "Flow Matching 的 t∈[0,1] 直接喂 TimestepEmbedding": sinusoidal 频率族 (max_period=10000) 是为跨度上千的位置设计的, [0,1] 内大部分频率几乎不动: 不缩放时 t=0.1 与 t=0.9 的嵌入余弦相似度高达 0.98, 模型根本分不清早晚。SD3 同样把 t ×1000。插值系数仍用 t∈[0,1], 只有 **给模型看的 t** 要缩放; 采样器必须用同一量纲 (`EulerFlowSampler.time_scale`)。
- "双流 = 两个独立 Transformer": 注意力是共享的一次 softmax, 文本 token 能看图像 token, 反之亦然。
- "velocity 依赖 t": 直线路径上 v = ε − x_0 与 t 无关; 依赖 t 的是模型的输入 x_t。

## 自测题

1. 为什么 Flow Matching 的初始 loss ≈ 2 而 DDPM ≈ 1? —— target 是 ε−x_0, 两个独立单位方差之差, 方差为 2。
2. 图像 64 token、文本 16 token, 联合注意力矩阵多大? —— 80×80 (每个头)。
3. Euler 20 步时最后一次前向的 t 为什么是 50 而不是 0? —— t 取每步的起点 1, 0.95, …, 0.05; 走完最后一步才到 0。
