# llm_basic — 从零手写一个最小 Transformer LM

零依赖（除了 numpy）的字符级语言模型，用来**演示训练和推理的全过程**：
前向、反向、优化器、采样、gradcheck — 全部肉眼可见。

适合作为读完 `ref/llama2.c` 后的下一步：把"工程实现的细节"剥掉，只看"数学是怎么流过模型的"。

---

## 文件结构

| 文件 | 作用 | 行数 |
|---|---|---|
| `prepare.py` | 下载 Tiny Shakespeare，做字符级编码，写 `train.bin` / `val.bin` / `meta.npz` | ~70 |
| `tokenizer.py` | 字符级 tokenizer（encode / decode，从 meta 加载） | ~40 |
| `model.py` | **核心。** 每个组件成对出现：`xxx_forward(...)` + `xxx_backward(...)`。Transformer = embedding + 1×Block(RMSNorm + 单头 causal attention + ReLU MLP + 残差) + RMSNorm + lm_head | ~370 |
| `optim.py` | Adam，30 行 | ~60 |
| `gradcheck.py` | 数值梯度 vs 解析梯度，逐参数验证 backward 正确 | ~140 |
| `train.py` | 训练循环：get_batch → forward → loss → backward → adam_step | ~150 |
| `sample.py` | 自回归采样（temperature + top-k） | ~110 |

---

## 为什么这样设计

- **不依赖 torch/jax**：每一次矩阵乘、每一行链式法则都看得见
- **numpy 当数组容器**：避开纯 Python list 的速度灾难，又不引入自动微分
- **每个组件 forward/backward 成对**：照着这个模式可以一键加任何新组件
- **参数用 dict 存、更新返回新 dict**：方便 gradcheck 临时改一个值再还原
- **float64 全程**：gradcheck 必须的精度

---

## 快速上手

```bash
cd llm_basic

# 1) 下载数据 + 编码（约 1MB，5 秒）
python prepare.py

# 2) 验证手写反向传播是否正确（必跑！约 1 秒）
python gradcheck.py

# 3) 训练（CPU 上 2000 步约 2 分钟，loss 4.17 → ~1.9）
python train.py

# 4) 生成
python sample.py "ROMEO:" --max-new 300 --temperature 0.8
python sample.py "" --top-k 10
```

---

## 模型结构

```
ids (B, T)
  │
  ▼
tok_emb(V, D) + pos_emb(T_max, D)         加法
  │
  ▼
┌─ Block ────────────────────────────┐
│  x → RMSNorm → Attn(单头, causal) ┐ │
│                                    + │
│  h → RMSNorm → MLP(ReLU) ──────────┐ │
│                                    + │
└──────────────────────────────────────┘
  │
  ▼
RMSNorm
  │
  ▼
lm_head(D, V) → logits (B, T, V)
  │
  ▼
softmax + cross-entropy → loss
```

默认超参（`train.py`）：
```
DIM = 64, HIDDEN_DIM = 128, SEQ_LEN = 64
BATCH_SIZE = 32, LR = 3e-4, MAX_ITERS = 2000
```
模型 ~45K 参数。

---

## gradcheck 输出示例

```
running gradcheck on a tiny model (vocab=8, dim=4, T=3, B=2) ...
criterion: |g_a - g_n| < atol + rtol * max(|g_a|, |g_n|)
           atol=1e-7, rtol=1e-4, eps=1e-4

  [OK ] block_0_attn_Wk  max_abs=2.42e-12  max_rel=8.53e-06
  [OK ] block_0_attn_Wo  max_abs=1.58e-09  max_rel=1.96e-07
  [OK ] block_0_attn_Wq  max_abs=3.22e-12  max_rel=1.74e-06
  ...
all gradients within tolerance — analytical backward looks correct.
```

如果哪一行写 `[BAD]`，去找对应的 `xxx_backward` 函数 —— 大概率是某个 transpose 写反、sum 维度搞错，或 softmax Jacobian 漏了一项。

---

## 阅读路径

1. 先读 `prepare.py` + `tokenizer.py` —— 看数据怎么进来
2. 读 `model.py` 从上往下：
   - `embedding` / `linear`：最简单的两个，热身
   - `rmsnorm`：注意 backward 里"耦合项"的来历
   - `attention`：最长，重点是 softmax 反向公式 `ds_i = a_i * (da_i - Σa_j da_j)`
   - `block` / `transformer`：把上面拼起来；注意残差的反向是"两条路梯度相加"
   - `cross_entropy_forward_backward`：fused loss，`dlogits = (probs - onehot)/N`
3. 读 `gradcheck.py` —— 理解为什么用 atol+rtol 组合判定
4. 读 `train.py` + `optim.py` —— 看循环怎么转
5. 读 `sample.py` —— 看推理怎么走

---

## 跟 `ref/llama2.c` 的对照

这个最小实现刻意做了简化，缺了什么、为什么：

| 简化项 | 真模型怎么做 | 为什么省略 |
|---|---|---|
| 学得式位置 embedding | RoPE 旋转编码 | RoPE 反向较复杂，先看绝对位置 emb |
| 单头 attention | Multi-Head / GQA | 多头只是 reshape 后的并行，原理一样 |
| ReLU + 普通 MLP | SwiGLU + 三个 linear | SwiGLU 反向多一项乘法，不影响骨架 |
| 1 层 | N 层 | 多层就是一个 for 循环 |
| 无 KV cache | 推理时缓存 K,V | 加上 cache 就能去掉 sample.py 里的"每步重算整个 forward" |
| Adam | AdamW + warmup + cosine | 学习率调度对 demo 不必要 |
| 全 float64 | bf16/fp16 | 学习模型时优先精度，不是速度 |

---

## 性能参考

- prepare.py: ~5 秒（含网络下载）
- gradcheck.py: ~1 秒
- train.py 2000 iters: ~2 分钟（M1 / 现代 CPU）
- sample.py 300 tokens: ~5 秒（朴素无 KV cache）

整个项目跑一遍也就 5 分钟，看代码 + 推公式才是大头。
