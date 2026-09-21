# llm\_basic — 只用 numpy 手写一个 GPT（前向 + 反向）

没有 torch、没有自动微分。字符级语言模型的训练与推理全过程——前向、反向、优化器、采样、梯度检查——每一步都是看得见的 numpy 代码。

## 直觉

- 语言模型只做一件事：给定前文，输出"下一个 token"的概率分布。训练 = 让真实下一个 token 的概率变大（交叉熵变小）。
- `loss.backward()` 不是魔法：每个算子写一对 `xxx_forward(...) -> (out, cache)` / `xxx_backward(dout, cache) -> (dx, *dparams)`，整个模型的反向就是**把前向倒着走一遍**，把 `dout` 一层层往回传。
- 残差 `out = x + f(x)` 的反向是"两条路的梯度相加"；一个张量被用了几次（x 同时喂给 Q/K/V、同一个 token 在 batch 里出现多次），它的梯度就是几路之和。
- 手写反向极易写错且**不会报错**，所以必须有 `gradcheck.py`：用中心差分算"不可能错"的数值梯度来对答案。

```
ids [B,T] → tok_emb[V,D] + pos_emb[T_max,D]
          → n_layer × { x + Attn(RMSNorm(x));  h + MLP(RMSNorm(h)) }     单头 causal attention，ReLU MLP
          → RMSNorm → lm_head[D,V] → logits [B,T,V] → softmax + CE
```

| 文件 | 作用 |
| --- | --- |
| `prepare.py` | 下载 Tiny Shakespeare，字符级编码，写 `train.bin` / `val.bin` / `meta.npz` |
| `tokenizer.py` | 字符级 tokenizer（训练流水线用的就是它，vocab=65） |
| `model.py` | **核心**：embedding / linear / RMSNorm / ReLU / attention / MLP / block / CE 的 forward + backward |
| `optim.py` | Adam；可选 AdamW 解耦权重衰减、全局范数梯度裁剪、warmup+cosine 学习率（默认全关） |
| `gradcheck.py` | 逐算子 + 端到端（n\_layer=1、2）数值梯度检查 |
| `train.py` / `sample.py` | 训练循环 / 自回归采样（temperature + top-k） |
| `bpe.py` | **独立演示**，不参与训练：手写 byte-level BPE。训练、采样和自带的 `ckpt.npz` 全部基于字符级 `tokenizer.py` |

## 核心公式

| 算子 | 前向 | 反向 |
| --- | --- | --- |
| Linear | `y = x @ W + b` | `dx = dy @ Wᵀ`，`dW = xᵀ @ dy`，`db = Σ dy` |
| RMSNorm | `y = g ⊙ x / rms`，`rms = √(mean(x²)+ε)` | `dx = g⊙dy/rms − x·Σⱼ(dyⱼgⱼxⱼ)/(D·rms³)` |
| Attention | `A = softmax(QKᵀ/√D + mask)`，`out = (A @ V) @ Wo` | `dV = Aᵀ @ dctx`，`dA = dctx @ Vᵀ`，`dS = A ⊙ (dA − Σⱼ AⱼdAⱼ)` |
| Cross-entropy | `L = −mean(log pᵧ)` | `dlogits = (p − onehot(y)) / N` |
| Embedding | `out = W[ids]` | `np.add.at(dW, ids, dout)`（重复 id 要累加） |
| Adam(W) | `m,v` 为 g、g² 的滑动平均，除以 `1−βᵗ` 做偏置修正 | `W ← W − lr·( m̂/(√v̂+ε) + λW )` |

warmup+cosine：`lr(t) = max_lr·t/warmup`（t ≤ warmup），之后 `min_lr + ½(max_lr−min_lr)(1+cos(π·progress))`。
梯度裁剪：`g ← g · min(1, max_norm/‖g‖)`，‖g‖ 是**所有参数**拼成一个向量的范数。

## 运行命令

所有脚本都在 `llm_basic/` 目录下直接运行（脚本之间用 `from model import ...` 互相导入，**不能**用 `python -m llm_basic.xxx`）：

```bash
cd llm_basic
python prepare.py            # 已自带数据，可跳过
python gradcheck.py          # 必跑：< 1 秒
python optim.py              # 优化器自检（裁剪 / cosine 端点 / AdamW）
python train.py --max-iters 300 --eval-interval 100 --out /tmp/c.npz   # 4 秒试跑，不覆盖自带 ckpt.npz
python train.py              # 完整 2000 步，约 27 秒，会覆盖 ckpt.npz
python train.py --n-layer 2 --weight-decay 0.1 --grad-clip 1.0 --cosine --out /tmp/c2.npz
python sample.py "ROMEO:" --max-new 120 --temperature 0.8              # 用自带 ckpt.npz
python sample.py --ckpt /tmp/c2.npz --top-k 10
python bpe.py --merges 300   # 独立的 BPE 演示
```

## 运行后应该看到什么

以下都是实测输出（Apple Silicon CPU，seed 固定，可复现）。

`python gradcheck.py`：7 个算子的相对误差在 1e-11 ~ 2e-10，端到端两种层数全部 `[OK ]`：

```
[1] 逐算子 gradcheck（全元素，断言 rel < 1e-06）
  [OK ] linear         max_rel=2.42e-11
  [OK ] rmsnorm        max_rel=8.16e-11
  [OK ] attention      max_rel=1.57e-10
  [OK ] cross_entropy  max_rel=1.81e-10      （另有 embedding / relu / mlp）
[2] 端到端 gradcheck  n_layer=2 ...
  [OK ] pos_emb          max_abs=1.00e-07  max_rel=1.14e-06      ← 最差的一项
all gradients within tolerance — analytical backward looks correct.
```

把 `rmsnorm_backward` 的耦合项故意删掉，逐算子检查立刻报 `rmsnorm: dx 相对误差 3.11e-01`——这就是它存在的意义。

`python optim.py`：`optim self-check OK: clip 37.14 → 1.000000, cosine lr 1.0e-04 → 1.0e-03 → 1.0e-04`

`python train.py`（默认 1 层，45,568 参数）。第 1 步 loss 必须 ≈ ln 65 = 4.174，代码里有断言：

```
step     1 | train_loss 4.1770 | val_loss 4.1453
step   500 | train_loss 2.4463 | val_loss 2.3920
step  1000 | train_loss 2.0278 | val_loss 2.1128
step  2000 | train_loss 1.9119 | val_loss 1.9809 | 26.8s
```

2 层 + AdamW(0.1) + clip(1.0) + cosine（78,656 参数）：2000 步 val\_loss 2.0452、45 秒——**并不比 1 层好**。步数这么少时 cosine 过早把 lr 降到 3e-5，多出来的一层还没学起来；这里要学的是"这些开关怎么实现"，不是"开了就一定更好"。

`python sample.py "ROMEO:" --max-new 120 --temperature 0.8`（自带 ckpt，val≈2.0 的水平：像英语，但还不是英语）：

```
ROMEO:
Gacking wich should bothess preath my him her patime have a fare don to whenth the is welle to crove to of I parath?
```

`python bpe.py`：`字符级需要 60 个 token, BPE 只要 24 个`，压缩率 2.50 字符/token。

## 常见误区

1. **"gradcheck 过了 = 训练一定没问题"**：它只证明 backward 与 forward 一致。forward 本身写错（比如 mask 方向反了）它查不出来，要靠初始 loss≈ln V、loss 曲线、采样结果来发现。
2. **ε 越小数值梯度越准**：不对。截断误差 ∝ ε²，舍入误差 ∝ 1/ε。本模型实测 ε=1e-4 时 `pos_emb` 相对误差 1.1e-4（不过关），ε=1e-5 降到 1.1e-6，ε=1e-6 时 `attn_Wq` 反而恶化到 1.3e-3。误差随 ε² 缩小说明是截断误差而不是 bug。
3. **embedding 反向写成 `dW[ids] += dout`**：重复 id 只会被加一次，梯度悄悄偏小。必须 `np.add.at`。
4. **AdamW = Adam + L2 正则**：L2 是把 `λW` 加进梯度，会被 `1/√v` 缩放，梯度大的参数几乎不衰减；AdamW 把 `λW` 直接加在更新量上，所有参数按同一比例衰减。另外 gain / bias（1 维参数）不做衰减。
5. **梯度裁剪是逐参数裁**：那样会改变梯度方向。全局范数裁剪是所有参数同乘一个系数，方向不变。
6. **多层需要新的反向推导**：不需要。前向 `for` 循环存下每层 cache，反向 `reversed` 再走一遍。层数直接从参数名 `block_{i}_*` 数出来，所以没有 `n_layer` 字段的旧 `ckpt.npz` 照常加载。

## 自测题

1. 为什么交叉熵对 logits 的梯度 `(p − onehot)/N` 每一行加起来恰好为 0？这意味着什么？
   **答**：`Σp = 1`、`Σonehot = 1`，相减为 0。意味着给一行 logits 同时加一个常数不改变 loss（softmax 平移不变），梯度在这个方向上没有分量。`gradcheck.py` 里有这条断言。
2. `attention_backward` 里没有再对 `dscores` 乘一次 causal mask，为什么被屏蔽位置的梯度仍然是 0？
   **答**：被屏蔽位置 `scores = −inf` → `A = 0`；softmax 反向 `dS = A ⊙ (…)` 带一个因子 A，所以自动为 0。
3. 训练第 1 步 loss 打印出 20 而不是 4.17，最可能哪里错了？
   **答**：初始化太大：logits 不再 ≈ 0，模型"自信地瞎猜"，loss ≫ ln V（实测把矩阵 std 从 0.02 改成 1，初始 loss = 19.98）。应检查 `init_weights` 的 std=0.02；`train.py` 的断言 `|loss − ln V| < 0.5` 会直接拦下。

## 刻意省略了什么

| 这里 | 真模型 | 到哪里看 |
| --- | --- | --- |
| 可学习绝对位置 embedding | RoPE | `llm_models/` |
| 单头 attention | Multi-Head / GQA / MLA | `llm_models/` |
| ReLU MLP | SwiGLU | `llm_models/` |
| 无 KV cache（每步重算整段前向） | 缓存 K、V | `llm_models/utils/generation.py` |
| float64 | bf16 / fp16 | — |
