# SFT — 全参监督微调

```bash
python -m llm_finetune.run_finetune.sft.train_sft      # ~15 s
```

## 直觉
预训练模型只会续写。SFT 给它看 (问题, 回答), 但**只对回答部分算 loss** —— 问题是人写的, 不是模型该生成的。
SFT 没有新的 loss 函数 (`SFTLoss = StandardLMLoss`), 全部差异都在 labels 里。

## 核心公式
`L = − Σ_{t ∈ 回复} log p(y_t | x, y_<t)`；prompt / pad 位置的 label = −100。

```
idx    :  x1    x2    SEP   y1    y2    EOS
labels : −100  −100   y1    y2    EOS   −100      # labels 已左移一位; SEP 这一列预测第一个回复 token
```

## 运行后应该看到什么
任务: 反转 6 个 token (prompt 空间 13⁶ ≈ 480 万, 训练集 / 留出集按 token 和 mod 5 不相交)。

| | 数值 |
|---|---|
| 初始 loss | 2.875 (ln 16 = 2.773) |
| 留出集 exact-match, 600 步 | 0.000 → **1.000** |
| 同配置, 但 prompt mask 多盖一位 `labels[:, :P]` | **0.000** |

## 常见误区
- **prompt mask 差一位**。labels 已经左移, `labels[:, P-1]` 就是第一个回复 token。写成 `[:, :P]` 后模型永远学不到 "看完 prompt 第一个词说什么", 后面全对也没用 —— 上表第三行。`make_labels` 里有断言: 被监督的位置数必须等于回复长度。
- "loss 下降 = 学会了"。在固定 batch 上 loss 下降只说明能背; 这里每步换新 batch, 用留出集 exact-match 验收。
- 忘了监督 EOS: 模型不会停。

## 自测题
1. 回复长度 R (含 EOS)、prompt 长度 P, labels 里有几个位置不是 −100? — **R 个**: 位置 P−1 … P+R−2。
2. 为什么不在 prompt 上也算 loss? — 那是在教模型 "生成用户的问题"; 容量和梯度都浪费在不需要的分布上, 回复部分的有效权重被稀释。
3. 把 `SFTLoss` 换成 `StandardLMLoss` 会怎样? — 完全一样, 它们是同一个类。
