# Reward Model — 把 "A 比 B 好" 变成标量分

```bash
python -m llm_finetune.run_finetune.rm.train_rm        # ~9 s
```

## 直觉
人只能稳定地标相对偏好, 在线 RL 却需要一个能随时调用的标量奖励。RM = LM 主干 + 标量头, 读完整条 (prompt, 回复) 后打分。

## 核心公式
`P(y_w ≻ y_l) = σ(r_w − r_l)`，`L = − log σ(r_w − r_l)`；`r = value_head(h[最后一个非 pad token])`。只有分差有意义。

## 运行后应该看到什么
偏好对 = (正确排序, 损坏的排序); 损坏 = 改错一个 token 或漏掉一个 token (后者变短 → batch 里有右 pad)。

| | 数值 |
|---|---|
| 第 1 步 loss | 0.6929 (≈ ln 2) |
| **留出集**偏好准确率, 300 步 | 0.549 → **0.930** |
| 同一序列右侧多垫 5 个 PAD: 取最后一个真 token | 分数偏移 5e-06 |
| 同上, 取 `scores[:, -1]` (旧实现) | 分数偏移 **4.54** |

训练走的是通用 `Trainer`: `PairwiseForward(rm)` 把 chosen / rejected 拼成一个 batch 前向, `BradleyTerryLoss` 是普通 `LossComputer`。

## 常见误区
- 右 pad 时读 `h[:, -1]`: 那是 pad 位置的隐状态。上表最后一行。
- 手抄一遍主干前向来拿隐状态: 主干一改 (mask / RoPE / cache) 就悄悄不一致。这里把 `lm_head` 换成 `Identity`, 直接复用公开的 `forward`。注意这会原地改掉传入的 backbone。
- 把 RM 分数当绝对质量: Bradley-Terry 只约束分差, 整体平移不改变 loss。
- chosen / rejected 的输入格式不对称 (比如只有一边带 EOS): RM 会学到这个捷径。

## 自测题
1. 给所有分数加常数 c, loss 变吗? — 不变。
2. 为什么 RM 通常从 SFT checkpoint 初始化? — 它要先 "读懂" 回复; 主干已有的表示让偏好数据只需教会打分头。
3. 左 pad 时 "最后一个真 token" 在哪? — 就是 `[:, -1]`; 本实现假设右 pad, 用 `attention_mask.sum(1) − 1`。
