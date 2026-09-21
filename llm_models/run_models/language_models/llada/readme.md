# LLaDA — 掩码扩散语言模型

论文: Nie et al. 2025, "Large Language Diffusion Models"。模型代码: `llm_models/models/language_models/llada.py`。

## 直觉

- 结构上就是 **去掉因果 mask 的 LLaMA** (GQA + SwiGLU + RMSNorm + RoPE), 词表最后一个 id 是 `[MASK]`。
- 训练: 每条序列抽一个遮蔽比例 `t ~ U(0,1)`, 每个 token 独立以概率 `t` 变成 `[MASK]`, 让模型还原。
- 生成: prompt 后面接一串 `[MASK]`, 分 N 步去噪。每步并行预测所有 `[MASK]`, 留下最有把握的, 把没把握的 **重新遮住** 留给下一步。
- 对比 BERT: BERT 固定遮 15%, 从没见过 "几乎全是 `[MASK]`" 的输入, 所以不能从零生成; LLaDA 的 t 覆盖 0→1, 再配上 1/t 权重, loss 就成了 −log p(x) 的上界 (ELBO) —— 一个正经的生成模型。
- 对比自回归 LM: 一步可以定多个 token (并行解码); 任意位置都能填 (续写、倒推、两头填是同一个函数), 没有 "只会 A→B 不会 B→A" 的反转诅咒; 代价是双向注意力下每步都要整段重算, **没有 KV cache**。

## 核心公式

```
前向加噪:   q(x_t | x_0): 每个 token 独立以概率 t 变成 [MASK]
训练 loss:  L = E_{t~U(ε,1)} E_{x_t} [ (1/t) · Σ_{i: x_t[i]=[MASK]} −log p_θ(x_0[i] | x_t) ] / L_seq   ≥  −log p_θ(x_0) / L_seq
采样日程:   第 s 步 (共 N 步) 结束后剩余 [MASK] 数 = round(n · (1 − s/N))
重遮策略:   low-confidence: 置信度 = 所选 token 的概率, 重遮最低的; 已定 token 置信度 = +∞ (永不重遮)
```

为什么除以 t: `E[被遮个数 / t] = L_seq`, 所以均匀猜测时 loss 期望恰为 ln V, 与自回归 CE 同量纲可比。

## 运行命令

```bash
python -m llm_models.run_models.language_models.llada.train_llada   # ~18s CPU
python -m llm_models.run_models.language_models.llada.infer_llada   # ~12s CPU, 内部自带 400 步训练, 不需要 checkpoint
```

## 运行后应该看到什么

任务是 "置换链": 固定一个 16-环置换 π, 序列满足 `x[i+1] = π(x[i])`, 长度 16。每步抽 **新数据 + 新遮蔽** (没用 Trainer 的固定 batch), 所以这里的 loss 下降是真的学会了 π, 评估也都在 held-out 序列上。

`train_llada` (seed 0 实测):

```
初始 loss 3.014  vs  ln V = 2.833
step  100 | loss 0.364   ...   step  400 | loss 0.188
训练后 held-out loss 0.189  (理论下界 ln P / L = 0.173)
每步剩余 [MASK] 数 [11, 8, 4, 0]  期望 [11, 8, 4, 0]
  给首 token 续写      | 1 步并行 0.887 | 15 步 low-confidence 1.000 | 15 步 random 0.910
  给末 token 倒推      | 1 步并行 0.909 | 15 步 low-confidence 1.000 | 15 步 random 0.910
  给中间 token 两头填  | 1 步并行 0.995 | 15 步 low-confidence 1.000 | 15 步 random 0.997
```

- 初始 loss 比 ln V 高约 0.1~0.2 不是 bug: weight tying 下 `[MASK]` 位置的隐状态 ≈ `[MASK]` 自己的 embedding, 初始最偏爱输出 `[MASK]` (CE ≈ ln(V−1+e^1.3) ≈ 2.95), 其余是 1/t 估计的噪声。
- 整条序列只有起点随机, 所以真实 NLL/token = ln 16 / 16 = 0.173; ELBO 压到 0.189, 离下界很近。
- 换种子 (1/2/3) 时 low-confidence 各方向仍在 0.87~1.00, random 在 0.74~0.97; 单个方向上 low-confidence 偶尔略输 random (自信地早定错一个 token 会传染), 所以脚本断言的是三个方向的均值。

`infer_llada` 打印一次 5 步去噪 (只给中间的 9, 两头都要填):

```
输入     :  _  _  _  _  _  _  _  _  9  _  _  _  _  _  _  _
第 1 步后 :  _  _  _  _  7  _  _  _  9  _  _  8  _  _  _ 14
第 3 步后 :  _  0  4  _  7  _ 12  _  9  _ 11  8  _  5  2 14
第 5 步后 : 15  0  4  3  7  1 12 10  9  6 11  8 13  5  2 14
5 步 low-confidence: 与真值一致 1.000 | 链规则成立 1.000 | chance 0.062
```

并断言: `[MASK]` 个数走日程、已定 token 不再变、输出无 `[MASK]`、padding mask 生效且不传 mask 时右侧能影响左侧 (双向)。

## 常见误区

1. **"LLaDA = 遮蔽率随机的 BERT"** —— 缺了 1/t 就不是似然界, 只是一个去噪器; 1/t 和 "除以总 token 数" 两处都不能改。
2. **"重遮 = 把已经生成的 token 再擦掉"** —— 这里 (和官方实现一样) 已定的 token 永不重遮, 重遮的只是本步刚预测、置信度垫底的那些。
3. **"步数越少越好, 反正是并行的"** —— 同一步里定下的 token 互相看不见对方, 1 步并行 0.887 vs 15 步 1.000 就是这个代价; 步数是质量/速度旋钮。
4. **"扩散 LM 能用 KV cache"** —— 每步输入里的 `[MASK]` 在变, 且注意力是双向的, 所有位置的 K/V 都会变, 只能整段重算。
5. **忘了屏蔽 `[MASK]` 的 logit** —— weight tying 下模型天然偏爱在 `[MASK]` 位置输出 `[MASK]`, 采样时必须置 −inf。

## 自测题

1. 为什么 loss 要除以 t? 如果不除会怎样?
   **答**: t 小的样本被遮 token 少, 不加权则它们几乎不贡献梯度, 且 loss 不再是 −log p(x) 的上界。除以 t 后每条序列的期望贡献都是 L·CE, 均匀猜测时正好 ln V。
2. 生成 15 个 token、4 步, 每步结束后还剩几个 `[MASK]`?
   **答**: round(15·(1−s/4)) = 11, 8, 4, 0 (脚本里有断言)。
3. 自回归 LM 学了 "A 的下一个是 B", 为什么答不出 "B 的上一个是谁", 而 LLaDA 可以?
   **答**: 自回归只建模 p(右 | 左), 从没训练过用右边预测左边; LLaDA 的随机遮蔽让每个 token 既当过条件也当过目标, 左右上下文都用 —— 实测 "给末 token 倒推" 准确率 1.000。
