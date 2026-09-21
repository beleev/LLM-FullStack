# M10 — Sampling: 把 logits 变成 token

## 直觉
模型给出 V 个 logit, 最后一步要选 1 个 token。纯 greedy 确定但容易复读; 纯按 softmax 采样有多样性,
但词表很大时长尾 token 虽然单个概率极小, 加起来却有可观质量, 隔一阵就会抽到一个"垃圾 token"把后文带偏。
所有策略都在做同一件事: **先改造分布 (调尖/调平、砍尾巴、罚重复), 再采样**。

## 核心数据结构或公式
每个 filter 都是 `logits (V,) → logits (V,)`, 被砍的置 `-inf`, 因此可以任意串联:

| 策略 | 规则 | 保留集合大小 |
|---|---|---|
| temperature | `softmax(logits / T)`; T→0 即 greedy | 全部 |
| top-k | 保留最大的 k 个 | 固定 k |
| top-p (nucleus) | 按概率降序累加, 保留累计质量首次 ≥ p 的最小集合 | 随分布自适应 |
| min-p | 保留 `p_i ≥ min_p · p_max` | 模型越确定砍得越狠 |
| repetition penalty | 出现过的 token: 正 logit `/penalty`, 负 logit `×penalty` | 全部 |

`sample()` 的顺序: `rep_penalty → /T → top_k → top_p → min_p → Gumbel-max`。
**Gumbel-max**: `argmax_i(log p_i + G_i)`, `G_i = -log(-log U_i)`, 与 `multinomial(p)` 同分布, 但只有
element-wise 运算 + 一次 argmax (没有 cumsum / 二分搜索), batch 维天然并行。

API (full_engine 依赖): `SamplingParams(temperature, top_k, top_p, min_p, repetition_penalty)`,
`sample(logits, params, history=None, rng=None) -> int`; `temperature=0` 即 greedy。

## 运行后应该看到什么
```bash
python -m llm_infer.m10_sampling.demo      # ~1.1 s
```
```
[1] token id              0     1     2     3     4  ...
    softmax(logits)·N  4156  2521  1529   562   341  ...
    temp=0.3 (更尖)     8142  1554   293     6     4  ...
    temp=2.0 (更平)     2476  1870  1535   898   707  ...
    top_k=3            5027  3091  1882     0     0  ...
    top_p=0.5          6183  3817     0     0     0  ...   (0.416 < 0.5 ≤ 0.416+0.252 → 留 2 个)
    min_p=0.1          4787  2846  1703   664     0  ...   (阈值 0.0416: p3=0.056 留, p4=0.034 砍)
[2] 200 组随机 logits (V=50, 含并列值) 性质断言全部通过
[3] TV(gumbel-max, softmax) = 0.0064   TV(multinomial, softmax) = 0.0066
[4] penalty=1.5: token 0 (logit +3) 0.4156 → 0.2082; token 9 (logit −1) 0.0076 → 0.0063
    错误写法 (一律除): token 9 0.0076 → 0.0144  ← 反而变大
```
断言: top-k 恰好 k 个有限值且是最大的 k 个; top-p 保留质量 ≥ p、去掉其中最小者就 < p;
min-p 集合 == `{p_i ≥ min_p·p_max}`; T=0 与 T=1e-4 都等于 argmax; Gumbel-max TV < 0.02 (seed 固定);
被罚 token 概率严格下降; 被砍 token 在 10000 次里出现 0 次。

## 与真实系统的差距
- 真实系统对整个 batch `(B, V)` 一次处理, 每行参数不同; top-p 的全词表排序很贵, FlashInfer 用
  rejection sampling 做免排序的 top-k/top-p。这里是单条 `(V,)` + Python。
- 顺序不统一: vLLM / HF / llama.cpp 对 min_p 与 top_k/top_p 的先后不完全一致, 同一组参数跨框架分布可能不同。
- 缺 presence / frequency penalty (OpenAI 风格, 与出现次数成正比地减 logit)、logit_bias、
  bad_words、per-request seed、以及 m14 的结构化输出 mask (也是一个 `-inf` filter, 插在同一条链上)。

## 常见误区
- "top-k 用 `logits >= 第k大值` 就行": 有并列值时会留下多于 k 个 (原实现就是这样, 已改为按下标保留)。
- "repetition penalty 就是 logit 除以 penalty": 负 logit 除以 >1 的数会变大, 反而鼓励重复 (见 [4])。
- "top-p 和温度谁先谁后无所谓": 温度在前时, T 越大分布越平, 同样的 p 会留下更多 token。
- "T=0 就是除以 0": 实现里必须特判成 argmax; 另外 greedy 也受 repetition penalty 影响 (penalty 在最前)。
- "Gumbel-max 是近似采样": 是精确的, [3] 里它与 multinomial 的 TV 处在同一噪声水平。

## 自测题
1. **概率 [0.5, 0.3, 0.1, 0.1], top_p=0.8 留几个? top_p=0.81 呢?**
   0.5+0.3=0.8 ≥ 0.8 → 2 个; 0.81 时 0.8 < 0.81, 需再加一个 → 3 个。
2. **模型非常确定 (p_max=0.95) 与非常犹豫 (p_max=0.05) 时, min_p=0.1 的阈值各是多少? 这比固定 top-k 好在哪?**
   0.095 与 0.005。确定时几乎只留 1 个, 犹豫时留下很多候选; top-k 在两种情况下都留 k 个, 要么放进垃圾要么砍掉合理候选。
3. **为什么 GPU 上偏爱 Gumbel-max 而不是 `multinomial`?**
   multinomial 需要 cumsum + 搜索 (串行依赖, 且常伴随 host 同步); Gumbel-max 只有逐元素加噪声 + argmax, 整个 batch 一个 kernel。
