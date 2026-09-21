# M17 — EAGLE 式投机解码: draft 看特征, 不只看 token

## 直觉
m07 的独立 draft 只看得到 token id。可 target 每次验证都顺手算出了每个位置的 hidden state,
里面已经写着"接下来想说什么" —— 扔掉太浪费。EAGLE 让 draft 在**特征空间**自回归:
拿 target 的真特征 h_t 和刚确定的 token 的 embedding, 预测下一个特征 ĥ, 再过 **target 自己的 lm_head**
得到 draft token。draft 本体很小 (真实 EAGLE 是一层 decoder), 不用另养一个小 LM。

## 核心数据结构或公式
- draft 一步: `ĥ_{t+1} = [h_t ; emb(x_{t+1}) ; 1] @ A`, `A` 形状 (2D+1, D); draft token = `argmax(ĥ @ lm_head)`。
- 第 1 步的 h 是 target 真特征 (`TinyLM.forward(..., return_hidden=True)` 在验证时白送的 `hid[n]`),
  第 2..K 步用 draft 自己的 ĥ → 误差累积, 越靠后的槽位越难接受。
- 训练: `collect_pairs` 跑 target 收集 (h_t, e_{t+1}) → h_{t+1}, `fit_draft` 一步最小二乘。
- 对照组 token-only: `ĥ = [emb(x) ; 1] @ A'`, 同样的数据、同样的头, 只是少了 h。
- 验证循环、接受规则、KV 回滚全部复用 `m07.speculative_decode`, 本模块只实现 drafter 协议
  `propose(out, K, temperature, rng, hidden) → (tokens, probs)`。

## 运行后应该看到什么
`python -m llm_infer.m17_eagle_speculative.demo` (约 3 s)
```
  draft 训练对 (lstsq 一步拟合)         = 6000 条
  特征拟合相对误差: EAGLE 式 [h_t; e_t+1] = 56.0%
  特征拟合相对误差: token-only [e_t+1]   = 87.1%
[对比] 32 个随机 prompt × 24 新 token, K=4, baseline target 调用 = 768
  EAGLE 式 [h_t; e_t+1]   每轮接受 0.60/4, 首槽命中 30%, target 调用 505 (1.52x)
  token-only [e_t+1]      每轮接受 0.36/4, 首槽命中 12%, target 调用 588 (1.31x)
```
assert: 两种 draft 的输出都与 `target.generate_greedy` 逐 token 相同;
`target_calls == 实际 forward 次数 == prompt 数 + 验证轮数` (用计数子类对账, 没有漏记的调用);
EAGLE 式每轮接受 > 1.3× token-only, 首槽命中 > 1.5×, target 调用更少。

## 与真实系统的差距
- 真 EAGLE 的 draft 是一层带注意力的 decoder, 能看全部历史 (h, e) 对, 且用 SGD 在真实语料上训练;
  这里是一个线性映射 + 随机权重 target, 所以接受率远低于论文的 ~80% / 3x。
- EAGLE-2 用 draft 置信度动态长**树** (见 m19 的固定形状树); EAGLE-3 融合多层特征并做 training-time test。
- 真实系统里 draft 也有自己的 KV cache 要回滚; 线性 draft 对 (h, token) 是马尔可夫的, 这里不需要。
- 只演示 greedy; 采样时 `EagleDrafter` 同样返回完整 draft 分布, 可直接走 m07 的 rejection sampling。

## 常见误区
- "EAGLE 的 draft 预测 token" —— 它预测的是**特征**, token 由共享的 lm_head 读出; 这也是它不用自带词表头的原因。
- "draft 要用 h_{t+1}" —— 拿不到。`out[-1]` 还没喂给 target, 只有产生它的那个位置的 h_t; 所以输入必须是 (h_t, emb(x_{t+1})) 这一对。
- "拒绝/bonus 后要额外跑一步 target 拿特征" —— 不用。验证那次 forward 已经给出所有槽位的特征, 取第 n 行即可; 任何额外的 target 计算都必须计入调用数。
- "特征拟合误差 56% 还能用?" —— 接受只要求 argmax 一致, 不要求 ĥ 精确。

## 自测题
1. 为什么喂 emb(x_{t+1}) 而不只喂 h_t?
   答: h_t 只决定下一个 token 的**分布**, 采样/纠错后实际选了哪个 token 是额外信息; 不喂它 draft 就无法消除这一不确定性 (EAGLE 论文的核心观察)。
2. 本 demo 中 draft 走 K=4 步花了多少次 target forward?
   答: 0 次。draft 每步只是一次 (2D+1)→D 矩阵乘加一次 lm_head; 每轮唯一的 target 调用是验证。
3. 把 draft 换成更差的, 输出会变吗?
   答: 不会, greedy 验证保证输出与 target greedy 逐 token 相同 (demo 对两种 draft 都 assert 了), 只是 target 调用变多。
