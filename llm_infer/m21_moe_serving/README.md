# M21 — MoE serving: 专家并行 (EP) 与 EPLB 负载均衡

## 直觉
MoE 层有 E 个专家 FFN, 每个 token 只过 router 选出的 top-k 个, 所以 "参数很多、每 token 计算很少"。
专家太多一张卡放不下 → **专家并行 (EP)**: 专家分散在各 rank, token 去找专家:
1. **dispatch** (all-to-all): 每个 (token, 专家) 分配被发到该专家所在的 rank;
2. 各 rank 对收到的 token 按专家分组, 批量算 FFN;
3. **combine** (all-to-all): 结果发回 token 原位置, 按 gate 加权求和。

问题: router 不均匀。热专家所在 rank 收到的 token 远多于其它 rank, 而 combine 要等所有 rank ——
**一步耗时 = 最慢的 rank**。所以指标是 `max/mean rank load`, 不是平均值。
**EPLB**: 给热专家多放几个副本 (冗余 slot), 把它的 token 拆给各副本, 再把副本贪心放到最轻的 rank。

## 核心数据结构或公式
```
route        logits = x W_r + bias (T, E) → top-k idx (T, k), gates = softmax(top-k logits) (T, k)
out[t]       = Σ_k gates[t,k] · FFN_{idx[t,k]}(x[t])
Placement    slot_expert (S,): 每个物理 slot 装哪个逻辑专家, S = E + n_redundant
             slot_rank   (S,): 每个 slot 在哪个 rank
slot_of      (T·k,): 每条分配去哪个 slot; 有副本的专家按 round-robin 拆分 token
rank_load[r] = #{分配 : slot_rank[slot_of] == r};   step time ∝ max_r rank_load[r]

EPLB 两步贪心 (eplb_placement):
  ① 副本数: 重复 n_redundant 次 —— 给 load/n_rep 最大的专家 +1 副本
  ② 放置:   slot 按每副本负载降序, 依次放到 "当前最轻且有空位" 的 rank
```

## 运行后应该看到什么
```bash
python -m llm_infer.m21_moe_serving.demo     # < 1 s
```
16 专家 / 4 rank / top-2 / 2048 tokens, router 加 Zipf 偏置 (专家 e 的先验 ∝ 1/(e+1)):
```
[1] EP dispatch/combine vs 朴素逐 token 循环   max|Δ| = 0.00e+00
[2] 每专家 token 数 = [1451, 854, 435, 276, 363, 187, 167, 134, 34, 35, 22, 34, 27, 22, 38, 17]
    每 rank token 数 = [3016, 851, 125, 104]   max/mean = 2.95x  (平均 rank 利用率只有 34%)
[3] greedy 放置, 无副本   rank load=[1517, 948, 770, 861]    max/mean=1.48x
    EPLB +4 副本         rank load=[1025, 1033, 992, 1046]  max/mean=1.02x   副本: 专家0 ×4, 专家1 ×2
    模拟 step 时间 (∝ max rank load) 缩短 65%
    最热专家 / 平均 rank 负载 = 1.42  (>1 ⇒ 不复制就不可能均衡)
```
断言: EP 输出 == 基线 (<1e-5, 三种放置都成立 → 复制不改变模型输出); Σ rank_load == T·k;
max/mean 严格下降 2.95 → 1.48 → 1.02。EPLB 的负载统计来自**另一个**历史 batch, 在新 batch 上评测。

## 与真实系统的差距
- all-to-all 用数组分组模拟, 没有通信成本。真实系统 (DeepEP) 里 dispatch/combine 的通信量和延迟
  是 EP 的主要开销, 还要 FP8 dispatch、通信计算重叠、节点内 NVLink / 节点间 RDMA 分层。
- "负载 = token 数" 是代价模型; 真实 grouped GEMM 的耗时对 token 数不是严格线性。
- DeepSeek EPLB 还有分层版本 (先在节点间均衡专家组, 减少跨节点流量), 副本间按负载动态选择而非 round-robin。
- 重新放置要搬专家权重 (GB 级), 真实系统每隔几分钟才做一次; 训练侧靠 aux-loss / bias 调整让路由本身更均衡。
- 这里只有一层, 无 shared expert, 无 capacity factor / token drop。

## 常见误区
- "平均负载一样就行" —— EP 是同步的, 慢的那个 rank 决定一切; 看 max/mean。
- "把专家重新排一下位置就能均衡" —— 当单个专家的负载 > 平均 rank 负载 (本例 1.42x) 时, 任何无副本
  的放置都不可能均衡 (本例最好 1.48x); 必须复制并拆分。
- "复制专家会改变模型输出" —— 副本权重相同, 一个 token 走哪个副本结果都一样 (demo 断言)。
- "gate 是对全部 E 个 logit 做 softmax" —— 常见做法是只在选中的 k 个上归一化 (各模型略有不同)。

## 自测题
1. 8 个 rank, 某专家独占全部分配的 30%, 无副本时 max/mean 的下界? **答**: 0.30 / (1/8) = 2.4x。
2. 为什么 EPLB 用历史统计而不是当前 batch 的路由结果? **答**: 放置要搬权重, 必须在 batch 到来之前定好;
   能这么做是因为专家热度在分钟级时间尺度上相对稳定。
3. dispatch 后同一 token 的 k 份结果来自不同 rank, combine 怎么保证正确? **答**: 每条分配带着 token 下标
   和 gate, combine 做 scatter-add (`np.add.at(out, tok, gate·y)`), 加法与到达顺序无关。
