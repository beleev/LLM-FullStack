// 阶段 2 · llm_models 的扩展章节: 2024–2025 的新零件。每章一个 (或两个) 实验台, 挂载关系见 data/labmap/models.js。
const A = 'llm_models/layers/core/attention.py'
const RUN = 'python -m llm_models.run_models'

export default {
  stage: 'models',
  chapters: [
    { route: 'models-generation', label: 'KV cache 生成', hint: 'GenerationMixin: prefill / decode, MLA 只缓存 latent' },
    { route: 'models-qknorm-yarn', label: 'QK-Norm 与 YaRN', hint: '管住 logit 幅度 + 把 RoPE 拉到更长上下文' },
    { route: 'models-mamba', label: 'Mamba 选择性 SSM', hint: '输入决定 Δ: 记谁、忘谁; O(1) 状态' },
    { route: 'models-moe-balance', label: 'Aux-loss-free 均衡', hint: '路由偏置只管选人, 不碰门控权重' },
    { route: 'models-dsa', label: 'DSA 稀疏注意力', hint: 'LightningIndexer 粗选 top-k + KL 对齐 loss' },
    { route: 'models-gptoss', label: 'GPT-OSS 结构', hint: '滑窗/全注意力交替 + 可学 sink logit + MoE' },
    { route: 'models-llada', label: 'LLaDA 扩散语言模型', hint: '随机比例遮盖训练, 低置信度重遮采样' },
    { route: 'models-var', label: 'VAR 逐尺度生成', hint: '1×1 → 2×2 → … 由粗到细, 级内并行' },
  ],
  pages: {
    'models-generation': {
      title: 'KV cache 生成 · 训练并行, 推理为什么不必每步重算',
      subtitle: '自回归每步只多 1 个 token, 旧 token 的 K/V 不会变。GenerationMixin 把 “prefill 一次 + 每步只喂 1 个 token” 做成所有语言模型共用的 generate()。',
      tldr: '无 cache 生成 N 个 token 要做 O(N²) 次 token 前向, 有 cache 是 O(N)。MHA/GQA 缓存 K、V 本体; MLA 只缓存低秩 latent c_kv 和共享的 k_rope, 每步现场升维。',
      question: '为什么只缓存 K/V 而不缓存 Q? 上下文窗口满了之后, cache 为什么会整个作废?',
      code: 'llm_models/utils/generation.py · llm_models/layers/core/attention.py',
      points: [
        { title: '只有 K/V 值得存', body: '第 t 步的输出只需要 “最新 token 的 Q” 去查 “所有历史 token 的 K/V”。旧 token 的 Q 再也用不上, 旧 K/V 在因果 mask 下永远不变 —— 所以缓存 K/V, 每步只算 1 个 token。' },
        { title: 'prefill 与 decode 是两种负载', body: 'prefill 一次处理整段 prompt (算力密集, 可并行); decode 每步 1 个 token (访存密集, 串行)。generate() 用 cache.pos 是否为 0 区分这两个分支。阶段 5 的调度器就是围绕这两种负载设计的。' },
        { title: 'MLA: 缓存 latent 而不是 K/V', body: 'MLA 的 cache 只有 c_kv [S, r] 和 post-RoPE 的共享 k_rope [S, rope]; K-nope 和 V 每步从 latent 升维得到。用一点重算换 cache 缩小一个数量级。c_kv 不能带 RoPE, 否则升维矩阵无法被吸收 —— 这就是 rope/nope 要拆开的原因。' },
      ],
      links: [
        { from: 'llm_basic/sample.py', to: 'GenerationMixin.generate', body: '阶段 1 的采样每步重算整段 forward; 这里加上 cache 分支, 输出逐 token 完全一致。' },
        { from: 'KVCache', to: 'llm_infer KV 与缓存内存', body: '这里的 cache 是每层一个 dict; 阶段 5 把它换成分页的 block 表来服务多请求。' },
        { from: 'MultiHeadLatentAttention', to: 'DSA', body: 'DSA 在 MLA 的 cache 上再多存一份 indexer 的 key, 用来给旧位置打分。' },
      ],
      sourceRows: [
        { concept: '三个分支', code: 'generation.py:GenerationMixin.generate', takeaway: '无 cache: 喂整个前缀; pos==0 或窗口已满: (重新) prefill; 否则 decode 只喂 idx[:, -1:]。' },
        { concept: '窗口满了就作废', code: 'cache.pos + 1 > max_len', takeaway: '窗口左移后每个 token 的绝对位置都变了, 旧 K/V (带着旧的 RoPE 角度) 不能复用。' },
        { concept: '等价性断言', code: 'generation.py:benchmark_kv_cache', takeaway: '贪心生成两遍, assert torch.equal —— cache 只许变快, 不许改结果。' },
        { concept: 'MLA 的 cache', code: 'cache["c_kv"], cache["k_rope"]', takeaway: '只追加 latent; k_up / v_up 每步对全部 S 个位置现场升维。' },
      ],
      snippetTitle: 'generate() 的骨架',
      snippet: `cache = KVCache(num_layers) if use_cache else None
for _ in range(max_new_tokens):
    if cache is None:
        inp = idx[:, -max_len:]          # 每步重算整个前缀: O(T) 个 token
    elif cache.pos == 0 or cache.pos + 1 > max_len:
        cache = KVCache(num_layers)      # prefill; 或窗口已满, 旧位置全部失效
        inp = idx[:, -max_len:]
    else:
        inp = idx[:, -1:]                # decode: 只喂 1 个新 token
    logits = model(inp, cache=cache)[:, -1]
    idx = cat([idx, sample(logits)], dim=1)

# attention 内部 (每层):
k = cat([cache["k"], k_new]); v = cat([cache["v"], v_new])   # MHA / GQA
c_kv = cat([cache["c_kv"], c_new]); k, v = k_up(c_kv), v_up(c_kv)  # MLA`,
      source: ['llm_models/utils/generation.py:GenerationMixin.generate'],
      run: `${RUN}.language_models.llama.infer_llama`,
    },

    'models-qknorm-yarn': {
      title: 'QK-Norm 与 YaRN · 两个 “让注意力在极端情况下不失控” 的补丁',
      subtitle: 'QK-Norm 管的是数值幅度: logit 不许随 q/k 范数无限长大。YaRN / NTK-aware 管的是位置: 让 RoPE 在超出训练长度时不看到没见过的角度。',
      tldr: 'QK-Norm: q、k 在 RoPE 之前各过一次 head_dim 上的 RMSNorm, logit 上界 g²√d。YaRN: 按 “训练期转了几圈” 给每个 RoPE 频率分段缩放 —— 高频不动, 低频 ÷s, 中间线性过渡, 再乘 mscale 补偿温度。',
      question: '为什么 logit 爆炸不能只靠 1/√d 解决? 为什么长度外推时出问题的是低频维度而不是高频?',
      code: `${A} · llm_models/layers/core/position_encoding.py`,
      points: [
        { title: 'logit 随范数二次增长', body: 'q·k/√d 里的 √d 只抵消维度, 不抵消范数。训练中 q、k 范数同时涨 2 倍, logit 涨 4 倍, softmax 塌成 one-hot, 雅可比 diag(p)−ppᵀ → 0, 这个 head 收不到梯度; 低精度下还会直接溢出。' },
        { title: 'QK-Norm 只留方向和一个增益', body: '对 q、k 各做 RMSNorm (Qwen3 / OLMo-2 / Gemma-3): logit = g²·√d·cosθ。模型仍能通过可学的 g 调节注意力有多 “尖”, 但只有这一个受控旋钮。必须放在 RoPE 之前 —— RoPE 是旋转, 不改变范数。' },
        { title: '低频维度没见过 “那么大的角度”', body: '波长大于训练长度 L 的维度, 训练时连一圈都没转完; 位置超过 L 后它们的角度落在训练分布之外。PI 全体 ÷s 会抹掉高频的局部分辨率; NTK-aware 改 base, 高频不动但中段压不够; YaRN 按圈数 r 分段, 两头都保住。' },
      ],
      links: [
        { from: 'position · RoPE', to: 'scaled_inv_freq', body: '位置编码一章讲 RoPE 为什么编码相对位置; 这里只改 θ_i 这一张表, 旋转公式一行不动。' },
        { from: 'RMSNorm', to: 'q_norm / k_norm', body: '和 Block 里的 RMSNorm 是同一个类, 只是作用在 head_dim 上, 每个 head 共享一组 g。' },
        { from: 'QK-Norm', to: 'llm_train 精度与稳定性', body: '阶段 3 从训练侧管稳定性 (clip、warmup、loss scaling); QK-Norm 是从结构侧去掉一个不稳定源。' },
      ],
      sourceRows: [
        { concept: 'QK-Norm 开关', code: 'attention.py:GroupedQueryAttention', takeaway: 'self.q_norm = RMSNorm(head_dim) if qk_norm else None; forward 里在 _call_rope 之前调用。' },
        { concept: 'NTK-aware', code: 'base * factor ** (d / (d - 2))', takeaway: '指数 d/(d−2) 正好让最后一个频率 ÷s, 第一个频率不变。' },
        { concept: 'YaRN 分段', code: 'gamma = ((rotations - beta_slow) / (beta_fast - beta_slow)).clamp(0, 1)', takeaway: 'r = L·θ/2π 是训练期圈数; γ=1 原样外推, γ=0 按 PI 内插。' },
        { concept: '温度补偿', code: '0.1 * math.log(factor) + 1.0', takeaway: '上下文变长后 softmax 分母项变多、分布变平, 用 mscale 把 logit 放大一点补回来。' },
      ],
      snippetTitle: '两处改动各只有几行',
      snippet: `# QK-Norm: RoPE 之前, 对每个 head 的 q / k 归一化
q = q_norm(q)            # RMSNorm over head_dim, 带可学增益 g
k = k_norm(k)
q, k = rope(q), rope(k)  # 旋转不改范数
scores = q @ k.T / sqrt(d)          # |score| <= g² · sqrt(d)

# YaRN: 只改 RoPE 的频率表
theta = base ** (-arange(0, d, 2) / d)
r = L_train * theta / (2 * pi)                       # 训练长度内转了几圈
gamma = clip((r - beta_slow) / (beta_fast - beta_slow), 0, 1)
theta_new = (1 - gamma) * theta / s + gamma * theta  # 低频内插, 高频外推
mscale = 0.1 * ln(s) + 1`,
      source: ['llm_models/layers/core/position_encoding.py:scaled_inv_freq'],
      run: `${RUN}.foundation.rope_scaling.infer_rope_scaling`,
    },

    'models-mamba': {
      title: 'Mamba · 让状态空间模型学会 “挑着记”',
      subtitle: '注意力把所有历史都留着再挑; SSM 只有一个固定大小的状态, 必须在写入的那一刻就决定留什么。Mamba 的答案: 让步长 Δ 由输入决定。',
      tldr: 'h_t = exp(Δ_t·A)·h_{t-1} + Δ_t·B_t·x_t, y_t = C_t·h_t。Δ_t、B_t、C_t 都由 x_t 线性投影得到 (选择性)。Δ 大 = 清掉旧状态并写入当前 token; Δ≈0 = 当前 token 被跳过。解码 O(1) 状态, 训练用并行 scan。',
      question: '线性时不变的 SSM (S4) 可以写成卷积、训练很快, 为什么 Mamba 宁可放弃这个性质?',
      code: 'llm_models/layers/sparse/ssm.py · llm_models/models/language_models/mamba.py',
      points: [
        { title: '一个 Δ 同时是写入门和遗忘门', body: '离散化后 Ā = exp(Δ·A) ∈ (0,1) 是旧状态的保留率, B̄ ≈ Δ·B 是新输入的写入强度。Δ 大: 保留率→0、写入强, “忘掉过去记住这个”; Δ→0: 保留率→1、写入→0, “这个 token 当没看见”。' },
        { title: 'LTI 只能按距离加权', body: 'Δ 固定时, token j 对最终状态的贡献只取决于它离结尾多远。它无法表达 “这个词重要、那个是废话” —— 而这正是语言建模 (和 induction / selective copying 任务) 需要的。' },
        { title: '状态大小与长度无关', body: '每层的 “cache” 就是 h [D_inner, N] 加一小段卷积缓冲, 生成 100 万 token 也不增长。代价: 信息是有损压缩的, 精确召回很久以前的某个 token 不如注意力 —— 所以出现了 Jamba / Qwen3-Next 这类混合架构。' },
      ],
      links: [
        { from: 'Gated DeltaNet', to: 'SelectiveSSM', body: '两者都是 “固定大小状态 + 输入相关的门”。DeltaNet 的状态是矩阵、用 delta rule 覆写; Mamba 的状态是对角 SSM、用 Δ 控制衰减。' },
        { from: 'KV cache', to: 'cache["h"]', body: '同一个 GenerationMixin.generate(): 注意力层往 cache 里追加 K/V, Mamba 层就地更新 h。' },
        { from: 'MambaBlock', to: 'Block 组装器', body: 'Mamba block = Pre-RMSNorm + MambaLayer + 残差, 没有单独的 FFN: 门控分支 SiLU(gate) 已经承担了这部分非线性。' },
      ],
      sourceRows: [
        { concept: '选择性参数', code: 'self.x_proj(x).split([dt_rank, N, N])', takeaway: 'Δ (低秩)、B、C 全部来自当前输入 x_t —— “选择性” 三个字的全部实现。' },
        { concept: 'Δ 恒正', code: 'F.softplus(self.dt_proj(dt_low))', takeaway: 'softplus 保证 Δ>0; 再配合 A = −exp(A_log) < 0, exp(Δ·A) 一定落在 (0,1)。' },
        { concept: '递推本体', code: 'h = A_bar[:, t] * h + Bx[:, t]', takeaway: '教学版顺序 scan; 生产实现用 work-efficient 并行 scan + kernel fusion。' },
        { concept: 'O(1) 解码', code: 'cache["h"] = h', takeaway: '给了 cache 就从上一步的 h 续跑, 与已读过多少 token 无关。' },
      ],
      snippetTitle: '选择性扫描',
      snippet: `dt, B_t, C_t = x_proj(x).split([r, N, N])     # 全部依赖输入
delta = softplus(dt_proj(dt))                   # [B, T, D, 1]  > 0
A = -exp(A_log)                                 # [D, N]        < 0

A_bar = exp(delta * A)                          # 保留率 ∈ (0, 1)
Bx    = delta * B_t * x                         # 写入量

h = zeros(B, D, N)
for t in range(T):                              # 训练时换成并行 scan
    h = A_bar[:, t] * h + Bx[:, t]
    y[t] = (h * C_t[:, t]).sum(-1)
return y + D_skip * x`,
      source: ['llm_models/layers/sparse/ssm.py:SelectiveSSM'],
      run: `${RUN}.language_models.mamba.infer_mamba`,
    },

    'models-moe-balance': {
      title: 'Aux-loss-free 负载均衡 · 会自己动的路由偏置',
      subtitle: 'MoE 必须均衡负载, 但传统 aux loss 是往语言建模目标里掺了一个不相干的梯度。DeepSeek-V3 的做法: 把 “均衡” 从 loss 里拿出来, 变成一个不收梯度的控制器。',
      tldr: '每个专家一个偏置 b_e: 选 top-k 时用 s+b, 算门控权重时只用 s。每个训练 step 之后 b_e += γ·sign(平均负载 − 专家 e 的负载)。负载被拉平, 路由分数和门控权重不被扭曲。',
      question: '同样是把热门专家 “压一压”, 为什么改偏置比加 aux loss 对模型质量更友好?',
      code: 'llm_models/models/moe/deepseekV3.py · llm_models/training/loss.py',
      points: [
        { title: '选人和加权解耦', body: 'top-k 的排序用 sigmoid(logit) + b, 但乘到专家输出上的权重取自不带 b 的 sigmoid 分数再归一化。b 只改变 “谁上场”, 不改变 “上场后信多少”。' },
        { title: '一个符号控制器, 不是一个 loss', body: 'b 不在计算图里 (@torch.no_grad), 更新规则只用 sign: 过载就 −γ, 欠载就 +γ。步长恒为 γ、与 batch 大小无关, 好调; 但 γ 太大会来回震荡, 太小追不上路由器的漂移。' },
        { title: 'aux loss 的代价', body: 'L_aux = α·E·Σ f_e·P_e 的梯度直接压低热门专家的路由 logit: α 大了干扰语言建模、小了均衡不住。教学代码保留了 MoELMLoss 作为对照, 真实的 V3 还留了一个极小权重的序列级 aux loss 防极端情况。' },
      ],
      links: [
        { from: 'moe · Mixtral vs DeepSeek', to: 'routing_bias', body: 'MoE 路由一章讲 top-k、共享专家、softmax vs sigmoid 门控; 这里只补 “怎么保持均衡” 这一块。' },
        { from: 'update_routing_bias', to: 'train loop', body: '它必须在 optimizer.step() 之后显式调用 —— 忘了调, b 永远是 0, 退化成无均衡。' },
        { from: '负载均衡', to: 'llm_train EP 与序列并行', body: '专家并行下不均衡 = 某张卡 all-to-all 收到的 token 爆掉容量, 直接拖慢整个 step。' },
      ],
      sourceRows: [
        { concept: '偏置只进排序', code: 'select_scores = sigmoid_scores + self.routing_bias', takeaway: 'topk 作用在 select_scores 上。' },
        { concept: '权重不带偏置', code: 'topk_sigmoid = sigmoid_scores.gather(-1, selected_experts)', takeaway: '从原始分数里取被选中专家的值再归一到和为 1。' },
        { concept: '更新规则', code: 'deepseekV3.py:update_routing_bias', takeaway: 'bincount 得到每个专家的负载; bias += γ·sign(mean − load)。' },
        { concept: '对照: aux loss', code: 'loss.py:MoELMLoss', takeaway: 'f_e (被选频率) × P_e (平均路由概率) 求和; 只有 P_e 这一支有梯度。' },
      ],
      snippetTitle: '前向选人 + 步后调偏置',
      snippet: `s = sigmoid(router(x))                       # [N, E] 原始亲和度
_, picked = topk(s + bias, k)                 # 偏置只影响 “选谁”
w = s.gather(-1, picked)
w = w / w.sum(-1, keepdim=True)               # 门控权重: 不含 bias
out = shared(x) + sum(w_i * expert_i(x))

# optimizer.step() 之后:
with no_grad():
    load = bincount(picked.flatten(), minlength=E)
    bias += gamma * sign(load.mean() - load)  # 过载 −γ, 欠载 +γ`,
      source: ['llm_models/models/moe/deepseekV3.py:update_routing_bias'],
      run: `${RUN}.moe.deepseek.train_deepseek`,
    },

    'models-dsa': {
      title: 'DSA · 用一个便宜的 indexer 决定 “值得算注意力的那 k 个”',
      subtitle: 'DeepSeek-V3.2 的稀疏注意力不改 mask 的形状规则, 而是让模型自己学: 每个 query 只在 indexer 选出的 top-k 个 key 上做 MLA。',
      tldr: 'LightningIndexer 用几个小 head 算 I[t,s] = Σ_h ReLU(q_h·k_h/√d), 每行留 top-k, 主注意力只看这 k 个。top-k 不可导, indexer 靠 KL(主注意力分布 ‖ softmax(I)) 单独训练, 且先 dense warmup 再切稀疏。',
      question: 'indexer 自己也要对所有 (t, s) 打分, 还是 O(T²) —— 那到底省在哪?',
      code: A,
      points: [
        { title: '省的是 “贵的那部分”', body: 'indexer 的 head 少、维度小、没有 value、不用 softmax, 可以用低精度算; 主注意力 (MLA 升维 + softmax + 乘 V) 从 O(T²) 降到 O(T·k)。长上下文下 k (如 2048) ≪ T (如 128K), 总成本由 indexer 这个小常数的 T² 主导。' },
        { title: 'indexer 是被 “蒸馏” 出来的', body: 'top-k 是离散选择, LM loss 的梯度传不到 indexer。于是单独加一项对齐 loss: 把各头平均后的主注意力分布当 teacher, 让 softmax(I) 去拟合。输入 detach, 对齐 loss 不改动主干表示。' },
        { title: '先 dense warmup', body: '一开始 indexer 是随机的, 直接稀疏 = 随机丢 key, 模型会被毁掉。所以先让主注意力看全部位置、只训 indexer 对齐 (set_dense_warmup), 对齐好了再切到稀疏继续训 —— 实验台里 “对齐程度” 滑杆演示的就是这个过程。' },
      ],
      links: [
        { from: 'attention · MLA', to: 'MultiHeadLatentSparseAttention', body: 'DSA = LightningIndexer + 一张 top-k mask + 原封不动的 MLA。' },
        { from: 'models-mtp · 掩码实验台', to: 'top-k 稀疏', body: '那里的 top-k 模式展示 mask 长什么样; 这里展示 mask 是怎么被 “学” 出来的, 以及 k 取多少才够。' },
        { from: 'cache["idx_k"]', to: 'KV cache 生成', body: '解码时历史 token 的 indexer key 也要缓存, 否则没法给旧位置打分。' },
      ],
      sourceRows: [
        { concept: 'indexer 打分', code: 'attention.py:LightningIndexer', takeaway: 'ReLU 后对 head 求和; 只需要排序, 不需要归一化。' },
        { concept: '输入 detach', code: 'self.indexer(q.detach(), mask=mask, cache=cache)', takeaway: '对齐 loss 只训练 indexer 自己的 w_q / w_k。' },
        { concept: 'top-k → mask', code: '_sparse_mask_from_topk', takeaway: '每行 scatter 出 k 个 True, 再与因果 mask 取交集。' },
        { concept: 'KL 对齐', code: 'kl = (p * (p.clamp_min(1e-9).log() - log_q)).sum(dim=-1)', takeaway: 'p = attn.mean(dim=1) 是 detach 的 teacher; 两边都限定在 used_mask 上。' },
        { concept: 'loss 汇总', code: 'loss.py:MoELMLoss', takeaway: 'total = LM + aux_loss_weight·aux + index_loss_weight·index_loss。' },
      ],
      snippetTitle: 'DSA 的三步',
      snippet: `# 1) 便宜的 indexer 给每对 (t, s) 打分
I = relu(q_idx @ k_idx.T / sqrt(d_idx)).sum(heads)     # [T, S]
I = I.masked_fill(~causal, -inf)

# 2) 每行只留 top-k, 昂贵的 MLA 只在这些位置上算
keep = zeros_like(I).scatter(-1, I.topk(k).indices, True) & causal
out, attn = mla(x, mask=causal if dense_warmup else keep)

# 3) top-k 不可导 → indexer 用单独的 KL 对齐 loss 训练
p = attn.mean(heads).detach()                          # teacher
index_loss = (p * (log(p) - log_softmax(I))).sum(-1).mean()`,
      source: ['llm_models/layers/core/attention.py:LightningIndexer'],
      run: `${RUN}.moe.deepseek_v3_2.train_deepseek_v3_2`,
    },

    'models-gptoss': {
      title: 'GPT-OSS 结构 · 滑窗/全注意力交替 + 可学 sink + MoE',
      subtitle: 'OpenAI 2025 年的开放权重模型没有发明新零件, 而是把三个已知零件拼得很讲究: 一半层用 128 的滑窗省 KV, 每个 head 一个可学的 sink logit, FFN 换成 MoE。',
      tldr: '偶数层滑窗 (KV 上限 W)、奇数层全注意力 (兜底长程), KV cache ≈ 全注意力模型的一半而感受野不打折。sink logit 拼进 softmax 分母、算完丢弃: 让 head 可以 “谁都不看”, 也让滑窗不再依赖开头那几个 token。',
      question: '滑窗会把开头 token 挤出 cache —— StreamingLLM 发现这会让模型崩溃, GPT-OSS 为什么不怕?',
      code: `llm_models/models/moe/gpt_oss.py · ${A} · llm_models/utils/masks.py`,
      points: [
        { title: '交替排布: 用一半的 KV 买全部感受野', body: '纯滑窗 L 层的感受野只有 L·(W−1)+1; 只要隔一层插一层全注意力, 任何位置一步就能够到全部历史。滑窗层的 KV 是滚动缓冲 (上限 W, 与上下文长度无关), 长上下文下 cache 几乎只由全注意力层决定。' },
        { title: 'softmax 不会 “弃权”', body: 'softmax 只看 logit 的相对差, 概率和恒为 1。一个 head 在某个位置没什么可看时, 也得输出一堆无关 value 的加权平均。没有 sink 的模型学会了把多余概率倒在第一个 token 上 —— 这就是 attention sink 现象。' },
        { title: '把 sink 做成参数而不是 token', body: 'StreamingLLM 的补救是永远保留开头 4 个 token (masks.py 的 sink_tokens 参数)。GPT-OSS 直接给每个 head 一个可学 logit 参与 softmax, 概率分给它就等于丢掉: 行和 < 1, 不占 KV, 也不依赖任何特定 token 留在窗口里。' },
      ],
      links: [
        { from: 'models-mtp · SWA', to: '交替排布', body: 'Mistral 全部层都是滑窗; GPT-OSS / Gemma 系列改成滑窗与全注意力交替。' },
        { from: 'build_sliding_window_mask', to: 'use_sink', body: 'sink_tokens 是 “mask 层面” 的 sink (保留开头 token); use_sink 是 “softmax 层面” 的 sink (可学 logit)。' },
        { from: 'moe · MixtralMoE', to: 'GPT-OSS FFN', body: 'FFN 槽位是标准的 top-k softmax 路由 MoE, 与 Mixtral 同族。' },
      ],
      sourceRows: [
        { concept: '带状 mask', code: 'masks.py:build_sliding_window_mask', takeaway: '(j <= i) & (j > i − W); sink_tokens > 0 时额外保留开头 S 列。' },
        { concept: 'sink 参数', code: 'self.sink = nn.Parameter(torch.zeros(num_heads))', takeaway: '每个 head 一个标量, 初始 0 = “多一个分数为 0 的空 key”。' },
        { concept: '参与 softmax 后丢弃', code: 'F.softmax(torch.cat([scores, sink], dim=-1), dim=-1)[..., :-1]', takeaway: '最后一列切掉 → 每行概率和 < 1。' },
        { concept: '滚动裁剪 cache', code: 'layer_cache["k"][:, :, -self.window_size:]', takeaway: '只有滑窗层裁: mini 模型各层 cache 长度 [8, 40, 8, 40], 比全注意力省 40%, 输出逐 token 不变。' },
        { concept: 'MoE FFN', code: 'gpt_oss.py:GPTOSSBlock', takeaway: '直接复用 MixtralBlock: “先 top-k 再 softmax” 与 “softmax → top-k → 重归一” 逐项相等。' },
      ],
      snippetTitle: '一层 GPT-OSS 风格的注意力',
      snippet: `# 层排布: 偶数层滑窗, 奇数层全注意力
mask = sliding_window_mask(T, W) if layer % 2 == 0 else causal_mask(T)

scores = q @ k.T / sqrt(d)                       # [H, T, S]
scores = scores.masked_fill(~mask, -inf)

sink = self.sink.view(H, 1, 1).expand(H, T, 1)   # 每个 head 一个可学 logit
probs = softmax(cat([scores, sink], -1), -1)     # sink 只进分母
probs = probs[..., :-1]                          # 丢掉 sink 列: 行和 < 1
out = probs @ v

# KV cache: 滑窗层只留最近 W 条, 全注意力层留全部
kv_entries = W if layer % 2 == 0 else T`,
      source: ['llm_models/models/moe/gpt_oss.py:is_swa', 'llm_models/utils/masks.py:build_sliding_window_mask'],
      run: `${RUN}.moe.gpt_oss.infer_gpt_oss`,
    },

    'models-llada': {
      title: 'LLaDA · 不从左到右写的语言模型',
      subtitle: '把 “扩散” 搬到离散 token 上: 加噪 = 随机把 token 换成 [MASK], 去噪 = 双向 Transformer 一次预测所有 [MASK]。和 BERT 的区别只在遮盖比例是随机的 t ∈ (0,1)。',
      tldr: '训练: 每条序列抽 t, 每个 token 以概率 t 被遮, 只在被遮位置算 CE 并乘 1/t (这是似然的上界, 不是启发式)。采样: 从全 [MASK] 出发, 每步预测全部 → 保留置信度高的 → 其余重新遮住, 剩余 [MASK] 数按线性日程递减。',
      question: 'BERT 也是遮盖再预测, 为什么 BERT 不能拿来生成而 LLaDA 能?',
      code: 'llm_models/models/language_models/llada.py',
      points: [
        { title: '随机遮盖比例 = 一族去噪任务', body: 'BERT 固定遮 15%, 只学会了 “补少量空”。LLaDA 的 t 均匀铺满 (0,1): t≈1 时几乎从零生成, t≈0 时只补一两个词。采样过程正是从 t=1 走到 t=0, 每一步遇到的遮盖比例训练时都见过。' },
        { title: '1/t 权重让 loss 成为似然上界', body: 't 小的样本被遮 token 少, 不加权的话几乎不贡献 loss; 乘 1/t 后 E[被遮数/t] = L, 均匀猜测时 loss 期望正好是 ln V, 与自回归 CE 同量纲、可比。代码里用分层采样铺 t 来压住 1/t 带来的方差。' },
        { title: '生成顺序由置信度决定', body: '低置信度重遮: 先定 “显然” 的 token, 它们成为上下文后再定难的。已定稿的 token 置信度记 +∞ 永不重遮。代价: 双向注意力没有 KV cache, 每步整段重算; 步数是质量/速度旋钮。续写和填空是同一个函数。' },
      ],
      links: [
        { from: 'BERT · MaskedLMLoss', to: 'LLaDALoss', body: '同样只在被遮位置算 CE; 区别是遮盖率随机 + 1/t 加权 + 除以总 token 数。' },
        { from: 'diffusion · 连续扩散', to: 'forward_process', body: '连续扩散加高斯噪声, 这里的 “噪声” 是 [MASK]; 两者都是 “训练时随机抽噪声强度 t, 采样时从 t=1 走回 0”。' },
        { from: 'KV cache 生成', to: 'LLaDA.sample', body: '自回归靠因果 mask 才有 KV cache; LLaDA 每步任何位置都可能变, 无法复用。' },
      ],
      sourceRows: [
        { concept: '加噪', code: 'llada.py:forward_process', takeaway: '每条序列一个 t, 每个 token 独立以概率 t 换成 [MASK]; batch 内 t 分层采样。' },
        { concept: 'ELBO loss', code: '(ce * masked / t[:, None]).sum() / labels.numel()', takeaway: '除以总 token 数而不是被遮 token 数, 1/t 才有正确的含义。' },
        { concept: '线性日程', code: 'n_masked = torch.round(n_gen * (1 - s / steps)).long()', takeaway: '第 s 步结束后应剩的 [MASK] 数。' },
        { concept: '重遮', code: 'x.masked_fill(rank < n_masked[:, None], self.mask_id)', takeaway: 'rank = 置信度升序名次; 已定稿位置 conf=+inf, 排在最后。' },
      ],
      snippetTitle: '训练一步 + 采样循环',
      snippet: `# 训练: 随机比例遮盖
t = uniform(eps, 1, size=[B])                    # 每条序列一个遮盖率
masked = rand(B, T) < t[:, None]
logits = model(x.masked_fill(masked, MASK))      # 双向注意力, 无因果 mask
loss = (CE(logits, x) * masked / t[:, None]).sum() / (B * T)

# 采样: 从全 [MASK] 迭代去噪
x = full([B, T], MASK)
for s in range(1, steps + 1):
    probs = softmax(model(x)); x0 = probs.argmax(-1)
    conf = where(x == MASK, probs.max(-1), +inf)   # 已定稿的永不重遮
    x = where(x == MASK, x0, x)                    # 先全部填上
    n = round(T * (1 - s / steps))                 # 本步后应剩的 [MASK] 数
    x[conf 最低的 n 个] = MASK                     # 没把握的遮回去`,
      source: ['llm_models/models/language_models/llada.py:forward_process'],
      run: `${RUN}.language_models.llada.infer_llada`,
    },

    'models-var': {
      title: 'VAR · 自回归的单位从 “下一个 token” 换成 “下一个分辨率”',
      subtitle: '把图像按光栅顺序展平再做 next-token, 既慢 (步数 = 像素数) 又破坏二维结构。VAR 让模型先画 1×1 的 “总体印象”, 再画 2×2、4×4 …, 每一级内部并行。',
      tldr: '多尺度 VQ 把图像编码成 token 金字塔: 每一级量化的是 “前面所有级还没解释掉的残差”。Transformer 用块状因果 mask (同级互相可见, 只能看更粗的级) 逐级预测, 生成整张图只要 K 次前向。',
      question: '同一尺度内的 token 是一次并行采样出来的, 彼此看不到对方的采样结果 —— 为什么画面不会乱?',
      code: 'llm_models/models/generative/var.py · llm_models/layers/diffusion/vq.py',
      points: [
        { title: '由粗到细符合图像的统计结构', body: '低分辨率决定构图和明暗, 高分辨率只补细节。先定粗的, 细的那一级每个 token 的不确定性已经很小, 并行独立采样带来的不一致也就很小。' },
        { title: '残差量化: 后面的级修正前面的级', body: '第 k 级的输入是 f − Σ_{j<k} upsample(z_j)。粗尺度量化得再粗糙, 误差都会留在残差里交给下一级; 所以 token 总数 (Σ s²) 虽然比单尺度多, 每一级的码本却可以共用同一个。' },
        { title: '块状因果 mask', body: '序列 = [1×1 | 2×2 | 4×4 | …] 拼接。mask 规则: 同一块内全可见, 只能看到更早 (更粗) 的块。训练时一次前向算完所有级的 loss; 采样时 K 次前向, 每次产出一整级。' },
      ],
      links: [
        { from: 'diffusion · DiT', to: 'VARModel', body: '扩散是 “在噪声强度上由粗到细”, VAR 是 “在分辨率上由粗到细”; 两者都避开了逐像素的串行。' },
        { from: 'MultiScaleVQ', to: 'ImageTokenizer', body: 'tokenizer 必须先单独训练 (重建 + vq_loss), 码本有意义之后才能训 VAR Transformer。' },
        { from: 'LLaDA', to: '级内并行采样', body: '两者都是 “一次前向定多个 token”; LLaDA 靠置信度挑, VAR 靠尺度结构分批。' },
      ],
      sourceRows: [
        { concept: 'token 金字塔', code: 'self.num_tokens = sum(s * s for s in scales)', takeaway: 'scales=(1,2,4) 时 L = 21; 序列长度是各级面积之和。' },
        { concept: '块状因果 mask', code: 'var.py:block_causal_mask', takeaway: '按 token 所属的级编号比较: 级号 ≤ 自己的都可见。' },
        { concept: '逐级采样', code: 'var.py:sample_tokens', takeaway: '每级: 累计重建 f_hat → 下采样成下一级的输入特征 → 一次前向 → 整级并行采样。' },
        { concept: '下一级的输入', code: 'feats.append(q._down(f_hat, s))', takeaway: '喂给 Transformer 的是 “到目前为止的重建” 在目标分辨率下的样子, 不是上一级的 token 本身。' },
      ],
      snippetTitle: 'next-scale 采样',
      snippet: `f_hat = 0                                   # 累计重建 [D, H, H]
feats, indices = [], []
for k, s in enumerate(scales):              # 1, 2, 4, 8 ...
    if k > 0:
        f_hat = f_hat + upsample(codebook[indices[-1]], H)
        feats.append(downsample(f_hat, s))  # 下一级的输入
    logits = transformer(feats, mask=block_causal)[-s*s:]
    idx = multinomial(softmax(logits / T))  # 整级 s×s 个 token 并行采样
    indices.append(idx.view(s, s))
image = decoder(f_hat + upsample(codebook[indices[-1]], H))

# tokenizer 侧 (多尺度残差量化):
#   r = f - f_hat;  z_k = quantize(downsample(r, s_k));  f_hat += upsample(z_k)`,
      source: ['llm_models/models/generative/var.py:sample_tokens'],
      run: `${RUN}.generative.var.infer_var`,
    },
  },
}
