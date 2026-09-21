// 阶段 4 · llm_finetune 的章节扩展。
// 前半: 覆盖已有章节的个别字段 (补 source / 修正过时文字); 后半: 新技术的新章节。
// source 只引用当前确实存在的文件+符号; 还没落地的 Python 模块先不写 source。
export default {
  stage: 'finetune',
  chapters: [
    { route: 'finetune-qlora', label: 'QLoRA · NF4 量化基座', hint: '分位数码本、block absmax、4.5 bit/参数' },
    { route: 'finetune-dora', label: 'DoRA 幅度/方向分解', hint: 'W = m · V/‖V‖, 低秩只管方向' },
    { route: 'finetune-simpo-orpo', label: 'SimPO · ORPO 无参考偏好优化', hint: '长度归一、目标间隔 γ、odds ratio' },
    { route: 'finetune-grpo-variants', label: 'GRPO 进阶: clip · DAPO · Dr.GRPO · GSPO', hint: '重要性比率、clip-higher、长度偏置、序列级比率' },
    { route: 'finetune-onpolicy-distill', label: 'on-policy 蒸馏', hint: '学生采样、逐 token reverse KL' },
    { route: 'finetune-rlvr', label: '看题的可验证奖励', hint: 'prompt 相关 reward 与常数基线' },
  ],
  pages: {
    // ───────────── 已有章节: 按字段覆盖 ─────────────
    'finetune-sft': {
      tldr: 'labels 已经整体左移一格 (labels = x[:, 1:]), 所以 prompt 长 P 时要 mask 的是前 P−1 格; 写成 [:P] 会把第一个 response token 也 mask 掉。cross_entropy 只在剩下的 response 位置上计算。',
      question: '为什么 prompt token 不应该贡献 loss? labels 已经错位之后, mask 的边界到底是 P 还是 P−1?',
      points: [
        { title: '数据目标改变', body: '预训练学接龙, SFT 学在用户指令后输出符合任务的 response。loss 形式不变, 变的只是哪些位置算数。' },
        { title: '边界是 P−1 不是 P', body: 'labels[t] = x[t+1]。最后一个 prompt token 所在的位置 P−1, 它的 label 正是第一个 response token —— 这一格必须留着。' },
        { title: 'bug 很隐蔽', body: '多 mask 一格 loss 照样下降, 只是模型从没学过"回答怎么开头"; 单 token 答案时则全部标签为 -100, loss = nan。' },
      ],
      sourceRows: [
        { concept: '错位', code: 'llm_finetune/data/tasks.py:make_labels', takeaway: 'labels = seq[:, 1:]: 输入与标签在数据侧就已经错开一格, loss 里不再 shift。' },
        { concept: 'prompt mask', code: 'labels[:, :P-1] = -100', takeaway: '第 P−1 格的目标是第一个 response token, 不能 mask; make_labels 里还有一条 assert 专门防这个差一位。' },
        { concept: 'ignore_index', code: 'methods/sft.py: SFTLoss = StandardLMLoss', takeaway: 'SFT 没有新 loss, 差异全在 labels; CE 对 -100 的位置既不算 loss 也不算进分母。' },
        { concept: '训练入口', code: 'run_finetune/sft/train_sft.py', takeaway: '把 dataset、loss、optimizer 接到一起。' },
      ],
      snippetTitle: 'SFT labels: 先错位, 再 mask',
      snippet: `x      = [BOS, p1, ..., p_{P-1}, r1, r2, EOS]     # 前 P 个是 prompt
idx    = x[:, :-1]                                # 模型输入
labels = x[:, 1:].clone()                         # 已左移一格: labels[t] = x[t+1]

labels[:, :P-1] = -100      # 正确: labels[P-1] = r1 要保留
# labels[:, :P] = -100      # 错误: 把 r1 也 mask 了 (off-by-one)

loss = cross_entropy(logits.reshape(-1, V), labels.reshape(-1),
                     ignore_index=-100)`,
      code: 'llm_finetune/data/tasks.py · llm_finetune/data/instruction_data.py · llm_finetune/methods/sft.py',
      source: ['llm_finetune/data/tasks.py:make_labels', 'llm_finetune/data/instruction_data.py:_sample'],
    },
    'finetune-lora': {
      subtitle: 'LoRA 不改变任务 loss, 改的是哪些参数允许更新。QLoRA 再把冻结的基座压到 4 bit, DoRA 把更新拆成幅度与方向 —— 各有专章。',
      code: 'llm_finetune/methods/lora.py · llm_finetune/methods/qlora.py · llm_finetune/utils/param_utils.py',
      links: [
        { from: 'SFT loss', to: 'LoRA SFT', body: 'loss 不变, optimizer 只看到 A/B。' },
        { from: 'target_modules', to: '模型结构', body: '通常注入 q/k/v/o 或 FFN linear。' },
        { from: 'methods/lora.py', to: 'methods/qlora.py', body: 'QLoRA 复用同一套 A/B 分支, 只把 base 权重换成 NF4 打包的 uint8 + 每 block 一个 scale。' },
        { from: 'LoRA adapter', to: 'infer Multi-LoRA', body: '推理服务可以让多个 adapter 共享同一基座。' },
      ],
      sourceRows: [
        { concept: 'LoRALinear', code: 'methods/lora.py:LoRALinear', takeaway: 'base 分支 + adapter 分支, B 初始化为 0。' },
        { concept: '注入', code: 'methods/lora.py:apply_lora', takeaway: '按模块名替换目标 Linear。' },
        { concept: 'merge', code: 'methods/lora.py:merge_lora_weights', takeaway: 'W ← W + (α/r)·B·A, 之后就是普通 Linear。' },
        { concept: '落盘', code: 'methods/lora.py:get_lora_state_dict', takeaway: '只保存 lora_A / lora_B。' },
        { concept: '4-bit 基座', code: 'methods/qlora.py:NF4Linear', takeaway: 'base 权重只以 uint8 buffer 存在, 每次前向反量化, 梯度只流向 A/B。' },
      ],
      source: ['llm_finetune/methods/lora.py:LoRALinear', 'llm_finetune/methods/lora.py:merge_lora_weights'],
    },
    'finetune-dpo': {
      source: ['llm_finetune/methods/dpo.py:DPOLoss', 'llm_finetune/methods/dpo.py:compute_sequence_logprobs'],
    },
    'finetune-rlhf': {
      source: ['llm_finetune/methods/reward_model.py:BradleyTerryLoss', 'llm_finetune/methods/distill.py:DistillLoss'],
    },
    'finetune-runs': {
      subtitle: 'methods/ 里的文件只定义算法, run_finetune/ 负责把数据、模型、优化器和保存策略接起来: sft · lora · qlora · dpo · rm · grpo · distill 各一个目录。',
      code: 'llm_finetune/run_finetune/{sft,lora,qlora,dpo,rm,grpo,distill}/train_*.py',
      sourceRows: [
        { concept: 'SFT run', code: 'run_finetune/sft/train_sft.py', takeaway: 'instruction batch + SFTLoss。' },
        { concept: 'LoRA run', code: 'run_finetune/lora/train_lora.py', takeaway: 'apply_lora 后只优化 adapter。' },
        { concept: 'QLoRA run', code: 'run_finetune/qlora/train_qlora.py', takeaway: '基座先 NF4 量化, 再挂 adapter。' },
        { concept: 'DPO run', code: 'run_finetune/dpo/train_dpo.py', takeaway: 'policy/ref 双模型路径。' },
        { concept: 'RM run', code: 'run_finetune/rm/train_rm.py', takeaway: '同一份偏好数据, 学的是标量分数。' },
        { concept: 'GRPO run', code: 'run_finetune/grpo/train_grpo.py', takeaway: '在线采样 + 规则奖励。' },
        { concept: '蒸馏 run', code: 'run_finetune/distill/train_distill.py', takeaway: 'teacher 冻结, student 学软标签。' },
      ],
    },

    // ───────────── 新章节 ─────────────
    'finetune-qlora': {
      title: 'QLoRA · 把冻结的基座压到 4 bit',
      subtitle: 'LoRA 省的是梯度和优化器状态, 基座权重本身还是 16 bit 躺在显存里。QLoRA 把它量化成 NF4, 只在前向时临时反量化。',
      tldr: '权重近似正态分布, 所以 16 个码点按正态分位数摆 (NF4) 比等间距 (INT4) 误差更小; 每 64 个权重共用一个 absmax scale, 合计约 4.5 bit/参数; 梯度只流向 LoRA 的 A/B。',
      question: '同样是 16 个码点, 为什么按分位数摆比等间距好? block 为什么不能太大也不能太小?',
      code: 'llm_finetune/methods/qlora.py · llm_finetune/run_finetune/qlora/train_qlora.py',
      points: [
        { title: '码点跟着密度走', body: '正态权重大多挤在 0 附近。NF4 在中间密、两端疏, 每个码被用到的概率接近 1/16; 等间距 INT4 两端的码几乎没人用。' },
        { title: 'block 隔离 outlier', body: '每个 block 除以自己的 absmax。一个 outlier 只会拉稀它所在 block 的码点, 代价是每 block 多存一个 scale: 4 + 32/B bit。' },
        { title: '量化的是基座, 不是训练', body: 'NF4 权重不更新、不需要梯度; 前向反量化成浮点再算, LoRA 分支全程高精度。误差由 adapter 在训练中顺带补偿。' },
      ],
      links: [
        { from: 'methods/lora.py', to: 'methods/qlora.py', body: 'A/B 分支完全复用, 只换 base 权重的存储方式。' },
        { from: 'nf4_quantize', to: 'nf4_dequantize', body: '分 block → 除 absmax → 最近邻查码本 → 两个 4-bit 索引拼一个 uint8; 反向就是查表乘 scale。' },
        { from: 'QLoRA', to: 'llm_infer 量化章节', body: '同样的 block-wise absmax 思路在推理侧的权重量化里再次出现。' },
      ],
      sourceRows: [
        { concept: '码本', code: 'methods/qlora.py: NF4_CODEBOOK', takeaway: '标准正态的 16 个等概率分位点, 归一化到 [-1,1], 0 被显式保留。' },
        { concept: '量化', code: 'methods/qlora.py:nf4_quantize', takeaway: 'absmax 缩放 + argmin 最近邻 + 4bit 打包。' },
        { concept: '反量化', code: 'methods/qlora.py:nf4_dequantize', takeaway: '拆包 → 查码本 → 乘回 block scale。' },
        { concept: '前向', code: 'methods/qlora.py:NF4Linear', takeaway: 'weight 属性即时反量化; 外面再套 LoRA: y = dequant(W)x + (α/r)·BAx。' },
        { concept: '显存账', code: 'methods/qlora.py:weight_bytes', takeaway: '0.5 字节/参数 + 每 64 个参数一个 FP32 scale ≈ 0.56 字节, 对 FP32 约 7.1×。' },
      ],
      snippetTitle: 'NF4 量化 / 反量化',
      snippet: `blocks = w.reshape(-1, 64)
scale  = blocks.abs().amax(dim=1, keepdim=True)     # 每 block 一个 absmax
normed = blocks / scale                             # ∈ [-1, 1]
idx    = (normed[..., None] - NF4_CODEBOOK).abs().argmin(-1)   # 最近的码点
packed = (idx[0::2] << 4) | idx[1::2]               # 两个 4bit 拼一个 uint8

# 前向时:
w_hat = NF4_CODEBOOK[unpack(packed)] * scale        # 反量化回浮点
y = x @ w_hat.T + (alpha / r) * (x @ A.T) @ B.T     # base 冻结, 只训 A/B`,
      source: ['llm_finetune/methods/qlora.py:nf4_quantize', 'llm_finetune/methods/qlora.py:NF4Linear'],
      run: 'python -m llm_finetune.run_finetune.qlora.train_qlora',
    },

    'finetune-dora': {
      title: 'DoRA · 把权重拆成幅度 × 方向',
      subtitle: 'LoRA 和全参微调的差距从哪来? DoRA 的回答: LoRA 把"改长度"和"转方向"绑在了同一个低秩更新上。',
      tldr: 'W′ = m · (W₀ + BA) / ‖W₀ + BA‖ (逐个权重向量取范数; 本仓库按输出行)。低秩更新 BA 只负责方向, 每个向量一个可训标量 m 负责幅度; 多出的参数只有 d 个, 推理前照样能 merge。',
      question: '同样的秩 r, 为什么多加一组"每个权重向量一个标量"就能更接近全参微调的学习行为?',
      code: 'llm_finetune/methods/dora.py · llm_finetune/methods/lora.py',
      points: [
        { title: '两个量被解耦', body: '全参微调里幅度变化 ΔM 和方向变化 ΔD 近乎独立; LoRA 里两者强正相关 —— 想只转方向, ΔV 必须精确落在一个点上。' },
        { title: '归一化 = 梯度投影', body: 'V 被除以自己的范数, 所以 loss 对 V 的梯度自动去掉径向分量: 低秩容量全部花在转方向上。' },
        { title: '几乎零成本', body: '比 LoRA 多 d 个参数 (相对 2dr 多 1/(2r)); 训练时每步要显式构造 W′ 并求范数; merge 后与普通 Linear 无异。' },
      ],
      links: [
        { from: 'LoRALinear', to: 'DoRALinear', body: 'base + BA 不变, 外面再套一层"按列归一化 × m"。' },
        { from: 'weight normalization', to: 'DoRA', body: '思路同源: 把参数化改成幅度/方向, 优化曲面更友好。' },
        { from: 'DoRA', to: 'merge', body: '训练完把 m·V/‖V‖ 算出来写回 W, 推理零开销。' },
      ],
      sourceRows: [
        { concept: '初始化', code: 'methods/dora.py:DoRALinear', takeaway: 'lora_magnitude = ‖W₀‖ (按输出行, [d_out]); B = 0 且 m = ‖W₀‖ ⇒ 第 0 步输出与原模型完全一致。' },
        { concept: '方向分支', code: 'V = W₀ + (α/r)·B·A', takeaway: '和 LoRA 一样的低秩更新, 但只有它的方向会被用到。' },
        { concept: '归一化', code: 'direction.norm(dim=1, keepdim=True).detach()', takeaway: '论文建议把范数当常数 (detach), 省一份反向显存, 效果几乎不变。' },
        { concept: '幅度', code: 'lora_magnitude: nn.Parameter([d_out])', takeaway: '每个输出行一个标量; 名字带 lora_ 以复用冻结 / 落盘工具。' },
      ],
      snippetTitle: 'DoRA forward (骨架)',
      snippet: `V = W0 + (alpha / r) * (B @ A)            # 方向分支: 与 LoRA 相同的低秩更新
norm = V.norm(dim=1, keepdim=True)         # 逐行范数 [d_out, 1]
W = m[:, None] * V / norm.detach()         # m: 每行一个可训幅度; 范数当常数
y = x @ W.T                                # 归一化作用于整行, 每步都要显式构造 W

# 初始化: B = 0, m = W0.norm(dim=1)  ->  W == W0
# merge:  W0 <- m[:, None] * V / V.norm(dim=1, keepdim=True)`,
      source: ['llm_finetune/methods/dora.py:DoRALinear'],
      run: 'python -m llm_finetune.run_finetune.dora.train_dora',
    },

    'finetune-simpo-orpo': {
      title: 'SimPO · ORPO — 不要 reference model 的偏好优化',
      subtitle: 'DPO 的 reference 要多占一份显存、多做两次前向。拿掉它之后, 得有别的东西防止模型"靠长度作弊"和"漂离 SFT 起点"。',
      tldr: 'SimPO: 用每 token 平均 log p 当隐式奖励 (长度归一化), 再减目标间隔 γ。ORPO: 在 SFT 的 NLL 上直接加一项 −log σ(log odds ratio), 一个阶段、一个模型完成 SFT + 对齐。',
      question: 'DPO 里的 reference 除了"别离 SFT 太远", 还顺带解决了什么? 拿掉它之后谁来接手?',
      code: 'llm_finetune/methods/simpo.py · llm_finetune/methods/orpo.py · llm_finetune/methods/dpo.py',
      points: [
        { title: 'sum log p 有长度红利', body: '序列越长 Σlog p 越负。不做任何校正时, rejected 更长就等于白送 margin, 模型什么都没学 loss 就到 0。' },
        { title: 'SimPO: 归一化 + γ', body: '按长度取平均后数值范围变小, 所以 β 取 2 左右; γ 要求 chosen 的平均 log p 至少领先 γ/β 才停手。奖励与生成时的打分 (平均 log p) 一致。' },
        { title: 'ORPO: SFT 项当锚', body: 'loss = NLL(chosen) + λ·(−log σ(log odds_c − log odds_r))。NLL 防漂移, odds ratio 压低 rejected; 连单独的 SFT 阶段都省了。' },
      ],
      links: [
        { from: 'DPOLoss', to: 'SimPO / ORPO', body: '同一份 (prompt, chosen, rejected) 数据, 同一个 logsigmoid 外壳, 换的是送进去的 z。' },
        { from: 'compute_sequence_logprobs', to: '平均 log p', body: 'SimPO / ORPO 都需要再除以 response 长度 (有效 label 数)。' },
        { from: 'ref 两次前向', to: '省掉', body: 'DPO: policy×2 + ref×2; SimPO / ORPO: policy×2。' },
      ],
      sourceRows: [
        { concept: 'DPO 的 z', code: 'methods/dpo.py:DPOLoss', takeaway: 'β·[(π_c − ref_c) − (π_r − ref_r)]: ref 同样"越长越负", 长度项成对抵消。' },
        { concept: 'SimPO 的 z', code: 'methods/simpo.py:SimPOLoss', takeaway: 'β·(logp_c/|y_c| − logp_r/|y_r|) − γ: 长度归一化顶替 ref; γ 是目标间隔 (本仓库默认 β=2, γ=1)。' },
        { concept: 'ORPO 的 odds', code: 'log(p/(1−p)), p = exp(mean logp)', takeaway: '用每 token 平均概率, 同样与长度无关。' },
        { concept: 'ORPO 总 loss', code: 'methods/orpo.py:ORPOLoss', takeaway: 'nll_chosen + λ·(−logsigmoid(log_or)); 论文 λ=0.1, 本仓库玩具任务默认 0.5; 直接从 base 模型起训。' },
      ],
      snippetTitle: '三种 z 并排',
      snippet: `lp_c, lp_r = seq_logprob(policy, chosen), seq_logprob(policy, rejected)   # [B] 求和
n_c, n_r   = len(chosen_resp), len(rejected_resp)

# DPO: 需要 ref 的两次前向
z_dpo   = beta * ((lp_c - ref_c) - (lp_r - ref_r))

# SimPO: 无 ref, 长度归一 + 目标间隔
z_simpo = beta * (lp_c / n_c - lp_r / n_r) - gamma

# ORPO: 无 ref, odds ratio + SFT 锚
odds    = lambda lp, n: (lp / n) - log1p(-exp(lp / n))     # log(p / (1-p))
loss_orpo = -(lp_c / n_c) + lam * -logsigmoid(odds(lp_c, n_c) - odds(lp_r, n_r))

loss = -logsigmoid(z).mean()      # DPO / SimPO 共用的外壳`,
      source: ['llm_finetune/methods/simpo.py:SimPOLoss', 'llm_finetune/methods/orpo.py:ORPOLoss'],
    },

    'finetune-grpo-variants': {
      title: 'GRPO 进阶 · 重要性比率、clip 与 DAPO / Dr.GRPO / GSPO',
      subtitle: '最小版 GRPO 每批样本只更新一次。想让一批昂贵的采样多用几轮, 就必须引入 ρ = π_new/π_old 和 clip —— 后面三个变体都是在修这个目标里的偏置。',
      tldr: '多轮内层更新 ⇒ 需要重要性比率 + clip。DAPO: 放宽上界 (clip-higher)、丢弃全对/全错的组 (动态采样)、token 级 loss。Dr.GRPO: 去掉 /σ 和 1/|o| 两个偏置。GSPO: 用序列级比率替代 token 级比率。',
      question: '同一条"又长又错"的回答, 在 GRPO / Dr.GRPO / DAPO 下每个 token 分到的惩罚分别是多少? 为什么上界 clip 只卡得住低概率 token?',
      code: 'llm_finetune/methods/grpo.py · llm_finetune/run_finetune/grpo/train_grpo.py',
      points: [
        { title: 'clip 是"停手线"', body: 'min(ρÂ, clip(ρ)Â) 在越界一侧是平的, 梯度为 0。ρ ≤ 1/π_old, 所以高概率 token 根本碰不到上界, 被卡住的只有低概率的探索 token → 熵塌缩; clip-higher 只放宽上界。' },
        { title: '两个隐藏偏置', body: '1/|o_i| 让长的错误回答每 token 罚得更轻 (越错越长); /σ 给几乎全对、几乎全错的题加权。Dr.GRPO 把两者都换成常数。' },
        { title: '比率的粒度', body: '奖励是序列级的, token 级 ρ_t 却是单样本噪声估计。GSPO 用 s = exp(mean log ρ_t), 整条序列共用一个权重、一起 clip。' },
      ],
      links: [
        { from: 'GRPOTrainer.step', to: '多轮内层更新', body: '采样一次, 存下 logp_old, 对同一批做 K 轮更新; 第 1 轮 ρ ≡ 1。' },
        { from: 'PPO clipped surrogate', to: 'GRPO', body: 'GRPO 只换了优势的来源 (组内归一化), 目标函数外形不变。' },
        { from: '组内奖励全相同', to: 'DAPO 动态采样', body: 'Â 全 0 的组不贡献梯度, 丢弃并继续采样直到 batch 填满有效组。' },
        { from: 'token 级 ρ_t', to: 'GSPO 序列级 s', body: 'MoE 路由抖动会放大 token 级比率噪声; 序列级几何平均稳定得多。' },
      ],
      sourceRows: [
        { concept: '四个变体 = 五个开关', code: 'methods/grpo.py:GRPOConfig', takeaway: 'clip_high / std_norm / loss_agg / seq_ratio / dynamic_sampling; VARIANTS 字典把 grpo · dapo · dr_grpo · gspo 写成开关组合。' },
        { concept: '组内优势', code: 'methods/grpo.py:group_advantages', takeaway: 'r − μ, std_norm=True 时再除 σ (torch.std, 无偏)。' },
        { concept: '重要性比率', code: 'ratio = exp(logp_new − logp_old.detach())', takeaway: 'logp_old 在采样后立刻算好并固定。' },
        { concept: 'clip-higher', code: 'clamp(ratio, 1−ε_low, 1+ε_high)', takeaway: 'DAPO 取 0.2 / 0.28, 只放宽上界。' },
        { concept: 'loss 聚合', code: 'methods/grpo.py:aggregate', takeaway: 'seq_mean (GRPO, 1/|o_i|) · token_mean (DAPO) · fixed_len (Dr.GRPO, 常数分母): 只差每个 token 的权重。' },
        { concept: 'Dr.GRPO', code: 'adv = r − μ;  / L_max', takeaway: '不除 σ, 不除 |o_i|。' },
      ],
      snippetTitle: '带 ratio + clip 的 GRPO 内层循环',
      snippet: `seqs, mask = sample_group(policy, prompts, G)
r    = reward_fn(prompts, seqs)                       # [B, G]
adv  = r - r.mean(1, keepdim=True)                    # Dr.GRPO 到此为止
adv  = adv / (r.std(1, keepdim=True) + 1e-4)          # GRPO / DAPO 再除 σ
keep = r.std(1) > 0                                   # DAPO: 全对/全错的组丢弃重采
logp_old = token_logprobs(policy, seqs).detach()

for _ in range(K):                                    # 同一批样本更新 K 轮
    logp  = token_logprobs(policy, seqs)
    ratio = (logp - logp_old).exp()                   # token 级 ρ_t
    # GSPO: ratio = (((logp - logp_old) * mask).sum(1) / mask.sum(1)).exp()[:, None]
    obj = torch.min(ratio * adv, ratio.clamp(1 - e_lo, 1 + e_hi) * adv)
    loss = -(obj * mask).sum(1) / mask.sum(1)         # GRPO: 先按序列平均 (1/|o_i|)
    # DAPO: -(obj * mask).sum() / mask.sum()          # token 级
    # Dr.GRPO: -(obj * mask).sum(1) / L_MAX           # 常数分母
    loss.mean().backward(); opt.step(); opt.zero_grad()`,
      source: ['llm_finetune/methods/grpo.py:GRPOConfig', 'llm_finetune/methods/grpo.py:group_advantages', 'llm_finetune/methods/grpo.py:aggregate'],
      run: 'python -m llm_finetune.run_finetune.grpo.train_grpo',
    },

    'finetune-onpolicy-distill': {
      title: 'on-policy 蒸馏 · 学生自己写, 老师逐 token 打分',
      subtitle: '离线蒸馏让学生模仿老师写的文本 (forward KL); on-policy 蒸馏让学生先自己生成, 再在自己走到的每个状态上向老师对齐 (reverse KL)。',
      tldr: '学生采样 y ~ π_s, 对每个位置计算 KL(π_s ‖ π_t) 并最小化。监督信号像蒸馏一样稠密 (每 token 一个), 数据分布像 RL 一样 on-policy (学生自己的状态), 没有 exposure bias。',
      question: '容量不够的学生, 是该"摊开盖住老师的所有答法", 还是"挑一种答法学到位"? 这和 KL 的方向有什么关系?',
      code: 'llm_finetune/methods/on_policy_distill.py · llm_finetune/methods/distill.py',
      points: [
        { title: 'KL 方向决定性格', body: 'forward KL 在老师的样本上求期望 → 老师有质量的地方学生必须有 (mode-covering, 会在低谷里胡说); reverse KL 在学生的样本上求期望 → 学生只为自己去的地方负责 (mode-seeking)。' },
        { title: '分布错配', body: '离线蒸馏只见过老师的前缀; 推理时学生走进自己的前缀, 错误逐步累积。on-policy 直接在学生的前缀上训练。' },
        { title: '稠密 + on-policy', body: 'RL 每条序列只有 1 个奖励; on-policy 蒸馏每个 token 都有老师的完整分布当"奖励", 样本效率高一到两个数量级。' },
      ],
      links: [
        { from: 'DistillLoss (离线)', to: 'on-policy 蒸馏', body: '温度软化 + T² 补偿仍然适用; 变的是样本来源和 KL 方向。' },
        { from: 'policy.generate', to: 'teacher 打分', body: '与 GRPO 共用采样管线, 只是 reward_fn 换成老师的逐 token log p。' },
        { from: 'reverse KL', to: 'RL 的 KL 惩罚', body: 'GRPO 里锚住 ref 的那一项就是同一个量: E_{y~π}[log π − log π_ref]。' },
      ],
      sourceRows: [
        { concept: '离线基线', code: 'methods/distill.py:DistillLoss', takeaway: 'α·CE + (1−α)·T²·KL(p_t^T ‖ p_s^T): 这是 forward KL, 数据来自固定语料。' },
        { concept: '学生采样', code: 'methods/on_policy_distill.py:on_policy_distill_step', takeaway: '采样不回传梯度; 梯度只来自随后的那次前向。' },
        { concept: '逐 token reverse KL', code: 'methods/on_policy_distill.py:token_kl', takeaway: 'Σ_v q_s(v)·(log q_s(v) − log p_t(v)); 在完整词表上精确求和, 不需要对 KL 做采样估计。' },
        { concept: 'T² 补偿', code: '* (T * T)', takeaway: '软标签梯度 ∝ 1/T², 不补偿的话调高 T 等于关掉蒸馏项。' },
      ],
      snippetTitle: 'on-policy 蒸馏一步',
      snippet: `with torch.no_grad():
    seqs = student.generate(prompts)                 # 1. 学生自己写 (on-policy)
    t_logp = log_softmax(teacher(seqs), -1)          # 2. 老师在学生的前缀上给出分布

s_logp = log_softmax(student(seqs), -1)              # 3. 学生带梯度重算一遍
# 4. 逐 token reverse KL: 期望在学生自己的分布上取
kl = (s_logp.exp() * (s_logp - t_logp)).sum(-1)      # [B, T]
loss = (kl * completion_mask).sum() / completion_mask.sum()
loss.backward(); opt.step()`,
      source: ['llm_finetune/methods/on_policy_distill.py:on_policy_distill_step', 'llm_finetune/methods/on_policy_distill.py:token_kl'],
      run: 'python -m llm_finetune.run_finetune.distill.train_distill',
    },

    'finetune-rlvr': {
      title: '看题的可验证奖励 · 别让 RL 只学会一个常数',
      subtitle: 'RLVR 用程序判分代替奖励模型。但如果判分规则不看 prompt, reward 涨到满分也说明不了模型学会了任务。',
      tldr: '奖励必须是 (prompt, completion) 的函数。"输出落在词表后半区就给分"这类区域奖励, 最优策略是无视 prompt 的常数输出; 换成"答案 = f(prompt)"之后, 常数策略的得分上限会掉到 1/类别数。',
      question: '一条一路上涨到 1.0 的 reward 曲线, 怎样区分"学会了按题作答"和"学会了一个 unigram 偏好"?',
      code: 'llm_finetune/data/tasks.py · llm_finetune/data/prompt_data.py · llm_finetune/methods/grpo.py',
      points: [
        { title: '常数基线', body: '让所有 prompt 输出同一个 completion 能拿到的最高分。它等于 100% 的奖励函数, 验证不了条件行为。' },
        { title: '塌缩是最短路径', body: '区域奖励下 RL 会迅速把概率压到某个高分 token 上: 熵归零、与输入无关, 曲线却很漂亮。' },
        { title: '真实 RLVR 天然看题', body: '数学答案比对、单元测试、格式校验都依赖题目。自造玩具任务时要主动保证这一点, 并单独报告常数基线。' },
      ],
      links: [
        { from: '旧版区域奖励', to: 'SeqTask.verify', body: 'reward_fn 的签名从 (completion) 变成 (prompts, completions); 旧的区域奖励已从仓库移除。' },
        { from: 'PromptDataGenerator', to: 'SeqTask', body: 'prompt 里编码题目 (copy / reverse / sort 一段序列), 答案由程序算出; 13^6 种 prompt, 不可能靠背。' },
        { from: '组内零方差', to: '动态采样', body: '看题任务早期常见"整组全错": 这些组没有梯度, 见 GRPO 进阶一章。' },
      ],
      sourceRows: [
        { concept: '区域奖励 (反例)', code: 'reward = (completion >= V//2).mean()', takeaway: '只看 completion 里落在 [V/2, V) 的比例 —— 与 prompt 无关; 本仓库早期版本用过, 已换掉。' },
        { concept: '看题奖励', code: 'llm_finetune/data/tasks.py:SeqTask', takeaway: 'verify(prompts, completions): 前 R 个 token 与 target(prompts) 完全一致 (含 EOS) 才得 1 分。' },
        { concept: '常数基线', code: 'max_c mean_prompt reward(prompt, c)', takeaway: '训练前先算一遍, 作为"没学会"的参考线。' },
        { concept: '评估', code: '按 prompt 分桶的准确率', takeaway: '只看平均 reward 会被常数策略骗过。' },
      ],
      snippetTitle: '两种 reward_fn 的签名',
      snippet: `def region_reward(completion):                  # 不看题: 常数策略即可满分
    return (completion >= V // 2).float().mean(-1)

def sum_reward(prompt, completion):             # 看题: 答案随 prompt 变
    target = (prompt[:, -2] + prompt[:, -1]) % 10
    return (completion[:, 0] == target).float()

# 训练前的体检: 常数基线
best_const = max(sum_reward(prompts, full_like(c)).mean() for c in range(10))
# region_reward 的 best_const = 1.0  ->  这个任务验证不了条件生成`,
      source: ['llm_finetune/data/tasks.py:SeqTask'],
      run: 'python -m llm_finetune.run_finetune.grpo.train_grpo',
    },
  },
}
