// 阶段 4 · llm_finetune 的全部章节 (11 章)。
// 每章都写全 StageTopic 要渲染的字段, 不再依赖 models.js 里的旧副本。
// source 只写真实存在的 def / class —— npm run check:sources 会逐条校验。
// 页面里的数字全部来自 llm_finetune/run_finetune/*/readme.md, 可以用对应的 run 命令复现。
export default {
  stage: 'finetune',
  chapters: [
    { route: 'finetune-qlora', label: 'QLoRA · NF4 量化基座', hint: '4 bit 存基座, 前向再解开; 整模型 389 KB → 59 KB' },
    { route: 'finetune-dora', label: 'DoRA 幅度/方向分解', hint: '低秩只管转方向, 每行长度单独学一个标量' },
    { route: 'finetune-simpo-orpo', label: 'SimPO · ORPO 无参考偏好优化', hint: '拿掉 ref: 前向次数减半, 显存减半' },
    { route: 'finetune-grpo-variants', label: 'GRPO 进阶: clip · DAPO · Dr.GRPO · GSPO', hint: '一批采样更新多次, 才需要比率和 clip' },
    { route: 'finetune-onpolicy-distill', label: 'on-policy 蒸馏', hint: '学生自己写, 老师逐 token 打分' },
    { route: 'finetune-rlvr', label: '看题的可验证奖励', hint: '奖励不看题, RL 就只学会一个常数' },
  ],
  pages: {
    // ───────────────────────── SFT ─────────────────────────
    'finetune-sft': {
      title: 'SFT 数据与 loss · 只教模型回答, 不教模型提问',
      subtitle: '全章围着一行 mask 转: 哪一格该盖 -100, 哪一格盖了就等于从没教模型怎么开口。',
      tldr: 'labels 在数据侧已经左移一格, 所以 prompt 长 P 时只能盖前 P−1 格。盖成 P 格会把第一个回复 token 一起盖掉 —— 同配置实测: 正确写法留出集 exact-match 1.000, 多盖一位 0.000。',
      question: 'labels 已经错位之后, mask 的边界到底是 P 还是 P−1? 多盖一格为什么从 loss 曲线上看不出来?',
      code: 'llm_finetune/data/tasks.py · llm_finetune/data/instruction_data.py · llm_finetune/methods/sft.py',
      points: [
        { title: 'SFT 没有新 loss', body: '还是那个带 ignore_index=-100 的交叉熵, SFTLoss 就是 StandardLMLoss 的别名。预训练学接龙, SFT 学"看到问题就回答", 全部差别只在 labels: 哪些位置算数。' },
        {
          title: '边界是 P−1',
          key: true,
          body: 'labels[t] 存的是 x[t+1]。最后一个 prompt token 坐在位置 P−1, 它要预测的恰好是第一个回复 token, 这一格必须留着。make_labels 里有一条 assert 专门守这个: 被监督的位置数必须等于回复长度。',
        },
        { title: '差一格看不出来', body: '多盖一格只少一个监督位, loss 照样往下掉, 但模型从没学过"看完问题第一个词说什么"。反转任务上重跑: 正确写法留出集 EM 1.000, 多盖一位 0.000。答案只有 1 个 token 时全部标签都是 -100, loss 直接 nan。' },
      ],
      links: [
        { from: '预训练 CE', to: 'SFTLoss', body: '同一个类, 换的只是 labels 里哪些位置是 -100。' },
        { from: 'make_labels', to: 'DPO / SimPO / ORPO', body: '偏好方法的 log π(y|x) 也从这套 labels 来。边界写错, 区分 chosen 与 rejected 的 log p(y₁|x) 这一项就丢了。' },
        { from: 'SFT 终态', to: 'policy / reference', body: 'DPO 的 policy 和 ref 都从 SFT checkpoint 复制。' },
      ],
      sourceRows: [
        { concept: '错位', code: 'llm_finetune/data/tasks.py:make_labels', takeaway: 'labels = seq[:, 1:] 后面补一格 -100。输入和标签在数据侧就错开了, loss 里不再 shift。' },
        { concept: 'prompt mask', code: 'labels[:, :P-1] = -100', takeaway: '只盖 prompt 内部的转移。第 P−1 格的目标是第一个回复 token, 盖了就白训。' },
        { concept: '差一位的守卫', code: 'assert (labels != IGNORE).sum(1) == n_response', takeaway: '被监督的位置数必须恰好等于回复长度 (含 EOS), 写错立刻炸在这里。' },
        { concept: 'ignore_index', code: 'llm_finetune/methods/sft.py: SFTLoss = StandardLMLoss', takeaway: 'CE 对 -100 的位置既不算 loss 也不算进分母。' },
        { concept: '验收', code: 'llm_finetune/data/tasks.py:exact_match', takeaway: '留出集与训练集按 prompt 的 token 和 mod 5 不相交, 所以报的是泛化不是背诵。' },
      ],
      snippetTitle: 'SFT labels: 先错位, 再 mask',
      snippet: `x      = [BOS, p1, ..., p_{P-1}, r1, r2, EOS]     # 前 P 个是 prompt
idx    = x[:, :-1]                                # 模型输入
labels = x[:, 1:].clone()                         # 已左移一格: labels[t] = x[t+1]

labels[:, :P-1] = -100      # 正确: labels[P-1] = r1 要保留
# labels[:, :P] = -100      # 错误: 把 r1 也 mask 了 (off-by-one)

loss = cross_entropy(logits.reshape(-1, V), labels.reshape(-1),
                     ignore_index=-100)`,
      source: ['llm_finetune/data/tasks.py:make_labels', 'llm_finetune/data/instruction_data.py:_sample'],
      run: 'python -m llm_finetune.run_finetune.sft.train_sft',
    },

    // ───────────────────────── LoRA ─────────────────────────
    'finetune-lora': {
      title: 'LoRA 参数高效微调 · 冻结 W, 只学低秩 ΔW',
      subtitle: 'LoRA 换来了什么、又赔上了什么 —— 这一章不照抄那句"LoRA 收敛更快"。',
      tldr: '把 y = Wx 换成 y = Wx + (α/r)·BAx: W 冻结, 只训 A、B。省下的是梯度、Adam 状态和"每个任务一整份权重"; 不省训练步数 —— 同样 300 步, 全参留出集 EM 0.809, LoRA(r=8) 只有 0.352。',
      question: 'LoRA 到底买到了什么? 为什么 B 必须初始化成 0, 而 A 不行?',
      code: 'llm_finetune/methods/lora.py · llm_finetune/methods/qlora.py · llm_finetune/utils/param_utils.py',
      points: [
        { title: '低秩的是 ΔW, 不是 W', body: '微调带来的权重变化内在维度很低, 所以拿 B(d_out×r)·A(r×d_in) 去装它。参数从 d_in·d_out 降到 r·(d_in+d_out): d=4096、r=8 时一层从 16M 降到 65K, 占 0.39%。任务离预训练分布越远 ΔW 越不低秩, LoRA 就越吃亏。' },
        { title: 'B = 0 是无害启动', body: 'B 全零 ⇒ 第 0 步 BA = 0 ⇒ 输出与基座逐位相同, 微调起点就是预训练终点。A 随机初始化保证 B 第 1 步就有梯度; 两个都置零, 梯度永远是 0。' },
        {
          title: '买的是显存和分发, 不是速度',
          key: true,
          body: '同一基座从 copy 适配到 sort, 各 300 步: 全参 99,648 个可训参数 / Adam 状态 778 KB / 留出集 EM 0.809; LoRA r=8 只有 19,456 个参数 / 152 KB, lr 拉到 1e-2 也才 EM 0.352。旧版 README 里"LoRA 收敛更快"是病态初始化加 10× 学习率造出来的假象, 已经删掉。',
        },
      ],
      links: [
        { from: 'SFT loss', to: 'LoRA SFT', body: 'loss 一行不改, 只是 optimizer 只看得见 A/B。' },
        { from: 'target_modules', to: '模型结构', body: 'apply_lora 按属性名末段匹配 (w_q / w_v …), 与层路径无关, 所以同一份代码能注到任何沿用同名属性的模型上。' },
        { from: 'methods/lora.py', to: 'methods/qlora.py', body: 'QLoRA 复用同一套 A/B 分支, 只把 base 权重换成 NF4 打包的 uint8 加每 block 一个 scale。' },
        { from: 'adapter', to: 'llm_infer Multi-LoRA', body: '不 merge, 多个 adapter 就能共享同一份基座, 按请求切换。' },
      ],
      sourceRows: [
        { concept: 'LoRALinear', code: 'llm_finetune/methods/lora.py:LoRALinear', takeaway: 'base 分支加 adapter 分支; A 用 Kaiming uniform, B 全零。' },
        { concept: '注入', code: 'llm_finetune/methods/lora.py:apply_lora', takeaway: '按模块名把目标 Linear 换成 LoRALinear。' },
        { concept: '只训 adapter', code: 'llm_finetune/methods/lora.py:mark_only_lora_as_trainable', takeaway: '把 base 的 requires_grad 关掉; 再用 print_trainable_parameters 核一遍才算数, loss 下降证明不了 base 被冻住。' },
        { concept: 'merge', code: 'llm_finetune/methods/lora.py:merge_lora_weights', takeaway: 'W ← W + (α/r)·B·A, 换回普通 Linear。要断言数值相等 (实测前后 logits 最大差 6.4e-06); 只断言形状会漏掉转置和漏乘 α/r。' },
        { concept: '落盘', code: 'llm_finetune/methods/lora.py:get_lora_state_dict', takeaway: '只抽出 lora_A / lora_B: 本仓库 adapter 76 KB, 整模型 389 KB。' },
      ],
      snippetTitle: 'LoRA forward 与 merge',
      snippet: `base = linear(x, W, b)              # W frozen
delta = lora_B(lora_A(dropout(x))) * (alpha / r)
y = base + delta

# merge:
W <- W + (alpha / r) * (B @ A)`,
      source: ['llm_finetune/methods/lora.py:LoRALinear', 'llm_finetune/methods/lora.py:merge_lora_weights'],
      run: 'python -m llm_finetune.run_finetune.lora.train_lora',
    },

    // ───────────────────────── QLoRA ─────────────────────────
    'finetune-qlora': {
      title: 'QLoRA · 把冻结的基座压到 4 bit',
      subtitle: '学会算一个模型量化之后到底占多少字节, 顺带搞清楚为什么 embedding 和 lm_head 要放过。',
      tldr: 'LoRA 省掉了梯度和优化器状态, 冻结的基座本身还整份躺在显存里。QLoRA 把它存成 NF4, 前向时现场反量化: 本仓库整模型 389 KB → 59 KB, 压了 6.57×, 基座原任务留出集 EM 仍是 1.000。',
      question: '同样 16 个码点, 为什么按分位数摆比等间距好? block 为什么不能太大也不能太小?',
      code: 'llm_finetune/methods/qlora.py · llm_finetune/run_finetune/qlora/train_qlora.py',
      points: [
        { title: '码点跟着密度走', body: '权重大多挤在 0 附近。NF4 的 16 个码点取标准正态的等概率分位数, 中间密两端疏, 每个码被用到的概率都接近 1/16; 等间距 INT4 有好几个码浪费在几乎没有权重的两端。' },
        {
          title: '账要算整个模型',
          key: true,
          body: '存储 = 0.5 字节/参数 + 每 64 个参数一个 fp32 scale = 0.5625 字节/参数, 对 fp32 的理论上限 7.1×。实测只有 6.57×, 差额就是没量化的 embedding / lm_head / RMSNorm。只报"被量化那几层"的压缩比是在骗自己。',
        },
        { title: '量化的是基座, 不是训练', body: 'NF4 权重是只读 buffer, 不更新也不要梯度; 前向反量化成浮点再算, LoRA 分支全程高精度。9.0% 的单层相对量化误差由 adapter 在训练里顺带补掉 —— 适配 sort 的留出集 EM 0.402, 反而略高于同配置的 fp32 LoRA (0.352)。' },
      ],
      links: [
        { from: 'methods/lora.py', to: 'methods/qlora.py', body: 'A/B 分支完全复用, 只换 base 权重的存储方式。' },
        { from: 'nf4_quantize', to: 'nf4_dequantize', body: '分 block → 除 absmax → 最近邻查码本 → 两个 4 bit 拼一个 uint8; 反向就是拆包查表再乘回 scale。' },
        { from: 'block_size', to: '误差 / 开销', body: 'block 小则 scale 更贴合局部、误差更小, 但 scale 开销 4/B 字节每参数变大 —— 论文再对 scale 做一次双重量化就是为了压这块。' },
        { from: 'QLoRA', to: 'llm_infer 权重量化', body: '同一套 block-wise absmax 在推理侧再出现一次。' },
      ],
      sourceRows: [
        { concept: '码本', code: 'llm_finetune/methods/qlora.py: NF4_CODEBOOK', takeaway: '标准正态的 16 个等概率分位点, 归一化到 [-1, 1], 0 被显式保留。' },
        { concept: '量化', code: 'llm_finetune/methods/qlora.py:nf4_quantize', takeaway: '按 block 除 absmax, argmin 找最近码点, 两个 4 bit 打一个 uint8。block_size 为奇数时打包会错位, 脚本开头有单测。' },
        { concept: '反量化', code: 'llm_finetune/methods/qlora.py:nf4_dequantize', takeaway: '拆包查码本再乘回 block scale。' },
        { concept: '前向', code: 'llm_finetune/methods/qlora.py:NF4Linear', takeaway: 'weight 是个 property, 每次取都现场反量化; 外面再套 LoRA: y = dequant(W)x + (α/r)·BAx。' },
        { concept: '哪些层放过', code: 'llm_finetune/methods/qlora.py:apply_qlora', takeaway: '只量化 14 个线性层。embedding 与 lm_head 共享权重、查表的行分布不像正态, 且误差直接进 logits; bitsandbytes 默认也跳过。' },
        { concept: '显存账', code: 'llm_finetune/methods/qlora.py:weight_bytes', takeaway: '0.5625 字节/参数。合并时先反量化再加 ΔW, 保持高精度, 否则刚学到的更新会被量化噪声抹掉一部分。' },
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

    // ───────────────────────── DoRA ─────────────────────────
    'finetune-dora': {
      title: 'DoRA · 把权重拆成幅度 × 方向',
      subtitle: '把 LoRA 与全参微调的差距定位到具体一步, 再用一个 d_out 维向量补上一截。',
      tldr: '把每一行权重看成"长度 × 方向": W′ = m ⊙ (W₀ + BA) / ‖W₀ + BA‖_row。低秩更新只管转方向, 长度由 m 单独学。每层只多 d_out 个参数, 同 r=8、3 个种子平均: LoRA 留出集 EM 0.382 → DoRA 0.522。',
      question: '同样的秩 r, 为什么多加一组"每行一个标量"就能更接近全参微调?',
      code: 'llm_finetune/methods/dora.py · llm_finetune/methods/lora.py',
      points: [
        {
          title: '两个量在 LoRA 里被绑住了',
          key: true,
          body: '全参微调时, 一行权重的长度变化和方向变化基本各走各的; LoRA 的 ΔW = BA 是加性的, 想"只转方向不改长度", ΔW 必须精确落在一个点上。DoRA 把两件事拆开: 归一化之后 BA 只剩方向, 长度交给 m。',
        },
        { title: '归一化顺手做了梯度投影', body: 'V = W₀ + (α/r)BA 被除以自己的行范数, loss 对 V 的梯度里径向那一份自动消掉 —— 低秩容量全花在转方向上。反传时分母 .detach(), 省一整份 [d_out, d_in] 的梯度显存, 效果几乎不变。' },
        { title: '参数几乎白送, 时间不白送', body: '比 LoRA 每层多 d_out 个参数 (本仓库 19,456 → 20,736, 多 1,280 个)。代价在训练: 归一化作用在整行上, 每步都要显式构造 [d_out, d_in] 的 W′, 拆不成两次小 matmul。merge 完就是普通 Linear, 推理零开销。' },
      ],
      links: [
        { from: 'LoRALinear', to: 'DoRALinear', body: '继承同一个类, base + BA 不变, 外面再套一层"按行归一化 × m"。' },
        { from: 'weight normalization', to: 'DoRA', body: '思路同源: 把参数化改成幅度 / 方向, 优化曲面更好走。' },
        { from: 'DoRA', to: 'merge', body: '训完把 m·V/‖V‖ 算出来写回 W, 推理和 LoRA 一样没有额外开销。' },
      ],
      sourceRows: [
        { concept: '初始化', code: 'llm_finetune/methods/dora.py:DoRALinear', takeaway: 'lora_magnitude = ‖W₀‖_row, 形状 [d_out]; 配上 B = 0, 第 0 步 W′ 与 W₀ 逐位相同。m 初始化成 1 就没这个性质了。' },
        { concept: '方向分支', code: 'V = W₀ + (α/r)·B·A', takeaway: '和 LoRA 一模一样的低秩更新, 但只有它的方向会被用到。' },
        { concept: '归一化', code: 'direction.norm(dim=1, keepdim=True).detach()', takeaway: '论文 §4.3 把范数当常数; 忘了 detach 结果差不多, 但多一份全尺寸梯度显存。' },
        { concept: '幅度', code: 'lora_magnitude: nn.Parameter([d_out])', takeaway: '名字带 lora_ 前缀, 直接复用冻结和落盘工具。训练后 layers.0.attn.w_q 的 m 相对初值平均变了 59.7% —— 这个任务确实要改长度。' },
        { concept: '诚实的读数', code: 'run_finetune/dora/readme.md', takeaway: '3 个种子平均 loss 0.369 → 0.296, EM 0.382 → 0.522; 但换成 r=2 时 3 个种子里有 1 个是 LoRA 赢。玩具规模, 断言只要求平均值。' },
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

    // ───────────────────────── DPO ─────────────────────────
    'finetune-dpo': {
      title: 'DPO 偏好对齐 · 用 chosen/rejected 直接优化策略',
      subtitle: '这一章会让你多盯一个指标: reward margin 之外的 log π(chosen) —— 它掉下去的时候, 模型正在变差。',
      tldr: 'KL 约束下的最优策略满足 r(x,y) = β·log π/π_ref + const; 代回 Bradley-Terry, RM 和 PPO 两步就塌缩成一个对偏好对的二分类。policy 刚从 ref 复制时 Δ = 0, 第 1 步 loss 恰好是 ln 2 = 0.6931。',
      question: 'reward margin 一路拉大, 为什么生成质量反而掉了?',
      code: 'llm_finetune/methods/dpo.py · llm_finetune/data/preference_data.py',
      points: [
        { title: '数据形态变了, loss 外壳没变', body: '样本是 (prompt, chosen, rejected), 不是单条标准答案。送进 −log σ(·) 的是隐式奖励差 Δ = (log π_c − log ref_c) − (log π_r − log ref_r)。ref 对同一条长回复也给出同样低的 Σlog p, 相减之后长度红利被抵消掉。' },
        {
          title: 'margin 变大 ≠ 模型变好',
          key: true,
          body: 'loss 只看差值, "两边一起降、rejected 降得更快"同样让 loss 变小。200 步实测: 留出集偏好准确率 0.965 → 0.996, 可是 log π(chosen) 从 −4.03 掉到 −4.18, 贪心 exact-match 从 0.332 掉到 0.137。这就是 likelihood displacement, 脚本对它设了断言。',
        },
        { title: 'β 越大越保守', body: 'β 不是"用力程度"。β 大 → σ 更快饱和 → 排对一点点梯度就没了 → policy 贴着 ref 不动; β 小 → 要把 Δ 拉得很大才停手, 漂得更远。它就是 RLHF 目标里 KL 惩罚系数的化身。' },
      ],
      links: [
        { from: 'SFT 终态', to: 'policy / reference', body: 'policy 可训, ref 冻结 + eval + no_grad。ref 若被注册成子模块, 会进 optimizer、被 .train() 切模式, 所以代码里故意把它塞在 tuple 里。' },
        { from: 'make_labels', to: 'compute_sequence_logprobs', body: '同样忽略 prompt 和 pad。多盖一位会丢掉 log p(y₁|x) —— 上下文相同, 但 chosen 与 rejected 的 y₁ 可以不同, 这一项恰恰是区分两者的。' },
        { from: 'DPOLoss', to: 'SimPO / ORPO', body: '同一份数据、同一个 logsigmoid 外壳, 换的是送进去的 z 和要不要 ref。' },
        { from: 'PairwiseForward', to: '通用 Trainer', body: '把 "policy 前向 + ref 前向" 包成一个 nn.Module, Trainer 一行不改; 旧版 DPOTrainer 里复制的训练循环已删。' },
      ],
      sourceRows: [
        { concept: 'logprob 汇总', code: 'llm_finetune/methods/dpo.py:compute_sequence_logprobs', takeaway: '返回 [B], 保留逐样本粒度。用 cross_entropy(reduction="sum") 会把整个 batch 加成一个数, 没法逐对相减。' },
        { concept: 'DPO 的 z', code: 'beta * ((p_c - p_r) - (ref_c - ref_r))', takeaway: 'policy 相对 ref 的隐式奖励差。policy = ref 时恒为 0, 所以第 1 步 loss 必是 ln 2 —— 最好用的自检。' },
        { concept: '梯度权重', code: 'llm_finetune/methods/dpo.py:DPOLoss', takeaway: '每个样本的权重是 σ(−βΔ): 排对且已拉开的样本自动退出, 力气集中在排错的那些上。' },
        { concept: '一步的开销', code: 'llm_finetune/methods/dpo.py:PairwiseForward', takeaway: '序列数 ×2 (chosen + rejected), 再加一次不带梯度的 ref 前向; 常驻权重 ×2 = 778 KB。ref 不存激活, 所以实测每步只慢约 35%, 不是 100%。' },
        { concept: '偏好数据', code: 'llm_finetune/data/preference_data.py:PreferenceDataGenerator', takeaway: 'chosen = 正确回复, rejected = 改错一个 token 或漏掉一个。两边格式必须对称, 否则"只有一边带 EOS"这种捷径会被学走。' },
      ],
      snippetTitle: 'DPO 核心',
      snippet: `pi_logratios  = policy_chosen - policy_rejected
ref_logratios = ref_chosen - ref_rejected
logits = beta * (pi_logratios - ref_logratios)

loss = -F.logsigmoid(logits).mean()`,
      source: ['llm_finetune/methods/dpo.py:DPOLoss', 'llm_finetune/methods/dpo.py:compute_sequence_logprobs'],
      run: 'python -m llm_finetune.run_finetune.dpo.train_dpo',
    },

    // ───────────────────────── SimPO / ORPO ─────────────────────────
    'finetune-simpo-orpo': {
      title: 'SimPO · ORPO — 不要 reference model 的偏好优化',
      subtitle: '搞清 ref 在替你挡什么, 以及拿掉它之后得请谁来接班。',
      tldr: 'ref 要多占一份显存、每步多一次前向 (实测 DPO 2 次 / 778 KB / 5.4 s, SimPO 与 ORPO 1 次 / 389 KB / 4.0 s)。SimPO 用长度归一化的平均 log p 当奖励再减目标间隔 γ; ORPO 直接在 SFT 的 NLL 上加一项 odds ratio 惩罚, 一个阶段搞定。',
      question: 'DPO 里的 ref 除了"别离 SFT 太远", 还顺手解决了什么? 拿掉它谁来接手?',
      code: 'llm_finetune/methods/simpo.py · llm_finetune/methods/orpo.py · llm_finetune/methods/dpo.py',
      points: [
        { title: 'Σlog p 有长度红利', body: '序列越长, Σlog p 越负。不做任何校正的话, rejected 更长就等于白送 margin: 模型什么偏好都没学, loss 已经到 0。DPO 靠 ref 抵消这一项, 因为 ref 对同一条长回复也给出同样低的分。' },
        { title: 'SimPO: 归一化 + γ', body: '按长度取平均, 长度直接约掉, 奖励也和生成时的打分 (平均 log p) 对上了。代价是平均后数值范围小得多, β 要取 2 左右; γ 规定 chosen 至少领先 γ/β 才准停手 —— 没有 ref 就得自己画一条线。' },
        {
          title: '省下的显存, 要用别的东西换',
          key: true,
          body: '三者的偏好准确率都上去了 (0.988~0.996), 差别在 log π(chosen): DPO −4.25、SimPO −5.69、ORPO −2.51, 对应留出集 EM 0.121 / 0.023 / 0.543。SimPO 什么锚都没有, 掉得最惨; ORPO 的 NLL 项就是那个锚, 所以它能从零开始训 (300 步 EM 0.973, 与纯 SFT 打平, 还把 rejected 压得更低: −16.34 vs −14.85)。',
        },
      ],
      links: [
        { from: 'DPOLoss', to: 'SimPO / ORPO', body: '同一份 (prompt, chosen, rejected)、同一个 logsigmoid 外壳, 换的是送进去的 z。' },
        { from: 'compute_sequence_logprobs', to: '平均 log p', body: 'SimPO 与 ORPO 都要再除以回复的有效 label 数。' },
        { from: 'ref 的两次前向', to: '省掉', body: 'DPO: policy×2 + ref×2; SimPO / ORPO: policy×2。实测每步快约 25% —— 反传还在, 所以省不到一半。' },
        { from: 'ORPO 的 NLL 项', to: 'SFT', body: '前一项就是 SFT, 所以 ORPO 不需要先做一遍 SFT; 它同时充当防漂移的锚。' },
      ],
      sourceRows: [
        { concept: 'DPO 的 z', code: 'llm_finetune/methods/dpo.py:DPOLoss', takeaway: 'β·[(π_c − ref_c) − (π_r − ref_r)]: ref 同样"越长越负", 长度项成对抵消。' },
        { concept: 'SimPO 的 z', code: 'llm_finetune/methods/simpo.py:SimPOLoss', takeaway: 'β·(logp_c/|y_c| − logp_r/|y_r|) − γ; 本仓库默认 β=2、γ=1。沿用 DPO 的 β=0.1 会几乎推不动。' },
        { concept: 'γ = 0 且不归一化', code: 'z = logp_c − logp_r', takeaway: '退化成"π_ref 取均匀分布的 DPO", 长度红利全回来了。' },
        { concept: 'ORPO 的 odds', code: 'log(p/(1−p)), p = exp(mean logp)', takeaway: '用每 token 平均概率, 同样与长度无关。p→1 时 log(1−p) 要 clamp, 否则 log 0。' },
        { concept: 'ORPO 总 loss', code: 'llm_finetune/methods/orpo.py:ORPOLoss', takeaway: 'nll_chosen + λ·(−logsigmoid(log_or))。论文 λ=0.1, 本仓库玩具任务默认 0.5。odds 在 p→1 处发散, 对"两边都已经很自信"的差距比概率比更敏感。' },
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
      run: 'python -m llm_finetune.run_finetune.simpo_orpo.train_simpo_orpo',
    },

    // ───────────────────────── RM · GRPO · 蒸馏 ─────────────────────────
    'finetune-rlhf': {
      widgets: ['GrpoLab', 'SoftmaxTempLab'],
      title: 'RM · GRPO · 蒸馏 — 从偏好到能力迁移',
      subtitle: '三件事: 标注有噪声时 RM 最多学到多大分差、GRPO 的 baseline 从哪来、蒸馏的 KL 项为什么要乘 T²。',
      tldr: 'RM 把"A 比 B 好"学成一个能随时调用的标量分 (留出集偏好准确率 0.549 → 0.930); GRPO 用同题 G 条回复的组内均值当 baseline, 把 PPO 的 critic 整个省掉; 蒸馏用温度软化的 teacher 分布, 给 student 比硬标签密得多的监督。',
      question: 'GRPO 砍掉 critic 之后 baseline 从哪来? 标注有 20% 概率标反时, RM 学到的分差会停在哪?',
      code: 'llm_finetune/methods/reward_model.py · llm_finetune/methods/grpo.py · llm_finetune/methods/distill.py',
      points: [
        {
          title: 'RM 只学分差',
          key: true,
          body: 'Bradley-Terry: P(A≻B) = σ(r_A − r_B), loss = −log σ(r_w − r_l)。给所有分数加同一个常数, loss 一点不变 —— 分数没有量纲, 只有序。标注有 ε 的概率标反时, 期望 loss 的零梯度点落在 σ(Δ) = 1−ε, 即最优分差停在 ln((1−ε)/ε): ε=0.2 约 1.39, ε=0.5 时为 0。',
        },
        { title: '组内排名顶替 critic', body: '同一个 prompt 采 G 条, Â = (r − μ)/σ。组内均值就是"这道题的期望得分"的蒙特卡洛估计, 正是 value 网络想预测的东西。代价: 一组全对或全错时 Â 全为 0, 这批样本不产生任何梯度 —— 实测每步约 32% 的 prompt 落在这种情况里。' },
        { title: '暗知识在分布里', body: '硬标签只告诉 student 一个 token; teacher 软化后的分布把所有相似 token 的相对排序都交出来。KD 项要乘 T²: 软标签对 logits 的梯度按 1/T² 衰减, 不补偿的话调高温度等于偷偷关掉蒸馏项。数据有限时软标签明显更好 (forward KL 2.292 → 1.860), 数据无限时这个优势会消失。' },
      ],
      links: [
        { from: 'PreferenceDataGenerator', to: 'RewardModel', body: '与 DPO 同一份数据, 两种用法: RM 学打分, DPO 直接学策略。' },
        { from: 'RM 分数', to: 'GRPO 的 reward_fn', body: 'reward_fn(prompts, completions) 既可以是 RM, 也可以是程序 verifier —— 后者就是 RLVR。' },
        { from: 'policy.generate', to: '组内优势', body: '在线采样, 数据分布随 policy 漂移 (on-policy); 这也是 GRPO 接不进通用 Trainer 的原因。' },
        { from: 'teacher logits / T', to: 'student KL', body: 'T 放大暗知识, T² 补回 softmax 梯度的 1/T² 缩放, 软硬两项的相对权重才只由 α 决定。' },
      ],
      sourceRows: [
        { concept: 'value head', code: 'llm_finetune/methods/reward_model.py:RewardModel', takeaway: '复用 LLaMA 骨架, lm_head 换成 Identity 再接一个 [D]→[1] 的头。手抄一遍主干前向的话, 主干一改 (mask / RoPE / cache) 就会悄悄不一致。' },
        { concept: '读哪一个位置', code: 'attention_mask.sum(1) - 1', takeaway: '右 pad 时取最后一个真 token。取 [:, -1] 读到的是 pad: 同一序列多垫 5 位 PAD, 分数偏移从 5e-06 变成 4.54。' },
        { concept: 'BT loss', code: 'llm_finetune/methods/reward_model.py:BradleyTerryLoss', takeaway: '第 1 步 loss 0.6929 ≈ ln 2 (两边分数还没拉开)。' },
        { concept: '组内优势', code: 'llm_finetune/methods/grpo.py:group_advantages', takeaway: 'adv = (r − mean) / (std + eps); 组内奖励全相同时 adv 全为 0, 这组白占 batch —— DAPO 的动态采样就是冲它去的。' },
        { concept: '重要性比率', code: 'llm_finetune/methods/grpo.py:GRPOTrainer', takeaway: '一批 rollout 更新 μ=2 次: 第 1 个 epoch ρ ≡ 1 (实测 |ρ−1| = 0.000), clip 从第 2 个 epoch 才开始起作用。' },
        { concept: 'KL 的 k3 估计', code: 'd.exp() - d - 1', takeaway: '逐 token、恒 ≥ 0、方差比直接用 log-ratio 小。只有 β>0 时才需要常驻一份 ref。' },
        { concept: '蒸馏损失', code: 'llm_finetune/methods/distill.py:DistillLoss', takeaway: 'α·CE + (1−α)·T²·KL(p_t^T ‖ p_s^T); KD 项和 CE 项必须用同一个 mask, 否则两项优化的不是同一批位置。' },
      ],
      snippetTitle: 'GRPO 单步的完整控制流',
      snippet: `seqs = policy.generate(prompts × G)        # 1. 组内采样
rewards = reward_fn(prompts, seqs)         # 2. 规则验证打分 (RLVR)
adv = (r - r.mean(group)) / r.std(group)   # 3. 组内相对优势
loss = -(adv * logp.mean(-1)).mean()       # 4. 策略梯度
     + beta * kl(policy, ref)              #    + KL 锚定
loss.backward(); opt.step()                # 5. 一次更新`,
      source: [
        'llm_finetune/methods/reward_model.py:BradleyTerryLoss',
        'llm_finetune/methods/grpo.py:GRPOTrainer',
        'llm_finetune/methods/distill.py:DistillLoss',
      ],
      run: 'python -m llm_finetune.run_finetune.rm.train_rm',
    },

    // ───────────────────────── GRPO 变体 ─────────────────────────
    'finetune-grpo-variants': {
      title: 'GRPO 进阶 · 重要性比率、clip 与 DAPO / Dr.GRPO / GSPO',
      subtitle: '把三个变体各自在修的那一处偏置指出来; 也别把四条噪声之内的最终分数读成"谁更强"。',
      tldr: '一批采样只更新一次的话, ρ ≡ 1, clip 是死代码。想让昂贵的 rollout 多用几轮, 就得引入 ρ = π_new/π_old 和裁剪; 三个变体分别在修这个目标里的三处偏置: 熵塌缩、长度偏置、比率噪声。',
      question: '同一条"又长又错"的回答, 在 GRPO / Dr.GRPO / DAPO 下每个 token 分到的惩罚分别是多少? 为什么上界 clip 只卡得住低概率 token?',
      code: 'llm_finetune/methods/grpo.py · llm_finetune/run_finetune/grpo/train_grpo.py',
      points: [
        { title: 'clip 是停手线, 不是刹车', body: 'min(ρÂ, clip(ρ)Â) 在越界那一侧是平的, 梯度为 0。ρ 最大只能到 1/π_old, 所以 π_old=0.9 的 token 顶天涨到 1.11, 根本碰不到 1.2 的上界; 被卡住的全是想翻身的低概率探索 token → 熵塌缩。DAPO 的 clip-higher 因此只放宽上界 (0.2 / 0.28)。' },
        {
          title: '两处偏置藏在分母里',
          key: true,
          body: '1/|o_i| 让答错的长回复每个 token 罚得更轻 —— 实测 3-token 回复的单 token 权重是 9-token 的 3.0 倍, 等于鼓励"错就错得长一点"; ÷σ 给几乎全对、几乎全错的题加权 (难题比中等题的权重 0.66, 去掉 σ 后只有 0.44)。Dr.GRPO 把两个分母都换成常数。',
        },
        { title: '比率的粒度', body: '奖励是整条序列给的, token 级 ρ_t 却是单样本噪声估计。GSPO 改用 s = exp(mean_t log ρ_t), 整条回复共用一个权重一起裁剪: 实测 std log ρ 从 0.067 降到 0.027 (约小 √C 倍), 所以它的裁剪区间也要跟着收窄到 0.05。' },
      ],
      links: [
        { from: 'GRPOTrainer.step', to: '多轮内层更新', body: '采样一次, 存下 logp_old, 对同一批做 μ 次更新。实测第 1 个 epoch |ρ−1| = 0.000, 第 2 个 epoch 才偏离。' },
        { from: 'PPO clipped surrogate', to: 'GRPO', body: '目标函数外形完全一样, GRPO 只换了优势的来源 (组内归一化)。' },
        { from: '组内奖励全相同', to: 'DAPO 动态采样', body: 'Â 全 0 的组不贡献梯度。实测这类 prompt 占 32%; DAPO 把它们从 loss 里去掉 (进 loss 的 A=0 样本 32.1% → 0.0%)。本实现只过滤不补采, batch 会变小。' },
        { from: '采样温度 T', to: 'log-prob 的 T', body: '采样用 π^{1/T}, 算 log-prob 也必须用同一个 T, 否则 ρ 的分母不是行为策略, 第 1 个 epoch 就已经 off-policy。' },
      ],
      sourceRows: [
        { concept: '四个变体 = 五个开关', code: 'llm_finetune/methods/grpo.py:GRPOConfig', takeaway: 'clip_high / std_norm / loss_agg / seq_ratio / dynamic_sampling; VARIANTS 字典把 grpo · dapo · dr_grpo · gspo 写成开关组合。' },
        { concept: '组内优势', code: 'llm_finetune/methods/grpo.py:group_advantages', takeaway: 'r − μ; std_norm=True 时再除 σ。' },
        { concept: '重要性比率', code: 'ratio = exp(logp_new − logp_old.detach())', takeaway: 'logp_old 在采样后立刻用同一温度算好并冻结。' },
        { concept: 'loss 聚合', code: 'llm_finetune/methods/grpo.py:aggregate', takeaway: 'seq_mean (GRPO, ÷|o_i|) · token_mean (DAPO) · fixed_len (Dr.GRPO, ÷常数 C): 差别只在每个 token 的权重。' },
        { concept: '别误读最终分数', code: 'run_finetune/grpo/readme.md', takeaway: '60 步后留出集采样 pass@1: grpo 0.287 / dapo 0.326 / dr_grpo 0.324 / gspo 0.320, 差异在噪声内。贪心 EM 基本没动 (0.562 → 0.53) —— RLVR 主要把 pass@k 挤进 pass@1, 不是教新能力。' },
        { concept: '用什么验收', code: '采样 pass@1, 不是贪心 EM', takeaway: 'RL 优化的就是采样分布; 拿贪心解码去验收 RL, 什么都看不到。' },
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
      source: [
        'llm_finetune/methods/grpo.py:GRPOConfig',
        'llm_finetune/methods/grpo.py:group_advantages',
        'llm_finetune/methods/grpo.py:aggregate',
      ],
      run: 'python -m llm_finetune.run_finetune.grpo.train_grpo',
    },

    // ───────────────────────── on-policy 蒸馏 ─────────────────────────
    'finetune-onpolicy-distill': {
      title: 'on-policy 蒸馏 · 学生自己写, 老师逐 token 打分',
      subtitle: '用 KL 写在哪一侧解释两种蒸馏的性格差异, 再挑出容量不够的学生该用哪一种。',
      tldr: '离线蒸馏在老师写的前缀上教学生 (forward KL, 摊开盖住所有峰); on-policy 让学生先自己生成, 再在自己走到的每个位置上对齐老师 (reverse KL, 钻进一个峰)。实测样本合格率 0.059 → 0.402。',
      question: '容量不够的学生, 该摊开盖住老师的所有答法, 还是挑一种答到位? 这跟 KL 写在哪一侧有什么关系?',
      code: 'llm_finetune/methods/on_policy_distill.py · llm_finetune/methods/distill.py',
      points: [
        {
          title: 'KL 的方向决定性格',
          key: true,
          body: 'KL(p‖q) = Σ p·log(p/q): 谁写在前面, 期望就在谁的分布上取。forward KL 在老师的样本上取期望 —— 老师有质量而学生没有的地方惩罚趋于无穷, 学生被迫全覆盖, 代价是往两峰之间的低谷里也放概率, 采样时半路串台。reverse KL 在学生自己的样本上取期望 —— 学生不去的地方权重为 0, 于是它缩进一个峰。',
        },
        { title: '分布错配', body: '离线蒸馏只见过老师的前缀; 推理时学生走进自己的前缀, 那是从没被教过的状态, 错误一步步累积。on-policy 直接在学生的前缀上训练, 训练分布就是推理分布。' },
        { title: '稠密 + on-policy, 但会丢多样性', body: 'RL 每条序列只有 1 个标量奖励; on-policy 蒸馏每个 token 位置都有老师的完整 V 维分布, 而且是解析求 KL, 不用 REINFORCE。实测合格率 0.402 vs 离线 0.059, reverse KL 0.51 vs 2.04; 代价是老师 30% 概率的少数派答法被压到 log π = −33 (离线只到 −17)。' },
      ],
      links: [
        { from: 'DistillLoss (离线)', to: 'on-policy 蒸馏', body: '温度软化和 T² 补偿仍然适用; 变的是样本来源和 KL 写在哪一侧。' },
        { from: 'student.generate', to: 'teacher 打分', body: '与 GRPO 共用采样管线, 只是把 reward_fn 换成老师的逐 token 分布。采样不回传梯度, 它只决定"在哪些状态上算 KL"。' },
        { from: 'reverse KL', to: 'GRPO 的 KL 惩罚', body: '锚住 ref 的那一项就是同一个量: E_{y~π}[log π − log π_ref]。' },
        { from: '热身', to: 'on-policy', body: '从随机初始化直接上 on-policy 没用: 学生采的全是垃圾, 老师在这些分布外前缀上的输出也没意义。先用 SFT 或离线蒸馏热身 200 步。' },
      ],
      sourceRows: [
        { concept: '离线基线', code: 'llm_finetune/methods/distill.py:DistillLoss', takeaway: 'α·CE + (1−α)·T²·KL(p_t^T ‖ p_s^T)。这是 forward KL, 数据来自固定语料。' },
        { concept: '学生采样', code: 'llm_finetune/methods/on_policy_distill.py:on_policy_distill_step', takeaway: 'generate 在 no_grad 下跑; 梯度只来自随后那次带梯度的前向。' },
        { concept: '逐 token reverse KL', code: 'llm_finetune/methods/on_policy_distill.py:token_kl', takeaway: 'Σ_v q_s(v)·(log q_s(v) − log p_t(v)), 在完整词表上精确求和, 不需要对 KL 做采样估计。' },
        { concept: '各赢各的', code: 'run_finetune/on_policy_distill/readme.md', takeaway: 'off-policy forward KL 0.88 / reverse 2.04; on-policy forward 1.36 / reverse 0.51。谁优化哪个指标就赢哪个, 别只看一栏。' },
        { concept: '为什么不接 Trainer', code: 'run_finetune/on_policy_distill/train_on_policy_distill.py', takeaway: '训练数据由当前 student 现场生成, 数据生成器得拿到模型本身 —— 和 GRPO 一个原因。' },
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
      source: [
        'llm_finetune/methods/on_policy_distill.py:on_policy_distill_step',
        'llm_finetune/methods/on_policy_distill.py:token_kl',
      ],
      run: 'python -m llm_finetune.run_finetune.on_policy_distill.train_on_policy_distill',
    },

    // ───────────────────────── RLVR ─────────────────────────
    'finetune-rlvr': {
      title: '看题的可验证奖励 · 别让 RL 只学会一个常数',
      subtitle: '跑 RL 之前先算一个数: 常数基线。它决定那条漂亮的 reward 曲线值不值钱。',
      tldr: '奖励必须是 (prompt, completion) 的函数。"输出落在词表后半区就给分"这类区域奖励, 最优策略是无视 prompt 的常数输出, reward 能涨到 1.0 却什么都没证明; 换成"答案 = f(prompt)"之后, 常数策略的上限掉回 1/类别数。',
      question: '一条一路涨到 1.0 的 reward 曲线, 怎么分辨"学会按题作答"和"学会了一个 unigram 偏好"?',
      code: 'llm_finetune/data/tasks.py · llm_finetune/data/prompt_data.py · llm_finetune/methods/grpo.py',
      points: [
        {
          title: '先算常数基线',
          key: true,
          body: '让所有 prompt 都输出同一个 completion, 能拿到的最高平均奖励就是常数基线。它等于 1.0 的奖励函数验证不了任何条件行为; 看题任务里它通常只有 1/类别数, 模型超过它才说明在读 prompt。训练前先算一遍, 当作"没学会"的参考线。',
        },
        { title: '塌缩是最短路径', body: '区域奖励下, RL 会飞快把概率压到某个高分 token 上: 熵归零、输出与输入无关, 曲线却很好看。奖励不看题的时候, 这是拿高分最省力的走法, 不是模型偷懒。' },
        { title: '真实 RLVR 天然看题', body: '数学答案比对、单元测试、格式校验都依赖题目本身。自己造玩具任务时要主动保证这一点: 本仓库的 SeqTask 有 13⁶ ≈ 480 万种 prompt, 训练集与留出集按 token 和 mod 5 不相交, 背不下来。' },
      ],
      links: [
        { from: '旧版区域奖励', to: 'SeqTask.verify', body: 'reward_fn 的签名从 (completion) 变成 (prompts, completions); 旧的区域奖励已从仓库移除。' },
        { from: 'PromptDataGenerator', to: 'SeqTask', body: 'prompt 里编码题目 (copy / reverse / sort 一段序列), 答案由程序算出。' },
        { from: '组内零方差', to: 'DAPO 动态采样', body: '看题任务早期常见"整组全错": 这些组优势全为 0, 不产生梯度, 见 GRPO 进阶一章。' },
        { from: 'verify', to: '训练与验收', body: '同一个判分器既当奖励也当留出集指标 —— 前提是它真的依赖 prompt。' },
      ],
      sourceRows: [
        { concept: '区域奖励 (反例)', code: 'reward = (completion >= V//2).mean()', takeaway: '只看 completion 里落在 [V/2, V) 的比例, 与 prompt 无关。本仓库早期版本用过, 已换掉。' },
        { concept: '看题奖励', code: 'llm_finetune/data/tasks.py:SeqTask', takeaway: 'verify(prompts, completions): 前 R 个 token 与 target(prompts) 完全一致 (含 EOS) 才得 1 分。' },
        { concept: '判分器', code: 'llm_finetune/data/tasks.py:verify', takeaway: '0/1 判分, 不给部分分。任务太难时"整组全错"的比例会很高, 这时要么降难度、要么增大 G、要么给部分分。' },
        { concept: '留出集怎么切', code: 'llm_finetune/data/tasks.py:sample_prompts', takeaway: '留出集 = token 和 ≡ 0 (mod 5) 的 prompt, 与训练集严格不相交。' },
        { concept: '评估', code: '按 prompt 分桶的准确率', takeaway: '只看平均 reward 会被常数策略骗过去。' },
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
      source: ['llm_finetune/data/tasks.py:SeqTask', 'llm_finetune/data/tasks.py:verify'],
      run: 'python -m llm_finetune.run_finetune.grpo.train_grpo',
    },

    // ───────────────────────── 压轴: 训练脚本与选型 ─────────────────────────
    'finetune-runs': {
      title: '训练脚本与落盘 · 11 种方法的代价结构',
      subtitle: '在"有什么数据 / 有多少显存 / 能不能在线采样"之后直接点名方法, 并说出它每步几次前向、常驻几份权重、最后落盘什么。',
      tldr: '方法之间的真正区别不在 loss 写得好不好看, 在代价结构: 要不要常驻 reference model、要不要在线采样、落盘是全量权重还是 adapter。实测 DPO 每步 2 次前向 / 常驻 778 KB / 5.4 s, SimPO 与 ORPO 1 次 / 389 KB / 4.0 s。',
      question: '手上只有 (问, 答) 且显存紧张, 该用哪个? 换成成对偏好呢? 换成"答案能被程序判对错"呢?',
      code: 'llm_finetune/run_finetune/{sft,lora,dora,qlora,rm,dpo,simpo_orpo,grpo,distill,on_policy_distill}/train_*.py',
      points: [
        { title: '两个维度可以自由组合', body: 'SFT / DPO / GRPO 决定"优化什么目标", LoRA / QLoRA / DoRA 决定"更新哪些参数"。它们互相正交, LoRA-SFT、LoRA-DPO 都很常见。methods/ 放可复用算法, run_finetune/ 放一次实验的编排。' },
        {
          title: '代价写在三处',
          key: true,
          body: '每步 LM 前向次数 (DPO 2 次, SimPO / ORPO 1 次, GRPO 系还要先采 G 条)、常驻权重份数 (DPO 要 policy + ref = 778 KB, 无 ref 的只要 389 KB)、落盘产物 (全量权重 / adapter / value head)。数据形态先砍掉一半候选, 这三项再砍一半。',
        },
        { title: '别只存 merge 后的权重', body: 'LoRA adapter 只有 r·(d_in+d_out) 个参数: 本仓库 76 KB, 整模型 389 KB。存 adapter 才能独立分发、按请求热切换; 只留 merge 后的完整权重, 就退化成每个任务一份大模型。' },
      ],
      links: [
        { from: 'methods/*.py', to: 'run_finetune/*/train_*.py', body: '算法类被训练脚本实例化并喂 batch; 脚本还负责留出集评估和最后那条断言。' },
        { from: 'PairwiseForward / TeacherStudent', to: '通用 Trainer', body: '一步要跑多次前向的方法 (DPO / SimPO / ORPO / RM / 蒸馏) 都包成一个 nn.Module, Trainer 一行不用改。' },
        { from: 'GRPO / on-policy 蒸馏', to: '不用 Trainer', body: '数据由当前策略现场采样, GRPO 还要在同一批 rollout 上更新 μ 次 —— "取 batch → 更新一次"的约定不成立, 硬塞只会更难读。' },
        { from: 'print_trainable_parameters', to: '验收 PEFT', body: 'loss 下降和显存占用都证明不了 base 被冻住; 统计 requires_grad 的参数量才是直接证据。' },
        { from: 'adapter state_dict', to: 'llm_infer Multi-LoRA', body: '多租户服务靠 adapter 能独立切换。' },
      ],
      sourceRows: [
        { concept: '共用基座', code: 'llm_finetune/run_finetune/common.py:pretrained_base', takeaway: '所有脚本从同一个 copy 任务的基座出发, 指标之间才可比。' },
        { concept: 'SFT / LoRA / DoRA / QLoRA', code: 'run_finetune/{sft,lora,dora,qlora}/train_*.py', takeaway: '数据是 (x, y), 每步 1 次前向, 不要 ref 也不要采样; 区别只在"哪些参数可训、基座怎么存"。' },
        { concept: 'RM', code: 'run_finetune/rm/train_rm.py', takeaway: '同一份偏好数据, 学的是标量分。落盘多一个 value head; 换掉 lm_head 时会原地改掉传入的 backbone。' },
        { concept: 'DPO', code: 'run_finetune/dpo/train_dpo.py', takeaway: 'policy + ref 双份权重, 每步 2 次前向 (2B 条序列)。' },
        { concept: 'SimPO / ORPO', code: 'run_finetune/simpo_orpo/train_simpo_orpo.py', takeaway: '同样的数据, 去掉 ref: 常驻权重减半、前向减半, 实测每步快约 25%。' },
        { concept: 'GRPO 系', code: 'run_finetune/grpo/train_grpo.py', takeaway: '只要 prompt, 但必须有 verifier 或 RM, 且必须能在线采样 (G 条/prompt)。β>0 时才额外常驻一份 ref。' },
        { concept: '蒸馏', code: 'run_finetune/{distill,on_policy_distill}/train_*.py', takeaway: '都要一个更强的 teacher 常驻; 离线版从语料取数, on-policy 版由 student 现场采样。' },
        { concept: '参数统计', code: 'llm_finetune/utils/param_utils.py:print_trainable_parameters', takeaway: '打印可训 / 总参数与占比, 用来验收"真的只训了 adapter"。' },
      ],
      snippetTitle: '读一个 run_finetune 脚本, 盯这 5 行',
      snippet: `1. dataset / collate_fn 产出哪些字段?
2. labels 中哪些位置是 -100?
3. 哪些参数 requires_grad=True?
4. loss 需要 policy/ref/chosen/rejected 哪些输入?
5. checkpoint 保存 full model 还是 adapter?`,
      source: [
        'llm_finetune/run_finetune/common.py:pretrained_base',
        'llm_finetune/utils/param_utils.py:print_trainable_parameters',
      ],
      run: 'python -m llm_finetune.run_all',
    },
  },
}
