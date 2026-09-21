// 阶段 4 · llm_finetune 自测题: 每章 3 题, 考取舍与"为什么", 错误选项是常见误解。
export default {
  finetune: [
    {
      q: 'SFT、LoRA、DPO 三者的关系, 哪种说法是对的?',
      options: ['三选一, 互相替代', 'SFT/DPO 决定"优化什么目标", LoRA 决定"更新哪些参数", 两个维度可以自由组合', 'LoRA 是一种新的 loss, 专门用于小数据', 'DPO 只能用全参训练, 不能配 LoRA'],
      answer: 1,
      why: 'LoRA/QLoRA/DoRA 是参数化方式, SFT/DPO/GRPO 是目标函数; LoRA-SFT、LoRA-DPO 都很常见。',
    },
    {
      q: 'PPO 式 RLHF 要同时驻留 4 个模型, DPO 只要 2 个。被省掉的是哪两个?',
      options: ['reward model 和 critic', 'reference 和 critic', 'policy 和 reference', 'reward model 和 reference'],
      answer: 0,
      why: 'DPO 把奖励隐式写成 β·log(π/π_ref), 不需要显式 RM; 没有在线 RL 也就不需要 critic。reference 仍然保留。',
    },
    {
      q: '下面哪一项最适合用 GRPO 这类在线 RL, 而不是 SFT 或 DPO?',
      options: ['有大量人工写好的标准回答', '有成对的 chosen / rejected 偏好标注', '有一个强 teacher 模型的输出分布', '答案能被程序自动判对错 (数学、代码), 但没有标准解题过程'],
      answer: 3,
      why: '可验证奖励只告诉你"对不对", 不告诉你"怎么写"; 让模型自己采样、用结果当信号, 正是在线 RL 的用武之地。',
    },
  ],
  'finetune-sft': [
    {
      q: 'labels = x[:, 1:] 已经左移一格, prompt 占 x 的前 P 个 token。正确的 prompt mask 是?',
      options: ['labels[:, :P] = -100', 'labels[:, :P+1] = -100', 'labels[:, :P-1] = -100', 'labels[:, 1:P] = -100'],
      answer: 2,
      why: 'labels[t] = x[t+1], 第一个 response token x[P] 落在 labels 的第 P−1 格, 所以只能 mask 前 P−1 格。',
    },
    {
      q: '如果错写成 labels[:, :P] = -100, 训练时最可能观察到什么?',
      options: ['立刻报 shape 错误', 'loss 正常下降, 但模型从没学过回答的第一个 token; 单 token 答案时 loss 为 nan', 'loss 明显偏高, 一眼就能发现', '模型学会复述 prompt'],
      answer: 1,
      why: '多 mask 一格只是少了一个监督位置, 曲线看不出异常; 只有当 response 只有 1 个 token 时才会因为全部被 ignore 而得到 nan。',
    },
    {
      q: '为什么 SFT 不让 prompt token 贡献 loss?',
      options: ['我们要教的是"给定指令后怎么回答", 在 prompt 上算 loss 会把梯度花在学用户怎么提问上', 'prompt 太长会爆显存', 'prompt 的 loss 总是 0', 'PyTorch 的 cross_entropy 不支持太长的序列'],
      answer: 0,
      why: 'mask 不省显存也不省计算 (前向照样要算), 它只是把监督信号集中到 response 上。',
    },
  ],
  'finetune-lora': [
    {
      q: 'LoRA 敢用 r = 8 去替代 d = 4096 的全量更新, 依据是什么?',
      options: ['任何矩阵都能被秩 8 的矩阵精确表示', '预训练权重 W 是低秩的', '低秩矩阵的梯度更稳定', '微调带来的权重变化 ΔW 本身内在秩就很低'],
      answer: 3,
      why: '低秩的是"变化量"而不是 W 本身; 如果目标任务离预训练分布很远, ΔW 不再低秩, LoRA 就会明显落后于全参。',
    },
    {
      q: 'B 初始化为 0、A 随机初始化。如果反过来两个都随机初始化会怎样?',
      options: ['没区别, 只是惯例', '训练会更快', '第 0 步 BA ≠ 0, 模型输出立刻偏离预训练模型, 等于先破坏再修复', '无法 merge'],
      answer: 2,
      why: 'B = 0 保证起点与原模型完全一致; 同时 A 非零让 B 的梯度不为零, 训练能正常启动。两个都为 0 则梯度全为 0。',
    },
    {
      q: '同一个基座、同一份数据、同样 300 步, 本仓库实测全参留出集 EM 0.809, LoRA(r=8) 只有 0.352。这说明 LoRA 买到的是什么?',
      options: ['更快的收敛', '更高的效果上限', '显存、存储和多租户可插拔 —— 效果上限 ≤ 全参, 步数也不会更少', '更好的泛化'],
      answer: 2,
      why: '旧版 README 里"LoRA 收敛更快"来自病态初始化 (初始 loss ≈ 250) + 只背 2 条样本 + 10× 学习率。修好初始化、改用留出集重测后, 同步数下全参更快更好。',
    },
  ],
  'finetune-dpo': [
    {
      q: 'DPO 训练第 0 步 (policy 刚从 ref 复制), loss 应该是多少?',
      options: ['ln 2 ≈ 0.693', '0', '接近 1', '取决于 β'],
      answer: 0,
      why: 'policy = ref 时隐式奖励差 Δ = 0, −log σ(0) = ln 2, 与 β 无关。初始 loss 不是 0.693 通常意味着 ref/policy 没对齐或 mask 有 bug。',
    },
    {
      q: '本仓库 200 步 DPO: 留出集偏好准确率 0.965 → 0.996, log π(chosen) −4.03 → −4.18, 贪心 EM 0.332 → 0.137。怎么解释?',
      options: ['评估脚本有 bug', 'loss 只看 chosen 与 rejected 的差; 两边一起降、rejected 降得更快, loss 照样变小 —— margin 变大不等于模型变好', '步数不够, 再训一会儿就好了', 'β 设得太小'],
      answer: 1,
      why: 'likelihood displacement: 优化目标里没有任何一项要求 chosen 自己的概率上升。所以盯 margin 之外还要盯 log π(chosen); ORPO 的 NLL 项正是为此而设。',
    },
    {
      q: '把 β 调大, 效果是?',
      options: ['允许 policy 离 ref 更远', '学习率等比变大, 别无影响', '同样的 Δ 更快饱和, policy 被约束得更贴近 ref', '不再需要 ref 模型'],
      answer: 2,
      why: 'β 对应 RLHF 目标里的 KL 惩罚系数: β 大 → 小小的 Δ 就让 loss 饱和、停止推动; β 小 → 要把 Δ 拉得很大才停。',
    },
  ],
  'finetune-rlhf': [
    {
      q: '偏好标注有 ε = 20% 的概率标反。训练充分的 Bradley–Terry RM 在这类样本对上学到的分差趋近于?',
      options: ['+∞', 'ln(0.8/0.2) ≈ 1.39', '0', '0.8'],
      answer: 1,
      why: '期望 loss 在 σ(Δ) = 1−ε 处梯度为零, 即 Δ* = ln((1−ε)/ε)。无噪声时才会趋于无穷, 纯噪声 (ε = 0.5) 时为 0。',
    },
    {
      q: 'GRPO 去掉了 PPO 的 critic, baseline 从哪来?',
      options: ['同一个 prompt 采 G 条回答, 用组内平均奖励当 baseline', '从 reward model 的输出', '用上一步的奖励', '不需要 baseline'],
      answer: 0,
      why: '组内均值是对"这道题的期望得分"的蒙特卡洛估计, 正是 value 网络想预测的东西 —— 用采样换掉了一个模型。',
    },
    {
      q: '蒸馏 loss 的软标签项为什么要乘 T²?',
      options: ['让 loss 数值好看', '因为 KL 散度本身与 T² 成正比', '为了让 teacher 的梯度也放大', '软标签项对 logits 的梯度随 T 按 1/T² 衰减, 不补偿的话调高温度等于悄悄关掉蒸馏'],
      answer: 3,
      why: '梯度 = (1/T)(q^T − p^T), 而 q^T − p^T 在高温下又 ∝ 1/T; 乘 T² 后软硬两项的相对权重才只由 α 决定。',
    },
  ],
  'finetune-runs': [
    {
      q: '一个 LoRA 实验只保存了 merge 后的完整权重, 没存 adapter。丢掉了什么?',
      options: ['模型精度', '训练速度', '同一基座上热切换多个 adapter、以及几十 MB 级分发的能力', '什么也没丢'],
      answer: 2,
      why: 'adapter 只有 r·(d_in+d_out) 个参数, 可以独立分发、按请求加载; 只留完整权重就退化成每个任务一份大模型。',
    },
    {
      q: '检查 PEFT 训练脚本"真的只训了 adapter", 最直接的办法是?',
      options: ['看 loss 是否下降', '统计 requires_grad=True 的参数量占比, 并确认 optimizer 只拿到这些参数', '看显存占用', '看 checkpoint 文件大小'],
      answer: 1,
      why: 'loss 下降和显存都不能证明 base 被冻结; 参数统计 (本仓库的 print_trainable_parameters) 才是直接证据。',
    },
    {
      q: '手上只有 (问, 答), 显存只够常驻一份权重加上 adapter 的优化器状态。下面哪一组还能用?',
      options: ['DPO', 'GRPO 系', 'LoRA / QLoRA / DoRA', 'SimPO / ORPO'],
      answer: 2,
      why: 'DPO 要常驻 policy + ref 两份 (778 KB, 加全参 Adam 合计 1556 KB); GRPO 还得有 verifier 并能在线采样; SimPO / ORPO 要的是成对偏好。只有 (问, 答) 且显存紧, 剩下的就是 PEFT 三兄弟。',
    },
  ],
  'finetune-qlora': [
    {
      q: 'NF4 的 16 个码点为什么按正态分位数摆, 而不是等间距?',
      options: ['分位数码点计算更快', '等间距无法表示 0', '为了兼容 INT4 kernel', '权重近似正态, 分位数码点让每个码被用到的概率接近相等, 同样 4 bit 携带的信息更多、误差更小'],
      answer: 3,
      why: '码点密度跟着数据密度走; 等间距码本把好几个码浪费在几乎没有权重的两端。',
    },
    {
      q: '某个 block 里出现一个很大的 outlier, 会发生什么?',
      options: ['只有 outlier 自己误差变大', '整个模型的权重都受影响', '该 block 的 absmax 被撑大, 同 block 其余权重挤进中间少数几个码点, 误差一起变大', '量化直接失败'],
      answer: 2,
      why: 'scale 是按 block 取的 absmax, 所以伤害范围恰好是一个 block —— 这也是 block 不能取太大的原因。',
    },
    {
      q: 'block 大小从 64 缩到 16, 代价是什么?',
      options: ['精度下降', '每个 block 都要存一个 scale, 平均每参数的存储从 4.5 bit 涨到 6 bit', '无法反量化', '训练变慢 4 倍'],
      answer: 1,
      why: '4 + 32/B: B = 64 → 4.5 bit, B = 16 → 6 bit。QLoRA 的双重量化就是把 scale 再量化一次来压这部分开销。',
    },
  ],
  'finetune-dora': [
    {
      q: 'DoRA 的权重参数化是?',
      options: ['W = m · (W₀ + BA) / ‖W₀ + BA‖, m 为每个权重向量一个可训标量', 'W = W₀ + m·BA', 'W = (W₀ + BA) / m', 'W = m·W₀ + BA'],
      answer: 0,
      why: '低秩更新只决定方向, 归一化后乘上单独学习的幅度 m; 初始 m = ‖W₀‖、B = 0, 起点与原模型一致。',
    },
    {
      q: 'DoRA 论文指出 LoRA 与全参微调在学习行为上的关键差别是?',
      options: ['LoRA 学习率必须更小', 'LoRA 无法改变权重方向', '全参微调不改变幅度', 'LoRA 中幅度变化与方向变化强正相关 (被绑在一起), 全参微调中两者近乎独立'],
      answer: 3,
      why: '想"只转方向不改长度", 加性更新必须精确落在一个点上; 解耦后低秩容量可以专心用于方向。',
    },
    {
      q: '相对同秩 LoRA, DoRA 的额外代价是?',
      options: ['参数量翻倍', '推理时每层多一次归一化, 无法 merge', '每个被注入的层多 d 个幅度参数, 训练时多算一次按列范数; merge 后推理无差别', '需要 reference 模型'],
      answer: 2,
      why: '幅度向量只有 d 个元素, 相对 2dr 可以忽略; m·V/‖V‖ 算出来就是一个普通矩阵, 可以写回 W。',
    },
  ],
  'finetune-simpo-orpo': [
    {
      q: '去掉 reference、又不做长度归一化, 直接用 Σlog π(chosen) − Σlog π(rejected) 当 margin, 会出什么问题?',
      options: ['梯度爆炸', '序列越长 Σlog p 越负: rejected 更长就白得 margin, 模型学到的是"短的好"而不是偏好', '无法反向传播', '没问题, 这就是 SimPO'],
      answer: 1,
      why: 'DPO 里 ref 对同一条长序列也给出同样低的 Σlog p, 相减后抵消; 拿掉 ref 必须另想办法消除长度红利。',
    },
    {
      q: 'SimPO 里的 γ 起什么作用?',
      options: ['目标间隔: chosen 的平均 log p 必须领先 rejected 至少 γ/β, loss 才开始饱和', '学习率', 'KL 惩罚系数', '温度'],
      answer: 0,
      why: '没有 ref 提供"从哪出发"的参照, γ 明确规定要拉开多大差距才算够, 类似 SVM 的 margin。',
    },
    {
      q: 'ORPO 为什么可以直接从 base 模型起训, 不需要先做 SFT?',
      options: ['它不需要 chosen 数据', '它用了更大的学习率', '它内置了 reward model', '它的 loss 本身就包含 chosen 的 NLL (SFT 项), odds ratio 项只是附加的惩罚'],
      answer: 3,
      why: 'loss = NLL(chosen) + λ·(−log σ(log odds ratio)): 前一项就是 SFT, 同时充当防漂移的锚; 后一项压低 rejected。',
    },
  ],
  'finetune-grpo-variants': [
    {
      q: 'clip 的上界 1+ε 实际上约束的是哪类 token?',
      options: ['所有 token 一视同仁', '高概率 token, 因为它们涨得快', '低概率 token: ρ ≤ 1/π_old, 高概率 token 根本碰不到上界, 被卡住的只有想翻身的探索 token', '只约束 Â < 0 的 token'],
      answer: 2,
      why: 'π_old = 0.9 时 ρ 最大 1.11 < 1.2; π_old = 0.01 的 token 每轮最多涨到 0.012。DAPO 的 clip-higher 因此只放宽上界。另外: 一批样本只更新一次时 ρ ≡ 1 (实测第 1 个 epoch |ρ−1| = 0.000), clip 根本没上场。',
    },
    {
      q: 'GRPO 的 loss 先对每条回答内部按长度平均 (1/|o_i|)。它带来的偏置是?',
      options: ['短回答被过度惩罚, 模型越写越短', '又长又错的回答每个 token 挨的罚更轻, 模型倾向于"错的时候写长点"', '没有偏置', '长回答的奖励被放大'],
      answer: 1,
      why: '同样的负优势被摊到更多 token 上: 实测 3-token 回复的单 token 权重是 9-token 的 3.0 倍。Dr.GRPO 用常数分母, DAPO 用 token 级平均, 两者都把这个比值压回 1.0。',
    },
    {
      q: '一个 prompt 的 G 条回答全部正确。对这一组, 哪种处理是 DAPO 的做法?',
      options: ['优势全为 0、没有梯度: 丢弃该组继续采样, 直到 batch 里每组都有对有错', '给全组正优势', '把 σ 加一个大 ε 后照常训练', '只保留最长的一条'],
      answer: 0,
      why: '零方差的组不提供任何学习信号, 却占着 batch 名额。实测这类 prompt 占 32%: 开了动态采样后, 进 loss 的 A=0 样本从 32.1% 降到 0.0%。本仓库只过滤不补采, batch 会变小; DAPO 原版会继续采到凑满。',
    },
  ],
  'finetune-onpolicy-distill': [
    {
      q: '单峰的学生去拟合双峰的老师。最小化 forward KL(p_teacher ‖ q_student) 会得到?',
      options: ['学生缩进其中一个峰', '学生与老师完全一致', '学生退化成均匀分布', '学生摊开盖住两个峰, 并在两峰之间老师认为不可能的区域放上大量概率'],
      answer: 3,
      why: 'forward KL 在老师的样本上求期望: 老师有质量而学生没有的地方惩罚趋于无穷, 所以必须全覆盖 (mode-covering)。',
    },
    {
      q: 'on-policy 蒸馏相对离线蒸馏 (在老师写的文本上训练), 解决的核心问题是?',
      options: ['老师模型太大', '词表不一致', '训练与推理的分布错配: 离线只见过老师的前缀, 推理时学生走进自己的前缀, 错误会累积', 'KL 计算太慢'],
      answer: 2,
      why: '学生自己采样、老师在学生到达的状态上给分布, 训练分布 = 推理分布, 这正是 on-policy 的含义。',
    },
    {
      q: '相对 GRPO 这类 RL, on-policy 蒸馏的监督信号有什么不同?',
      options: ['更稀疏', '每个 token 都有老师的完整分布作为信号 (稠密), 而 RL 每条序列只有一个标量奖励', '需要人工标注', '只能用于分类任务'],
      answer: 1,
      why: '同样是 on-policy 采样, 蒸馏把"整条序列一个分"换成"每个位置一个分布", 样本效率高得多, 前提是有一个好老师。实测样本合格率: 离线 0.059 → on-policy 0.402; 代价是老师的少数派答法被压到 log π = −33。',
    },
  ],
  'finetune-rlvr': [
    {
      q: '奖励函数是"completion 中落在词表后半区的 token 比例"。训练后 reward 达到 1.0, 能得出什么结论?',
      options: ['RL 管线能跑通; 但最优策略是无视 prompt 的常数输出, 无法说明学会了条件生成', '模型学会了理解 prompt', '模型泛化能力很强', '奖励函数设计得很好'],
      answer: 0,
      why: '奖励不依赖 prompt 时, 常数策略就能拿满分, RL 会沿最短路径塌缩到它。',
    },
    {
      q: '"常数基线"指的是什么, 为什么训练前要先算它?',
      options: ['学习率恒定时的 loss', 'ref 模型的奖励', '奖励的最小值', '让所有 prompt 输出同一个 completion 能拿到的最高平均奖励; 它是"完全没读题"的得分参考线'],
      answer: 3,
      why: '常数基线 = 100% 的任务验证不了条件行为; 看题任务里它通常只有 1/类别数, 超过它才说明模型在读 prompt。',
    },
    {
      q: '换成看题奖励后, 训练早期经常出现"整组 G 条全错"。这时 GRPO 的梯度是?',
      options: ['很大, 因为全错', '为负', '为零: 组内奖励相同 → 优势全为 0', '不受影响, 照常更新'],
      answer: 2,
      why: '组内相对优势只看差异; 全错与全对一样没有信号。对策是降低任务难度 / 增大 G / 动态采样 / 给部分分。',
    },
  ],
}
