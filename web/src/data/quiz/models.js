// 阶段 2 · llm_models 自测题 (含手写总览页 models / attention / position / blocks / moe / diffusion)
export default {
  models: [
    {
      q: '阶段 2 说 “几十种主流模型 = 少数零件的不同组合”。这些零件槽位指的是?',
      options: ['优化器、学习率、batch size、精度', 'attention、FFN、norm、位置编码', 'tokenizer、数据集、loss、采样器', 'GPU、通信、并行策略、checkpoint'],
      answer: 1,
      why: 'Pre-LN Block 的骨架几乎不变; LLaMA、Mistral、DeepSeek 的差别都落在这四个槽位各选了哪个实现。',
    },
    {
      q: '为什么阶段 2 换成 PyTorch, 而不再像阶段 1 那样手写反向?',
      options: ['numpy 算不了注意力', '手写反向已经讲清了原理; 这一阶段的主角是结构, autograd 让 “换一个零件” 只需改前向', 'PyTorch 的梯度更准', '手写反向不支持多层'],
      answer: 1,
      why: '要比较十几种架构, 每种都手推反向成本太高且与主题无关; 阶段 1 已证明 autograd 做的事没有魔法。',
    },
    {
      q: '所有 decoder 语言模型都混入同一个 GenerationMixin, 它对宿主模型的要求是?',
      options: ['必须是 Transformer', '必须用 RoPE', '有 layers、max_len, 且 forward(idx, cache=None) 接受一个 cache', '必须共享 embedding 和 lm_head'],
      answer: 2,
      why: 'generate() 只管 “喂什么 token、怎么采样”; cache 里放 K/V、MLA latent 还是 Mamba 状态, 由各层自己决定, 所以 Mamba 也能复用它。',
    },
  ],
  attention: [
    {
      q: 'MHA → GQA 主要省的是什么?',
      options: ['Q 投影的参数量', '注意力分数矩阵的 O(T²) 计算', 'KV cache: 多个 Q 头共享一组 K/V, cache 按 num_kv_heads / num_heads 缩小', '位置编码的计算'],
      answer: 2,
      why: '推理瓶颈是每步都要读全部 KV cache (访存受限); GQA 几乎不掉点就把它缩小 4–8 倍, 分数矩阵的计算量并没有变。',
    },
    {
      q: 'MLA 的 KV cache 里存的是什么?',
      options: ['低秩 latent c_kv 和所有头共享的 post-RoPE k_rope', '每个头完整的 K 和 V', '只存 V, K 每步重算', '注意力概率矩阵'],
      answer: 0,
      why: 'K-nope 和 V 每步从 c_kv 现场升维; 教学配置下每 token 每层 MHA 1024 / GQA 256 / MLA 96 个数 (9.4%)。',
    },
    {
      q: 'MLA 为什么要把 Q/K 拆成 nope 和 rope 两段, 只旋转 rope 段?',
      options: ['为了减少 RoPE 的计算量', 'RoPE 只能作用在偶数维度', '为了让 Q 和 K 的维度不同', 'RoPE 与位置相关, 若作用在 latent 升维出的 K 上, 升维矩阵就无法被吸收合并, latent cache 失效'],
      answer: 3,
      why: '带位置的旋转矩阵夹在 W_UK 和 Q 之间, 无法预先合并; 所以单独留一小段带 RoPE 的共享 key 承担位置信息。',
    },
  ],
  position: [
    {
      q: 'RoPE 为什么说编码的是 “相对位置”?',
      options: ['它把相对距离加在 attention 分数上', '位置 m 的 q 和位置 n 的 k 各自旋转后, 点积只依赖 m − n', '它只对相邻 token 生效', '它的参数是按相对距离学习的'],
      answer: 1,
      why: '⟨R_m q, R_n k⟩ = ⟨q, R_{n−m} k⟩: 旋转矩阵的转置相乘只剩角度差。RoPE 没有任何可学参数。',
    },
    {
      q: 'RoPE 的不同维度对用不同频率 θ_i = base^(−2i/d), 高频和低频各管什么?',
      options: ['高频管远距离, 低频管近距离', '高频管语义, 低频管语法', '高频 (波长短) 分辨相邻 token 的先后, 低频 (波长长) 区分远距离', '两者没有分工'],
      answer: 2,
      why: '像时钟的秒针和时针: 秒针分辨得了 1 秒, 但 60 秒后就重复; 时针慢, 却能区分几个小时。长度外推出问题的正是 “时针” 那一端。',
    },
    {
      q: 'M-RoPE (Qwen2-VL) 处理纯文本 token 时会怎样?',
      options: ['三个轴的位置 id 相同, 严格退化为普通 1-D RoPE', '报错, 必须有图像输入', '只使用时间轴, 丢弃另外两段维度', '退化成 sinusoidal 绝对位置编码'],
      answer: 0,
      why: 'head_dim 的频率轴被切成 T/H/W 三段, 每段用各自的位置 id; 文本三轴 id 相同, 结果与 1-D RoPE 逐元素相等。',
    },
  ],
  blocks: [
    {
      q: 'Pre-LN (norm 放在子层之前) 相比原始 Transformer 的 Post-LN, 好处是?',
      options: ['参数更少', '推理更快', '残差主干上没有 norm, 梯度可以直通到底, 深层模型无需精细 warmup 也能稳定训练', '不再需要位置编码'],
      answer: 2,
      why: 'Post-LN 每层都对 “残差 + 子层” 整体归一化, 梯度每过一层被缩放一次; Pre-LN 的主干是纯加法。',
    },
    {
      q: 'SwiGLU 有三个线性层, 为什么参数量仍与两层 GELU FFN 大致持平?',
      options: ['其中一个线性层不含参数', '中间维度从 4d 缩到约 8d/3', '三个线性层共享权重', '它去掉了 bias 所以持平'],
      answer: 1,
      why: '3 × d × (8d/3) = 8d² = 2 × d × 4d。门控 (一路做开关、一路做内容) 在同等参数下效果更好。',
    },
    {
      q: 'RMSNorm 相比 LayerNorm 去掉了什么?',
      options: ['可学的缩放 γ', '除以标准差这一步', '对 batch 维的统计', '减均值 (以及偏置 β), 只保留按均方根缩放'],
      answer: 3,
      why: '实验表明重新居中贡献很小, 起作用的是重新缩放; 去掉后更省算、数值上也更简单。',
    },
  ],
  moe: [
    {
      q: 'MoE 的 “总参数量” 和 “激活参数量” 分别决定什么?',
      options: ['总参数决定显存占用, 激活参数决定每个 token 的计算量', '总参数决定速度, 激活参数决定显存', '两者都只影响训练', '两者相等'],
      answer: 0,
      why: '每个 token 只过 top-k 个专家, FLOPs 按激活参数算; 但所有专家都得放在显存里 —— MoE 用显存换算力。',
    },
    {
      q: 'DeepSeek MoE 的共享专家 (shared experts) 为什么存在?',
      options: ['为了减少总参数', '为了让路由器更容易训练到 one-hot', '通用知识每个 token 都需要, 放进总是激活的共享专家, 路由专家才能专心分化', '为了替代 attention'],
      answer: 2,
      why: '没有共享专家时, 每个路由专家都得重复学一份通用能力, 浪费容量; 细粒度专家 + 共享专家是 DeepSeek 路线的两个要点。',
    },
    {
      q: '不做任何负载均衡, MoE 训练会发生什么?',
      options: ['路由自动趋于均匀', '被选中多的专家学得更好 → 更容易被选中, 滚雪球直到少数专家包揽一切 (路由坍缩)', '所有专家学成一样', '只是训练变慢, 最终效果不变'],
      answer: 1,
      why: '这是一个正反馈回路; 坍缩后其余专家等于白占显存, 专家并行下热点卡还会拖慢整个 step。',
    },
  ],
  'models-mtp': [
    {
      q: '滑动窗口注意力 (SWA) 每层只看最近 W 个 token, 远处的信息怎么到达当前位置?',
      options: ['到不了, 这是 SWA 的硬限制', '靠位置编码', '靠更大的 batch', '跨层接力: L 层的理论感受野约 L·W, 但越远的信息被转手次数越多'],
      answer: 3,
      why: '第 l 层的位置 i 看到的是第 l−1 层的表示, 它们已经各自汇聚了前 W 个位置的信息。',
    },
    {
      q: 'MTP (多 token 预测) 的第 k 级为什么不破坏因果性?',
      options: ['因为它用了双向注意力', '因为它只在推理时启用', '第 k 级在位置 i 拼接的是真实 token t_{i+k} 的 embedding (teacher forcing), 预测的是更后面的 t_{i+k+1}', '因为它不计算 loss'],
      answer: 2,
      why: '每深一级多 “看” 一个真实 token、多预测一步, 因果链完整; 推理时这些额外的头可以直接当投机解码的草稿。',
    },
    {
      q: 'Gated DeltaNet 的状态是固定大小的 Dh×Dh 矩阵。相比 KV cache, 它的根本代价是?',
      options: ['有损: 最多只有 Dh 个互相正交的 key 槽位, 写入更多不同的 key 必然互相污染', '训练时无法并行', '不支持因果生成', '状态比 KV cache 还大'],
      answer: 0,
      why: 'delta rule 能精确覆写同一个 key, α 门能整体遗忘, 但容量上限摆在那; 所以 Qwen3-Next 每 4 层保留 1 层全注意力兜底精确召回。',
    },
  ],
  diffusion: [
    {
      q: 'DDPM 训练时, 网络的输入和预测目标分别是?',
      options: ['输入干净图, 预测噪声图', '输入加噪后的 x_t 和时间步 t, 预测加进去的噪声 ε', '输入纯噪声, 预测干净图', '输入 x_t, 预测 t'],
      answer: 1,
      why: '随机抽 t、按闭式公式一步加噪到 x_t, 让网络回归 ε; 采样时从纯噪声出发一步步减去预测的噪声。',
    },
    {
      q: 'Flow Matching (rectified flow) 的插值路径 x_t = (1−t)·x₀ + t·ε 有什么好处?',
      options: ['不需要训练', '网络可以更小', '路径是直线、速度目标 v = ε − x₀ 处处恒定, 少数几步欧拉积分就能走完', '不需要时间步输入'],
      answer: 2,
      why: 'DDPM 的路径是弯的, 要很多小步才能跟住; 直线路径让采样步数从上千降到几十甚至个位数, 所以 SD3 / Sora 一类模型都换了过来。',
    },
    {
      q: 'Flow Matching 的 t ∈ [0,1], 送进时间步 embedding 之前为什么要 ×1000?',
      options: ['sinusoidal embedding 的频率是按 “整数步” 设计的, t 只在 [0,1] 内变化时各频率几乎不动: cos(emb(0.1), emb(0.9)) ≈ 0.98', '为了和 DDPM 的 loss 数值对齐', '为了加快采样', '为了让插值系数变大'],
      answer: 0,
      why: '×1000 之后相似度降到约 0.17, 网络才分得清不同噪声强度; 注意只缩放 embedding 的输入, 插值公式里的 t 不动。',
    },
  ],
  'models-generation': [
    {
      q: '为什么 KV cache 只缓存 K 和 V, 不缓存 Q?',
      options: ['Q 太大存不下', '第 t 步只需要最新 token 的 Q 去查所有历史 K/V; 旧 token 的 Q 再也用不到', 'Q 每步都会变', 'Q 不经过 RoPE'],
      answer: 1,
      why: '因果 mask 下旧位置的输出不受新 token 影响, 不需要重算, 它们的 Q 自然没用; 而新 token 要和每一个旧 K 做点积。',
    },
    {
      q: 'prompt 6 个 token、生成 16 个, 无 cache 与有 cache 的 token 前向次数分别约为?',
      options: ['22 和 22', '216 和 21', '96 和 16', '256 和 16'],
      answer: 1,
      why: '无 cache 每步重算整个前缀: 6+7+…+21 = 216; 有 cache 只 prefill 6 个, 之后每步 1 个: 6+15 = 21。差距随长度平方增长。',
    },
    {
      q: '序列长度超过 max_len、窗口必须左移时, generate() 为什么丢掉整个 cache 重新 prefill?',
      options: ['因为显存不够', '因为采样温度变了', '因为 PyTorch 不支持裁剪张量', '左移后每个 token 的绝对位置都变了, 旧 K 是带着旧位置的 RoPE 角度缓存的, 不能复用'],
      answer: 3,
      why: 'K 是 RoPE 之后才缓存的。滑窗层不受此限: 它按绝对位置继续往后编号, 只是把窗口外再也看不到的 K/V 丢掉。',
    },
  ],
  'models-qknorm-yarn': [
    {
      q: '已经除以 √d 了, 为什么注意力 logit 还会爆炸?',
      options: ['√d 只抵消维度带来的方差, 不抵消 q、k 范数在训练中的增长; 范数各涨 2 倍, logit 涨 4 倍', '因为 softmax 不稳定', '因为 RoPE 放大了范数', '因为 batch 太大'],
      answer: 0,
      why: 'Python demo: 权重 ×10, 最大 logit 从 1.2 涨到 120; 开 QK-Norm 后稳定在 3.4 左右, 上界是 g²·√d_head。',
    },
    {
      q: 'QK-Norm 为什么必须放在 RoPE 之前?',
      options: ['放后面会报错', 'RoPE 是纯旋转、不改变范数, 先归一化再旋转才能同时保住 “范数受控” 和 “相对位置性质”', '为了省计算', 'RoPE 之后向量维度变了'],
      answer: 1,
      why: '若在 RoPE 之后再乘逐维增益 g, 会破坏成对维度的旋转结构, 点积不再只依赖 m − n。',
    },
    {
      q: '把上下文扩到 4 倍时, YaRN 对不同频率的处理是?',
      options: ['所有频率一律 ÷4', '只改 base, 其余不动', '训练期转够 32 圈的高频原样保留, 不足 1 圈的低频 ÷4, 中间线性过渡; 另乘 mscale', '只把最低的一个频率 ÷4'],
      answer: 2,
      why: '一律 ÷4 是 PI, 会抹掉局部分辨率; 只改 base 是 NTK-aware, 中段压不够。d_head=64、L=2048 时 YaRN 有 9 个最高频维度完全不动, mscale = 0.1·ln4+1 ≈ 1.139。',
    },
  ],
  'models-mamba': [
    {
      q: 'Mamba 里 Δ_t 很大意味着什么?',
      options: ['这个 token 被跳过', '状态被清零且不写入', '旧状态保留率 exp(Δ·A) → 0, 同时当前输入被强写入: “忘掉过去, 记住这个”', '学习率变大'],
      answer: 2,
      why: '一个标量同时控制遗忘门和写入门; Δ → 0 则相反: 保留率 → 1、写入 → 0, 当前 token 等于没出现过。',
    },
    {
      q: '线性时不变 (LTI) 的 SSM 为什么做不好 “从一堆废话里记住关键词”?',
      options: ['状态维度太小', '不支持因果', '训练太慢', '参数与输入无关, 每个 token 对最终状态的贡献只取决于离结尾多远, 无法按内容取舍'],
      answer: 3,
      why: '实验台里 LTI 模式下 “7” 的占比上限是 1/10; 让 Δ、B、C 依赖输入之后才能按内容选择。',
    },
    {
      q: 'Mamba 用 “选择性” 换来了能力, 放弃了什么?',
      options: ['可以写成全局卷积的性质 (LTI 才行), 训练必须改用并行 scan', 'O(1) 的解码状态', '因果性', '残差连接'],
      answer: 0,
      why: '参数随时间变化后系统不再是卷积; 好在递推满足结合律, 可以用并行 scan 在 O(log T) 深度内算完。解码仍是 O(1) 状态 (demo 里生成加速 16×)。',
    },
  ],
  'models-moe-balance': [
    {
      q: 'Aux-loss-free 均衡里, 路由偏置 b 参与了哪一步?',
      options: ['只参与 top-k 选谁; 门控权重仍取自不带 b 的原始分数', '只参与门控权重', '两者都参与', '只参与 loss'],
      answer: 0,
      why: '这样 b 只改变 “谁上场”, 不改变 “上场后信多少”, 均衡和语言建模目标互不干扰 —— 实验台里 “路由分数被改动量” 恒为 0。',
    },
    {
      q: '更新规则 b += γ·sign(mean − load) 只用符号、不用差值大小, 好处和风险分别是?',
      options: ['好处: 收敛更快; 风险: 无', '好处: 能精确求出最优 b; 风险: 计算贵', '好处: 不需要 γ; 风险: 偏置会无限增大', '好处: 步长恒为 γ、与 batch 大小无关, 好调; 风险: γ 太大时每步都矫枉过正, 负载来回震荡'],
      answer: 3,
      why: '过载多少都只 −γ。sigmoid 分数之间的差距也就 0.1 量级, 所以 γ 要小 (DeepSeek-V3 用 0.001)。',
    },
    {
      q: 'aux loss = E·Σ f_e·P_e 在完全均衡和完全坍缩时分别等于多少 (E=8, K=2)?',
      options: ['0 和 1', '1 和 8', '2 和 8', '8 和 2'],
      answer: 2,
      why: '均衡时 f_e = K/E、ΣP_e = 1 → K; 坍缩时趋向 E。梯度只从 P_e (路由概率) 这一支流回路由器, f_e 是计数、不可导。',
    },
  ],
  'models-dsa': [
    {
      q: '只用语言模型 loss 训练, LightningIndexer 能学到东西吗?',
      options: ['能, 梯度经过 top-k 传回去', '能, 但很慢', '不能: top-k 是离散选择、不可导, indexer 的梯度是 None, 必须靠单独的 KL 对齐 loss', '不能, 因为 indexer 没有参数'],
      answer: 2,
      why: 'KL(主注意力 ‖ softmax(indexer 分数)) 把主注意力当 teacher。demo 里 KL 从 0.114 降到 0.005, top-8 召回从 0.45 升到 0.92。',
    },
    {
      q: 'DSA 省下的是什么?',
      options: ['KV cache 的大小', '主注意力的计算量: 从 O(T²) 降到 O(T·k); cache 不但没省, 还多存了一份 indexer 的 key', '模型参数量', '训练数据量'],
      answer: 1,
      why: '每个历史 token 仍可能被未来某个 query 选中, 所以 K/V 都得留着; 省的是每个 query 只对 k 个位置做昂贵的 MLA。',
    },
    {
      q: '为什么要先 dense warmup (主注意力看全部位置、只训 indexer), 再切换到稀疏?',
      options: ['为了省显存', '为了让 top-k 变得可导', '因为稀疏注意力不能反向传播', '刚初始化的 indexer 等于随机挑 key, 直接稀疏会漏掉主注意力质量的大头、毁掉模型'],
      answer: 3,
      why: '实验台里把 “对齐程度” 拖到 0: top-8 的输出误差接近 100%; 对齐之后同样的 k 误差只有百分之几。',
    },
  ],
  'models-gptoss': [
    {
      q: 'GPT-OSS 把滑窗层和全注意力层交替排布, 相比 “全部滑窗” 和 “全部全注意力” 各赢在哪?',
      options: ['比全滑窗: 任何位置一层就能直达全部历史; 比全注意力: 一半的层 KV 封顶在 W', '比全滑窗: 参数更少; 比全注意力: 精度更高', '比全滑窗: 训练更快; 比全注意力: 不需要位置编码', '两边都不赢, 只是实现简单'],
      answer: 0,
      why: 'mini 模型 (T=40, W=8) 各层 cache 长度 [8, 40, 8, 40], 比全注意力省 40%; 裁掉的 K/V 本来就被 mask 成 −inf, 输出不变。',
    },
    {
      q: '标准 softmax 注意力的一个 head 在 “没什么值得看” 时会怎样?',
      options: ['输出零向量', '概率全为 0', '概率和仍被迫为 1, 只能输出无关 value 的平均; 实际训练出的模型会把多余概率倒在开头的 token 上', '自动跳过这一层'],
      answer: 2,
      why: 'softmax 只认 logit 的相对差。这就是 attention sink 现象, 也是滑窗把开头 token 挤出 cache 后模型崩溃的原因。',
    },
    {
      q: 'GPT-OSS 的可学 sink logit 与 StreamingLLM “保留开头 4 个 token” 的区别是?',
      options: ['没有区别', 'sink logit 需要额外的 KV cache', 'sink logit 只在推理时生效', 'sink 是每个 head 一个标量, 只进 softmax 分母、没有 value、不占 cache, 也不依赖任何特定 token 留在窗口里'],
      answer: 3,
      why: '3 个分数为 0 的 key 加一个 ln 3 的 sink: 真实 key 合计只分到 0.5, 其余一半被直接丢弃, 每行概率和 < 1。',
    },
  ],
  'models-llada': [
    {
      q: 'LLaDA 和 BERT 都是 “遮盖再预测”, 为什么 LLaDA 能用来生成?',
      options: ['LLaDA 用了因果 mask', 'LLaDA 的遮盖比例 t 在 (0,1) 上随机, 覆盖了从 “几乎全遮” 到 “只遮一点” 的整族去噪任务, 采样正是从 t=1 走到 0', 'LLaDA 的模型更大', 'LLaDA 有 KV cache'],
      answer: 1,
      why: 'BERT 固定遮 15%, 从没见过 “几乎全是 [MASK]” 的输入; 配上 1/t 加权, LLaDA 的 loss 还是负对数似然的上界。',
    },
    {
      q: '采样时 “低置信度重遮” 相比 “随机重遮” 好在哪?',
      options: ['前向次数更少', '不需要日程表', '先定稿最有把握的位置, 它们成为上下文后再定难的; 定稿永不重遮, 所以没把握的预测不该被过早固定', '可以使用 KV cache'],
      answer: 2,
      why: 'demo 的填空准确率: 低置信度重遮 1.00、随机重遮 0.91、一步定完 0.89。生成顺序由置信度决定, 而不是从左到右。',
    },
    {
      q: 'LLaDA 采样为什么用不了 KV cache?',
      options: ['因为它没有注意力层', '因为词表里有 [MASK]', '因为步数太少不值得', '注意力是双向的, 且每一步输入序列的任意位置都可能变化, 所有位置的 K/V 都要重算'],
      answer: 3,
      why: 'KV cache 成立的前提是因果 mask 下旧位置的表示不再改变。LLaDA 赚的是串行步数可以少于 token 数, 赔的是每步整段重算。',
    },
  ],
  'models-var': [
    {
      q: 'VAR 的自回归单位是什么?',
      options: ['下一个分辨率的整张 token map (1×1 → 2×2 → 4×4 …), 级内所有 token 并行生成', '下一个像素', '下一个 patch (光栅顺序)', '下一个去噪时间步'],
      answer: 0,
      why: '生成整张图只需 K 次前向 (demo: 1→2→4 共 3 次, 光栅要 16 次), 并且保留了二维邻接结构。',
    },
    {
      q: '多尺度 VQ 的第 k 级量化的是什么?',
      options: ['原图下采样到 s_k × s_k', '上一级 token 的 embedding', '目标特征减去前面所有级上采样之和的残差', '随机噪声'],
      answer: 2,
      why: '残差量化让细尺度能修正粗尺度留下的误差; 代价是 token 总数 Σs² 比单尺度多 (8×8 时 85 对 64)。',
    },
    {
      q: 'VAR Transformer 用的 “块状因果 mask” 规则是?',
      options: ['每个 token 只能看左上方的 token', '同一尺度内互相可见, 并且只能看更粗的尺度', '全部互相可见', '只能看同一尺度'],
      answer: 1,
      why: '训练时一次前向就能算出所有尺度的 loss; demo 里改动第 2 级的一个 token 只会影响第 3 级的 logits。最细一级的 token 从不作为输入。',
    },
  ],
}
