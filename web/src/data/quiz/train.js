// 阶段 3 · llm_train 自测题。每章 3 题, 考取舍和"为什么", 错误选项都是常见误解。
export default {
  train: [
    {
      q: '模型放得进单卡, 只是训练太慢。第一步该上哪种并行?',
      options: ['数据并行 (DDP)', '张量并行 (TP)', '流水线并行 (PP)', 'ZeRO-3'],
      answer: 0,
      why: '模型放得下时, DDP 通信最少 (每步一次梯度 all-reduce) 且几乎线性扩展; TP/PP/ZeRO-3 都是"放不下"时才付的通信代价。',
    },
    {
      q: 'llm_train 的每个 demo 都先算一个 dense / 单机基线再 assert 一致, 这在验证什么?',
      options: ['并行版本跑得更快', 'numpy 的精度足够', '随机种子设置正确', '分布式路径与单机训练在数学上等价'],
      answer: 3,
      why: '并行只是换了"谁算哪一块、什么时候通信", 不能改变 loss → grad → update 的结果; 等价性是所有并行策略的验收标准。',
    },
    {
      q: '下面哪一对"技术 → 解决的瓶颈"是错的?',
      options: ['梯度累积 → 激活显存限制了 batch', 'ZeRO → 优化器状态占显存', '1F1B → 流水线气泡太大', '激活重算 → 反向要存的中间激活太多'],
      answer: 2,
      why: '1F1B 的气泡与 GPipe 完全相同, 它省的是在途激活显存; 减气泡要靠增大 micro-batch 数或交错调度。',
    },
  ],
  'train-batch-ddp': [
    {
      q: '把 batch=32 拆成 8 + 24 两个 micro-batch 做梯度累积, 每个 micro 的 loss 已各自取均值。怎样累加才等于整 batch 的梯度?',
      options: ['直接相加', '按样本数加权: 8/32 和 24/32', '相加后除以 2', '只用较大的那个 micro 的梯度'],
      answer: 1,
      why: 'full-batch 的 mean 是对 32 个样本平均; 每个 micro 自己的均值要再乘 micro_size/full_size 才能拼回去, 等分时才退化成"除以 N"。',
    },
    {
      q: '梯度累积和 DDP 都在"切 batch", 本质区别是什么?',
      options: ['梯度累积在时间上串行换显存, DDP 在空间上并行换吞吐', '梯度累积更快', 'DDP 的结果与单卡不等价', '梯度累积需要 all-reduce'],
      answer: 0,
      why: '两者得到的梯度在数学上相同; 累积不需要通信但不提速, DDP 提速但每步要同步一次全模型梯度。',
    },
    {
      q: 'DDP 从 8 卡扩到 64 卡, 每张卡每步要发送的梯度字节数大约怎么变?',
      options: ['变成 8 倍', '变成 1/8', '与卡数的平方成正比', '几乎不变 (2(N−1)/N·S, 上限 2S)'],
      answer: 3,
      why: 'ring all-reduce 把张量切 N 块沿环传, 每卡只发 2(N−1) 块、每块 S/N; 这正是 DDP 能扩到上千卡的原因。',
    },
  ],
  'train-model-parallel': [
    {
      q: 'PP=4、M=8 时, 1F1B 相比 GPipe 到底省了什么?',
      options: ['总时间槽更少', '气泡比例更低', 'stage 0 的激活峰值从 8 降到 4', '通信量减半'],
      answer: 2,
      why: '两种调度的总时间和气泡比例 (PP-1)/(PP-1+M) 完全相同。1F1B 只是限制每张卡在途的 micro-batch 不超过 PP-s, 所以省的是激活显存。',
    },
    {
      q: 'Megatron 张量并行里, MLP 的两个矩阵为什么是"先列切、后行切"?',
      options: ['这样参数量最小', '中间的激活函数可以在各卡本地独立算, 前向只需一次 all-reduce', '行切比列切更快', '为了让每张卡的 batch 不同'],
      answer: 1,
      why: '列切后每张卡拿到完整输入、算出一段隐藏维, GeLU 逐元素可本地做; 行切再把各段乘回来, 只在最后求和处 all-reduce 一次。',
    },
    {
      q: 'TP 的前向已经 all-reduce 过输出了, 反向还需要通信吗?',
      options: ['需要: 每卡算出的 dX 只含自己那部分列的贡献, 必须 all-reduce 求和', '不需要, 梯度各卡独立', '只有 bias 的梯度需要', '只在最后一层需要'],
      answer: 0,
      why: '输入 X 被每张卡都用过, 它的梯度是各卡贡献之和; 省掉这次 all-reduce, 传给上一层的梯度就是错的 (m03 里差 1.8e-2)。所以一层 Transformer 共 4 次 all-reduce。',
    },
  ],
  'train-memory': [
    {
      q: '混合精度 Adam 训练, 每个参数的 16 字节里最大的一块是什么?',
      options: ['fp16 参数 (2 字节)', 'fp16 梯度 (2 字节)', '激活', 'fp32 master + Adam m + v (12 字节)'],
      answer: 3,
      why: '2 + 2 + 12: 优化器状态占 3/4, 所以 ZeRO-1 只切它就能把 16Ψ 降到接近 4Ψ; 激活不在这 16 字节里, 要另算。',
    },
    {
      q: 'ZeRO-2 把梯度也切成了 1/N, 通信量比 DDP 多多少?',
      options: ['多 N 倍', '多 50%', '不多: all-reduce 本来就等于 reduce-scatter + all-gather', '少一半'],
      answer: 2,
      why: 'DDP 的 all-reduce 拆开就是 reduce-scatter + all-gather; ZeRO-2 只是在 reduce-scatter 之后先各自更新分片、再 all-gather 参数。只有 ZeRO-3 因为每层都要 gather 参数才涨到约 1.5×。',
    },
    {
      q: 'L 层网络做激活重算, 把段长 k 设成 L (只保存输入) 能把激活显存降到最低吗?',
      options: ['能, 只存 1 份', '不能: 反向时要一次性重算出整段 L 层的激活, 峰值又回到 ~L', '能, 但计算量翻 L 倍', '不能, 因为梯度会不精确'],
      answer: 1,
      why: '峰值 ≈ L/k (常驻边界) + k (当前段瞬时激活), k=√L 时最小; 重算的梯度与全存逐位相同, 总计算只多约一次前向 (+33% 以内)。',
    },
  ],
  'train-precision-stability': [
    {
      q: 'BF16 相比 FP16, 赢在哪、输在哪?',
      options: ['范围同 FP32 所以不需要 loss scaling, 但尾数少 3 位、精度更低', '范围和精度都更好', '精度更高但范围更小', '两者只是名字不同'],
      answer: 0,
      why: 'BF16 = 8 位指数 + 7 位尾数: 1e-8 不下溢、70000 不上溢, 但 1+2^-9 会被舍回 1; FP16 = 5 + 10 位, 精度高 8 倍但小于 6e-8 就归零。',
    },
    {
      q: '动态 loss scaling 检测到梯度里有 Inf 时, 正确的做法是?',
      options: ['把 Inf 替换成 0 继续更新', '把 Inf 裁剪到 65504 继续更新', '停止训练并回滚 checkpoint', '跳过这一步不更新, 并把 scale 减半'],
      answer: 3,
      why: '溢出的梯度没有任何可信信息, 任何"修补"都会污染 fp32 master; 跳过一步几乎没有代价, scale 减半后下一步就能恢复正常。',
    },
    {
      q: 'm13 的 FP8 消融里, "去掉 scaling"和"去掉 fp32 master weights"哪个伤害更大?',
      options: ['去掉 scaling (差 29×)', '两者一样', '去掉 master (差 584×)', '都没有影响'],
      answer: 2,
      why: 'scaling 决定数值能不能进 FP8 的网格, 而没有高精度 master 时小更新每一步都被舍掉, 训练根本不收敛; 两者缺一不可, 但 master 更要命。',
    },
  ],
  'train-moe-seq': [
    {
      q: 'EP 的 all-to-all 和 DDP 的 all-reduce, 通信量上最本质的区别是?',
      options: ['all-to-all 总是更小', 'all-to-all 的通信量由路由结果 (数据) 决定, 不均衡时会出现热点卡', 'all-reduce 由数据决定', '没有区别'],
      answer: 1,
      why: 'all-reduce 每步搬的字节数由模型大小固定。all-to-all 发多少、发给谁取决于 gating: 路由倾斜 = 某些卡收爆、其余闲着, 所以均衡是训练目标的一部分。',
    },
    {
      q: '因果注意力下做 Ring Attention, 序列连续切给 4 张卡。mask 省掉了近一半计算, 墙钟时间省了多少?',
      options: ['只省约 11%: 最后一张卡每轮都满载, 其他卡在等它', '也省一半', '一点没省', '省 75%'],
      answer: 0,
      why: '环上每一轮的耗时由最忙的卡决定 (228 vs 256)。zigzag 把序列切 2D 段、一头一尾配对, 每卡每轮工作量相同, 墙钟降到 132, 快 1.73×。',
    },
    {
      q: 'Ring Attention 分块算出的注意力与一次性算的相比?',
      options: ['是近似, 误差随块数增大', '只在无 mask 时等价', '只在块数为 2 时等价', '数值上精确等价: online softmax 增量合并时修正了指数基准'],
      answer: 3,
      why: '每收到一块就用新的最大值重新缩放旧的累计量 (m, 分母, 加权和), 这与 FlashAttention 是同一个恒等式, 不是近似。',
    },
  ],
  'train-collectives-loop': [
    {
      q: 'Ring all-reduce 为什么要把张量切成 N 块?',
      options: ['为了压缩数据', '为了减少步数', '让每条链路每一步都在传不同的块, 流量均摊, 每卡只发 2(N−1)/N·S', '因为 GPU 显存不够'],
      answer: 2,
      why: '不切块的话每步要传整个张量; 切成 N 块后所有链路同时满载且互不重复。代价是步数变成 2(N−1), 所以小张量要先攒成 bucket。',
    },
    {
      q: 'ZeRO-2 / FSDP 的梯度同步用的是哪个原语?',
      options: ['all-reduce', 'reduce-scatter', 'broadcast', 'all-to-all'],
      answer: 1,
      why: 'reduce-scatter = 求和后每卡只留自己那 1/N, 正好是"每卡只负责更新自己那片参数"所需要的; 它也是 ring all-reduce 的前半段。',
    },
    {
      q: 'full_loop 的 40 步里有 4 步被跳过。这说明了什么?',
      options: ['fp16 溢出和坏 batch 是常态, 训练系统必须能跳步且跳步后状态仍一致', '实现有 bug', '学习率太大', '数据有问题, 应该停止训练'],
      answer: 0,
      why: '3 次真实 fp16 溢出 + 1 次注入的 NaN batch 都被 guard 拦下, master 未被污染; resume 后轨迹逐位一致 —— 容错是主循环的一部分, 不是补丁。',
    },
  ],
  'train-pipeline-schedules': [
    {
      q: '既然 1F1B 不减少气泡, 它为什么能间接让训练更快?',
      options: ['它减少了通信', '它让反向更快', '它减少了 stage 数', '省下的激活显存可以用来开更大的 micro-batch 数 M, 气泡 (PP−1)/(M+PP−1) 随之下降'],
      answer: 3,
      why: 'GPipe 的在途激活 = M, M 一大就爆显存; 1F1B 把它钉在 PP, M 可以放心加大, PP=8、M=64 时气泡只剩 9.9%。',
    },
    {
      q: '交错 1F1B (v=2) 把气泡从 27.3% 降到 15.8%, 代价是什么?',
      options: ['没有代价', '参数量翻倍', '跨卡 P2P 次数约 v 倍, 且在途激活更多 (5.5 vs 4 个 stage)', '梯度不再精确'],
      answer: 2,
      why: '每卡拿 v 个不相邻的小 stage, micro-batch 要绕卡 v 圈; warmup 更长导致更多激活在途。所以只在 M 加不上去、PP 又必须跨机时才值得开。',
    },
    {
      q: '为什么气泡公式里是 v·M?',
      options: ['因为 batch 变大了 v 倍', '每个小 stage 只有 1/v 的计算量, 灌满/排空流水线的空等时间也缩成 1/v, 相对有效工作就小了 v 倍', '因为卡数变成了 v 倍', '因为反向快了 v 倍'],
      answer: 1,
      why: '气泡的绝对长度 ∝ (PP−1) × 单个 stage 的耗时; 交错把"单个 stage"切小了 v 倍, 而总有效工作量不变。',
    },
  ],
  'train-lr-schedule': [
    {
      q: '用 warmup + cosine 训完 100% 后想再加训 60%, 最大的麻烦是?',
      options: ['LR 已经退到接近 0, 继续训要么 re-warmup (loss 反弹) 要么按新总步数从头来', 'cosine 不支持大 batch', '优化器状态丢失', '没有任何麻烦'],
      answer: 0,
      why: 'cosine 的 lr = f(step/total), total 一变整条曲线都变 (m10: 前 90 步最大差 5.96e-4); 已经走过的退火无法"撤销"。',
    },
    {
      q: 'WSD 为什么能随时从主干分叉出一个"成品"模型?',
      options: ['因为它不需要 warmup', '因为它的 LR 一直很小', '因为它不用 Adam', '稳定段 LR 恒为峰值、与总步数无关, 任何稳定段 checkpoint 接一段短退火即可'],
      answer: 3,
      why: '只有最后 ~10% 的退火段依赖总步数; 主干继续高 LR 往前训, 分支各自退火, 加数据、做 scaling law 实验都不用重训。',
    },
    {
      q: 'warmup 主要在防什么?',
      options: ['防止过拟合', '防止显存溢出', '训练最初 Adam 的二阶矩估计很不准, 大 LR 容易一步走飞', '让 loss 曲线更好看'],
      answer: 2,
      why: 'v 的滑动平均在前几百步还没"热"起来, 归一化后的步长方差很大; 线性 warmup 用小步子走过这段最危险的路。',
    },
  ],
  'train-low-precision': [
    {
      q: 'FP8 E4M3 训练里, per-tensor scale 和 block-128 scale 在普通张量上的差距有多大?',
      options: ['block 好 10 倍以上', '基本打平 (4.54e-4 vs 4.61e-4); outlier 超过 ~1e5 倍才拉开', 'per-tensor 完全不能用', 'block 反而明显更差'],
      answer: 1,
      why: 'E4M3 自带 2^15 以上的动态范围, 没有极端 outlier 时一个 scale 就够; 细粒度 scale 是给 outlier 上的保险 (1e5 倍时误差 0.126 vs 0.026)。',
    },
    {
      q: '同为 block 16 的 FP4, E4M3 scale (NVFP4) 为什么比 E8M0 scale (MXFP4) 准?',
      options: ['E8M0 只能是 2 的幂, amax/scale 落在 [4,8) 的哪里全凭运气, 大于 6 还会饱和; E4M3 能把 amax 几乎精确贴到 6', 'E4M3 的范围更大', 'E8M0 占的位更多', '两者其实一样准'],
      answer: 0,
      why: 'E2M1 的最大值是 6。scale 对不准 amax, 要么浪费量程要么截断最大值 (42% 的 block 被饱和); 0.114 vs 0.095 的差距全部来自 scale 的精度。',
    },
    {
      q: '"FP4 的对数网格一定比 INT4 的均匀网格好" —— 对吗?',
      options: ['对, 浮点永远更好', '对, 因为 FP4 有指数位', '不对, INT4 在所有数据上都更好', '不对: 高斯分布的权重上 INT4 (0.096) 反而赢 MXFP4 (0.113); 重尾的激活/梯度上浮点才占优'],
      answer: 3,
      why: '对数网格在 0 附近密、远处疏, 适合偶发大值的重尾分布; 集中的高斯分布用均匀网格更划算。格式要对着数据分布选。',
    },
  ],
  'train-muon': [
    {
      q: 'Muon 对动量矩阵做 Newton–Schulz 正交化, 效果是什么?',
      options: ['把梯度裁剪到固定范数', '把梯度稀疏化', '把所有奇异值推到 ~1, 让每个奇异方向的步长大致相同', '让梯度逐元素变成 ±1'],
      answer: 2,
      why: 'NS(M) ≈ UVᵀ: 保留奇异向量、抹平奇异值。原本被头几个大奇异方向主导的更新, 变成各方向齐步走 (5 步把 1e-4 抬到 0.041, 放大数百倍)。',
    },
    {
      q: 'Adam 已经逐元素自适应了, 为什么在"旋转过的病态问题"上还是输给 Muon 9.2×?',
      options: ['Adam 的学习率没调好', 'Adam 的逐元素缩放只能修与坐标轴对齐 (对角) 的病态; 正交化与基无关', 'Adam 的动量太大', 'Muon 用了更多显存'],
      answer: 1,
      why: '病态方向一旦是多个坐标的线性组合, 逐元素的 1/√v 就无能为力。反过来, 病态恰好轴对齐时 Adam 赢 1.9× —— 两者是互补的归纳偏置。',
    },
    {
      q: '为什么 Muon 只用于 2-D 权重矩阵, embedding / 输出头 / norm / bias 仍用 AdamW?',
      options: ['正交化是矩阵概念, 对向量无意义; embedding 和输出头的行是独立的 token 向量, 不该被当作一个线性映射来正交化', '因为 AdamW 更省显存', '因为 Muon 太慢', '因为这些参数不需要训练'],
      answer: 0,
      why: 'Muon 的前提是"这个参数是一个线性映射, 谱范数是它合适的几何"。Muon 状态只有 1 份动量, 比 Adam 的 m+v 省一半, 并不更费显存。',
    },
  ],
  'train-ulysses': [
    {
      q: 'Ulysses 在注意力前做的 all-to-all, 把每卡的张量从什么形状变成什么形状?',
      options: ['[T, H, d] → [T/P, H, d]', '[T/P, H, d] → [T/P, H/P, d]', '[T, H/P, d] → [T, H, d]', '[T/P, H, d] → [T, H/P, d]'],
      answer: 3,
      why: '从"每卡一段序列、全部头"换成"每卡完整序列、一部分头"。头与头互不相干, 所以每卡直接跑普通 FlashAttention, 结果与完整注意力逐位相同。',
    },
    {
      q: 'P 从 2 增到 8, Ulysses 和 Ring 每卡的通信量分别怎么变?',
      options: ['都上升', '都下降', 'Ulysses 下降 (4(P−1)/P²), Ring 基本不降 (2(P−1)/P → 2)', 'Ulysses 上升, Ring 下降'],
      answer: 2,
      why: 'demo: Ulysses 16384 → 7168 字节, Ring 16384 → 28672, P=8 时差 4 倍。Ulysses 每次只发自己 1/P 里的 (P−1)/P; Ring 要把 KV 块传 P−1 轮。',
    },
    {
      q: '既然 Ulysses 通信更省, 为什么长序列训练还常常要叠一层 Ring?',
      options: ['Ring 的结果更精确', 'Ulysses 要求 P 整除头数 (GQA 下 KV 头更少), 且 all-to-all 吃对分带宽、跨机不友好', 'Ulysses 不支持因果 mask', 'Ring 不需要通信'],
      answer: 1,
      why: '8 个头最多 8 路 Ulysses; 再往上只能靠 Ring。常见组合是机内 (NVLink) 8 路 Ulysses × 机间 4 路 Ring, 各取所长。',
    },
  ],
}
