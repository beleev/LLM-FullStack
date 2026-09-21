// 阶段 3 · llm_train 的全部章节页 (完整定义, 不再依赖 models.js 里的旧副本)。
// 数字全部取自对应 demo 的实际输出, 改 Python 后请重新核对。
// points 里 key: true 的那一条 = 这一章只让人记住一句话时, 记这句。
export default {
  stage: 'train',
  chapters: [
    { route: 'train-pipeline-schedules', label: '1F1B 与交错调度', hint: '气泡公式、在途激活、interleaved 1F1B' },
    { route: 'train-lr-schedule', label: 'WSD 学习率调度', hint: 'warmup / cosine / WSD, 为什么能随时加训' },
    { route: 'train-low-precision', label: 'FP8 与 FP4 微缩放', hint: 'block scaling、MXFP4 / NVFP4' },
    { route: 'train-muon', label: 'Muon 优化器', hint: 'Newton–Schulz 正交化、QK-clip' },
    { route: 'train-ulysses', label: 'Ulysses 序列并行', hint: 'all-to-all 把切序列换成切头' },
  ],
  pages: {
    // ---------------------------------------------------------------- 切 batch
    'train-batch-ddp': {
      title: 'batch 与 DDP · 同一个梯度的三种算法',
      subtitle: '读完你能说清 DDP 为什么"就是一个更大的 batch", 以及这句话在什么条件下会悄悄失效。',
      tldr: '同一个全局梯度有三种算法: 单卡一次算完、梯度累积用 K 倍串行时间换 1/K 的激活、DDP 用每步 2(P−1)/P 份梯度的通信换 1/P 的时间。三者相等的前提是每一份的样本数相同。',
      question: 'micro-batch 和 DDP 的 rank 都在切 batch, 为什么一个不用通信、另一个每步都要 all-reduce?',
      code: 'llm_train/m01_gradient_accumulation/demo.py · llm_train/m02_data_parallel/demo.py',
      points: [
        {
          title: '切法不同, 梯度相同',
          key: true,
          body: '全局 batch 的梯度就是 N 个样本各自梯度的平均。谁先算、在哪张卡上算、分几批算, 都不改这个平均值。m01 实测累积梯度与整 batch 差 2.98e-08, m02 实测 DDP 走完一步后参数与单卡差 3.73e-09 —— 都是 float32 求和顺序留下的零头, 两边都用 assert 卡着。所以"DDP 就是一个更大的 batch"不是比喻, 是等式。',
        },
        {
          title: '累积换显存, DDP 换时间',
          body: '梯度累积把激活峰值按 micro-batch 算 (m01: 8 个样本 → 2 个), 代价是 K 倍串行, 算力和通信一点没省。DDP 把每卡样本数除以 P, 代价是每步一次全模型梯度 all-reduce, 每卡发 2(P−1)/P 份 —— 8 卡 1.75 份、64 卡 1.97 份, 几乎不随卡数涨。DDP 不省显存: 参数、梯度、Adam 状态每卡仍是一整份, 这正是 ZeRO 的入口。',
        },
        {
          title: '等价挂在"按样本数加权"上',
          body: 'micro 的 loss 已经在本地取过均值, 累加时要再乘 n_k/N。等分时它退化成 1/K, 所以漏掉也不报错。m01 里"直接相加"恰好把梯度放大 K=4 倍, 等于偷偷把学习率乘 4。换成 [3,3,2] 这种不等长切法立刻露馅。跨卡同理: all-reduce(mean) 做的是各卡局部均值的简单平均, 只有每卡样本数相同时它才等于全局均值。',
        },
      ],
      links: [
        { from: 'llm_basic train.py', to: '梯度累积', body: 'update 从每个 batch 一次变成每 K 个 micro 一次, 其余一行没改。' },
        { from: 'local grads', to: 'all_reduce_mean', body: '本地梯度只是全局 batch 的一片; 漏一次同步, 各副本当场分家。' },
        { from: 'DDP', to: 'ZeRO', body: 'DDP 每卡都存完整参数、梯度、Adam 状态; 存不下时才轮到分片。' },
        { from: '2(N−1)/N', to: 'ring all-reduce', body: '这个系数怎么来的, 见「通信与 full_loop」一章。' },
      ],
      sourceRows: [
        { concept: 'micro 权重', code: 'm01_gradient_accumulation/demo.py:accumulate', takeaway: 'scale=n/len(x) 这一行就是 full-batch mean 的全部; weighted=False 是故意写错的对照组。' },
        { concept: '不等长 micro', code: 'sizes=[3, 3, 2]', takeaway: '最后一个 micro 不满时, 按样本数加权和按 1/K 平均给出不同答案。' },
        { concept: '梯度同步', code: 'core/collectives.py:all_reduce_mean', takeaway: 'all-reduce(sum) 再除以 world: 之后每张卡拿到逐位相同的梯度。' },
        { concept: '副本一致性', code: 'assert max_abs_diff(replicas[0], r) == 0', takeaway: '同步后所有副本必须逐位相同, 否则训练已经分叉, 而且不会报错。' },
        { concept: '通信账', code: 'core/collectives.py:CommCounter', takeaway: '每 rank 每步 2(N−1)/N × 全模型梯度字节, 上限 2 份, 与卡数几乎无关。' },
      ],
      snippetTitle: '累积的一步 vs DDP 的一步',
      snippet: `# 累积: 时间上串行, 每个 micro 只占 n/N 的激活
accum = zeros_like(params)
for xb, yb in micro_batches:                 # K 次前向 + 反向
    _, g = model.loss_and_grads(xb, yb)      # 这个 loss 已经是 xb 内部的均值
    add_inplace(accum, g, scale=len(xb) / N) # ★ 权重是样本数占比, 不是 1/K

# DDP: 空间上并行, 每卡只吃 N/P 个样本
local = [rep.loss_and_grads(xs, ys)[1] for rep, xs, ys in zip(replicas, x_shards, y_shards)]
synced = all_reduce_mean(local)              # 之后所有副本逐位相同
for rep in replicas:
    rep.apply_grads(synced, lr)`,
      source: ['llm_train/m01_gradient_accumulation/demo.py:accumulate', 'llm_train/core/collectives.py:all_reduce_sum'],
      run: 'python -m llm_train.m02_data_parallel.demo',
    },

    // ---------------------------------------------------------------- 切模型
    'train-model-parallel': {
      title: 'TP / PP 切模型 · 层内切矩阵, 层间切流水线',
      subtitle: '读完你能判断一个放不进单卡的模型该切矩阵还是切层, 以及各自要在哪里付通信。',
      tldr: '单层太宽就切矩阵 (TP: W1 列切、W2 行切, 前向 1 次 + 反向 1 次 all-reduce), 层数太多就切层 (PP: 用 micro-batch 把气泡填小)。',
      question: 'TP 的通信为什么在层内、PP 的只在 stage 边界? 前向已经 all-reduce 过了, 反向为什么还要再来一次?',
      code: 'llm_train/m03_tensor_parallel/demo.py · llm_train/m04_pipeline_parallel/demo.py',
      points: [
        {
          title: 'TP: 先列切, 后行切',
          key: true,
          body: 'W1 按列切, 每卡拿到完整输入、算出隐藏维的一段 —— 中间的 GeLU 是逐元素的, 各卡本地算就行, 不用通信。W2 再按行切, 每卡得到一个部分和, 最后一次 all-reduce 把它们加起来。顺序反过来 (先行切) 就得在激活函数前先同步一次。bias b2 只在 all-reduce 之后加一次, 否则会被加 N 遍。',
        },
        {
          title: '反向也要 all-reduce',
          body: '输入 X 被每张卡都用过, 所以它的梯度是各卡贡献之和。每卡自己算出的 dX 只含它那 H/N 列的部分, 不求和就直接传给上一层, 梯度是错的 —— m03 里这个差是 1.8e-2。一个 MLP 块 = 前向 1 次 + 反向 1 次, attention 块同样两次, 一层 Transformer 共 4 次 all-reduce。',
        },
        {
          title: 'PP: 气泡与在途激活',
          body: '按连续层切 stage, 通信只在 stage 边界传一份激活, 比 TP 省得多; 代价是首尾的卡在空等。气泡 = (PP−1)/(M+PP−1), GPipe 与 1F1B 完全相同; 两者的区别是 stage 0 攥着的在途激活从 M 降到 PP−s。细节见「1F1B 与交错调度」。',
        },
      ],
      links: [
        { from: 'llm_models Block', to: 'TP', body: 'TP 落在 attention 和 MLP 的几个大矩阵上, 逐元素算子一律不动。' },
        { from: 'num_layers', to: 'PP', body: '层越深, 按连续层切 stage 越自然, 边界上只传一份激活。' },
        { from: 'micro-batch 数 M', to: '气泡占比', body: 'M 越大气泡越小; M 能开多大取决于在途激活放不放得下。' },
        { from: 'PP', to: '1F1B 与交错调度', body: '三种调度的时间表和显存账在下一章摊开。' },
      ],
      sourceRows: [
        { concept: 'column parallel', code: 'W1_s = np.split(W1, world, axis=1)', takeaway: '[D,H] → N×[D,H/N], 每个 rank 计算 hidden 的一段。' },
        { concept: 'row parallel + g 算子', code: 'tp_out = all_reduce_sum(partial)[0] + b2', takeaway: '部分和求和得到 dense 输出; b2 只加一次。' },
        { concept: 'f 算子的反向', code: 'tp_dx = all_reduce_sum(dx_partial)[0]', takeaway: '前向恒等、反向 all-reduce; 省掉它 dX 就错 1.8e-2。' },
        { concept: '流水线排程', code: 'm04_pipeline_parallel/demo.py:simulate', takeaway: '每卡按固定顺序执行, 每个 op 在"卡空闲且依赖已完成"的最早时刻开始。' },
      ],
      snippetTitle: 'TP MLP 的前向与反向',
      snippet: `z_s = [x @ w + b for w, b in zip(W1_s, b1_s)]     # 列切: N × [B, H/N]
h_s = [relu(z) for z in z_s]                      # 逐元素, 无需通信
partial = [h @ w for h, w in zip(h_s, W2_s)]      # 行切: 每份只是部分和
tp_out = all_reduce_sum(partial)[0] + b2          # 前向唯一一次通信

dx_partial = [dz @ w.T for dz, w in zip(d_z_s, W1_s)]
tp_dx = all_reduce_sum(dx_partial)[0]             # 反向唯一一次通信`,
      source: ['llm_train/m03_tensor_parallel/demo.py:main'],
      run: 'python -m llm_train.m03_tensor_parallel.demo',
    },

    // ---------------------------------------------------------------- 切状态 / 省显存 / 可恢复
    'train-memory': {
      title: '状态与显存 · ZeRO、激活重算与可恢复',
      subtitle: '读完你能自己算: 一个 7B 模型开某个 ZeRO stage 之后, 每卡还剩多少显存留给激活。',
      tldr: '混合精度 Adam 每参数 16 字节 = 2 (fp16 参数) + 2 (fp16 梯度) + 12 (fp32 master + m + v); ZeRO-1/2/3 依次把 12 / 14 / 16 除以卡数。激活不在这 16 字节里, 用 +33% 的重算把它从 O(L) 压到 O(√L)。',
      question: '7B 模型在 64 张卡上开 ZeRO-1, 每卡还要多少 GB? 为什么再加卡也降不下去?',
      code: 'llm_train/m05_zero_fsdp/demo.py · llm_train/m07_activation_checkpointing/demo.py · llm_train/m08_checkpoint_resume/demo.py',
      points: [
        {
          title: '2 + 2 + 12',
          key: true,
          body: '优化器状态占了这 16 字节里的 12。DDP 是 16Ψ, ZeRO-1 切掉 12 变成 4Ψ+12Ψ/N, ZeRO-2 再切梯度变成 2Ψ+14Ψ/N, ZeRO-3/FSDP 连参数一起切成 16Ψ/N。m05 里 Ψ=192、N=4, 每 rank 常驻字节恰好是 3072 / 1344 / 1056 / 768。Adam 是逐元素更新的, 每卡只更新自己那一片, 结果与 DDP 逐位相同 (差 0.0e+00)。7B 开 ZeRO-1、64 卡: 7e9 × (4 + 12/64) ≈ 29 GB —— 那个 4Ψ 是加多少卡都除不掉的地板。',
        },
        {
          title: '通信账: ZeRO-2 不比 DDP 贵',
          body: 'all-reduce 本来就等于 reduce-scatter + all-gather, 所以 ZeRO-2 只是把这两半中间插了一步"各自更新自己的分片", 总字节和 DDP 持平 (m05 实测 1.00×)。ZeRO-3 每层前向、反向都要先 all-gather 一次参数才能算, 约 1.5×; 用完立刻 free, 瞬时只多一层的完整 fp16 参数 (demo 里 +256 B)。',
        },
        {
          title: '激活重算: 用 33% 计算换 √L 显存',
          body: '不存所有层的中间激活, 只存段边界, 反向前重跑段内前向。峰值 ≈ L/k + k, k=√L 时最小。m07 里 L=16: k=1 要 17 份, k=4 只要 8 份, 梯度逐位相同 (0.0e+00)。代价是多跑一次前向 —— 前反向本来是 3 个单位, 变成 4 个, +33% 上限; demo 里 L=16、k=4 实测 +25%。',
        },
        {
          title: '可恢复 = 四样齐全',
          body: '参数、优化器状态、数据游标、RNG。m08 把它们一样样拿掉: 漏 RNG 差 3.1e-2, 漏优化器差 5.8e-2, 漏数据游标差 1.5e-1 —— 全都不报错, 只是轨迹悄悄偏了。顺带一提, 那个 checkpoint 里 RNG 状态约 2.5KB, 比玩具模型本身还大。',
        },
      ],
      links: [
        { from: 'DDP 每卡全量', to: 'ZeRO 分片', body: '复制换成分片: 显存下降, 通信上升 (stage 3 约 1.5×)。' },
        { from: 'llm_basic 的 cache', to: '激活重算', body: '小模型把前向中间量全存; 大模型只存段边界, 其余反向前重跑。' },
        { from: 'ToyDataStream.cursor', to: 'resume', body: '数据读到哪也是训练状态, 不存就接不回同一条轨迹。' },
        { from: 'reduce-scatter', to: '通信原语', body: 'all-reduce 拆成两半看, 就明白 ZeRO-2 为什么不涨通信。' },
      ],
      sourceRows: [
        { concept: '状态怎么切', code: 'm05_zero_fsdp/demo.py:init_rank_state', takeaway: 'stage 决定哪些数组存整份、哪些只存 1/N —— 这就是 ZeRO 的全部数据结构。' },
        { concept: '梯度同步的分水岭', code: 'reduce_scatter_sum(layer_g)', takeaway: 'stage ≤ 1 用 all-reduce 留整份梯度; stage ≥ 2 只留自己那片。' },
        { concept: 'FSDP gather → compute → free', code: 'full = all_gather([rk["p16"][i] for rk in ranks])', takeaway: '用到哪层 gather 哪层, del full 立刻释放; 瞬时只多一层的完整参数。' },
        { concept: '激活账本', code: 'm07_activation_checkpointing/demo.py:run', takeaway: '重算出来的段内激活也占显存, 必须计进峰值。' },
        { concept: 'resume payload', code: 'm08_checkpoint_resume/demo.py:snapshot', takeaway: '保存 step、model、optimizer、数据游标、RNG; 缺一样轨迹就对不上。' },
      ],
      snippetTitle: 'ZeRO 一步 (stage 2/3)',
      snippet: `# 每 rank 常驻: p16 (stage 3 只存 1/N), master/m/v (stage>=1 只存 1/N)
for i in range(L):                                  # FSDP 前向: 逐层 gather → 算 → free
    full = all_gather([rk.p16[i] for rk in ranks])
    h = layer_fwd(h, full)
    del full

g_shard = reduce_scatter_sum(layer_grads) / world   # 每 rank 只留自己那片梯度
adam_update(master_shard, g_shard, m_shard, v_shard, t, lr)
p16_shard = master_shard.astype(float16)            # 下一步再按需 gather`,
      source: ['llm_train/m05_zero_fsdp/demo.py:init_rank_state', 'llm_train/m07_activation_checkpointing/demo.py:run'],
      run: 'python -m llm_train.m05_zero_fsdp.demo',
    },

    // ---------------------------------------------------------------- 数值格式
    'train-precision-stability': {
      title: '精度与稳定性 · FP16 / BF16 / FP8 与坏 step 防护',
      subtitle: '读完你能解释现代训练为什么默认 BF16、fp32 master 为什么删不得, 以及一个 Inf 冒出来时框架在做什么。',
      tldr: '低精度只负责算, 更新永远发生在 fp32 master 上。FP16 要配动态 loss scaling (溢出就跳过这一步并把 scale 减半), BF16 用精度换范围、省掉 scaling, FP8 再砍一半后 scale 变成必选项。',
      question: '为什么 FP16 训练不是把所有数组 astype(np.float16) 就完事? BF16 和 FP8 又各改了什么?',
      code: 'llm_train/m06_mixed_precision/demo.py · llm_train/m10_training_stability/demo.py · llm_train/m13_fp8_training/demo.py · llm_train/core/numerics.py',
      points: [
        {
          title: '范围 vs 精度: 16 位的两种分法',
          body: 'FP16 是 5 位指数 + 10 位尾数: 1e-8 的梯度直接变 0, 70000 的激活直接变 Inf。BF16 是 8 + 7, 指数位和 FP32 一样宽, 所以 1e-8 存成 1.001e-08、70000 存成 70144, 两头都不炸; 代价是尾数少 3 位, 1+2⁻⁹ 会被舍回 1 (FP16 能存成 1.001953125)。BF16 的精度比 FP16 低, 赢的只有范围 —— 但范围恰好是训练最怕丢的东西。',
        },
        {
          title: 'loss scaling + fp32 master',
          key: true,
          body: '反向前把 loss 放大, 更新前再除回去, 小梯度就不会在 fp16 里归零: 1e-8 直接转 fp16 是 0, 先 ×32768 再转、再除回来得到 9.997e-09。出现 Inf 就跳过这一步、scale 减半, 连续若干个好 step 再翻倍 —— m06 的 24 步里跳了 5 步。更新本身永远落在 fp32 master 上: 纯 fp16 权重每步加 1e-5 的更新, 100 步后还是 1.0, 一动没动。',
        },
        {
          title: 'FP8 的两套配方',
          body: 'Transformer Engine 走前向 E4M3、反向 E5M2、per-tensor scale; DeepSeek-V3 走全程 E4M3、细粒度 block scale。E4M3 自带约 2^15 的动态范围, 所以普通张量上两种 scale 打平 (4.54e-4 vs 4.61e-4); 要等 outlier 超过约 1e5 倍, block scale 才拉开 (0.126 vs 0.026)。真正要命的是 master: 去掉 scaling 差 29×, 去掉 fp32 master 差 584×。',
        },
      ],
      links: [
        { from: 'Adam state', to: 'fp32 master', body: '参数更新必须保留足够精度, FP16 / FP8 都一样。' },
        { from: 'global grad norm', to: 'DDP/FSDP', body: '裁剪要排在全局同步之后、unscale 之后。' },
        { from: 'LossScaler.scale', to: 'checkpoint', body: 'scale 和 good_steps 也是训练状态, 要写进 checkpoint。' },
        { from: 'FP8 block scaling', to: 'FP4 微缩放', body: '位宽再砍半后 block 必须缩到 16~32, 见「FP8 与 FP4 微缩放」。' },
      ],
      sourceRows: [
        { concept: '动态 loss scale', code: 'm06_mixed_precision/demo.py:LossScaler', takeaway: '溢出 → 减半并跳过; 连续 growth_interval (真实默认 2000) 个好 step → 翻倍。' },
        { concept: '浮点格式三要素', code: 'core/numerics.py:fake_quant_float', takeaway: '尾数位数、最大值、最小 normal 指数 —— BF16 / FP8 / FP4 共用这一个函数。' },
        { concept: 'NaN guard + clip', code: 'm10_training_stability/demo.py:guarded_step', takeaway: '坏 step 返回 False 且一个参数都不碰; 好 step 先按全局范数裁剪再更新。' },
        { concept: '裁剪不改方向', code: 'g * min(1, max_norm / ‖g‖)', takeaway: 'm10 里 ‖g‖ 111.83 → 5.00, 裁剪前后的方向余弦是 1.000000。' },
        { concept: 'FP8 消融的两个旋钮', code: 'm13_fp8_training/demo.py:make_quantizer', takeaway: 'scaling ∈ {none, tensor, block} × 是否保留 master, 每个消融臂只动一个变量。' },
        { concept: 'E4M3 max = 448', code: 'FLOAT_FORMATS["e4m3"]', takeaway: '不是 480: 最高的那个尾数编码留给了 NaN。' },
      ],
      snippetTitle: 'AMP 的一步 (含跳步)',
      snippet: `half = model.astype(float16)                     # fp16 计算副本
loss, g16 = half.loss_and_grads(x, y, loss_scale=scaler.scale)

if has_overflow(g16):                            # Inf 来自 fp16 乘法本身超过 65504
    skipped += 1                                 # 跳过: master 不能被 Inf 污染
else:
    g32 = {k: g.astype(float32) / scaler.scale for k, g in g16.items()}
    clipped, _, _ = clip_by_global_norm(g32, max_norm)
    master.apply_grads(clipped, lr)              # 更新发生在 fp32 master 上
scaler.update(overflow)                          # 溢出减半, 连续好 step 翻倍`,
      source: ['llm_train/m06_mixed_precision/demo.py:LossScaler', 'llm_train/core/numerics.py:fake_quant_float'],
      run: 'python -m llm_train.m06_mixed_precision.demo',
    },

    // ---------------------------------------------------------------- 切专家 / 切序列
    'train-moe-seq': {
      widgets: ['MoeRouteLab', 'RingAttnLab'],
      title: 'EP 与序列并行 · 切专家, 切序列',
      subtitle: '读完你能说清 MoE 的通信量为什么随数据变, 以及一条放不进单卡的序列怎么切给多卡还能算出精确的注意力。',
      tldr: 'MoE 的通信量由 gating 结果决定, 不是模型结构: token 经两次 all-to-all 出门找专家再回家。Ring Attention 的通信量固定, 但因果 mask 下必须 zigzag 切, 否则省下的计算变不成省下的时间。',
      question: 'all-to-all 的通信量为什么由数据决定而不是模型结构? Ring Attention 凭什么敢说自己和完整注意力精确相等?',
      code: 'llm_train/m11_expert_parallel/demo.py · llm_train/m12_sequence_parallel/demo.py',
      points: [
        {
          title: '路由决定通信',
          key: true,
          body: 'DDP 每步搬多少字节由模型大小定死; EP 的 all-to-all 发多少、发给谁, 全看 gating 这一次把 token 路由到了哪。m11 里 64 个 token 分 4 卡, 每卡实际收到 [11, 18, 11, 24] 个 (均匀应为 16), 最热那张卡多干 50% 的活, 其余三张等它。换成刻意倾斜的 router、每专家容量设 10, 直接有 5 个 token 装不下、走残差绕过专家。所以"均衡"在 MoE 里不是调参, 是训练目标的一部分。',
        },
        {
          title: '两种压平路由的手段',
          body: 'Switch / Mixtral 加一个 aux loss: L = E·Σ f_e·P_e, 把 f 当常数对 P 求导, 梯度就在推平路由概率 (m11: 60 步后 max/mean 从 1.62× 降到 1.12×, 丢弃 5 → 0)。DeepSeek-V3 改成给每个专家一个可调 bias, 热门的减、冷门的加, 不往 loss 里掺东西 (m11: max/mean 降到 1.00×)。目标一样, 副作用不同。',
        },
        {
          title: '同一个 online softmax',
          body: '分块算注意力再增量合并: 每收到一块就用新的最大值把旧的 (m, 分母, 加权和) 重新缩放一遍。这是恒等变形不是近似, m12 实测与完整注意力差 4.4e-16。单卡内做叫 FlashAttention, 跨卡传块就叫 Ring Attention, 任何时刻每卡只持有 1/D 的 KV。因果 mask 下连续切分会让最后一张卡每轮满载、其余人等它 (墙钟 228)。一头一尾配对的 zigzag 切法让每卡每轮工作量相同, 墙钟降到 132, 快 1.73×。',
        },
      ],
      links: [
        { from: 'gating top-k', to: 'all-to-all dispatch', body: 'token 在哪张卡 ≠ 它的专家在哪张卡, 必须重排一次再排回来。' },
        { from: 'capacity factor', to: 'token dropping', body: '64 token、8 专家、cf=1.25 → 每专家容量 10, 溢出的 token 直接走残差。' },
        { from: 'KV 环传递', to: 'online softmax (m, l, acc)', body: '收到新块就增量合并, 任何时刻只持有 1/D 的 KV。' },
        { from: '因果 mask', to: 'zigzag 切分', body: '连续切分时最后一张卡每轮满载、大家等它; 一头一尾配对后每卡每轮等量, 墙钟快 1.73×。' },
      ],
      sourceRows: [
        { concept: 'dispatch 矩阵', code: 'm11_expert_parallel/demo.py:moe_forward_ep', takeaway: '行 = 源卡, 列 = 目标卡, 这张表就是 all-to-all 的发货单。' },
        { concept: 'aux loss 的梯度', code: 'm11_expert_parallel/demo.py:train_router_aux', takeaway: 'L = E·Σ f_e·P_e, f 视为常数, 对 P 求导就在推平路由。' },
        { concept: 'aux-loss-free', code: 'm11_expert_parallel/demo.py:balance_bias', takeaway: '热门专家的 bias 往下调、冷门往上调, 不往 loss 里掺东西。' },
        { concept: '环上的一步', code: 'm12_sequence_parallel/demo.py:ring_attention', takeaway: '卡 r 在第 s 轮处理来自卡 (r−s)%D 的 KV 块, 算完把块传给右邻居。' },
        { concept: '增量合并', code: 'm_new / scale', takeaway: '用新的最大值修正旧累计量的指数基准, 与一次性 softmax 差 4.4e-16。' },
        { concept: 'zigzag 切分', code: 'm12_sequence_parallel/demo.py:shard_positions', takeaway: '序列切 2D 段, 卡 r 拿第 r 段和第 (2D−1−r) 段 —— 一头一尾配对。' },
      ],
      snippetTitle: 'EP 与 Ring Attention 的控制流',
      snippet: `# EP: 两次 all-to-all 夹一段本地专家计算
recv = all_to_all(tokens_by_dst)      # dispatch: token 去找自己的专家
out_local = expert(recv)              # 只算自己持有的那几个专家
out = all_to_all(out_local)           # combine: 结果送回原卡

# Ring: D 轮之后每张卡都见过完整序列
for step in range(D):
    src = (rank - step) % D           # 本轮处理谁的 KV 块
    m, l, acc = online_merge(Q_local @ K[src].T, V[src])
    send_to_next(K[src], V[src])      # 环传, 可与计算重叠`,
      source: ['llm_train/m12_sequence_parallel/demo.py:shard_positions'],
      run: 'python -m llm_train.m11_expert_parallel.demo',
    },

    // ---------------------------------------------------------------- 通信原语与主循环
    'train-collectives-loop': {
      title: '通信与 full_loop · 从四个原语到完整主循环',
      subtitle: '读完你拿到任何一个训练框架, 都能先把它的通信路径还原成四条原语, 再去看它的封装。',
      tldr: '所有并行策略最后都落到 all-reduce / reduce-scatter / all-gather / all-to-all 四个原语; ring 实现让每卡每步只发 2(N−1)/N·S 字节, 与卡数几乎无关。',
      question: '拿到一个陌生的训练框架, 从哪里开始读才最快看懂它的并行方式?',
      code: 'llm_train/core/collectives.py · llm_train/m09_collectives/demo.py · llm_train/full_loop/demo.py',
      points: [
        {
          title: '通信原语是成本中心',
          key: true,
          body: '并行策略的差别常常不在数学公式, 而在每一步搬什么、搬多少、什么时候搬。ring all-reduce 把张量切 N 块沿环传, 每卡只发 2(N−1)/N·S, 上限 2S; 朴素的"全发给 0 号卡再广播"要让 0 号卡收发 2(N−1)·S, N=4 时就已经是 4 倍且随 N 线性涨。四个原语各有归属: all-reduce → DDP, reduce-scatter + all-gather → ZeRO/FSDP, all-to-all → MoE 和 Ulysses。',
        },
        {
          title: 'full_loop 是合成章',
          body: '它把 rank 切分、micro 累积、AMP 放大与跳步、全局裁剪、ZeRO 分片 Adam、分片 checkpoint 放进同一个 train_step 里。40 步跳了 4 步 (3 次真实 fp16 溢出 + 1 次注入的 NaN batch), 最终 scale 稳在 2^18; 中途存盘再续训, master 权重差 0.0e+00。同一个函数传 world=1、micro=1、amp=False 就退化成单卡基线, 所以等价性断言是这份代码的自洽检验。',
        },
        {
          title: '读真实框架的入口',
          body: '先回答七个问题: batch 怎么切、层怎么切、状态怎么切、序列怎么切、数用几位存、通信怎么走、坏 step 怎么恢复。答完这七条, 框架的封装就只是命名问题了。',
        },
      ],
      links: [
        { from: 'all-reduce', to: 'DDP', body: '每卡拿到完整平均梯度; 拆开就是 reduce-scatter + all-gather。' },
        { from: 'reduce-scatter + all-gather', to: 'ZeRO / FSDP', body: '梯度和参数在"只留自己那片"与"完整视图"之间来回切换。' },
        { from: 'all-to-all', to: 'MoE / Ulysses', body: '同一个原语: out[dst][src] = in[src][dst], 发货单的转置。' },
        { from: 'full_loop', to: 'llm_finetune', body: '主循环骨架不再变, 后面换的是数据和 loss。' },
      ],
      sourceRows: [
        { concept: 'ring all-reduce', code: 'core/collectives.py:ring_all_reduce_sum', takeaway: '第 t 步 rank r 发 chunk (r−t)%N; N=8 共 14 步, 每 rank 发 2(N−1)/N·S。' },
        { concept: '通信计数器', code: 'core/collectives.py:CommCounter', takeaway: '每个原语都记下每 rank 平均发送字节, 各 demo 打印的通信账都来自它。' },
        { concept: '完整一步', code: 'full_loop/demo.py:train_step', takeaway: '组合 DDP、累积、AMP 跳步、全局裁剪、ZeRO 分片更新。' },
        { concept: '跨分片求范数', code: 'all_reduce_sum([sum(s ** 2) for s in shards])', takeaway: '梯度已经切碎了, 全局范数只能靠每 rank 贡献自己的平方和再 all-reduce。' },
        { concept: '分片 checkpoint', code: 'full_loop/demo.py:save_sharded', takeaway: '每 rank 存自己的优化器分片 + 一个全局 meta; 恢复后轨迹逐位一致。' },
      ],
      snippetTitle: 'full_loop 的一步',
      snippet: `flat_lp = all_gather([rk["master"].astype(float16) for rk in ranks])  # ① 算副本, 通信的是 fp16
for r in range(world):                                 # ② 每 rank 累积自己的 micro-batch
    for xb, yb in micro_batches(r):
        acc[r] += grads(xb, yb, loss_scale=scaler.scale) / micro

if has_overflow(local):                                # ③ 任一 rank 坏了, 所有 rank 一起跳过
    scaler.update(True); return

shards = [g / (scale * world) for g in reduce_scatter_sum(local)]   # ④ 每 rank 只留自己那片
sq = all_reduce_sum([sum(s ** 2) for s in shards])     # ⑤ 全局范数要跨分片求和
clip = min(1.0, MAX_NORM / sqrt(sq))
for rk, g in zip(ranks, shards):                       # ⑥ 分片 Adam, 只碰自己的 master/m/v
    adam_update(rk["master"], g * clip, rk["m"], rk["v"], opt_steps, lr)`,
      source: ['llm_train/core/collectives.py:ring_all_reduce_sum', 'llm_train/full_loop/demo.py:train_step'],
      run: 'python -m llm_train.full_loop.demo',
    },

    // ---------------------------------------------------------------- 新章节
    'train-pipeline-schedules': {
      title: '1F1B 与交错调度 · 气泡和在途激活的账',
      subtitle: '读完你能分清流水线的两笔成本 —— 卡在空等和已前向未反向的激活 —— 以及三种调度各自动了哪一笔。',
      tldr: 'GPipe 和 1F1B 的气泡完全一样, 都是 (PP−1)/(M+PP−1); 1F1B 省的是在途激活。要真的减气泡只有两条路: 把 M 开大, 或者交错。',
      question: '1F1B 明明不减少气泡, 为什么大家都说它"更快"?',
      code: 'llm_train/m04_pipeline_parallel/demo.py',
      points: [
        {
          title: '1F1B 省显存, 不省时间',
          key: true,
          body: 'PP=4、M=8 时两种调度的气泡都是 27.3%, 但 stage 的激活峰值一个是 [8,8,8,8], 一个是 [4,3,2,1]。1F1B 对时间的帮助是间接的: 显存省下来 → M 可以开大 → 气泡跟着掉 (PP=8、M=64 时只剩 9.9%)。规则只有一行: 卡 s 上"已前向未反向"的 micro-batch 不许超过 PP−s。',
        },
        {
          title: '交错: 气泡再除以 v',
          body: '每张卡不再拿一段连续的层, 而是拿 v 个不相邻的小 stage, micro-batch 绕着卡转 v 圈。PP=4、M=8、v=2: 27.3% → 15.8%。直觉是气泡的绝对长度正比于"单个 stage 的计算量", 交错把单个 stage 切小了 v 倍, 而总有效工作量没变, 所以公式里的 M 变成 v·M。',
        },
        {
          title: '交错的代价',
          body: 'warmup 段更长, 在途激活从 4 个 stage 涨到 [11,9,7,5] 份 1/2-stage = 5.5 个 stage; 每个 micro-batch 的跨卡传输从 2(PP−1) 次变成 2(PP·v−1) 次; 还要求 M 是 PP 的整数倍。所以它只在 M 已经加不上去、PP 又必须跨机的时候才划算。',
        },
      ],
      links: [
        { from: 'train-model-parallel', to: 'device_order', body: '上一章看时间表, 这一章看时间表是怎么由"每卡的执行顺序"生成的。' },
        { from: 'warmup 个 F', to: '在途激活上限', body: 'warmup+1 就是一张卡上"已 F 未 B"的 micro-batch 数上限。' },
        { from: '在途激活', to: '激活重算', body: '交错多出来的那部分激活, 通常用 activation checkpointing 换回来。' },
      ],
      sourceRows: [
        { concept: '每卡的执行顺序', code: 'm04_pipeline_parallel/demo.py:device_order', takeaway: 'GPipe = 全部 F 再全部 B; 1F1B = warmup 个 F 之后严格一 F 一 B。' },
        { concept: '交错的 warmup', code: '(PP - 1 - s) * 2 + (v - 1) * PP', takeaway: '比普通 1F1B 的 PP−1−s 长得多 —— 在途激活变多就来自这里。' },
        { concept: '虚拟 stage', code: 'k = c * PP + s', takeaway: '共 PP·v 个虚拟 stage, 第 k 个住在卡 k % PP 上。' },
        { concept: '事件模拟', code: 'm04_pipeline_parallel/demo.py:simulate', takeaway: 'op 的开始时刻 = max(卡空闲, 依赖完成); 气泡 = 1 − 有效工作量 / 总时长。' },
      ],
      snippetTitle: '三种调度只差一个 warmup',
      snippet: `def device_order(kind, s, PP, M, v=1):
    fwd = [F(m, chunk) ...]              # 每 PP 个 micro-batch 一组, 组内先 chunk 0 再 chunk 1
    bwd = [B(m, chunk) ...]              # chunk 倒序
    if kind == "gpipe":
        return fwd + bwd                 # 在途 = M
    warmup = PP-1-s if v == 1 else (PP-1-s)*2 + (v-1)*PP
    order = fwd[:warmup]
    while bwd 没做完:                     # 稳态: 一个 F 一个 B
        order += [next(fwd), next(bwd)]
    return order

bubble = (PP - 1) / (v * M + PP - 1)`,
      source: ['llm_train/m04_pipeline_parallel/demo.py:device_order', 'llm_train/m04_pipeline_parallel/demo.py:simulate'],
      run: 'python -m llm_train.m04_pipeline_parallel.demo',
    },

    'train-lr-schedule': {
      title: 'WSD 学习率调度 · 不用提前承诺总步数',
      subtitle: '读完你能解释为什么一条 WSD 主干可以随时分叉出成品模型, 而 cosine 训练做不到。',
      tldr: 'cosine 的每一步 lr 都写成 f(step/total), 总步数一改整条曲线都变。WSD 的稳定段里没有 total, 所以任何一个稳定段 checkpoint 都能接着训, 或分叉出一段短退火。',
      question: '训到 100% 发现 loss 还在降, 想加训: cosine 和 WSD 各要付出什么?',
      code: 'llm_train/m10_training_stability/demo.py',
      points: [
        {
          title: 'warmup 在防什么',
          body: 'Adam 的二阶矩 v 是滑动平均, 最初几百步还没"热"起来, 归一化后的步长方差很大, 配上大 lr 容易一步走飞。线性 warmup 就是用小步子把这段最危险的路走完, 没有别的玄机。',
        },
        {
          title: 'cosine 把总步数写进了每一步',
          key: true,
          body: 'lr = f((step − warmup)/(total − warmup)), total 出现在公式里, 所以它一变整条曲线都跟着变。m10 里 total 从 100 改成 200, 前 90 步的 lr 最大差 5.96e-4; 同样的改动下 WSD 差 0.00e+00。续训 cosine 只剩两条路: re-warmup (loss 先反弹), 或者按新的 total 从头再来。',
        },
        {
          title: 'WSD 的工程价值',
          body: 'MiniCPM / DeepSeek-V3 / Kimi 用的是同一类调度: 主干一直保持峰值 lr 往前训, 想要"成品"就从任意 checkpoint 分叉出一段短退火 (m10 里退火占最后 10%)。数据追加、scaling law 实验都不用重训 —— 退火起点前的那个 checkpoint 是所有分支的公共祖先。',
        },
      ],
      links: [
        { from: 'train-precision-stability', to: 'warmup', body: 'warmup、clip、NaN guard 同属"别让坏 step 毁掉整条训练"。' },
        { from: 'decay_start', to: 'checkpoint', body: '退火开始前的那个 checkpoint 最值钱: 它是所有分支的公共祖先。' },
        { from: 'WSD 退火', to: 'Muon / AdamW', body: '调度和优化器正交; Kimi K2 用的就是 Muon + WSD。' },
      ],
      sourceRows: [
        { concept: 'warmup + cosine', code: 'm10_training_stability/demo.py:warmup_cosine_lr', takeaway: 'progress = (step − warmup)/(total − warmup): total 出现在每一步的公式里。' },
        { concept: 'WSD', code: 'm10_training_stability/demo.py:wsd_lr', takeaway: '稳定段直接 return base_lr —— 这一行里没有 total_steps。' },
        { concept: '退火起点', code: 'decay_start = total_steps - int(total_steps * decay_frac)', takeaway: '只有这一个量依赖总步数, 而且只影响最后约 10%。' },
        { concept: '两种调度的实测差', code: 'total=100 → 200, 前 90 步 lr 最大变化', takeaway: 'cosine 5.96e-04, WSD 0.00e+00 —— 这两个数就是全部论据。' },
      ],
      snippetTitle: 'WSD 三段式',
      snippet: `def wsd_lr(step, total_steps, base_lr, warmup_steps, decay_frac=0.1, min_ratio=0.0):
    decay_start = total_steps - int(total_steps * decay_frac)
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    if step < decay_start:
        return base_lr                              # 稳定段: 与 total_steps 无关
    progress = (step - decay_start) / max(1, total_steps - decay_start)
    return base_lr * (1 - (1 - min_ratio) * progress)`,
      source: ['llm_train/m10_training_stability/demo.py:wsd_lr', 'llm_train/m10_training_stability/demo.py:warmup_cosine_lr'],
      run: 'python -m llm_train.m10_training_stability.demo',
    },

    'train-low-precision': {
      title: 'FP8 与 FP4 微缩放 · scale 的粒度决定能砍到几位',
      subtitle: '读完你能解释 4 bit 为什么必须配 block scale, 以及 MXFP4 和 NVFP4 的差距全部来自哪一处。',
      tldr: '位宽砍到 4 bit 后, 单个元素的动态范围只剩 12 倍, 只能靠每 16~32 个元素共享一个 scale 撑回来; scale 自己的精度决定了 MXFP4 (0.113) 和 NVFP4 (0.095) 的差距。',
      question: '同样是 4 bit + block scale, 为什么 NVFP4 的误差明显低于 MXFP4? INT4 什么时候反而更好?',
      code: 'llm_train/m13_fp8_training/demo.py · llm_train/m15_fp4_microscaling/demo.py · llm_train/core/numerics.py',
      points: [
        {
          title: 'scale 由 block 里最大的那个元素定',
          key: true,
          body: '一个 block 里的 amax 决定 scale, 其余元素按它缩放后再舍入。所以 outlier 越大, 同 block 的邻居被挤得越靠近 0 —— block 越小, 被连累的邻居越少。FP8 上这件事要等 outlier 超过约 1e5 倍才显形 (per-tensor 0.126 vs block 0.026); FP4 的网格只有 ±{0.5,1,1.5,2,3,4,6}, 动态范围 12 倍, 不切小 block 根本用不了。',
        },
        {
          title: 'scale 自己也有精度',
          body: 'MXFP4 的 E8M0 scale 只能取 2 的幂, amax/scale 落在 [4,8) 的哪个位置全凭运气, 大于 6 的直接饱和 —— m15 里 42% 的 block 最大值被截断了。NVFP4 的 E4M3 scale 带 3 位尾数, 能把 amax 几乎精确贴到 6。同为 block 16 的对照: 0.1138 vs 0.0947, 差距全部来自这一处。',
        },
        {
          title: '浮点网格不是白送的',
          body: '高斯分布的权重上 INT4 (0.096) 反而赢 MXFP4 (0.113): 数据集中时均匀网格更划算。重尾数据 (激活、梯度) 上对数间距的浮点网格才占优 (0.142 vs 0.147)。格式要对着数据分布选, 不是位数相同就等价。',
        },
      ],
      links: [
        { from: 'train-precision-stability', to: 'FP8 block scaling', body: '同一个 fake_quant_float, 只是换了一组 (尾数位, 最大值, 最小指数)。' },
        { from: 'quant_blockwise', to: 'mxfp4 / nvfp4', body: '两个函数只有 scale 那一行不一样。' },
        { from: 'FP4 权重', to: 'llm_infer 量化', body: 'FP4 目前主要用于推理权重和 QAT; 全程 FP4 预训练还要随机舍入、Hadamard 旋转等技巧。' },
      ],
      sourceRows: [
        { concept: 'block scaling', code: 'core/numerics.py:quant_blockwise', takeaway: '每 block 个元素共享 scale = amax / max_val; block ≥ 张量大小就退化成 per-tensor。' },
        { concept: 'MXFP4 的 scale', code: 'scale = 2.0 ** (np.floor(np.log2(amax)) - 2)', takeaway: '纯 2 的幂 (E8M0); −2 是因为 E2M1 的最大指数是 2 (6 = 1.5·2²)。' },
        { concept: 'NVFP4 的 scale', code: 'm15_fp4_microscaling/demo.py:nvfp4', takeaway: 'block scale 本身量化成 E4M3, 再乘一个整张量的 FP32 scale 把它搬进 E4M3 的范围。' },
        { concept: 'FP8 消融', code: 'm13_fp8_training/demo.py:train', takeaway: 'per-tensor 4.54e-4 vs block 4.61e-4 打平; 真正要命的是没有 master (差 584×)。' },
      ],
      snippetTitle: 'MXFP4 与 NVFP4 只差 scale 一行',
      snippet: `def mxfp4(x, block=32):
    b = x.reshape(-1, block)
    amax = abs(b).max(axis=1, keepdims=True)
    scale = 2.0 ** (floor(log2(amax)) - 2)            # E8M0: 只能是 2 的幂
    return quant(b / scale, "e2m1") * scale

def nvfp4(x, block=16):
    b = x.reshape(-1, block)
    amax = abs(b).max(axis=1, keepdims=True)
    tensor_scale = abs(x).max() / (448 * 6)           # 整张量 1 个 FP32
    scale = quant(amax / 6 / tensor_scale, "e4m3") * tensor_scale
    return quant(b / scale, "e2m1") * scale`,
      source: ['llm_train/m15_fp4_microscaling/demo.py:mxfp4', 'llm_train/m15_fp4_microscaling/demo.py:nvfp4', 'llm_train/core/numerics.py:quant_blockwise'],
      run: 'python -m llm_train.m15_fp4_microscaling.demo',
    },

    'train-muon': {
      title: 'Muon 优化器 · 把更新矩阵正交化',
      subtitle: '读完你能说清 Muon 赢在哪、输在哪, 以及为什么它只用在 2-D 权重上。',
      tldr: 'Adam 逐元素看梯度; Muon 把 2-D 权重的动量当成一个矩阵正交化后再更新, 让每个奇异方向迈同样大的步子。病态方向不与坐标轴对齐时 Muon 赢 AdamW 9.2×, 对齐时反而输 1.9×。',
      question: 'Adam 已经逐元素自适应了, 为什么还会被"病态方向"拖住? Muon 什么时候反而不如 Adam?',
      code: 'llm_train/m14_muon_optimizer/demo.py',
      points: [
        {
          title: '梯度谱极不均匀',
          body: '梯度矩阵的奇异值常常差几个数量级, 更新就被头几个大方向吃掉了。Newton–Schulz 迭代 X ← aX + (bA + cA²)X 只用矩阵乘, 5 步把最小奇异值从 1e-4 抬到 0.041, 最大值压在 1.16 附近 —— 效果约等于 UVᵀ, 相当于在谱范数下做最陡下降。',
        },
        {
          title: '与基无关',
          key: true,
          body: 'Adam 的 1/√v 是逐元素的, 只能修"对角"的病态; 一旦病态方向是多个坐标的线性组合, 它就无能为力。m14 把同一个病态问题旋转一下: 非轴对齐时 Muon 4.93e-04 胜过调好 lr 的 AdamW 4.52e-03 (9.2×), 轴对齐时 AdamW 5.54e-04 反过来赢 Muon 1.08e-03 (1.9×)。这不是谁更强, 是两种不同的归纳偏置。',
        },
        {
          title: '大规模要配 QK-clip',
          body: 'Muon 的更新是满秩的, attention logit 容易爆 —— m14 里冲到 752, softmax 最大概率均值 1.000, 已经是 one-hot, 梯度全没了。Kimi K2 的 MuonClip: 每步后若 max logit 超过 τ, 就把 Wq、Wk 各乘 √(τ/S_max), 压回 100。另外 embedding、输出头、norm、bias 仍然用 AdamW —— 正交化是矩阵概念, 对向量参数没有意义。',
        },
      ],
      links: [
        { from: 'llm_basic optim.py', to: 'Muon', body: 'Adam 存 m、v 两份状态; Muon 只存动量一份, 反而更省。' },
        { from: 'newton_schulz', to: 'GPU 友好', body: '精确的 UVᵀ 要做 SVD; NS 只用矩阵乘, 5 步的开销相对前反向不到 1%。' },
        { from: '0.2·√max(n,m)', to: 'AdamW 学习率', body: 'Moonlight 的缩放: 让 Muon 更新的 RMS 接近 AdamW, 学习率和 weight decay 可以直接复用。' },
      ],
      sourceRows: [
        { concept: 'Newton–Schulz', code: 'm14_muon_optimizer/demo.py:newton_schulz', takeaway: '先除以 Frobenius 范数保证 σ ≤ 1, 再对每个奇异值作用 aσ + bσ³ + cσ⁵。' },
        { concept: '一步更新', code: 'm14_muon_optimizer/demo.py:Muon', takeaway: 'Nesterov 式动量 → 正交化 → 乘 0.2·√max(n,m) → 更新。' },
        { concept: 'QK-clip', code: 'm14_muon_optimizer/demo.py:qk_clip', takeaway: 'η = min(1, τ/S_max), Wq、Wk 各乘 √η。' },
        { concept: '对照实验', code: 'm14_muon_optimizer/demo.py:run', takeaway: 'rotate=True/False 两种病态, 各自扫 5 个 lr 取最好的再比 —— 否则比的是调参。' },
      ],
      snippetTitle: 'Muon 的核心',
      snippet: `def newton_schulz(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (norm(G) + 1e-7)               # 奇异值全部 ≤ 1, 迭代才收敛
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X  # 每个 σ ← aσ + bσ³ + cσ⁵
    return X                               # ≈ U Vᵀ

buf = mu * buf + G                         # 唯一的优化器状态
O = newton_schulz(G + mu * buf)            # Nesterov 动量, 再正交化
W -= lr * 0.2 * sqrt(max(W.shape)) * O`,
      source: ['llm_train/m14_muon_optimizer/demo.py:newton_schulz', 'llm_train/m14_muon_optimizer/demo.py:Muon'],
      run: 'python -m llm_train.m14_muon_optimizer.demo',
    },

    'train-ulysses': {
      title: 'Ulysses 序列并行 · 用 all-to-all 把切序列换成切头',
      subtitle: '读完你能算出 Ulysses 和 Ring 各自的每卡通信量, 并说清为什么实战里两者要叠着用。',
      tldr: 'Ulysses 在注意力前后各做一次 all-to-all, 把"每卡一段序列、全部头"换成"每卡完整序列、一部分头", 注意力核一行都不用改。',
      question: 'Ulysses 的通信量随 P 增大反而下降, 为什么大家没有全部换成它?',
      code: 'llm_train/m16_ulysses_sequence_parallel/demo.py · llm_train/m12_sequence_parallel/demo.py',
      points: [
        {
          title: '换切法, 不换数学',
          key: true,
          body: '注意力之外每卡持有 [T/P, H, d]; 进注意力前 all-to-all 成 [T, H/P, d] —— 每卡看到完整序列但只算 H/P 个头, 直接跑普通 FlashAttention, 算完再换回来。头与头之间本来就互不相干, 所以结果和完整注意力对得上 (m16: P=2/4 差 0.0e+00, P=8 差 4.4e-16)。因果 mask 天然均衡, 不需要 zigzag。',
        },
        {
          title: '通信量随 P 下降',
          body: '每卡发 4(P−1)/P² 份张量: 4 次 all-to-all (Q、K、V、输出), 每次只发不属于自己的 (P−1)/P。m16 里 T=32、H=8、d=8, 每 rank 字节 P=2/4/8 → 16384 / 12288 / 7168; 同样条件下 Ring 是 16384 / 24576 / 28672, P=8 时整整 4 倍。',
        },
        {
          title: '两个硬约束',
          body: 'P 必须整除头数 (m16 里 P=16、H=8 直接 AssertionError; GQA/MQA 的 KV 头更少, 限制更紧), 而且 all-to-all 吃的是对分带宽, 跨机不如 Ring 的邻居 P2P 友好。所以实战常见的是机内 8 路 Ulysses × 机间 4 路 Ring。',
        },
      ],
      links: [
        { from: 'train-moe-seq', to: 'Ring Attention', body: 'Ring 让 KV 块流动、Q 不动; Ulysses 让所有人一次性换位。' },
        { from: 'core/collectives.py:all_to_all', to: 'seq_to_head', body: '和 MoE dispatch 是同一个原语: out[dst][src] = in[src][dst]。' },
        { from: '[T/P, H, d]', to: '[T, H/P, d]', body: '盯住这个 shape 变化, 整个算法就看懂了。' },
      ],
      sourceRows: [
        { concept: '序列 → 头', code: 'm16_ulysses_sequence_parallel/demo.py:seq_to_head', takeaway: 'rank s 把自己的第 j 组头发给 rank j; 收到后沿序列维拼起来。' },
        { concept: '头 → 序列', code: 'm16_ulysses_sequence_parallel/demo.py:head_to_seq', takeaway: '完全对称的逆变换。' },
        { concept: '整除约束', code: 'assert H % P == 0', takeaway: '8 个头切不成 16 份 —— demo 里故意触发这个 AssertionError。' },
        { concept: '通信账', code: '4 * (P - 1) / P * shard_bytes', takeaway: '4 次 all-to-all, 每次只发不属于自己的 (P−1)/P。' },
      ],
      snippetTitle: 'Ulysses 注意力',
      snippet: `def ulysses_attention(q, k, v, P):
    assert H % P == 0                                  # 头数必须能被 P 整除
    # 3 次 all-to-all: P × [T/P, H, d] → P × [T, H/P, d]
    qs, ks, vs = (seq_to_head(split(a, P, axis=0)) for a in (q, k, v))
    # 每卡: 完整序列 × H/P 个头, 普通因果注意力, 核不用改
    outs = [causal_mha(qs[r], ks[r], vs[r]) for r in range(P)]
    # 第 4 次 all-to-all: 换回按序列切
    return concat(head_to_seq(outs), axis=0)`,
      source: ['llm_train/m16_ulysses_sequence_parallel/demo.py:seq_to_head', 'llm_train/m16_ulysses_sequence_parallel/demo.py:ulysses_attention'],
      run: 'python -m llm_train.m16_ulysses_sequence_parallel.demo',
    },
  },
}
