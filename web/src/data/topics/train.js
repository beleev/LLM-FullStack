// 阶段 3 · llm_train 的扩展章节 + 对已有章节的字段覆盖 (补 source / 修正过期文字)。
// 数字全部取自对应 demo 的实际输出, 改 Python 后请重新核对。
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
    // ---------------------------------------------------------------- 已有章节: 字段覆盖
    'train-batch-ddp': {
      sourceRows: [
        { concept: 'micro 权重', code: 'm01_gradient_accumulation/demo.py:accumulate', takeaway: 'micro loss 已在本地取均值, 累加时要再乘 micro_size / full_size, 才等于 full-batch mean。' },
        { concept: '梯度同步', code: 'core/collectives.py:all_reduce_mean', takeaway: 'all-reduce(sum) 再除以 world: 每个 rank 得到逐位相同的梯度。' },
        { concept: '通信成本', code: 'core/collectives.py:CommCounter', takeaway: '每 rank 每步发 2(N−1)/N × 全模型梯度字节, 与卡数几乎无关。' },
      ],
      source: ['llm_train/m01_gradient_accumulation/demo.py:accumulate', 'llm_train/core/collectives.py:all_reduce_sum'],
    },
    'train-model-parallel': {
      tldr: 'Tensor Parallel 切单层矩阵: 先列切后行切, 前向 1 次 + 反向 1 次 all-reduce (一层 Transformer 共 4 次)。Pipeline Parallel 切层, 用 micro-batch 填气泡; 1F1B 省的是显存不是时间。',
      points: [
        { title: 'TP: 列切 → 行切', body: 'W1 按列切, 每卡得到隐藏层的一段, 激活函数逐元素可本地算; W2 按行切得到部分和, 最后一次 all-reduce。bias b2 只在 all-reduce 之后加一次。' },
        { title: '反向也要 all-reduce', body: '每卡算出的 dX 只含自己那 H/N 列的贡献, 不求和上一层拿到的梯度就是错的 (demo 里差 1.8e-2)。MLP + attention 各 2 次, 一层 4 次。' },
        { title: 'PP: 气泡与在途激活', body: '气泡 = (PP−1)/(M+PP−1), GPipe 与 1F1B 相同; 区别是 stage 0 的在途激活从 M 降到 PP。细节见「1F1B 与交错调度」一章。' },
      ],
      sourceRows: [
        { concept: 'column parallel', code: 'W1_s = np.split(W1, world, axis=1)', takeaway: '[D,H] → N×[D,H/N], 每个 rank 计算 hidden 的一段。' },
        { concept: 'row parallel + g 算子', code: 'tp_out = all_reduce_sum(partial)[0] + b2', takeaway: '部分和求和得到 dense 输出; b2 只加一次。' },
        { concept: 'f 算子的反向', code: 'tp_dx = all_reduce_sum(dx_partial)[0]', takeaway: '前向恒等、反向 all-reduce; 省掉它 dX 就错。' },
        { concept: '流水线排程', code: 'm04_pipeline_parallel/demo.py:simulate', takeaway: '每卡按固定顺序执行, op 在「卡空闲且依赖完成」的最早时刻开始。' },
      ],
      snippet: `z_s = [x @ w + b for w, b in zip(W1_s, b1_s)]     # 列切: N × [B, H/N]
h_s = [relu(z) for z in z_s]                      # 逐元素, 无需通信
partial = [h @ w for h, w in zip(h_s, W2_s)]      # 行切: 每份只是部分和
tp_out = all_reduce_sum(partial)[0] + b2          # 前向唯一一次通信

dx_partial = [dz @ w.T for dz, w in zip(d_z_s, W1_s)]
tp_dx = all_reduce_sum(dx_partial)[0]             # 反向唯一一次通信`,
      source: ['llm_train/m03_tensor_parallel/demo.py:main'],
    },
    'train-memory': {
      tldr: '混合精度 Adam 每参数 16 字节 = 2 (fp16 参数) + 2 (fp16 梯度) + 12 (fp32 master + m + v)。ZeRO-1/2/3 依次把 12 / 2 / 2 除以 N; 激活重算用约 +33% 计算把激活从 O(L) 压到 O(√L); resume 要恢复模型、优化器、数据游标和随机状态。',
      question: '7B 模型在 64 张卡上开 ZeRO-1, 每卡还要多少 GB? 为什么再加卡也降不下去?',
      points: [
        { title: '2 + 2 + 12', body: 'DDP 16Ψ → ZeRO-1 4Ψ+12Ψ/N → ZeRO-2 2Ψ+14Ψ/N → ZeRO-3/FSDP 16Ψ/N。Adam 逐元素更新, 每卡只更新自己那一片, 结果与 DDP 逐位相同。' },
        { title: '通信账', body: 'ZeRO-2 不比 DDP 贵: all-reduce 本来就 = reduce-scatter + all-gather。ZeRO-3 每层前向、反向各多一次参数 all-gather, 约 1.5×; 用完立刻 free。' },
        { title: '激活重算', body: '只存段边界, 反向时重跑段内前向。峰值 ≈ L/k + k, k=√L 最小 (L=64 → 16 份); 代价是总计算 3 → 4 个单位 (+33%), 梯度逐位不变。' },
      ],
      sourceRows: [
        { concept: '状态怎么切', code: 'm05_zero_fsdp/demo.py:init_rank_state', takeaway: 'stage 决定哪些数组存整份、哪些只存 1/N —— 这就是 ZeRO 的全部数据结构。' },
        { concept: '梯度同步的分水岭', code: 'reduce_scatter_sum(layer_g)', takeaway: 'stage ≤ 1 用 all-reduce 留整份梯度; stage ≥ 2 只留自己那片。' },
        { concept: 'FSDP gather → compute → free', code: 'full = all_gather([rk["p16"][i] for rk in ranks])', takeaway: '用到哪层 gather 哪层, del full 立刻释放; 瞬时只多一层的完整参数。' },
        { concept: '激活账本', code: 'm07_activation_checkpointing/demo.py:run', takeaway: '重算出来的段内激活也占显存, 必须计入 peak。' },
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
    },
    'train-precision-stability': {
      title: '精度与稳定性 · FP16 / BF16 / FP8 与坏 step 防护',
      subtitle: '低精度省显存、省带宽、提吞吐, 但范围和精度都变窄; loss scaling、master weights 和 guard 负责兜底。',
      tldr: '真实 AMP = fp16 计算副本 + fp32 master + 动态 loss scaling (溢出就跳过这一步并把 scale 减半)。BF16 范围同 FP32 所以不需要 loss scale, 但精度比 FP16 低。FP8 再砍一半: 必须配 scale, 而 fp32 master 比 scale 更要命 (消融: scaling 值 29×, master 值 584×)。',
      question: '为什么 FP16 训练不是简单把所有数组 astype(np.float16)? BF16 和 FP8 又各自改变了什么?',
      code: 'llm_train/m06_mixed_precision/demo.py · llm_train/m10_training_stability/demo.py · llm_train/m13_fp8_training/demo.py · llm_train/core/numerics.py',
      points: [
        { title: '范围 vs 精度', body: 'FP16 最小只到 6e-8, 小梯度直接变 0; BF16 指数位与 FP32 相同, 不下溢也不上溢, 但尾数少 3 位, 1+2^-9 会被舍回 1。' },
        { title: 'loss scaling + master', body: '反向前放大 loss, 更新前除回去; 出现 Inf 就跳过这一步、scale 减半, 连续 N 个好 step 再翻倍。更新永远发生在 fp32 master 上 —— 纯 fp16 权重加 1e-5 的更新, 100 步都纹丝不动。' },
        { title: 'FP8 的两套配方', body: 'Transformer Engine: 前向 E4M3、反向 E5M2、per-tensor scale; DeepSeek-V3: 全程 E4M3、细粒度 block scale。E4M3 自带 ~2^15 动态范围, 普通张量上两种 scale 打平, outlier 超过 ~1e5 倍时 block scale 才拉开差距。' },
      ],
      links: [
        { from: 'Adam state', to: 'fp32 master', body: '参数更新必须保留足够精度, FP16 / FP8 都一样。' },
        { from: 'global grad norm', to: 'DDP/FSDP', body: '裁剪应发生在全局同步梯度之后、unscale 之后。' },
        { from: 'LossScaler.scale', to: 'checkpoint', body: 'scale 和 good_steps 也是训练状态, 要进 checkpoint。' },
        { from: 'FP8 block scaling', to: 'FP4 微缩放', body: '位宽再砍半后 block 必须缩到 16~32, 见「FP8 与 FP4 微缩放」一章。' },
      ],
      sourceRows: [
        { concept: '动态 loss scale', code: 'm06_mixed_precision/demo.py:LossScaler', takeaway: '溢出 → 减半并跳过; 连续 growth_interval (真实默认 2000) 个好 step → 翻倍。' },
        { concept: '浮点格式三要素', code: 'core/numerics.py:fake_quant_float', takeaway: '尾数位数、最大值、最小 normal 指数 —— BF16 / FP8 / FP4 共用这一个函数。' },
        { concept: 'NaN guard + clip', code: 'm10_training_stability/demo.py:guarded_step', takeaway: '坏 step 返回 False 且不碰参数; 好 step 先按全局范数裁剪再更新。' },
        { concept: 'FP8 消融的两个旋钮', code: 'm13_fp8_training/demo.py:make_quantizer', takeaway: 'scaling ∈ {none, tensor, block} × 是否保留 master, 每个消融臂只动一个变量。' },
        { concept: 'E4M3 max = 448', code: 'FLOAT_FORMATS["e4m3"]', takeaway: '不是 480: 最高的尾数编码留给了 NaN。' },
      ],
      snippetTitle: 'AMP 一步 (含跳步)',
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
    },
    'train-moe-seq': {
      links: [
        { from: 'gating top-k', to: 'all-to-all dispatch', body: 'token 在哪张卡 ≠ 它的专家在哪张卡, 必须重排。' },
        { from: 'capacity factor', to: 'token dropping', body: '64 token、8 专家、cf=1.25 → 每专家容量 10, 溢出 token 走残差。' },
        { from: 'KV 环传递', to: 'online softmax (m, l, acc)', body: '收到新块就增量合并, 任何时刻只持有 1/D 的 KV。' },
        { from: '因果 mask', to: 'zigzag 切分', body: '连续切分时最后一张卡每轮满载、大家等它; 一头一尾配对后每卡每轮工作量相同, 墙钟快 1.73×。' },
      ],
      source: ['llm_train/m12_sequence_parallel/demo.py:shard_positions'],
    },
    'train-collectives-loop': {
      points: [
        { title: '通信原语是成本中心', body: '并行策略的主要差别常常不是数学公式, 而是每一步要搬什么、搬多少、何时搬。ring all-reduce 每卡发 2(N−1)/N·S, 与卡数几乎无关。' },
        { title: 'full_loop 是合成章', body: '它把 rank split、micro accumulation、AMP scale、clip、ZeRO update、分片 checkpoint 放进同一步; 40 步里跳过 4 步 (3 次真实 fp16 溢出 + 1 次注入的 NaN batch), resume 后逐位一致。' },
        { title: '读真实框架的入口', body: '先找 batch 怎么切、状态怎么切、通信怎么走, 再看框架封装。' },
      ],
      sourceRows: [
        { concept: 'ring all-reduce', code: 'core/collectives.py:ring_all_reduce_sum', takeaway: '第 t 步 rank r 发 chunk (r−t)%N; N=8 共 14 步, 每 rank 发 2(N−1)/N·S。' },
        { concept: '通信计数器', code: 'core/collectives.py:CommCounter', takeaway: '每个原语都记下每 rank 平均发送字节, demo 里的通信账都来自它。' },
        { concept: '完整一步', code: 'full_loop/demo.py:train_step', takeaway: '组合 DDP、累积、AMP 跳步、clip、ZeRO 分片更新。' },
        { concept: '分片 checkpoint', code: 'full_loop/demo.py:save_sharded', takeaway: '每 rank 存自己的优化器分片; 恢复后轨迹逐位一致。' },
      ],
      source: ['llm_train/core/collectives.py:ring_all_reduce_sum', 'llm_train/full_loop/demo.py:train_step'],
    },

    // ---------------------------------------------------------------- 新章节
    'train-pipeline-schedules': {
      title: '1F1B 与交错调度 · 气泡和在途激活的账',
      subtitle: '流水线并行的两个成本: 卡在空等 (气泡), 和已前向未反向的 micro-batch 攥着的激活。三种调度各动了哪一个?',
      tldr: 'GPipe 与 1F1B 气泡相同, 都是 (PP−1)/(M+PP−1); 1F1B 只是把 stage s 的在途激活从 M 压到 PP−s。交错 1F1B 让每卡拿 v 个小 stage, 气泡变成 (PP−1)/(v·M+PP−1), 代价是 v 倍 P2P 和更多在途激活。',
      question: '1F1B 明明不减少气泡, 为什么大家都说它「更快」?',
      code: 'llm_train/m04_pipeline_parallel/demo.py',
      points: [
        { title: '1F1B 省显存, 不省时间', body: 'PP=4、M=8: 气泡都是 27.3%, 峰值 GPipe [8,8,8,8] vs 1F1B [4,3,2,1]。它对时间的帮助是间接的: 显存省下来 → M 可以开大 → 气泡变小 (PP=8、M=64 → 9.9%)。' },
        { title: '交错: 气泡再除以 v', body: '每卡持有 v 个不相邻的小 stage, micro-batch 绕卡 v 圈。PP=4、M=8、v=2: 27.3% → 15.8%。直觉: 灌满和排空流水线的那段空等, 长度与单个 stage 的计算量成正比。' },
        { title: '交错的代价', body: 'warmup 更长, 在途激活 [11,9,7,5] 份 1/2-stage = 5.5 个 stage (1F1B 为 4); 每个 micro-batch 的跨卡传输从 2(PP−1) 次变成 2(PP·v−1) 次; 还要求 M 是 PP 的整数倍。' },
      ],
      links: [
        { from: 'train-model-parallel', to: 'device_order', body: '上一章看时间表, 这一章看时间表是怎么由「每卡的执行顺序」生成的。' },
        { from: 'warmup 个 F', to: '在途激活上限', body: 'warmup+1 就是一张卡上「已 F 未 B」的 micro-batch 数上限。' },
        { from: '在途激活', to: '激活重算', body: '交错调度多出来的激活, 通常用 activation checkpointing 换回来。' },
      ],
      sourceRows: [
        { concept: '每卡的执行顺序', code: 'm04_pipeline_parallel/demo.py:device_order', takeaway: 'GPipe = 全部 F 再全部 B; 1F1B = warmup 个 F 之后严格一 F 一 B。' },
        { concept: '交错的 warmup', code: '(PP - 1 - s) * 2 + (v - 1) * PP', takeaway: '比普通 1F1B 的 PP−1−s 长得多 —— 这就是在途激活变多的来源。' },
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
      subtitle: 'warmup + cosine 是几年来的默认, 但它的每一步 LR 都依赖总步数。WSD 把「稳定段」和「退火段」拆开, 一条主干随时分叉。',
      tldr: 'Cosine: lr = f(step / total), 总步数一改整条曲线都变。WSD (Warmup-Stable-Decay): warmup 后保持峰值 LR, 只在最后 ~10% 线性退火; 稳定段与总步数无关, 任何稳定段 checkpoint 都能接着训或分叉出退火分支。',
      question: '训到 100% 发现 loss 还在降, 想加训: cosine 和 WSD 各需要付出什么?',
      code: 'llm_train/m10_training_stability/demo.py',
      points: [
        { title: 'warmup 为什么必要', body: 'Adam 的二阶矩估计在最初几百步很不准, 配大 LR 容易一步走飞; 线性 warmup 把最危险的阶段用小步子走过去。' },
        { title: 'cosine 绑死总步数', body: 'demo 里 total 从 100 改到 200, cosine 前 90 步的 LR 最大相差 5.96e-4, WSD 相差 0。续训 cosine 只能 re-warmup (loss 先反弹) 或从头来。' },
        { title: 'WSD 的工程价值', body: 'MiniCPM / DeepSeek-V3 / Kimi 同类调度: 主干一直保持高 LR, 要「成品」时从任意 checkpoint 分叉一段短退火; 数据追加、scaling law 实验都不用重训。' },
      ],
      links: [
        { from: 'train-precision-stability', to: 'warmup', body: 'warmup、clip、NaN guard 同属「别让坏 step 毁掉训练」。' },
        { from: 'decay_start', to: 'checkpoint', body: '退火开始前的那个 checkpoint 最值钱: 它是所有分支的公共祖先。' },
        { from: 'WSD 退火', to: 'Muon / AdamW', body: '调度与优化器正交; Kimi K2 用的就是 Muon + WSD。' },
      ],
      sourceRows: [
        { concept: 'warmup + cosine', code: 'm10_training_stability/demo.py:warmup_cosine_lr', takeaway: 'progress = (step − warmup) / (total − warmup): total 出现在每一步的公式里。' },
        { concept: 'WSD', code: 'm10_training_stability/demo.py:wsd_lr', takeaway: '稳定段直接 return base_lr —— 这一行里没有 total_steps。' },
        { concept: '退火起点', code: 'decay_start = total_steps - int(total_steps * decay_frac)', takeaway: '只有这一个量依赖总步数, 而且只影响最后 ~10%。' },
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
      subtitle: '位宽每砍一半, 单个元素能表达的动态范围就塌一截; 靠越来越细的共享 scale 把它撑回来。',
      tldr: 'FP8 E4M3 自带 ~2^15 动态范围, per-tensor 与 block scale 在普通张量上打平, outlier 超过 ~1e5 倍才见分晓。FP4 E2M1 只有 ±{0.5,1,1.5,2,3,4,6}, 动态范围 12 倍, 必须每 16~32 个元素共享一个 scale: MXFP4 = block 32 + E8M0 (2 的幂) scale = 4.25 bit; NVFP4 = block 16 + E4M3 scale = 4.5 bit。',
      question: '同样是 4 bit + block scale, 为什么 NVFP4 的误差 (0.095) 明显低于 MXFP4 (0.113)? INT4 什么时候反而更好?',
      code: 'llm_train/m13_fp8_training/demo.py · llm_train/m15_fp4_microscaling/demo.py · llm_train/core/numerics.py',
      points: [
        { title: 'scale 由 amax 决定', body: '一个 block 里最大的元素决定 scale, 其余元素按它缩放后再舍入。outlier 越大, 同 block 的邻居越靠近 0 —— block 越小, 受连累的邻居越少。' },
        { title: 'scale 自己的精度', body: 'E8M0 只能是 2 的幂: amax/scale 落在 [4,8), 大于 6 的会饱和 (demo 里 42% 的 block 最大值被截断)。E4M3 scale 带 3 位尾数, 能把 amax 几乎精确贴到 6。同为 block 16: 0.114 vs 0.095。' },
        { title: '浮点网格 vs 整数网格', body: '高斯数据上 INT4 (0.096) 反而赢 MXFP4 (0.113); 重尾数据 (激活/梯度) 上对数间距的浮点网格才占优 (0.142 vs 0.147)。FP4 不是白送的。' },
      ],
      links: [
        { from: 'train-precision-stability', to: 'FP8 block scaling', body: '同一个 fake_quant_float, 只是 (尾数位, 最大值, 最小指数) 换了一组。' },
        { from: 'quant_blockwise', to: 'mxfp4 / nvfp4', body: '两个函数只有 scale 那一行不同。' },
        { from: 'FP4 权重', to: 'llm_infer 量化', body: 'FP4 目前主要用于推理权重和 QAT; 全程 FP4 预训练还需要随机舍入、Hadamard 旋转等技巧。' },
      ],
      sourceRows: [
        { concept: 'block scaling', code: 'core/numerics.py:quant_blockwise', takeaway: '每 block 个元素共享 scale = amax / max_val; block ≥ 张量大小即 per-tensor。' },
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
      subtitle: 'Adam 逐元素地看梯度; Muon 把 2-D 权重的梯度当成一个矩阵来看, 让每个奇异方向迈同样大的步子。',
      tldr: '对 2-D 权重: W ← W − lr·NS(M), NS(M) ≈ UVᵀ。Newton–Schulz 迭代 X ← aX + (bA + cA²)X 只用矩阵乘, 5 步把奇异值推到 ~1 附近 (1e-4 → 0.041)。优化器状态只有 1 份动量 (Adam 的一半); embedding / 输出头 / norm / bias 仍用 AdamW。',
      question: 'Adam 已经有逐元素的自适应步长了, 为什么还会被「病态方向」拖住? Muon 什么时候反而不如 Adam?',
      code: 'llm_train/m14_muon_optimizer/demo.py',
      points: [
        { title: '梯度谱极不均匀', body: '梯度矩阵的奇异值常差几个数量级, SGD/Adam 的更新被头几个方向主导。正交化之后每个方向步长相同 —— 相当于在谱范数下做最陡下降。' },
        { title: '与基无关', body: 'Adam 的逐元素缩放只能修「对角」的病态。demo: 病态方向不与坐标轴对齐时 Muon 比调好 lr 的 AdamW 好 9.2×; 轴对齐时 Adam 反而赢 1.9×。' },
        { title: '大规模要配 QK-clip', body: 'Muon 的更新是满秩的, attention logit 容易爆 (demo 里 752)。Kimi K2 的 MuonClip: 每步后若 max logit > τ, 把 Wq、Wk 各缩 √(τ/S_max), 压回 100。' },
      ],
      links: [
        { from: 'llm_basic optim.py', to: 'Muon', body: 'Adam 存 m、v 两份状态; Muon 只存动量一份。' },
        { from: 'newton_schulz', to: 'GPU 友好', body: '精确 UVᵀ 要 SVD; NS 只用矩阵乘, 5 步开销相对前反向不到 1%。' },
        { from: '0.2·√max(n,m)', to: 'AdamW 学习率', body: 'Moonlight 的缩放: 让 Muon 更新的 RMS ≈ AdamW, 学习率和 weight decay 可直接复用。' },
      ],
      sourceRows: [
        { concept: 'Newton–Schulz', code: 'm14_muon_optimizer/demo.py:newton_schulz', takeaway: '先除以 Frobenius 范数保证 σ ≤ 1, 再对每个奇异值作用 aσ + bσ³ + cσ⁵。' },
        { concept: '一步更新', code: 'm14_muon_optimizer/demo.py:Muon', takeaway: 'Nesterov 式动量 → 正交化 → 乘 0.2·√max(n,m) → 更新。' },
        { concept: 'QK-clip', code: 'm14_muon_optimizer/demo.py:qk_clip', takeaway: 'η = min(1, τ/S_max), Wq、Wk 各乘 √η。' },
        { concept: '对照实验', code: 'm14_muon_optimizer/demo.py:run', takeaway: 'rotate=True/False 两种病态, 各自扫 5 个 lr 取最好再比。' },
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
      subtitle: '和 Ring Attention 目标相同 —— 一条序列放不进一张卡; 但它不改注意力核, 只在注意力前后各做一次 all-to-all。',
      tldr: '注意力之外每卡持有 [T/P, H, d]; 进注意力前 all-to-all 成 [T, H/P, d]: 每卡看到完整序列但只算 H/P 个头, 直接跑普通 FlashAttention; 算完再换回来。每卡通信 4(P−1)/P² 份张量, 随 P 下降 (Ring 是 2(P−1)/P, 不降)。硬约束: P 必须整除头数。',
      question: 'Ulysses 的通信量随 P 增大反而下降, 为什么大家没有全部换成它?',
      code: 'llm_train/m16_ulysses_sequence_parallel/demo.py · llm_train/m12_sequence_parallel/demo.py',
      points: [
        { title: '换切法, 不换数学', body: '头与头之间本来就互不相干, 所以按头切的注意力与完整注意力逐位相同 (demo: max|Δ| < 1e-12)。因果 mask 天然均衡, 不需要 zigzag。' },
        { title: '通信量随 P 下降', body: 'demo (T=32, H=8, d=8) 每 rank 字节: P=2/4/8 → Ulysses 16384/12288/7168, Ring 16384/24576/28672。P=8 时 Ring 是 Ulysses 的 4 倍。' },
        { title: '两个硬约束', body: 'P ≤ 头数且整除 (GQA/MQA 的 KV 头更少, 限制更紧); all-to-all 吃对分带宽, 跨机不如 Ring 的邻居 P2P 友好。实战: 机内 8 路 Ulysses × 机间 4 路 Ring。' },
      ],
      links: [
        { from: 'train-moe-seq', to: 'Ring Attention', body: 'Ring 让 KV 块流动、Q 不动; Ulysses 让所有人一次性换位。' },
        { from: 'core/collectives.py:all_to_all', to: 'seq_to_head', body: '和 MoE dispatch 是同一个原语, out[dst][src] = in[src][dst]。' },
        { from: '[T/P, H, d]', to: '[T, H/P, d]', body: '盯住这个 shape 变化就看懂了整个算法。' },
      ],
      sourceRows: [
        { concept: '序列 → 头', code: 'm16_ulysses_sequence_parallel/demo.py:seq_to_head', takeaway: 'rank s 把自己的第 j 组头发给 rank j; 收到后沿序列维拼起来。' },
        { concept: '头 → 序列', code: 'm16_ulysses_sequence_parallel/demo.py:head_to_seq', takeaway: '完全对称的逆变换。' },
        { concept: '整除约束', code: 'assert H % P == 0', takeaway: '8 个头切不成 16 份 —— demo 里故意触发这个 AssertionError。' },
        { concept: '通信账', code: '4 * (P - 1) / P * shard_bytes', takeaway: '4 次 all-to-all (Q、K、V、输出), 每次只发不属于自己的 (P−1)/P。' },
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
