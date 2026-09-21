// 阶段 5 · llm_infer 自测题 (每章 3 题; 错误选项都是常见误解)
const bank = {
  infer: [
    {
      q: '训练时最贵的是反向传播, 推理服务里 decode 阶段最常见的瓶颈是什么?',
      options: ['矩阵乘的 FLOPs', '显存带宽: 每出 1 个 token 都要把权重和 KV 读一遍', '反向传播的激活', 'tokenizer 的速度'],
      answer: 1,
      why: 'decode 每步只算 1 个 token, 算术强度极低, 时间花在"搬数据"上; 所以量化、GQA/MLA、投机解码都在减少每个 token 要搬的字节或摊薄这次搬运。',
    },
    {
      q: '下面哪一项优化会改变模型的输出 (greedy 下)?',
      options: ['KV cache', 'PagedAttention', '权重 INT4 量化', 'continuous batching'],
      answer: 2,
      why: 'cache、分页、调度、前缀复用、投机解码都只改执行路径, 输出逐 token 相同; 量化、sink cache、稀疏 attention 是有损的, 要看误差。',
    },
    {
      q: 'mini-vLLM 的 Engine.step 每一步做的第一件事是什么?',
      options: ['model.forward', '采样', 'scheduler.schedule(): 决定这一步哪些序列、各算几个 token', '释放已完成请求的 block'],
      answer: 2,
      why: '推理引擎首先是调度器和资源管理器: 先决定 batch = [(seq, n)], 才轮到前向和采样。',
    },
  ],
  'infer-kv-memory': [
    {
      q: 'block_size=8, 某条序列的页表是 [9, 4, 6, 1, 3]。第 37 个位置 (pos=37) 的 KV 存在哪里?',
      options: ['block 6 的第 5 格', 'block 3 的第 5 格', 'block 1 的第 5 格', 'block 37 的第 0 格'],
      answer: 1,
      why: '逻辑页 = 37 // 8 = 4 → 页表第 4 项是物理 block 3; 页内偏移 = 37 % 8 = 5。和操作系统的虚拟内存翻译完全一样。',
    },
    {
      q: 'PagedAttention 让 attention 算得更快了吗?',
      options: ['是, 分块后可以并行', '是, 减少了 FLOPs', '没有 —— 它买到的是显存利用率, 进而是更大的 batch', '没有, 它只是为了支持更长的上下文'],
      answer: 2,
      why: '多一层页表间接寻址, 单次 attention 甚至略慢。收益在于不再按 max_len 预留连续显存, 浪费只剩每条序列最后一个 block 的尾巴, 同样显存能塞下更多请求。',
    },
    {
      q: '有了 KV cache 之后, 第 t 步 decode 还有什么开销是随 t 增长的?',
      options: ['没有了, 每步严格 O(1)', '新 token 的 query 仍要和全部 t 个缓存的 key 做点积 (读一遍 KV)', '要重新计算旧 token 的 K/V', 'MLP 要对所有旧 token 重算'],
      answer: 1,
      why: 'cache 省掉的是旧 token 的 K/V 投影和 MLP; 新 query 对 t 个 key 的 attention 仍是 O(t) 的访存, 这正是长上下文 decode 变慢、需要 m22 稀疏读取的原因。',
    },
  ],
  'infer-scheduler': [
    {
      q: '队首请求因为 block 不够而无法被接纳, prefill 优先的调度器这一步应该怎么办?',
      options: ['返回空 batch, 等显存释放', '跳过队首, 接纳后面更短的请求', '落到 decode: 继续推进 running 的序列', '抢占所有 running 序列给它腾位置'],
      answer: 2,
      why: '返回空 batch 会活锁: running 不前进 → block 永远不释放 → 队首永远进不来。Python 里就是 `_admit(budget) or _schedule_running(budget)` 这一行。',
    },
    {
      q: 'token 预算 B=256, 当前有 10 条序列在 decode。这一步最多能塞多大的 prefill chunk?',
      options: ['256', '246', '128', '25'],
      answer: 1,
      why: 'decode 优先: 每条 running 先拿 1 个 token, 剩下 256 − 10 = 246 个预算给 prefill chunk。预算封顶了每步耗时, 也就封顶了 TBT。',
    },
    {
      q: '开启 chunked prefill (B=128) 后, demo 里 decode 用户的最大 TBT 从 297 ms 降到 52 ms。代价是什么?',
      options: ['没有代价', '总吞吐下降一半', '长请求自己的 TTFT 从 276 ms 涨到 445 ms', '输出结果会变'],
      answer: 2,
      why: '长 prompt 被切成多步, 每步还要付一次固定开销, 所以新请求的首 token 更晚。chunked prefill 是拿新请求的 TTFT 换其他人的延迟平稳。',
    },
  ],
  'infer-decode-control': [
    {
      q: 'draft 模型很差 (接受率很低), 投机解码的输出质量会怎样?',
      options: ['变差, 因为混入了 draft 的 token', '不变, 只是变慢 (甚至比不投机还慢)', '变好, 因为多了一次验证', '取决于温度'],
      answer: 1,
      why: '每个被接受的 token 都经过 target 验证; 拒绝时从残差 max(0, p−q) 重采样, 合起来的分布恰好是 target 的 p。draft 只影响速度。',
    },
    {
      q: '接受率 α=0.8 时, 把 draft 长度 K 从 4 加到 100, 每次 target 调用的期望 token 数最多到多少?',
      options: ['101', '约 80', '5', '约 3.4'],
      answer: 2,
      why: '(1−α^(K+1))/(1−α) 的上限是 1/(1−α) = 5。第一个猜错后面全废, 所以链式 draft 加长很快就没有收益, 这正是树形投机的动机。',
    },
    {
      q: '采样模式下, 如果把接受规则改成"draft 的 token 全部接受", 会发生什么?',
      options: ['更快且分布不变', '输出分布变成 draft 的分布, 不再无损', '只影响 bonus token', '和 rejection sampling 等价'],
      answer: 1,
      why: 'demo 里这个错误规则的 χ² 是 585 (临界值 37.7), 而正确的 min(1, p/q) + 残差重采样 ≤ 13.6。"无损"完全来自这条接受规则。',
    },
  ],
  'infer-compute': [
    {
      q: 'FlashAttention 的核心收益是什么?',
      options: ['attention 的 FLOPs 从 O(N²) 降到 O(N)', '不物化 N×N 矩阵: 显存 O(N), 并大幅减少 HBM 读写', '用近似 softmax 换速度', '把 attention 改成了线性注意力'],
      answer: 1,
      why: 'FLOPs 没少 (甚至略多), 结果也是精确的。快是因为每个 tile 只进 SRAM 一次, 省掉了对 N×N 中间矩阵的反复读写。',
    },
    {
      q: 'online softmax 处理到一个新 block, 发现了更大的 max。之前累计的分母 l 要怎么处理?',
      options: ['清零重算', '乘以 exp(m_old − m_new) 再加上新 block 的和', '不用动', '除以 block 数'],
      answer: 1,
      why: '旧的和是以 m_old 为基准算的 Σexp(s − m_old); 换基准到 m_new 只需整体乘 exp(m_old − m_new)。这一行让分块结果与整块 softmax 逐位一致。',
    },
    {
      q: 'CUDA Graph 在什么情况下收益最大?',
      options: ['大 batch 的 prefill', '小 batch decode: 每个 kernel 很短, launch 开销占比高', '任何情况都一样', '只对量化模型有效'],
      answer: 1,
      why: 'graph 省的是 CPU 逐个 launch kernel 的开销。kernel 本身越短这部分占比越高; 大 prefill 时 GPU 计算远大于 launch 开销, 几乎没收益。(本仓库的数字是注入开销的模型, 不是实测。)',
    },
  ],
  'infer-engine': [
    {
      q: '一条请求的 prompt 完全命中了 prefix cache。还需要做前向吗?',
      options: ['不需要, 直接 decode', '需要: 至少真算最后 1 个 token, 否则没有 logits 可以采样首 token', '需要重算整个 prompt', '只需要重算第一个 block'],
      answer: 1,
      why: 'cache 里存的是 KV, 不是 logits。match_prefix 故意只匹配 (len−1)//bs 个 block, 保证至少留 1 个 token 真算。',
    },
    {
      q: '引擎用 recompute 式抢占: 被抢占的序列已经生成的 token 怎么办?',
      options: ['全部丢弃, 从头生成', '保留文本, 归还 block; 重新接纳时连同已生成 token 一起重算 KV', '把 KV 换出到 CPU', '继续占着 block 等待'],
      answer: 1,
      why: 'output_ids 原样保留, num_computed 归 0 回 waiting 队首。重算的是 KV 不是文本, 所以 greedy 输出仍与朴素生成逐 token 相同 (demo: 9 个 block, 4 次抢占)。',
    },
    {
      q: 'demo 第一轮共 305 个待算 token, 前缀命中 72 个。runner.tokens_computed 应该是多少?',
      options: ['305', '377', '233', '72'],
      answer: 2,
      why: '305 = 233 个真前向 + 72 个命中跳过。引擎把这两个数分开统计并要求对得上账 —— "命中"必须是真的没算, 而不是算了再扔。',
    },
  ],
  'infer-prefix-radix': [
    {
      q: 'block_size=16, 两条请求的前 100 个 token 相同。链式 block hash 能复用多少个 token 的 KV?',
      options: ['100', '96', '112', '0, 因为后面不同'],
      answer: 1,
      why: '只能命中完整 block: ⌊100/16⌋·16 = 96。radix tree 以 token 为粒度, 能命中全部 100 个 —— 代价是实现更复杂 (边分裂)。',
    },
    {
      q: '为什么每个 block 的 hash 要把"父 block 的 hash"也算进去?',
      options: ['为了让 hash 更随机', '因为 KV 依赖它之前的所有 token: 同一段 token 接在不同前缀后, KV 并不相同', '为了节省内存', '为了支持 LRU'],
      answer: 1,
      why: 'attention 让位置 i 的 KV 取决于整个前缀。链式 hash 使一个 hash 唯一标识"到此为止的整段前缀", 不会把别的上下文里的同名 block 误当命中。',
    },
    {
      q: 'radix cache 满了, 为什么只能从叶子开始驱逐?',
      options: ['叶子占的显存最大', '中间节点被所有后代依赖, 扔掉它后代的 KV 就都无法复用了', '叶子的访问时间最旧', '根节点不能删除'],
      answer: 1,
      why: '前缀是链式依赖: 命中必须从根连续走到某处。驱逐中间节点等于把整棵子树作废, 所以按 LRU 挑 ref_count=0 的叶子。',
    },
  ],
  'infer-structured-output': [
    {
      q: '为什么不在每步解码时对 V 个 token 逐字符试走 FSM?',
      options: ['结果会不正确', '这是 O(V·len) 的 CPU 开销, 卡在 GPU 前向和采样之间, 直接吃掉 decode 延迟', 'FSM 不支持多字符', 'GPU 上无法做 mask'],
      answer: 1,
      why: '词表十几万, 每步都试走不可接受。FSM 状态有限, 离线对每个 (状态, token) 试走一次存成表, 在线只查一行。',
    },
    {
      q: 'token \'":\' 在"正在写 key"的状态下合法吗?',
      options: ['不合法, 因为它跨了语法边界', '合法: 逐字符试走都走得通, 落点是"等待 value"', '只有第一个字符合法', '取决于模型的 logits'],
      answer: 1,
      why: '多字符 token 可以一口气跨几个状态: \'"\' 结束 key, \':\' 进入 value。判定标准只有一条 —— 整段字符全部走得通。',
    },
    {
      q: '只靠 prompt 写"请输出 JSON"为什么不够?',
      options: ['模型看不懂中文', 'prompt 只是提高概率; 只有在 logits 上把非法 token 置 −inf 才是 100% 保证', 'JSON 太长', '会降低吞吐'],
      answer: 1,
      why: '采样总有概率落到非法 token。mask 让非法 token 概率严格为 0, 再配合 need 表保证在长度上限内能收尾, 输出一定可解析。',
    },
  ],
  'infer-quant-awq': [
    {
      q: 'AWQ 凭什么说"激活大的通道上的权重更重要"?',
      options: ['这些权重的数值更大', '输出误差 = Σ x_i·ΔW_i: 同样的权重舍入误差, 乘上大激活后对输出的影响更大', '这些权重更新得更频繁', '它们在第一层'],
      answer: 1,
      why: '量化真正要最小化的是 ‖XW − XŴ‖ 而不是 ‖W − Ŵ‖。demo: INT4 g32 的 RTN 误差 0.0759, AWQ 降到 0.0273。',
    },
    {
      q: 'AWQ 的缩放指数 α 为什么不是越大越好 (demo 最优 α=0.4)?',
      options: ['α 大了会溢出', '被放大的行撑大了所在 group 的 min/max 范围, 同组其它权重的格点变粗', 'α 只能取 0 或 1', 'α 大了推理会变慢'],
      answer: 1,
      why: '放大 s 倍让该行相对误差缩小约 s 倍, 但 group 共用一组 scale: range 变大 → 其它行误差上升。所以要在校准集上网格搜索。',
    },
    {
      q: 'KIVI 为什么对 K 按通道分组、对 V 按 token 分组?',
      options: ['K 比 V 大', 'K 有少数固定通道在所有 token 上都是大值; 按 token 分组时每一行的 scale 都被它撑大', 'V 不需要量化', '按通道更快'],
      answer: 1,
      why: 'per-channel 把离群通道关在自己的组里 (demo INT4: 误差 0.026 vs per-token 0.18)。V 没有固定离群通道, 按 token 分组还能在每个新 token 写入时独立量化。',
    },
  ],
  'infer-kv-footprint': [
    {
      q: 'LLaMA-3-8B (32 层, 8 个 KV 头, d_head=128, fp16) 每个 token 的 KV 是多少?',
      options: ['16 KiB', '128 KiB', '512 KiB', '2 MiB'],
      answer: 1,
      why: '2·8·128·32·2 字节 = 131072 B = 128 KiB。同形状的 MHA (32 个 KV 头) 是 512 KiB, 差的正好是 n_head/n_kv = 4 倍。',
    },
    {
      q: 'MLA 每 token 的 cache 公式里为什么没有"2·"?',
      options: ['它不缓存 V', 'K 和 V 共用同一个 latent c_kv, 用到时才各自上投影还原', '它用了 int8', '它只缓存一半的层'],
      answer: 1,
      why: 'cache 里只有 (d_c + d_rope) 维: latent 同时是 K 和 V 的来源。DeepSeek-V3: (512+64)·61·2 = 68.6 KiB, 比同尺寸 MHA 省 56.9×。',
    },
    {
      q: '40 GiB 的 KV 预算、上下文 8192: MHA 的 7B 只能同时服务 10 条, 换成 GQA-8 呢?',
      options: ['10 条, 并发由算力决定', '20 条', '40 条', '80 条'],
      answer: 2,
      why: '并发上限 = KV 预算 ÷ (每 token 字节 × 上下文长度)。每 token 字节降 4 倍, 并发就是 4 倍。结构选择在推理侧直接变成吞吐。',
    },
  ],
  'infer-attention-sinks': [
    {
      q: '纯滑动窗口把最开头几个 token 逐出 cache 后, 模型为什么会崩?',
      options: ['开头的 token 语义最重要', 'softmax 权重和必须为 1, 模型学会把"多余的注意力"倒在开头 token 上; 它们没了, 其余权重被迫整体膨胀', '窗口太小放不下句子', 'RoPE 不支持滑动'],
      answer: 1,
      why: 'sink token 收走的是"无处安放"的注意力质量, 和语义无关。逐出它们等于突然改掉 softmax 的分母, 这是训练时从未见过的分布。',
    },
    {
      q: 'SinkCache 为什么存未旋转 (pre-RoPE) 的 K?',
      options: ['节省显存', '位置要按 cache 槽位重新编号, 每步按当前槽位现转; 存了转好的就没法改位置了', 'RoPE 在推理时不需要', '为了支持量化'],
      answer: 1,
      why: '逐出中间 token 后, 留下的 token 槽位会前移。用槽位 0..L−1 当位置, 才能保证永远不超过训练长度。代价是每步 O(L·D) 的重转。',
    },
    {
      q: '用了 attention sinks, 模型就能"记住"无限长的上下文了吗?',
      options: ['能, 这就是它的目的', '不能: 窗口之外的内容彻底丢了; sink 只保证流式生成不崩', '能, 但只对英文', '能记住一半'],
      answer: 1,
      why: 'StreamingLLM 解决的是"无限输入下稳定生成 + 显存有界", 不是长程记忆。要记得就得保留 KV (offload) 或者检索。',
    },
  ],
  'infer-tree-speculation': [
    {
      q: '同样让 target 一次验证 16 个 token, 树 [3,2,1] 为什么比 K=15 的链接受得多 (demo 1.80 vs 1.40)?',
      options: ['树的 draft 模型更大', '链第一个猜错后面全废; 树在最容易错的前几层留了 top-k 备选', '树的验证更宽松', '树不需要 KV 回滚'],
      answer: 1,
      why: '接受长度由"第一处不一致"决定。把验证预算花在浅层的多个候选上, 比花在一条很深但早早断掉的链上划算。',
    },
    {
      q: 'tree attention mask 里, 一个节点能看到哪些位置?',
      options: ['上下文 + 所有编号比它小的节点', '上下文 + 它的祖先 + 它自己', '只有它的父节点', '整棵树'],
      answer: 1,
      why: '每个节点看到的恰好是"根到自己"这条链, 等价于把每条路径单独顺序 decode 一次。兄弟互不可见, 且同深度共享同一个 RoPE 位置。',
    },
    {
      q: 'EAGLE 的 draft 和 m07 的独立小模型 draft, 关键区别是什么?',
      options: ['EAGLE 的 draft 更大', 'EAGLE 吃 target 已经算好的 hidden state 并共享 target 的 lm_head, 信息比 token id 多得多', 'EAGLE 不需要验证', 'EAGLE 只能 greedy'],
      answer: 1,
      why: '验证那次 forward 白送了下一轮需要的特征。验证循环和接受规则完全复用 m07, 只换 drafter; 误差随 draft 步数累积, 所以越靠后的槽位越难接受。',
    },
  ],
  'infer-kv-offload': [
    {
      q: 'demo 里从磁盘层命中 1 个 block: 加载 2.70 ms, 重算 2.00 ms。该怎么办?',
      options: ['加载, 因为命中了就该用', '重算: 固定延迟摊不开, 命中块数超过 n* ≈ 1.54 才值得加载', '两个都做取先到的', '丢弃这个 block'],
      answer: 1,
      why: 'load = latency + n·每块搬运, recompute = n·每块重算。n* = latency / (每块重算 − 每块搬运)。每层都要做这个判断。',
    },
    {
      q: '加了一层 0.5 GB/s 的远端缓存, "命中就加载"的 TTFT 从 34.8 ms 变成 68.9 ms。原因是?',
      options: ['远端容量太小', '这条链路每 block 的加载 (4.19 ms) 比重算 (2.00 ms) 还慢, 命中越多越亏', '命中率下降了', 'hash 冲突'],
      answer: 1,
      why: '缓存层不是越多越好: 链路比重算慢时, 命中是负收益。策略改成"加载/重算取小"后回到 34.8 ms。',
    },
    {
      q: '沿 hash 链查找时, 第 3 块在所有层都 miss, 但第 4、5 块还在 CPU 层。它们能用吗?',
      options: ['能, 直接加载第 4、5 块', '不能: KV 依赖整个前缀, 链断了后面的块就不可用', '能, 但要重新计算 hash', '取决于 block 大小'],
      answer: 1,
      why: 'lookup 在第一个全层 miss 处停。第 4 块的 hash 标识的是"前 4 块的整段前缀"; 前缀的 KV 不全, 后面的也接不上。',
    },
  ],
  'infer-moe-serving': [
    {
      q: '专家并行下, 评价负载均衡为什么看 max/mean 而不是方差或平均值?',
      options: ['max 更容易算', '每层 combine 要等所有 rank: 一步的耗时 = 最忙 rank 的耗时', '平均值总是相同的', '方差没有单位'],
      answer: 1,
      why: 'max/mean = 2.95 意味着平均 rank 利用率只有 1/2.95 ≈ 34%, 其余时间都在等最慢的那张卡。',
    },
    {
      q: '最热专家的负载是平均 rank 负载的 1.42 倍。只重新摆放专家 (不复制) 能做到均衡吗?',
      options: ['能, 把它单独放一张卡', '不能: 它一个就超过一张卡该分到的量, 必须复制并拆分它的 token', '能, 用更好的贪心', '不能, 因为 top-k=2'],
      answer: 1,
      why: '单个专家不可分 → 它所在的 rank 至少 1.42× 均值。demo: 贪心无副本 1.48×, EPLB 给专家 0 加 3 个副本、专家 1 加 1 个后 1.02×。',
    },
    {
      q: '给热专家加冗余副本后, 模型输出会变吗?',
      options: ['会, 因为路由变了', '不会: 副本权重相同, 只是同一专家的 token 被分到不同 slot 去算', '会有很小的数值误差', '只有 top-1 路由不变'],
      answer: 1,
      why: 'EPLB 只改"逻辑专家 → 物理 slot"的映射, 路由和 gate 都不变; demo 断言输出与朴素实现 max|Δ| = 0。',
    },
  ],
  'infer-sparse-decode': [
    {
      q: 'Quest 给 block 打分用的是 q·k 的"上界"。为什么要上界而不是均值这类估计?',
      options: ['上界算得更快', '上界 ≥ block 内任何真实 q·k, 保证含有高分 token 的 block 不会被低估而漏掉', '上界更省显存', '均值无法计算'],
      answer: 1,
      why: '逐维取 max(q_i·kmin_i, q_i·kmax_i) 再求和。代价是偏松 (demo 平均 6.7×), 但"不漏针"比"估得准"重要。',
    },
    {
      q: 'needle 负载上 k 从 4 加到 8 (读 1.6% → 3.1% 的 KV), 误差从 0.83 断崖降到 0.023。说明什么?',
      options: ['误差与 k 成反比', '注意力质量集中在少数 block 上; 它们全部进了 top-k 的那一刻, 其余 KV 几乎无关紧要', 'k 必须是 8 的倍数', '随机选块也一样'],
      answer: 1,
      why: '稀疏 decode 的前提是注意力高度集中。同样读 3.1%, 随机选块误差是 1.06。',
    },
    {
      q: '在随机权重的 TinyLM 上, Quest 选块并不比随机选块好。为什么?',
      options: ['实现有 bug', '随机权重模型的注意力近乎均匀弥散, 没有"少数重要 token", 稀疏化的前提不成立', 'TinyLM 太小', 'block 太大'],
      answer: 1,
      why: '方法的收益来自数据分布而不是算法本身。训练过的 LLM 注意力很尖, 才有 Quest / NSA / DSA 的空间 —— demo 把这个反例也如实打印了出来。',
    },
  ],
}

// 源文件里正确项大多写在第 2 个 (方便对照着写 why); 展示前按题号确定性地轮转选项, 免得读者靠位置猜
const rotate = (q, k) => ({ ...q, options: q.options.map((_, j) => q.options[(j - k + 4) % 4]), answer: (q.answer + k) % 4 })
export default Object.fromEntries(
  Object.entries(bank).map(([route, qs], r) => [route, qs.map((q, i) => rotate(q, (r + 2 * i + 1) % 4))]),
)
