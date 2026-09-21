// 阶段 5 · llm_infer 章节内容。
// 本文件持有本阶段全部 14 章的完整 page 对象 (models.js 不再保留副本)。
// 贯穿全阶段的一条主线: decode 是带宽瓶颈, 不是算力瓶颈 ——
// 所以几乎每一项优化都在省 KV 的"搬运"和"存储", 而不是省乘法。
const I = 'llm_infer/'

export default {
  stage: 'infer',
  chapters: [
    { route: 'infer-prefix-radix', label: '前缀复用 · hash 与 radix', hint: '链式 block hash、radix tree、命中即跳过前向' },
    { route: 'infer-structured-output', label: 'token 级语法约束', hint: 'FSM × 多字符词表、预编译 mask 表' },
    { route: 'infer-quant-awq', label: 'INT4 · AWQ · KIVI', hint: 'group-wise、激活感知缩放、K 按通道 V 按 token' },
    { route: 'infer-kv-footprint', label: 'KV 体积 · MHA→MLA', hint: 'bytes/token 公式、latent cache decode' },
    { route: 'infer-attention-sinks', label: 'Attention sinks', hint: 'StreamingLLM: sink + 窗口 + 位置重编号' },
    { route: 'infer-tree-speculation', label: 'EAGLE 与树形投机', hint: '特征级 draft、tree attention mask、最长接受路径' },
    { route: 'infer-kv-offload', label: '分层 KV offload', hint: 'GPU→CPU→disk, 加载 vs 重算' },
    { route: 'infer-moe-serving', label: 'MoE serving · EPLB', hint: 'EP dispatch/combine、冗余专家' },
    { route: 'infer-sparse-decode', label: '稀疏注意力 decode', hint: 'block 打分 → top-k, 只读一小部分 KV' },
  ],
  pages: {
    // ---------------- 主线五章 ---------------- //
    'infer-kv-memory': {
      widgets: ['AttnMaskLab'],
      title: 'KV 与缓存内存 · 推理优化从"别重算"开始',
      subtitle: '读完你能算出一张卡同时放得下多少条序列, 并说清分页到底买到了什么。',
      tldr: '因果 mask 下旧 token 的 K/V 不会因为新 token 到来而改变, 所以存下来就别再算了: 累计过一遍模型的 token 数从 688 掉到 37 (18.6×)。省下的算力换成了显存压力, 于是再把 KV 切成定长 block 用页表管 —— 分页不让 attention 变快, 它买的是显存利用率。',
      question: '有了 KV cache 之后, decode 每步是不是就 O(1) 了?',
      code: 'llm_infer/m01_kv_cache · llm_infer/m02_paged_attention (前缀复用另见「前缀复用 · hash 与 radix」)',
      points: [
        {
          title: '只有 K/V 值得存',
          body: '因果 mask 保证第 i 个 token 的每层 hidden 只依赖 ≤ i 的 token, 后面来什么都不会改它。所以 prefill 把每层的 K/V 存下来, decode 每步只算新 token 的 q/k/v 再追加。q 不用存: 历史 token 的 q 只在它自己那一步用过一次。',
        },
        {
          title: 'cache 没省掉的那部分',
          key: true,
          body: '新 token 的 query 仍要和全部 t 个 key 做点积。cache 省的是旧 token 的 K/V 投影和 MLP 重算, 不是 attention 本身 —— 所以长上下文 decode 依然是 O(t) 的访存, 这正是后面 GQA/MLA、KV 量化、稀疏读取都要解决的事。',
        },
        {
          title: '分页买的是显存, 不是速度',
          body: 'KV 切成定长 block, 每条序列一张页表: 位置 pos 的 KV 在 pool[table[pos // bs], pos % bs]。多一层间接寻址, 单次 attention 反而略慢。收益是不再按 max_length 预留连续显存, 浪费只剩每条序列末页那几个空槽。',
        },
      ],
      links: [
        { from: 'sample.py 逐 token 生成', to: 'KV cache', body: '同样是自回归, 但不再每步把整段 prefix 重跑一遍。' },
        { from: 'BlockManager.can_allocate', to: 'Scheduler._admit', body: '显存块够不够, 决定这一步能不能接新请求; 不够就抢占。' },
        { from: 'free_list 归还顺序', to: 'PrefixCache 的 LRU', body: 'block 被 free 只是 ref_count 归零进队列, 内容留到被覆盖为止 —— 前缀复用正是靠这一点。' },
      ],
      sourceRows: [
        { concept: '对照组', code: 'm01_kv_cache/demo.py:generate_with_cache', takeaway: '有 cache 与无 cache 的输出 ids 必须逐个相同 (logits 差 2.80e-06); 累计过模型的 token 688 → 37。' },
        { concept: 'KV 字节公式', code: '2 · n_layer · T · D · sizeof(dtype)', takeaway: 'demo 断言公式算的 75776 B == numpy 实际 nbytes。代入 LLaMA-7B fp16 = 0.50 MiB/token, T=4096 就是 2.00 GiB 一条。' },
        { concept: '地址翻译', code: 'm02_paged_attention/paged_attention.py:paged_attention', takeaway: 'pos → (block_table[pos // bs], pos % bs), 和操作系统的虚拟内存一模一样; 与连续 KV 的结果 max|Δ| = 0.00e+00。' },
        { concept: '碎片上界', code: 'BlockManager.stats', takeaway: '每条序列最多浪费 bs−1 个槽 (demo 实测 1/12, 上界 3); 按 max_len 连续预留的方案, vLLM 论文测得只有 20~40% 装着真数据。' },
        { concept: '要不到块', code: 'MemoryError: need 25 new blocks, only 3 free', takeaway: '这个异常就是调度器触发抢占的信号。' },
      ],
      snippetTitle: 'KV cache 路径',
      snippet: `logits, kv_cache = lm.prefill(prompt_ids)
next_id = argmax(logits[-1])

for _ in range(max_new - 1):
    logits, kv_cache = lm.decode_step(next_id, kv_cache)   # 只喂 1 个 token
    next_id = argmax(logits)`,
      source: [`${I}m01_kv_cache/demo.py:generate_with_cache`, `${I}m02_paged_attention/paged_attention.py:paged_attention`],
      run: 'python -m llm_infer.m01_kv_cache.demo',
    },

    'infer-scheduler': {
      title: '调度与 prefill · 每一步都重新组 batch',
      subtitle: '读完你能说清"每步 token 预算"这一个旋钮, 怎么同时拧动 TBT 和 TTFT。',
      tldr: '静态 batch 要等最长的那条; 连续批把调度粒度从"一个请求"缩到"一步前向", 每步重问一遍谁进谁出。一条长 prompt 会把所有正在 decode 的用户卡住, 所以再加两道。chunked prefill 把长 prompt 切块混进 decode 的 batch, 最大卡顿 297 → 52 ms。P/D 分离干脆把两类负载放到不同节点。',
      question: '队首请求因为 block 不够而进不来, 这一步调度器该干什么?',
      code: 'llm_infer/m03_continuous_batching · m06_chunked_prefill · m15_pd_disaggregation',
      points: [
        {
          title: 'batch = [(seq, n)]',
          key: true,
          body: '一个列表说清了全部: n>1 是 prefill (或它的一个 chunk), n=1 是 decode, 两者可以出现在同一步。没有"prefill 步"和"decode 步"之分, 只有"这一步谁算几个 token"。',
        },
        {
          title: '永远不能返回空 batch',
          body: '队首拿不到 block 时如果直接返回空, running 就不前进 → 没有序列结束 → block 永远不释放 → 队首永远进不来。修法只有一行: prefill 进不来就落到 decode, 也就是 _admit(budget) or _schedule_running(budget) 的那个 or。抢占也只踢最年轻的, 保证最老的那条总能前进。',
        },
        {
          title: 'token 预算就是 TBT 上限',
          body: '代价模型 step_ms = 20 + 0.25 × 本步 token 数, 所以每步算多少 token 直接封顶用户两个 token 之间的间隔。B=128 时 decode 用户最大卡顿 297 → 52 ms, 代价是那条长请求自己的 TTFT 从 276 涨到 445 ms —— 不是白赚。',
        },
      ],
      links: [
        { from: 'BlockManager.can_allocate', to: 'Scheduler._admit', body: '接不接这条请求, 先问 block 够不够。' },
        { from: 'max_batch_tokens', to: '每步耗时 → TBT', body: '预算是 TBT 与 TTFT 之间的那个旋钮, 调小压卡顿、调大保首 token。' },
        { from: 'KVLink.send', to: 'P/D 分离', body: 'prefill 节点把 KV 按 byte 打包发给 decode 节点; 链路按 bit (Gbps) 标, 少除一个 8 就低估 8 倍。' },
      ],
      sourceRows: [
        { concept: '每步重组 batch', code: 'm03_continuous_batching/scheduler.py:Scheduler.schedule', takeaway: 'decode 优先时先给每条 running 1 个 token, 剩余预算切给 prefill chunk。' },
        { concept: '抢占', code: 'Scheduler._preempt', takeaway: 'block 全还、num_computed 归 0、回 waiting 队首; 已生成的文本原样保留, 回来时连同它一起重算 KV。' },
        { concept: 'TBT 代价模型', code: 'm06_chunked_prefill/chunked_prefill.py:simulate', takeaway: 'step_ms = 固定开销 + 每 token 开销 × batch token 数, 是模型不是实测; 预算 B 决定 TBT 上限。' },
        { concept: '分块不改结果', code: 'chunked_prefill(lm, ids, chunk)', takeaway: 'chunk ≥ 16 时 logits 与整段 prefill 逐位相同 (chunk=8 差 2.86e-06, 只来自分块求和顺序); 分数矩阵峰值 4096 → 512。' },
        { concept: '干扰有多大', code: 'm15_pd_disaggregation/pd.py:KVLink', takeaway: '同卡混跑时被长 prefill 插队的那一步, decode 延迟 0.253 → 4.68 ms (18× 抖动); 分离后回到 0.253 ms。' },
      ],
      snippetTitle: '调度骨架 (decode 优先 + 分块 prefill)',
      snippet: `def schedule(self):
    budget = cfg.max_batch_tokens
    batch = self._schedule_running(budget)   # 每条 running 先拿 1 个 token
    budget -= sum(n for _, n in batch)       # block 不够 → 抢占最年轻的
    if not self._just_preempted:
        batch += self._admit(budget)         # 剩余预算切给 prefill chunk
    return batch                             # [(seq, n_tokens)]`,
      source: [`${I}m03_continuous_batching/scheduler.py:schedule`, `${I}m15_pd_disaggregation/pd.py:KVLink`],
      run: 'python -m llm_infer.m03_continuous_batching.demo',
    },

    'infer-decode-control': {
      widgets: ['SoftmaxTempLab'],
      title: '解码加速与采样 · 更少 target 调用, 更可控的分布',
      subtitle: '读完你能解释为什么"让小模型先猜"不会让输出变差, 以及采样参数的先后顺序为什么会改结果。',
      tldr: 'decode 每步只出 1 个 token 却要把整份权重读一遍, 算力大量闲着。投机解码让便宜的 draft 连猜 K 个, target 一次 forward 验 K+1 个槽位: greedy 下逐位比 argmax, 采样下用 rejection sampling, 输出分布一点不变。采样这一侧则是 logits 上的一串后处理, 顺序会改结果。',
      question: '为什么投机解码在采样模式下仍然"无损"—— 输出分布和只用 target 采样完全一样?',
      code: 'llm_infer/m07_speculative_decoding · m10_sampling (EAGLE / 树形投机、语法约束、attention sinks 各有独立章节)',
      points: [
        {
          title: '一次验 K+1 个槽位',
          body: 'target 喂 [out[-1], d_0..d_{K-1}], 因果 mask 让第 i 行只看到 d_{<i}; 接受最长的正确前缀, 全对再白送 1 个 bonus token。draft 与 target 完全一致时, 48 个 token 只要 11 次 target 调用 (4.36×); 蒸馏式 draft 1.85×, 随便找个小模型 1.12×。',
        },
        {
          title: 'rejection sampling 让它无损',
          key: true,
          body: '以 min(1, p/q) 接受 draft 给的 token; 拒绝就从归一化残差 max(0, p−q) 重采样并停止。两步合起来恰好还是 p, 所以 draft 再差也只影响速度。demo 跑了 χ² 检验: 正确规则 ≤ 13.6 (临界值 37.7), 故意改成"全收 draft"立刻涨到 585。',
        },
        {
          title: '采样是一串顺序敏感的 filter',
          body: 'rep penalty → temperature → top-k → top-p → min-p → Gumbel-max。温度在前, 同样的 p 会留下更多 token; repetition penalty 对负 logit 要乘不是除, 一律除会把 token 9 的概率从 0.0076 抬到 0.0144, 反而鼓励重复。各框架顺序不完全一致, 同一组参数跨框架结果可能不同。',
        },
      ],
      links: [
        { from: 'sample.py 的 temperature/top-k', to: 'm10_sampling', body: '基础采样长成一整套服务端参数。' },
        { from: 'accept_greedy / accept_sampling', to: 'EAGLE · 树形投机', body: '后面两章只换 drafter 和验证形状, 接受规则原样复用。' },
        { from: 'truncate_kv', to: 'KV 回滚', body: '被拒 draft 的 KV 直接截掉, 从不重新 prefill —— 因果性保证前缀的 KV 与后面无关。' },
      ],
      sourceRows: [
        { concept: 'greedy 接受', code: 'm07_speculative_decoding/speculative.py:accept_greedy', takeaway: 'target 的 argmax 与 draft token 逐位比对, 第一个不一致处停下, 返回 target 自己的 token。' },
        { concept: '采样接受', code: 'm07_speculative_decoding/speculative.py:accept_sampling', takeaway: 'u < p_t/p_d 就接受; 否则从 max(0, p_t − p_d) 归一化后重采样。全接受时 bonus 直接采自 target。' },
        { concept: '期望产出', code: '(1 − α^(K+1)) / (1 − α)', takeaway: '每次 target 调用的期望 token 数; α 是每 token 接受率。K 再大也被 1/(1−α) 封顶, 这正是树形投机的动机。' },
        { concept: '采样顺序', code: 'm10_sampling/samplers.py:sample', takeaway: '六步链条里每一步都是"把被砍的置 -inf", 所以可以任意串联。' },
        { concept: 'Gumbel-max 是精确的', code: 'argmax(log p + G), G = −log(−log U)', takeaway: '与 multinomial 同分布, TV 0.0064 vs 参照噪声 0.0066; 只有逐元素运算 + 一次 argmax, 整个 batch 一个 kernel。' },
      ],
      snippetTitle: 'rejection sampling 接受规则',
      snippet: `def accept_sampling(d_tokens, d_probs, t_probs, rng):
    for i, tok in enumerate(d_tokens):
        if rng.random() < t_probs[i, tok] / d_probs[i, tok]:
            continue                                  # 以 min(1, p/q) 接受
        residual = maximum(t_probs[i] - d_probs[i], 0)
        return i, sample(residual / residual.sum())   # 拒绝: 从残差重采样并停止
    return len(d_tokens), sample(t_probs[-1])         # 全接受: bonus 采自 target`,
      source: [`${I}m07_speculative_decoding/speculative.py:accept_sampling`, `${I}m10_sampling/samplers.py:sample`],
      run: 'python -m llm_infer.m07_speculative_decoding.demo',
    },

    'infer-compute': {
      title: '算子与调度开销 · 一样的数学, 少搬几趟',
      subtitle: '读完你能说清 FlashAttention 快在哪一步, 以及 CUDA Graph 省的是谁的时间。',
      tldr: 'FlashAttention 把 Q 和 K/V 都切块, 用 online softmax 增量维护最大值、分母和输出, 从不把 T×T 的分数矩阵写进显存。工作集恒为 64×64, T=4096 时比 T² 小 4096 倍, 结果与朴素实现只差 1.78e-06。CUDA Graph 把一步几百次 kernel 提交压成 1 次。推理 TP 把每层权重切给多张卡, 一个 block 只需 2 次 all-reduce。',
      question: 'FlashAttention 的 FLOPs 一点没少, 凭什么还能快?',
      code: 'llm_infer/m09_tensor_parallel · m11_flash_attention · m12_cuda_graph (量化见「INT4 · AWQ · KIVI」)',
      points: [
        {
          title: 'softmax 的分母可以增量维护',
          key: true,
          body: '这是全部技巧。每个 query 行只带三个运行量: 最大值 m、分母 l、输出 O。来一块新分数就更新它们, 出现更大的 max 时把旧的和整体乘 exp(m_old − m_new) 换基准。于是每个 tile 只进一次 SRAM, 从不物化 T×T —— 快不是因为算得少, 是因为搬得少。',
        },
        {
          title: 'LSE 是可以拼接的凭证',
          body: 'lse = m + log l。拿两段各自的 (O, lse) 就能精确合成全量结果 (误差 2.68e-07); 直接取平均则错到 0.212, 因为两段的 softmax 质量不相等。ring attention、chunked prefill、跨卡分段 KV 全靠这一个原语。',
        },
        {
          title: 'CUDA Graph 省的是 host',
          body: '固定形状的 decode 捕获一次, 之后每步只 replay 一次, kernel 本身一点没变快。本仓库的 launch 开销是显式注入的参数: 0 / 10 / 50 / 200 µs 分别对应 1.3× / 6.4× / 12.2× / 15.0× —— 这些加速比是这个代价模型的产物, 不是实测。batch 越大单 kernel 越久, 占比越小, 所以大 prefill 通常回退 eager。',
        },
      ],
      links: [
        { from: '朴素 attention', to: 'flash_attention', body: '数学上严格相等 (30 组配置最坏 1.78e-06), 省的是对 N×N 中间矩阵的反复读写。' },
        { from: 'merge_attention(O, lse)', to: 'chunked prefill / ring attention', body: '能分段算完再合并, prefill 才切得动、KV 才跨得了卡。' },
        { from: 'tp_block', to: '多卡推理', body: '列切→行切配对, 中间结果天然是切开的, 只在子层末尾 all-reduce 一次。' },
      ],
      sourceRows: [
        { concept: 'online softmax', code: 'm11_flash_attention/flash_attention.py:flash_attention', takeaway: 'm / l / O 随 KV block 增量更新; 外层 Q 块、内层 K/V 块 (FA-2), Q 块的状态常驻 SRAM 只写回一次。' },
        { concept: 'causal 整块跳过', code: 'flash_attention(..., causal=True)', takeaway: 'K 块最早的 key 晚于 Q 块最晚的 query 就整块不算。T=4096 时 2016/4096 块被跳过 (49.2%), 省计算靠跳块不靠 mask。' },
        { concept: '分段合并', code: 'm11_flash_attention/flash_attention.py:merge_attention', takeaway: '按 lse 加权合并两段输出; 只需传 (O, lse), 不传 T×T。' },
        { concept: '图只认地址', code: 'm12_cuda_graph/graph.py:CudaGraph', takeaway: 'host 提交 16 → 1 次, 输出逐位相同。static_input 必须拷贝写入; 重新绑定名字的写法算的还是旧数据 (demo 复现了这个 bug)。' },
        { concept: 'TP 切在 head 边界', code: 'm09_tensor_parallel/parallel_linear.py:tp_block', takeaway: '每 block 2 次 all-reduce, 载荷恒为 T×D (8192 B) 与 tp 无关; 切进单个 head 内部会错 5.63e-01, 因为 softmax 不能跨 rank 拆。' },
      ],
      snippetTitle: 'online softmax 的三个运行量',
      snippet: `m_new = maximum(m, max(S_block))
P_b = exp(S_block - m_new)
l = exp(m - m_new) * l + sum(P_b)       # 旧分母换基准
O = exp(m - m_new) * O + P_b @ V_block  # 旧输出同样换基准
m = m_new
# 收尾: lse = m + log(l); out = O / l`,
      source: [
        `${I}m11_flash_attention/flash_attention.py:flash_attention`,
        `${I}m11_flash_attention/flash_attention.py:merge_attention`,
        `${I}m12_cuda_graph/graph.py:CudaGraph`,
      ],
      run: 'python -m llm_infer.m11_flash_attention.demo',
    },

    'infer-engine': {
      title: 'mini-vLLM 引擎 · 把前面 22 章接成一个循环',
      subtitle: '读完你能在 vLLM 的 step() 里认出每一行对应前面哪一章。',
      tldr: 'Engine.step 永远是同四件事: 调度 → 前向 → 采样 → 后处理。KV 不挂在序列上, 而是写进全局分页 pool, 所以前缀命中的 block 真的跳过前向 —— demo 里 305 个待算 token = 233 实算 + 72 命中, 同一批 prompt 再来一遍只实算 105。9 个 block 的小池逼出 4 次抢占, 输出仍与朴素 greedy 逐 token 相同。',
      question: '为什么说推理引擎首先是调度器和资源管理器, 其次才是 model.forward 的包装?',
      code: 'llm_infer/full_engine/engine.py · llm_infer/full_engine/model_runner.py · llm_infer/m13_lora_serving/lora.py',
      points: [
        {
          title: '真分页 KV pool',
          body: 'KV 写进 pool[layer] (num_blocks, block_size, D), attention 经 block_table 读回。所以"命中"是真的没算: ModelRunner.run 只对 ids[start_pos:] 做 Q/K/V, start_pos 之前的 KV 直接从 pool 里取 —— 它们来自前缀命中、上一个 chunk 或之前的 decode 步。',
        },
        {
          title: '难的只有调度那一步',
          key: true,
          body: '前向、采样、后处理都是固定动作。调度要同时管住三件事。一是 block 够不够, 不够就抢占最年轻的。二是队首会不会饿死, 进不来必须落到 decode。三是这一步算多少 token, 用预算封顶每步耗时。引擎的复杂度全在这里, 主循环只有四行。',
        },
        {
          title: '账必须对得上',
          body: 'prefix_hit_tokens + tokens_computed 必须等于全部需要 KV 的 token 数, 抢占后的重算也如实计入。这条 assert 是"省下的算力是真的"的唯一证据 —— 命中了却照样整段 prefill 的引擎, 账一对就露馅。',
        },
      ],
      links: [
        { from: 'Engine.add_request', to: 'Scheduler.waiting', body: 'prompt encode 后进入调度系统, 每条请求绑定自己的 SamplingParams。' },
        { from: 'Scheduler._admit', to: 'PrefixCache.match_prefix', body: '命中的 token 不占预算、不用算, seq.num_computed 直接从 n_hit 起步。' },
        { from: 'ModelRunner.run', to: 'm02.paged_attention', body: 'q 只有新 token 那几行, K/V 是经页表读回的整段前缀 (含别人算好的公共部分)。' },
        { from: 'Multi-LoRA (m13)', to: '同一 batch 多个 adapter', body: '底模那次 gemm 全 batch 共享, 每个 token 按自己的 adapter id 取 A/B。1000 个 adapter: 合并权重要 24.58 MB, 不合并只要 2.58 MB。' },
      ],
      sourceRows: [
        { concept: '一步', code: 'full_engine/engine.py:Engine.step', takeaway: 'schedule → 逐序列 run → 追平了才 sample → postprocess, 一共四行。' },
        { concept: '只算新 token', code: 'full_engine/model_runner.py:ModelRunner.run', takeaway: 'run(ids, block_table, start_pos): Q 只有 len(ids) − start_pos 行, K/V 是完整的 len(ids) 行。' },
        { concept: '中途不采样', code: 'done_prefill = num_computed + n == num_tokens', takeaway: 'prefill chunk 最后一个位置预测的是 prompt 里已知的下一个 token, 没有新 token 可采, 所以返回 None。' },
        { concept: '抢占', code: 'm03_continuous_batching/scheduler.py:Scheduler._preempt', takeaway: '还 block + 回队首, output_ids 原样保留; 只踢最年轻的, 所以最老的序列永远能前进, 不会活锁。' },
        { concept: '统计', code: 'Engine.report_stats', takeaway: 'steps / tokens_computed / prefix_hit_tokens / preempt / pool, 一眼看完这一轮的资源账。' },
        { concept: 'fuzz', code: 'demo [5]: 30 组随机配置', takeaway: '214 条请求、34 次抢占, 全部与 generate_greedy 逐 token 相同, 无活锁、无 block 泄漏。' },
      ],
      snippetTitle: 'Engine.step 的四件事',
      snippet: `def step(self):
    batch = self.scheduler.schedule()            # ① 调度: [(seq, n)], prefill chunk 与 decode 混批
    tokens = []
    for seq, n in batch:
        ids = seq.all_ids[:seq.num_computed + n]
        logits = self.runner.run(ids, block_table(seq), seq.num_computed)   # ② 前向: 只算 n 个
        caught_up = seq.num_computed + n == seq.num_tokens
        tokens.append(sample(logits) if caught_up else None)                # ③ 采样
    return self.scheduler.postprocess(batch, tokens)                        # ④ 后处理`,
      source: [`${I}full_engine/engine.py:step`, `${I}full_engine/model_runner.py:run`],
      run: 'python -m llm_infer.full_engine.demo',
    },

    // ---------------- 深入九章 ---------------- //
    'infer-prefix-radix': {
      title: '前缀复用 · 从 block hash 到 radix tree',
      subtitle: '多轮对话、few-shot、共享 system prompt —— 请求之间的公共前缀只该算一次。',
      tldr: 'KV 只依赖它之前的 token, 所以相同前缀的 KV 可以跨请求复用。m04 用链式 block hash, 只能命中整块; m05 用 radix tree, 能命中任意长度, 代价是要处理边分裂。容量满了从叶子按 LRU 驱逐。',
      question: '驱逐时为什么只能从叶子开始, 不能直接扔掉最久没用的中间节点?',
      code: 'llm_infer/m04_prefix_cache/prefix_cache.py · llm_infer/m05_radix_cache/radix_tree.py',
      points: [
        {
          title: '一个 hash 标识一整段前缀',
          key: true,
          body: 'h_i = H(h_{i-1} ‖ block_i)。父 hash 参与计算, 所以同一段 token 接在不同前缀后面不会误命中 —— 因为 attention 让位置 i 的 KV 取决于它前面的全部 token。查表 O(块数), 但只能命中 ⌊n/bs⌋·bs 个 token。',
        },
        {
          title: 'radix tree 换到 token 粒度',
          body: '边上存一段 token 和等长的 KV 槽位, 新请求沿树走到最长公共前缀。分叉落在边中间时 _split 把 (tokens, slots) 同步切成两半, 新的中间节点继承 ref_count。前 100 个 token 相同、bs=16 时: 链式 hash 命中 96 个, radix 命中全部 100 个。',
        },
        {
          title: '只驱逐叶子',
          body: '中间节点的 KV 被它所有后代依赖, 先扔它, 后代就全部作废 —— 前缀必须从根连续走下来。所以只挑 ref_count=0 的叶子, 按 last_access 从旧到新。',
        },
      ],
      links: [
        { from: 'BlockManager.ref_count', to: 'PrefixCache', body: '共享 block 靠引用计数, 最后一个使用者释放后才可回收。' },
        { from: 'RadixCache.match_prefix', to: 'Scheduler._admit', body: '命中的 token 不占 token 预算, num_computed 从 n_hit 起步。' },
        { from: 'radix tree', to: '分层 KV offload', body: '被 GPU 挤出去的前缀不丢弃, 降级到 CPU / 磁盘。' },
      ],
      sourceRows: [
        { concept: '链式 hash', code: 'm04_prefix_cache/prefix_cache.py:_block_hash', takeaway: 'parent_hash 参与计算, 所以一个 hash 唯一标识"到第 i 块为止"的整段前缀。' },
        { concept: '至少留一个', code: 'match_prefix 只匹配 (len−1)//bs 块', takeaway: '前缀全命中也要真算最后一个 token, 否则拿不到 logits, 首 token 无从采起 —— cache 里存的是 KV, 不是 logits。' },
        { concept: '最长前缀匹配', code: 'm05_radix_cache/radix_tree.py:RadixCache.match_prefix', takeaway: '按子边的首 token 索引孩子, 再沿边逐 token 比较。' },
        { concept: '边分裂', code: 'RadixCache._split', takeaway: 'tokens 与 slots 必须同步切; 新中间节点继承原节点的 ref_count。' },
        { concept: '驱逐', code: 'RadixCache.evict', takeaway: '只看叶子, 按 last_access 从旧到新。' },
      ],
      snippetTitle: 'radix 匹配 + 分裂',
      snippet: `def match_prefix(self, ids):
    node, i, slots = self.root, 0, []
    while i < len(ids) and ids[i] in node.children:
        child = node.children[ids[i]]
        n = lcp(child.edge_tokens, ids[i:])      # 沿这条边能走多远
        slots += child.slots[:n]
        if n < len(child.edge_tokens):
            child = self._split(child, n)        # 分叉落在边中间 → 切成两段
        node, i = child, i + n
    return slots                                 # 这些 KV 槽位直接复用, 不用前向`,
      source: [`${I}m05_radix_cache/radix_tree.py:RadixCache`, `${I}m04_prefix_cache/prefix_cache.py:PrefixCache`],
      run: 'python -m llm_infer.m05_radix_cache.demo',
    },

    'infer-structured-output': {
      title: 'token 级语法约束 · 把 FSM 预编译成 mask 表',
      subtitle: '读完你能解释为什么"合法字符"和"合法 token"是两回事。',
      tldr: '字符级 FSM 能保证语法合法, 但模型吐的是多字符 token (如 \'":\' 、\'true\')。对每个 (状态, token) 离线试走一遍: 整段字符都走得通才算合法, 同时记下落点状态。在线每步只查一行表, 把非法 token 的 logit 置 −inf。',
      question: '为什么不能每步在线对 V 个 token 逐字符试走一遍 FSM?',
      code: 'llm_infer/m14_structured_output/grammar.py',
      points: [
        {
          title: '合法性是 token 级的, 不是字符级的',
          key: true,
          body: '真实词表里 \'":\' 是一个 token, 一口气跨过"结束 key"和"进入 value"两个状态。判定只有一条标准: 这段字符从状态 s 出发能不能全部走通。next_state[s, t] 记下落点, −1 表示走不通; mask_table = next_state ≥ 0。',
        },
        {
          title: '为什么必须离线编译',
          body: '在线现算是 O(V·len) 的纯 CPU 开销, 卡在 GPU 前向和采样中间, 词表十几万时直接吃掉 decode 延迟。FSM 状态数有限, 离线各跑一遍就够, 在线只查一行, O(1)。',
        },
        {
          title: '还要能收尾',
          body: 'need[s, t] = 选了 t 之后最少还要几个 token 才能结束。剩余长度不够时提前屏蔽那些"收不了尾"的 token, 保证输出一定是完整可解析的 JSON, 而不是被长度上限截断的半截。',
        },
      ],
      links: [
        { from: 'JsonFSM (字符级)', to: 'compile_char_dfa', body: '先把字符级 FSM 枚举成 DFA 转移表。' },
        { from: 'token_row', to: 'compile_token_table', body: '对每个状态跑一遍"逐 token 逐字符试走", 存成 (S, V) 表。' },
        { from: 'mask_table[s]', to: 'm10 sample', body: 'mask 也只是一个把 logit 置 -inf 的 filter, 插在同一条采样链上, 之后的温度 / top-p 照常。' },
      ],
      sourceRows: [
        { concept: '词表', code: 'm14_structured_output/grammar.py:build_vocab', takeaway: '单字符 + 类 BPE 多字符片段, 特意含跨语法边界的 \'":\' 、\'e"\' 和永远非法的垃圾 token。' },
        { concept: '试走', code: 'm14_structured_output/grammar.py:token_row', takeaway: 'EOS 只在接受态合法; 其它 token 逐字符走转移表, 中途走不通就整个 token 作废。' },
        { concept: '预编译', code: 'm14_structured_output/grammar.py:compile_token_table', takeaway: 'next_state / mask_table / need 三张表, 一次算好反复用。' },
        { concept: '光靠 prompt 不够', code: 'logits[~mask] = -inf', takeaway: 'prompt 只能提高概率; 只有在 logits 上把非法 token 置 −inf, 概率才严格为 0。' },
        { concept: '与 xgrammar 的差距', code: 'CFG / 下推自动机', takeaway: '有栈的语法只能预编译"与栈无关"的那部分 token, 其余运行时再查。' },
      ],
      snippetTitle: '离线编译 + 在线查表',
      snippet: `# 离线: 每个状态 × 每个 token 试走一遍
for s in states:
    for t, piece in enumerate(vocab):
        cur = s
        for ch in piece:                  # 多字符 token 可以连跨几个状态
            cur = trans[cur].get(ch, -1)
            if cur < 0: break
        next_state[s, t] = cur            # -1 = 非法

# 在线: 每步 O(1)
logits[~mask_table[state]] = -inf
tok = sample(logits)
state = next_state[state, tok]`,
      source: [`${I}m14_structured_output/grammar.py:token_row`, `${I}m14_structured_output/grammar.py:compile_token_table`],
      run: 'python -m llm_infer.m14_structured_output.demo',
    },

    'infer-quant-awq': {
      title: 'INT4 · AWQ · KIVI · 比特越低, "怎么分组"越要紧',
      subtitle: 'decode 是带宽受限的: 权重和 KV 越小, 每步要搬的字节越少。',
      tldr: 'group-wise INT4 让每 g 个权重共用一组 scale/zero, 离群值只污染自己那一组。AWQ 更进一步: 输出误差 = Σ x_i·ΔW_i, 所以量化前把激活大的输入通道放大 s_i、激活同步缩小, 数学等价但误差从 0.0759 降到 0.0273。KV 量化里 K 有固定的离群通道, 按通道分组; V 没有, 按 token 分组。',
      question: '量化要最小化的为什么是 ‖XW − XŴ‖ 而不是 ‖W − Ŵ‖?',
      code: 'llm_infer/m08_quantization/{int8_weight.py,int4_awq.py,kv_quant.py}',
      points: [
        {
          title: '分组把离群值关起来',
          body: '一组一套 scale/zero, 一个离群值只撑大自己这一组的 range。g 越小越准, 但 scale/zero 的额外开销越大: g=128 时是 4 + 32/128 = 4.25 bit/权重, g=32 时约 4.5 bit。',
        },
        {
          title: '重要的是输出误差, 不是权重误差',
          key: true,
          body: '模型在意的是 XŴ 和 XW 差多少, 而误差 = Σ x_i·ΔW_i —— 同样的舍入误差, 乘上大激活后对输出的影响就大。AWQ 用恒等式 XW = (X/s)·(s⊙W): 把激活大的那些行放大 s 倍, 该行相对舍入误差缩小约 s 倍。代价是撑大了同组的 range, 所以 α 要在校准集上网格搜 (demo 最优 α=0.4, α=0 就是 RTN)。',
        },
        {
          title: 'K 和 V 的离群结构不一样',
          body: 'K 的少数通道在所有 token 上都是大值: 按 token 分组时每一行的 scale 都被它撑大, 误差 0.18; 按通道分组把它关在自己组里, 误差 0.026。V 没有这种固定离群通道, 按 token 分组还能在每个新 token 写入时独立量化, 不用回头改旧的。',
        },
      ],
      links: [
        { from: 'quantize_int8 (per-channel)', to: 'quantize_groupwise', body: '从每通道一组, 缩小到每 g 个权重一组。' },
        { from: 'mean|x_i|', to: 's_i = mean|x_i|^α', body: '只用校准激活的逐通道统计量, 不需要反向传播, 所以 AWQ 很便宜。' },
        { from: 'KV 量化', to: 'KV 体积一章', body: '量化减的是每个元素的字节数, GQA/MLA 减的是元素个数, 两者相乘。' },
      ],
      sourceRows: [
        { concept: '分组量化', code: 'm08_quantization/int4_awq.py:quantize_groupwise', takeaway: 'reshape 成 (D_in/g, g, D_out), 在 g 那一维上统计 min/max。' },
        { concept: '真正的目标', code: 'm08_quantization/int4_awq.py:output_err', takeaway: '‖XW − XŴ‖_F / ‖XW‖_F, 拿校准激活算出来的相对输出误差。' },
        { concept: 'AWQ 搜索', code: 'm08_quantization/int4_awq.py:awq_quantize', takeaway: 'Ŵ = Q(s ⊙ W) / s; 1/s 离线折进上一层的权重, 推理时零额外开销。' },
        { concept: 'KV 三种分组', code: 'm08_quantization/kv_quant.py:quantize_kv', takeaway: '三种 scheme 的实现只差 reshape 与 axis, 误差却差一个数量级。' },
      ],
      snippetTitle: 'AWQ: 量化前先缩放',
      snippet: `act = mean(abs(X_calib), axis=0)          # (D_in,) 每个输入通道的激活幅度
for alpha in arange(0, 1.01, 0.1):        # α = 0 就是 RTN
    s = act ** alpha
    s = s / sqrt(s.max() * s.min())       # 几何中心归一, 不整体放大 W
    W_hat = quant_dequant(W * s[:, None]) / s[:, None]
    err[alpha] = norm(X @ W - X @ W_hat) / norm(X @ W)
best = argmin(err)`,
      source: [`${I}m08_quantization/int4_awq.py:awq_quantize`, `${I}m08_quantization/kv_quant.py:quantize_kv`],
      run: 'python -m llm_infer.m08_quantization.demo',
    },

    'infer-kv-footprint': {
      title: 'KV 体积 · MHA / MQA / GQA / MLA',
      subtitle: '同一套 attention 数学, 四种"cache 里存什么"—— 直接决定一张卡能同时服务多少条序列。',
      tldr: '每 token KV 字节 = 2·n_kv·d_head·n_layer·bytes; MLA 只缓存 latent, 是 (d_c + d_rope)·n_layer·bytes。LLaMA-2-7B (MHA) 512 KiB → LLaMA-3-8B (GQA-8) 128 KiB → DeepSeek-V3 (MLA) 68.6 KiB, 比同尺寸 MHA 省 56.9×。',
      question: 'MLA 的 latent 为什么不能带 RoPE, 而要另外留一份解耦的 RoPE key?',
      code: 'llm_infer/m18_kv_attention_variants/attention_variants.py',
      points: [
        {
          title: '一个参数 n_kv 分出前三种',
          key: true,
          body: 'n_kv = n_head 是 MHA, 1 < n_kv < n_head 是 GQA, n_kv = 1 是 MQA (MLA 是另一条路: 改存 latent)。每组 query 头靠广播共享 KV, kernel 里不真复制。把显存预算除以每 token 字节、再除以上下文长度, 得到的就是并发上限 —— 结构选择在推理侧直接变成吞吐。',
        },
        {
          title: 'MLA 的 cache 里只有 latent',
          body: '只存 (T, d_c) 和 (T, d_rope) 两份。per-head 的 K/V 在 attention 时才由 C @ W_UK / W_UV 现场还原, 从不进 cache —— 所以公式里没有那个 "2·", K 和 V 共用同一个 latent。',
        },
        {
          title: 'absorb 形式',
          body: 'q·(C W_UK)ᵀ = (q W_UKᵀ)·Cᵀ: 把 W_UK 乘到 q 上, K 就是 cache 本身。于是 MLA 的 decode 等价于一个 head_dim = d_c + d_rope 的 MQA。latent 一旦带上 RoPE, 位置相关的旋转就夹在 W_UK 前面, 这个吸收做不成了。',
        },
      ],
      links: [
        { from: '阶段 2 注意力演进', to: 'KV bytes/token', body: '当年为省参数做的结构选择, 在推理侧变成显存账。' },
        { from: 'bytes/token', to: 'BlockManager.num_blocks', body: '显存 ÷ 每 token 字节 = 能放多少 token = 并发上限。' },
        { from: 'MLA absorb', to: 'FlashMLA', body: '真实 kernel 走吸收形式, 全程不物化 per-head K/V。' },
      ],
      sourceRows: [
        { concept: 'GQA 公式', code: 'm18_kv_attention_variants/attention_variants.py:kv_bytes_per_token', takeaway: '开头那个 2 = K 和 V 两份。' },
        { concept: 'MLA 公式', code: 'm18_kv_attention_variants/attention_variants.py:mla_bytes_per_token', takeaway: '没有 "2·": K/V 共用同一个 latent。' },
        { concept: 'latent decode', code: 'MLALayer.forward', takeaway: 'latent 不加 RoPE, 位置信息全放在另一份解耦的 RoPE key 里。' },
        { concept: '对拍', code: 'cache_nbytes', takeaway: 'demo 断言实际 cache 字节数 == 公式值, 公式不是纸上谈兵。' },
      ],
      snippetTitle: '两条公式',
      snippet: `def kv_bytes_per_token(n_kv, d_head, n_layer, nbytes=2):
    return 2 * n_kv * d_head * n_layer * nbytes     # 2 = K 和 V

def mla_bytes_per_token(d_c, d_rope, n_layer, nbytes=2):
    return (d_c + d_rope) * n_layer * nbytes        # K/V 共用 latent

# LLaMA-2-7B  MHA 32kv×128×32L → 512 KiB
# LLaMA-3-8B  GQA  8kv×128×32L → 128 KiB
# DeepSeek-V3 MLA (512+64)×61L → 68.6 KiB
max_seqs = kv_budget_bytes // (bytes_per_token * context_len)`,
      source: [`${I}m18_kv_attention_variants/attention_variants.py:kv_bytes_per_token`, `${I}m18_kv_attention_variants/attention_variants.py:mla_bytes_per_token`],
      run: 'python -m llm_infer.m18_kv_attention_variants.demo',
    },

    'infer-attention-sinks': {
      title: 'Attention sinks · 流式长上下文的有界 KV',
      subtitle: '纯滑动窗口一旦把开头几个 token 挤出去, 模型立刻崩 —— 因为 softmax 必须把那个 1 分给某个人。',
      tldr: 'softmax 的权重和恒为 1, 当前 token 没什么可看时, 多余的注意力会倒在开头几个 token 上, 它们被训练成了垃圾桶 (sink)。StreamingLLM 永远保留开头 n_sink 个 + 最近 window 个; 位置按 cache 槽位重新编号, 所以 K 必须存未旋转的版本。',
      question: 'SinkCache 里的 K 为什么要存 pre-RoPE 的, 和普通 KV cache 正好相反?',
      code: 'llm_infer/m16_attention_sinks/sink_cache.py',
      points: [
        {
          title: 'sink 是 softmax 的排污口',
          key: true,
          body: '开头的 token 对所有后续位置都可见, 于是被学成了"注意力没处放时就倒这儿"。逐出它们, softmax 的分母骤变, 其余权重被迫整体膨胀 —— 这是训练时从未见过的分布, 输出立刻漂掉。和语义无关, 换成别的开头 token 也一样。',
        },
        {
          title: '有界 cache',
          body: 'cache 条目恒 ≤ n_sink + window。超预算时逐出槽位 n_sink, 也就是窗口里最老的那个, sink 那几个永远不动。',
        },
        {
          title: '位置按槽位重编号',
          body: '位置取 cache 槽位 0..L−1, 而不是它在数据流里的绝对位置, 这样永远不会超过训练长度。代价是每步要把 cache 里的 K 整体重转一遍 (O(L·D)), 所以 K 只能存旋转前的。',
        },
      ],
      links: [
        { from: 'KV cache (m01)', to: 'SinkCache', body: '从 O(T) 无限增长变成 O(n_sink + window) 有界。' },
        { from: 'RoPE', to: 'positions = arange(L)', body: '相对位置只看槽位差, 中间 token 被逐出后"距离"就被压缩了。' },
        { from: 'select_blocks 的 forced', to: '稀疏 decode', body: '稀疏选块同样强制保留第 0 块 —— 是同一个 sink。' },
      ],
      sourceRows: [
        { concept: '逐出规则', code: 'm16_attention_sinks/sink_cache.py:SinkCache.append', takeaway: 'keep = [0:n_sink] + [n_sink+1:], 永远去掉槽位 n_sink。' },
        { concept: '重编号', code: 'm16_attention_sinks/sink_cache.py:stream_step', takeaway: 'pos = arange(L); K 每步按当前槽位现转, query 放在最后一个槽位。' },
        { concept: '不是长上下文', code: 'window 之外的 token', takeaway: '被逐出的内容彻底丢失。sink 保证的是"不崩", 不是"记得"; 要记得就得保留 KV (offload) 或者去检索。' },
        { concept: '随机权重上看不到', code: 'TinyLM 的 attention', takeaway: '未训练模型的注意力近乎均匀, 没有 sink 现象; 这一章的效应要在植入 sink 的合成数据上才成立。' },
      ],
      snippetTitle: 'SinkCache 一步',
      snippet: `def append(self, k_raw, v):                 # k_raw: 未旋转的 K
    K = concat([self.k, k_raw]); V = concat([self.v, v])
    if len(K) > self.n_sink + self.window:
        keep = r_[0:self.n_sink, self.n_sink + 1:len(K)]   # 去掉窗口里最老的
        K, V = K[keep], V[keep]
    self.k, self.v = K, V

pos = arange(len(cache))                    # 位置 = 槽位, 不是绝对位置
K = apply_rope(cache.k, positions=pos)      # 每步整体重转
q = apply_rope(q, positions=pos[-1:])`,
      source: [`${I}m16_attention_sinks/sink_cache.py:SinkCache`, `${I}m16_attention_sinks/sink_cache.py:stream_step`],
      run: 'python -m llm_infer.m16_attention_sinks.demo',
    },

    'infer-tree-speculation': {
      title: 'EAGLE 与树形投机 · 让一次 target forward 验更多',
      subtitle: '链式 draft 第一个猜错, 后面 K−1 个全废。两条改进路线: 猜得更准, 或者一次验更多分支。',
      tldr: 'EAGLE 让 draft 吃 target 白送的 hidden state、共享 target 的 lm_head, 接受率更高。树形投机则在每个节点留下 draft 的 top-k 候选长成一棵树, target 用 tree attention mask 一次验完, 接受与 target greedy 一致的最长路径。同样的验证预算: 链 K=15 每次 1.86 token, 树 [3,2,1] 2.58。',
      question: '同一深度的兄弟节点为什么共享同一个 RoPE 位置?',
      code: 'llm_infer/m17_eagle_speculative/eagle.py · llm_infer/m19_tree_speculation/tree_spec.py',
      points: [
        {
          title: '特征级 draft',
          body: 'ĥ_{t+1} = Draft(h_t, emb(x_{t+1})), 再过 target 自己的 lm_head。第 1 步的 h 是 target 验证时白送的真特征, 之后是 draft 自己的 ĥ, 误差逐步累积 —— 所以越靠后的槽位越难接受。',
        },
        {
          title: 'tree mask',
          key: true,
          body: 'anc[i, j] = "j 是 i 的祖先或就是 i"。每个节点看到的恰好是从根到自己那一条链, 等价于把每条路径各跑一次顺序 decode。兄弟之间互不可见, 同深度共享同一个位置。一次 forward 就把整棵树验完。',
        },
        {
          title: '算力换延迟',
          body: 'widths=[3,2,1] 长出 16 个节点, 只要 1 次 target 调用 + 3 次 draft 调用。decode 是带宽受限的, 多验几个 token 几乎不花额外时间; 但大 batch 下算力吃紧时, 这笔账就不再免费。',
        },
      ],
      links: [
        { from: 'm07 accept_greedy', to: 'accept_tree', body: '从"沿一条链比对"变成"从根往下走, 命中哪个孩子就进哪个"。' },
        { from: 'TinyLM.forward(return_hidden)', to: 'EagleDrafter.propose', body: '验证那次 forward 白送了下一轮 draft 要用的特征。' },
        { from: 'gather_kv', to: 'KV 回滚', body: '只留被接受路径上那几行 KV; K 已是 post-RoPE, 挑行就行。' },
      ],
      sourceRows: [
        { concept: '树形状', code: 'm19_tree_speculation/tree_spec.py:tree_shape', takeaway: 'BFS 编号, 父先于子; [3,2,1] → 1+3+6+6 = 16 个节点。' },
        { concept: '树 mask', code: 'm19_tree_speculation/tree_spec.py:tree_mask', takeaway: '上下文全可见, 树内只看祖先和自己; positions = n_ctx + depth。' },
        { concept: '最长路径', code: 'm19_tree_speculation/tree_spec.py:accept_tree', takeaway: '同一父节点的 top-k 候选互不相同, 所以每层至多命中一个孩子。' },
        { concept: 'EAGLE drafter', code: 'm17_eagle_speculative/eagle.py:EagleDrafter', takeaway: '只换 drafter, 验证循环与接受规则完全复用 m07.speculative_decode。' },
        { concept: '实测', code: 'm19 demo', takeaway: '同样的验证 token 数: chain K=15 每轮接受 1.40 (1.86 token/次 target 调用), tree [3,2,1] 1.80 (2.58)。' },
      ],
      snippetTitle: '树 mask + 接受最长路径',
      snippet: `anc = eye(n, dtype=bool)
for i in range(1, n):
    anc[i] |= anc[parents[i]]             # 祖先闭包 (BFS 序: 父先于子)
mask = where(anc, 0, -inf)                # 节点只看祖先 + 自己
logits = target.forward(tokens, kv, positions=n_ctx + depth, mask=mask)

t_pred, path = logits.argmax(-1), [0]
while True:
    kids = where(parents == path[-1])
    hit = [k for k in kids if tokens[k] == t_pred[path[-1]]]
    if not hit: break                     # 断了: t_pred[path[-1]] 就是纠错 token
    path.append(hit[0])`,
      source: [`${I}m19_tree_speculation/tree_spec.py:accept_tree`, `${I}m19_tree_speculation/tree_spec.py:tree_mask`, `${I}m17_eagle_speculative/eagle.py:EagleDrafter`],
      run: 'python -m llm_infer.m19_tree_speculation.demo',
    },

    'infer-kv-offload': {
      title: '分层 KV offload · GPU → CPU → 磁盘',
      subtitle: '多轮对话每一轮的 prompt 都是整段历史, GPU 装不下所有用户的历史。被挤出去的 KV 别扔, 降级存起来。',
      tldr: '搬 1 个 token 的 KV (128 KiB) 走 PCIe 25 GB/s 约 5 µs, 重算约 125 µs —— 搬比算便宜 25 倍。但每次加载还有一笔固定延迟, 链路一慢, 命中反而比重算更贵, 所以每一层都要单独判断"加载还是重算"。以下时间全部来自代价模型, 不是实测。',
      question: '为什么"多加一层缓存"有时反而让 TTFT 变差?',
      code: 'llm_infer/m20_kv_offload/tiered_cache.py',
      points: [
        {
          title: '降级, 不是丢弃',
          body: '每一层一个 LRU。GPU 层溢出的最旧 block 降到 CPU, CPU 溢出降到磁盘; 再次命中时提升回 GPU 层。',
        },
        {
          title: '链断了后面就用不了',
          body: '沿 hash 链逐块查找, 在第一个全层都 miss 的块处停下。后面的块即使还在也不可用 —— 第 4 块的 hash 标识的是"前 4 块的整段前缀", 前缀 KV 不全, 后面接不上。',
        },
        {
          title: '交叉点 n*',
          key: true,
          body: 'load = 固定延迟 + n × 每块搬运, recompute = n × 每块重算, 两条线交在 n* = latency / (每块重算 − 每块加载)。demo 里磁盘层的 n* = 1.54 块: 只命中 1 块时重算更快。所以策略必须是"两者取小", 而不是"命中就加载"。',
        },
      ],
      links: [
        { from: 'm04 PrefixCache', to: 'TieredKVCache', body: '同样的链式 block hash, 只是多了几层存储。' },
        { from: 'Tier.bandwidth_GBps', to: 'load_ms', body: '这里带宽单位是 GB/s (byte); 对照 P/D 分离那一章的 Gbps (bit), 差一个 8。' },
        { from: 'use_load', to: 'TTFT', body: '接一条 0.5 GB/s 的慢远端: "命中就加载"要 68.9 ms, "取小"回到 34.8 ms。' },
      ],
      sourceRows: [
        { concept: '代价模型', code: 'm20_kv_offload/tiered_cache.py:CostModel', takeaway: 'load = latency + bytes / bandwidth; recompute = tokens / 8000 tok/s。' },
        { concept: '降级', code: 'TieredKVCache._insert', takeaway: '溢出的最旧 block 递归插入下一层, 而不是直接删掉。' },
        { concept: '逐层决策', code: 'm20_kv_offload/tiered_cache.py:TieredKVCache.serve', takeaway: 'use_load = (load ≤ recompute), 每层各判断一次。' },
        { concept: 'demo 结果 (模拟)', code: '8 users × 6 turns', takeaway: '平均 TTFT: 只有 GPU 51.5 → 加 CPU 层 34.8 → 再加磁盘层 19.5 ms。' },
      ],
      snippetTitle: 'serve: 逐层"加载 vs 重算"',
      snippet: `where = self.lookup(block_hashes(prompt))     # 每个命中块在哪一层; 首个 miss 处停
ttft = recompute_ms(len(prompt) - len(where) * bs)
for tier, nb in zip(tiers, hits_per_tier):
    load = tier.latency_ms + nb * block_bytes / tier.bandwidth   # 固定延迟 + 搬运
    recompute = recompute_ms(nb * bs)
    ttft += min(load, recompute)              # 慢链路 / 命中太少 → 重算更快
self.store(prompt)                            # 命中的提升回 GPU 层, 溢出的逐层降级`,
      source: [`${I}m20_kv_offload/tiered_cache.py:serve`, `${I}m20_kv_offload/tiered_cache.py:CostModel`],
      run: 'python -m llm_infer.m20_kv_offload.demo',
    },

    'infer-moe-serving': {
      title: 'MoE serving · 专家并行与 EPLB',
      subtitle: '专家分布在多张卡上, 每层都要同步 —— 所以一步的耗时等于最忙那张卡的耗时。',
      tldr: 'token 经 all-to-all 发到专家所在的 rank (dispatch), 算完再发回按 gate 加权 (combine)。路由一倾斜, 热专家所在 rank 的负载能到均值的 2.95×, 平均利用率只剩 34%。EPLB 给热专家加冗余副本、把它的 token 拆开, 把 max/mean 压到 1.02×, 输出逐位不变。',
      question: '为什么只靠"重新摆放专家"不够, 必须复制热专家?',
      code: 'llm_infer/m21_moe_serving/moe.py',
      points: [
        {
          title: '看 max, 不看 mean',
          key: true,
          body: 'EP 每层 combine 都要等齐所有 rank, 所以一步的耗时由最忙的那张卡决定。max/mean = 2.95 意味着平均 rank 只有 1/2.95 ≈ 34% 的时间在干活, 其余都在等。',
        },
        {
          title: '两步贪心',
          body: '① 反复给"每副本负载"最大的那个专家再加一个副本。② 把所有 slot 按负载从大到小, 依次放到当前最轻且还有空位的 rank 上。',
        },
        {
          title: '不可分的热点',
          body: '最热专家的负载 / 平均 rank 负载 = 1.42 > 1: 它一个就超过一张卡该分到的量。单个专家不可分, 怎么摆都不可能均衡 (贪心无副本只能做到 1.48×), 只能复制并拆分它的 token。',
        },
      ],
      links: [
        { from: '阶段 2 MoE 路由', to: 'MoELayer.route', body: '同一个 top-k 路由, 这里只关心它在多卡上造成的负载分布。' },
        { from: '阶段 3 EP all-to-all', to: 'ep_forward', body: '训练可以用辅助 loss 压倾斜; 推理时路由已经定死, 只能靠放置。' },
        { from: 'eplb_placement', to: 'slot_of', body: '有副本的专家把自己的 token 轮流分给各个副本。' },
      ],
      sourceRows: [
        { concept: '路由', code: 'm21_moe_serving/moe.py:MoELayer.route', takeaway: 'gate 只在被选中的 k 个 logit 上做 softmax。' },
        { concept: 'dispatch/combine', code: 'm21_moe_serving/moe.py:ep_forward', takeaway: 'rank_load[r] = 发到 rank r 的 (token, k) 分配数。' },
        { concept: 'EPLB', code: 'm21_moe_serving/moe.py:eplb_placement', takeaway: 'n_rep[argmax(load / n_rep)] += 1, 一次加一个副本。' },
        { concept: '输出不变', code: 'max|Δ| = 0', takeaway: 'EPLB 只改"逻辑专家 → 物理 slot"的映射, 路由和 gate 都没动, 所以输出逐位相同。' },
        { concept: 'demo 结果', code: '16 专家 / 4 rank / top-2', takeaway: 'max/mean: 连续摆放 2.95× → 贪心无副本 1.48× → EPLB 加 4 个副本 1.02×。' },
      ],
      snippetTitle: 'EPLB 两步贪心',
      snippet: `n_rep = ones(E)
for _ in range(n_redundant):
    n_rep[argmax(expert_load / n_rep)] += 1     # ① 最热的"每副本负载"再加一个副本

slot_expert = repeat(arange(E), n_rep)
slot_load = (expert_load / n_rep)[slot_expert]  # 副本间均分 token
for s in argsort(-slot_load):                   # ② 从重到轻
    r = argmin(where(rank_free > 0, rank_load, inf))
    slot_rank[s] = r; rank_load[r] += slot_load[s]; rank_free[r] -= 1`,
      source: [`${I}m21_moe_serving/moe.py:eplb_placement`, `${I}m21_moe_serving/moe.py:ep_forward`],
      run: 'python -m llm_infer.m21_moe_serving.demo',
    },

    'infer-sparse-decode': {
      title: '稀疏注意力 decode · 只读 top-k 个 KV block',
      subtitle: '长上下文 decode 是访存瓶颈: 每步要把全部 KV 读一遍, 而注意力质量其实集中在很少的 token 上。',
      tldr: '把 KV 切成 block, 每块常驻一份很小的摘要 (Quest 用逐维 min/max; NSA/DSA 用压缩 key 或低维 indexer)。用 q 给摘要打分选 top-k 块, 只对它们做 attention。needle 负载上读 3.1% 的 KV 误差就降到 0.023; 但随机权重模型注意力弥散, 这个前提根本不成立。',
      question: 'Quest 为什么用"上界"打分, 而不是直接用 block 的均值?',
      code: 'llm_infer/m22_sparse_attention/sparse_attention.py',
      points: [
        {
          title: '廉价的上界打分',
          body: 'Quest: Σ_i max(q_i·kmin_i, q_i·kmax_i) ≥ 该 block 内任何一个 q·k。上界保证含有"针"的块不会被低估而漏掉, 代价是偏松 (demo 平均松 6.7×)。不漏比估得准重要。',
        },
        {
          title: '两块必须保留',
          body: '第 0 块 (attention sink) 和最后一块 (最近的 token) 永远选中, 并且算在 k 之内。丢掉第 0 块, softmax 的分母就崩了。',
        },
        {
          title: '前提是注意力足够尖',
          key: true,
          body: '收益完全来自"质量集中在少数 block"。k 从 4 加到 8 (读 1.6% → 3.1% 的 KV), 误差从 0.83 断崖掉到 0.023 —— 针全进 top-k 的那一刻其余 KV 就无关紧要了。但在随机权重的 TinyLM 上注意力近乎均匀, Quest 打分并不优于随机选块; 这个方法的收益来自数据分布, 不是算法本身。',
        },
      ],
      links: [
        { from: 'm02 KV block', to: 'block 摘要', body: '分页 KV 天然按 block 组织, 摘要大约是 KV 的 1/16 ~ 1/8。' },
        { from: 'attention sink', to: 'select_blocks 的 forced', body: '首块必须留, 否则 softmax 分母崩。' },
        { from: '阶段 2 DSA', to: 'lightning indexer', body: 'DeepSeek-V3.2 用训练出来的低维 indexer 按 token 选; 这里只有免训练的选块。' },
      ],
      sourceRows: [
        { concept: 'Quest 上界', code: 'm22_sparse_attention/sparse_attention.py:quest_upper_bound', takeaway: 'q_i 为正取 kmax, 为负取 kmin, 逐维求和。' },
        { concept: '选块', code: 'm22_sparse_attention/sparse_attention.py:select_blocks', takeaway: 'forced = {0, nb−1}, 其余按分数取 top-(k−2)。' },
        { concept: '评测', code: 'm22_sparse_attention/sparse_attention.py:evaluate', takeaway: '相对 L2 误差 + 选中 block 覆盖的真实注意力质量 (recall), 两个指标一起看。' },
        { concept: 'demo 结果', code: 'T=4096, 256 blocks', takeaway: 'k=4 (1.6%) 误差 0.83 → k=8 (3.1%) 误差 0.023; 同样读 3.1%, 随机选块误差 1.06。' },
      ],
      snippetTitle: '打分 → 选块 → 只读选中的 KV',
      snippet: `Kb = K.reshape(nb, bs, d)
kmin, kmax = Kb.min(1), Kb.max(1)                   # 常驻显存的摘要 (nb, d)
scores = maximum(q * kmin, q * kmax).sum(-1)        # ≥ block 内任何 q·k

forced = {0, nb - 1}                                # sink 块 + 最近块
rest = [b for b in argsort(-scores) if b not in forced]
blocks = sorted(list(forced) + rest[:k - 2])

idx = (blocks[:, None] * bs + arange(bs)).ravel()   # 只 gather k·bs 个 token
out = softmax(q @ K[idx].T / sqrt(d)) @ V[idx]`,
      source: [`${I}m22_sparse_attention/sparse_attention.py:quest_upper_bound`, `${I}m22_sparse_attention/sparse_attention.py:select_blocks`],
      run: 'python -m llm_infer.m22_sparse_attention.demo',
    },
  },
}
