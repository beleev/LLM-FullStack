// 阶段 5 · llm_infer 章节扩展: 新技术 (m16–m22、AWQ/KIVI、token 级语法、radix) 各成一章,
// 并按字段覆盖 models.js 里已有的 5 章 (修正漂移的文字 + 挂真源码)。
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
    // ---------------- 已有章节: 按字段覆盖 ---------------- //
    'infer-kv-memory': {
      source: [`${I}m01_kv_cache/demo.py:generate_with_cache`, `${I}m02_paged_attention/paged_attention.py:paged_attention`],
    },
    'infer-scheduler': {
      tldr: 'continuous batching 每步重组 batch; chunked prefill 把长 prompt 切块与 decode 混批, 用 token 预算封顶 TBT; block 不够时 recompute 式抢占; P/D 分离把 prefill 和 decode 放到不同资源池。',
      sourceRows: [
        { concept: '每步重组 batch', code: 'm03_continuous_batching/scheduler.py:Scheduler.schedule', takeaway: 'batch = [(seq, n)]: n>1 是 prefill chunk, n=1 是 decode, 可以混在同一步。' },
        { concept: '抢占', code: 'Scheduler._preempt', takeaway: 'block 全还、num_computed 归 0、回 waiting 队首; 已生成文本保留, 之后连同它一起重算。' },
        { concept: 'TBT 代价模型', code: 'm06_chunked_prefill/chunked_prefill.py:simulate', takeaway: '每步耗时 = 固定开销 + 每 token 开销 × batch token 数, 预算 B 决定 TBT 上限。' },
        { concept: '跨节点 KV', code: 'm15_pd_disaggregation/pd.py:KVLink', takeaway: '带宽按 Gbps (bit) 标, KV 按 byte 算: bytes/s = Gbps·1e9/8, 少除这个 8 会低估 8 倍。' },
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
    },
    'infer-decode-control': {
      title: '解码加速与采样 · 更少 target calls, 更可控分布',
      tldr: '投机解码用 draft 连猜 K 个 token, target 一次 forward 验 K+1 个槽位; greedy 下比 argmax, 采样下用 rejection sampling 保持分布不变。sampling 则是 logits 上的一串后处理。(EAGLE / 树形投机、token 级语法约束、attention sinks 各有独立章节。)',
      question: '为什么投机解码在采样模式下仍然"无损"—— 输出分布和只用 target 采样完全一样?',
      code: 'llm_infer/m07_speculative_decoding · m10_sampling (延伸: m14 · m16 · m17 · m19 见后续章节)',
      points: [
        { title: '一次验 K+1 个槽位', body: 'target 喂 [out[-1], d_0..d_{K-1}], 因果 mask 让第 i 行只看到 d_{<i}; 接受最长前缀, 再白送 1 个纠错/bonus token。' },
        { title: 'rejection sampling', body: '以 min(1, p/q) 接受 draft token; 拒绝则从归一化残差 max(0, p−q) 重采样并停止。两步合起来恰好是 p。' },
        { title: '采样顺序', body: 'rep penalty → temperature → top-k → top-p → min-p → 采样。各框架顺序不完全一致, 同一组参数结果可能不同。' },
      ],
      links: [
        { from: 'sample.py temperature/top-k', to: 'm10_sampling', body: '基础采样扩展为服务端参数。' },
        { from: 'accept_greedy / accept_sampling', to: 'm17 · m19', body: 'EAGLE 只换 drafter, 树形投机只换验证形状, 接受规则复用 m07。' },
        { from: 'truncate_kv', to: 'KV 回滚', body: '被拒 draft 的 KV 直接截掉, 从不重新 prefill。' },
      ],
      sourceRows: [
        { concept: 'greedy 接受', code: 'm07_speculative_decoding/speculative.py:accept_greedy', takeaway: 'target argmax 与 draft token 逐位比对, 第一个不一致处停, 返回 target 自己的 token。' },
        { concept: '采样接受', code: 'm07_speculative_decoding/speculative.py:accept_sampling', takeaway: 'u < p_t/p_d 接受; 否则从 max(0, p_t − p_d) 归一化后重采样。全接受时 bonus 直接采自 target。' },
        { concept: '期望产出', code: '(1 − α^(K+1)) / (1 − α)', takeaway: '每次 target 调用的期望 token 数; α 是每 token 接受率。K 再大也被 1/(1−α) 封顶。' },
        { concept: '采样顺序', code: 'm10_sampling/samplers.py:sample', takeaway: 'rep penalty → temp → top-k → top-p → min-p → Gumbel-max。' },
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
      tldr: 'FlashAttention 对 Q/K 双向分块、用 online softmax 累计, 不落完整 T×T, 并输出 LSE 供分段合并; CUDA Graph 复用固定形状执行图; 推理 TP 切大矩阵。(量化见「INT4 · AWQ · KIVI」一章。)',
      code: 'llm_infer/m09_tensor_parallel · m11_flash_attention · m12_cuda_graph (m08 量化见独立章节)',
      points: [
        { title: 'FlashAttention', body: '每个 tile 只进 SRAM 一次; 运行最大值 m、运行分母 l、运行输出 O 三个量随 KV block 增量更新, 结果与整块 softmax 逐位一致。' },
        { title: 'LSE 输出', body: 'lse = m + log l。两段 attention 结果可用 LSE 加权合并 (merge_attention), ring attention / chunked prefill 都靠它。' },
        { title: 'CUDA Graph', body: '固定 shape decode 捕获后 replay, 减少小 batch kernel launch 开销。本仓库 launch 开销是显式模拟参数, 不是实测。' },
      ],
      sourceRows: [
        { concept: 'online softmax', code: 'm11_flash_attention/flash_attention.py:flash_attention', takeaway: 'm/l/O 随 KV block 增量更新; 新最大值出现时旧的和要乘 exp(m_old − m_new)。' },
        { concept: '分段合并', code: 'm11_flash_attention/flash_attention.py:merge_attention', takeaway: '用两段的 LSE 做 softmax 权重合并输出。' },
        { concept: '图捕获', code: 'm12_cuda_graph/graph.py:CudaGraph', takeaway: 'capture 固定计算, replay 降调度开销; batch 要 pad 到桶大小。' },
        { concept: 'TP 切分', code: 'm09_tensor_parallel/parallel_linear.py:tp_block', takeaway: 'column→row, 每层 2 次 all-reduce。' },
      ],
      source: [`${I}m11_flash_attention/flash_attention.py:flash_attention`],
    },
    'infer-engine': {
      tldr: 'Engine.step = schedule → run → sample → postprocess。KV 写在全局物理分页 pool 里, 前缀命中的 block 真的跳过前向; prefill chunk 与 decode 混在同一个 batch; block 不够时 recompute 式抢占。greedy 下输出与朴素生成逐 token 相同。',
      points: [
        { title: '真分页 KV pool', body: 'KV 不挂在序列上, 而是写进 pool[layer] (num_blocks, block_size, D), attention 经 block_table 读回 —— 所以共享前缀的 block 不用再算。' },
        { title: '混合 batch', body: 'batch = [(seq, n)]: n>1 的 prefill chunk 与 n=1 的 decode 在同一步; token 预算 max_batch_tokens 封顶每步耗时。' },
        { title: '对得上的账', body: 'prefix_hit_tokens + runner.tokens_computed 必须等于全部 token 数; 抢占后的重算也如实计入。' },
      ],
      links: [
        { from: 'add_request', to: 'Scheduler.waiting', body: 'prompt encode 后进入调度系统。' },
        { from: 'Scheduler._admit', to: 'PrefixCache.match_prefix', body: '命中的 token 不占预算、不用算, seq.num_computed 直接从 n_hit 起步。' },
        { from: 'ModelRunner.run', to: 'm02.paged_attention', body: '只算 ids[start_pos:] 的 Q/K/V, 其余 KV 经页表从 pool 读回。' },
      ],
      sourceRows: [
        { concept: '一步', code: 'full_engine/engine.py:Engine.step', takeaway: 'schedule → 逐序列 run → 追平了才 sample → postprocess。' },
        { concept: '只算新 token', code: 'full_engine/model_runner.py:ModelRunner.run', takeaway: 'start_pos 之前的 KV 来自前缀命中 / 上一个 chunk / 之前的 decode 步。' },
        { concept: '抢占', code: 'm03_continuous_batching/scheduler.py:Scheduler._preempt', takeaway: '还 block + 回队首; 最老的序列永远能前进, 所以不会活锁。' },
        { concept: '统计', code: 'Engine.report_stats', takeaway: 'tokens_computed 与 prefix_hit_tokens 是"诚实的省算力账"。' },
      ],
      snippet: `def step(self):
    batch = self.scheduler.schedule()            # [(seq, n)]: prefill chunk 与 decode 混批
    tokens = []
    for seq, n in batch:
        ids = seq.all_ids[:seq.num_computed + n]
        logits = self.runner.run(ids, block_table(seq), seq.num_computed)
        caught_up = seq.num_computed + n == seq.num_tokens
        tokens.append(sample(logits) if caught_up else None)
    return self.scheduler.postprocess(batch, tokens)`,
      source: [`${I}full_engine/engine.py:step`, `${I}full_engine/model_runner.py:run`],
    },

    // ---------------- 新章节 ---------------- //
    'infer-prefix-radix': {
      title: '前缀复用 · 从 block hash 到 radix tree',
      subtitle: '多轮对话、few-shot、共享 system prompt —— 请求之间的公共前缀只该算一次。',
      tldr: 'KV 只依赖它之前的 token, 所以相同前缀的 KV 可以跨请求复用。m04 用链式 block hash (只能命中整块), m05 用 radix tree (命中任意长度, 边压缩 + 分裂), 容量满了从叶子按 LRU 驱逐。',
      question: '为什么驱逐只能从叶子开始, 不能直接扔掉最久没用的中间节点?',
      code: 'llm_infer/m04_prefix_cache/prefix_cache.py · llm_infer/m05_radix_cache/radix_tree.py',
      points: [
        { title: '链式 hash', body: 'h_i = H(h_{i-1} ‖ block_i): 一个 hash 唯一标识"到第 i 块为止的整个前缀", 查表 O(块数)。只能命中 ⌊n/block⌋·block 个 token。' },
        { title: 'radix tree', body: '边上存一段 token + 等长的 KV 槽位; 新请求沿树走到最长公共前缀。分叉落在边中间时 _split 把 (tokens, slots) 同步切两半。' },
        { title: '叶子优先 LRU', body: '中间节点的 KV 被所有后代依赖, 先扔它后代就全废了; 所以只驱逐 ref_count=0 的叶子。' },
      ],
      links: [
        { from: 'BlockManager.ref_count', to: 'PrefixCache', body: '共享 block 靠引用计数, 最后一个使用者释放后才可回收。' },
        { from: 'RadixCache.match_prefix', to: 'Scheduler._admit', body: '命中的 token 不占 token 预算, num_computed 从 n_hit 起步。' },
        { from: 'radix tree', to: 'm20 分层 offload', body: '被 GPU 挤出去的前缀不丢弃, 降级到 CPU / 磁盘。' },
      ],
      sourceRows: [
        { concept: '链式 hash', code: 'm04_prefix_cache/prefix_cache.py:_block_hash', takeaway: 'parent_hash 参与计算, 所以同一段 token 出现在不同前缀后面不会误命中。' },
        { concept: '最长前缀匹配', code: 'm05_radix_cache/radix_tree.py:RadixCache.match_prefix', takeaway: '按子边首 token 索引孩子, 沿边逐 token 比较。' },
        { concept: '边分裂', code: 'RadixCache._split', takeaway: '新中间节点继承 ref_count; tokens 与 slots 必须同步切。' },
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
      subtitle: '真实词表是多字符 token, 一个 token 可能一口气跨过好几个语法状态。',
      tldr: '字符级 FSM 保证合法, 但真实 token 是多字符的 (如 \'":\' 、\'true\')。对每个 (状态, token) 离线试走一遍: 整段字符都走得通才合法, 并记下落点状态。在线每步只查一行表, 把非法 token 的 logit 置 −inf。',
      question: '为什么不能每步在线对 V 个 token 逐字符试走 FSM?',
      code: 'llm_infer/m14_structured_output/grammar.py',
      points: [
        { title: 'next_state[s, t]', body: '(S, V) 整数表: 状态 s 吃下 token t 后的状态, −1 = 走不通。mask_table = next_state ≥ 0。' },
        { title: '为什么预编译', body: '在线现算是 O(V·len) 的纯 CPU 开销, 卡在 GPU 前向和采样之间; FSM 状态有限, 离线各跑一遍即可 O(1) 查表。' },
        { title: '能收尾', body: 'need[s, t] = 选 t 后最少还要几个 token 才能结束; 剩余长度不够时提前屏蔽"收不了尾"的 token, 保证输出一定是完整 JSON。' },
      ],
      links: [
        { from: 'JsonFSM (字符级)', to: 'compile_char_dfa', body: '先把字符级 FSM 枚举成 DFA 转移表。' },
        { from: 'token_row', to: 'compile_token_table', body: '对每个状态跑一遍"逐 token 逐字符试走", 存成 (S, V) 表。' },
        { from: 'mask_table[s]', to: 'm10 sample', body: 'mask 加在 logits 上, 之后的温度 / top-p 照常。' },
      ],
      sourceRows: [
        { concept: '词表', code: 'm14_structured_output/grammar.py:build_vocab', takeaway: '单字符 + 类 BPE 多字符片段, 含跨语法边界的 \'":\' 、\'e"\' 和永远非法的垃圾 token。' },
        { concept: '试走', code: 'm14_structured_output/grammar.py:token_row', takeaway: 'EOS 只在接受态合法; 其它 token 逐字符走转移表。' },
        { concept: '预编译', code: 'm14_structured_output/grammar.py:compile_token_table', takeaway: 'next_state / mask_table / need 三张表。' },
        { concept: '与 xgrammar 的差距', code: 'CFG / 下推自动机', takeaway: '有栈的语法只能预编译"与栈无关"的 token, 其余运行时再查。' },
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
      title: 'INT4 · AWQ · KIVI · 比特越低, "组怎么划"越重要',
      subtitle: 'decode 是带宽受限的: 权重和 KV 越小, 每步要搬的字节越少。',
      tldr: 'group-wise INT4 让每 g 个权重共用一组 scale/zero; AWQ 发现输出误差 = Σ x_i·ΔW_i, 于是量化前把激活大的输入通道放大 s_i、激活同步缩小, 数学等价但误差更小。KV 量化里 K 有固定离群通道 → 按通道分组, V 没有 → 按 token 分组 (KIVI)。',
      question: '为什么量化要最小化的是 ‖XW − XŴ‖ 而不是 ‖W − Ŵ‖?',
      code: 'llm_infer/m08_quantization/{int8_weight.py,int4_awq.py,kv_quant.py}',
      points: [
        { title: 'group-wise', body: '离群值只污染自己那一组的 scale。g 越小越准, 但 scale/zero 的开销越大: g=32 时 INT4 实际约 4.5 bit/权重。' },
        { title: 'AWQ 缩放', body: 'X W = (X/s)·(s⊙W)。放大后该行的相对舍入误差缩小约 s 倍; 代价是撑大同组的 range, 所以 α 要在校准集上网格搜索 (α=0 即 RTN)。' },
        { title: 'KIVI', body: 'K 的少数通道在所有 token 上都是大值: per-token 分组每行都被它撑大 scale; per-channel 把离群值关在自己的组里。' },
      ],
      links: [
        { from: 'quantize_int8 (per-channel)', to: 'quantize_groupwise', body: '从每通道一组缩小到每 g 个权重一组。' },
        { from: 'mean|x_i|', to: 's_i = mean|x_i|^α', body: '只用校准激活的逐通道统计量, 不需要反向传播。' },
        { from: 'KV quant', to: 'm18 KV 体积', body: '量化减字节/元素, GQA/MLA 减元素个数, 两者相乘。' },
      ],
      sourceRows: [
        { concept: '分组量化', code: 'm08_quantization/int4_awq.py:quantize_groupwise', takeaway: 'reshape 成 (D_in/g, g, D_out) 在 g 维上统计 min/max。' },
        { concept: '真正的目标', code: 'm08_quantization/int4_awq.py:output_err', takeaway: '‖XW − XŴ‖_F / ‖XW‖_F。' },
        { concept: 'AWQ 搜索', code: 'm08_quantization/int4_awq.py:awq_quantize', takeaway: 'Ŵ = Q(s ⊙ W) / s; 1/s 离线折进上一层, 推理零开销。' },
        { concept: 'KV 三种分组', code: 'm08_quantization/kv_quant.py:quantize_kv', takeaway: '三种 scheme 只差 reshape 与 axis。' },
      ],
      snippetTitle: 'AWQ: 量化前缩放',
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
      subtitle: '同一个 attention 数学, 四种"cache 里存什么"—— 决定一张卡能同时服务多少条序列。',
      tldr: '每 token KV 字节 = 2·n_kv·d_head·n_layer·bytes; MLA 只缓存 latent: (d_c + d_rope)·n_layer·bytes。LLaMA-2-7B (MHA) 512 KiB → LLaMA-3-8B (GQA-8) 128 KiB → DeepSeek-V3 (MLA) 68.6 KiB, 比同尺寸 MHA 省 56.9×。',
      question: 'MLA 的 latent 为什么不能带 RoPE, 而要另外留一份解耦的 RoPE key?',
      code: 'llm_infer/m18_kv_attention_variants/attention_variants.py',
      points: [
        { title: '一个参数', body: 'n_kv = n_head 是 MHA, 1 < n_kv < n_head 是 GQA, n_kv = 1 是 MQA。每组 query 头靠广播共享 KV, kernel 里不复制。' },
        { title: 'latent cache', body: 'MLA 的 cache 只有 (T, d_c) 和 (T, d_rope)。per-head K/V 在 attention 时才由 C @ W_UK / W_UV 现场还原, 不进 cache。' },
        { title: 'absorb 形式', body: 'q·(C W_UK)ᵀ = (q W_UKᵀ)·Cᵀ: 把 W_UK 乘到 q 上, K 就是 cache 本身 —— MLA decode ≡ head_dim = d_c+d_rope 的 MQA。' },
      ],
      links: [
        { from: '阶段 2 注意力演进', to: 'KV bytes/token', body: '结构选择在推理侧变成显存账。' },
        { from: 'bytes/token', to: 'BlockManager.num_blocks', body: '显存 ÷ 每 token 字节 = 能放多少 token = 并发上限。' },
        { from: 'MLA absorb', to: 'FlashMLA', body: '真实 kernel 走吸收形式, 全程不物化 per-head K/V。' },
      ],
      sourceRows: [
        { concept: 'GQA 公式', code: 'm18_kv_attention_variants/attention_variants.py:kv_bytes_per_token', takeaway: '2 = K 和 V。' },
        { concept: 'MLA 公式', code: 'm18_kv_attention_variants/attention_variants.py:mla_bytes_per_token', takeaway: '没有"2·": K/V 共用同一个 latent。' },
        { concept: 'latent decode', code: 'MLALayer.forward', takeaway: 'latent 不加 RoPE, 否则位置相关的旋转夹在 W_UK 前面, 就无法吸收进 W_Q。' },
        { concept: '对拍', code: 'cache_nbytes', takeaway: 'demo 断言实际 cache 字节数 == 公式。' },
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
      subtitle: '纯滑动窗口一旦把开头几个 token 挤出去, 模型立刻崩 —— 因为 softmax 必须把 1 分给某个人。',
      tldr: 'softmax 的权重和恒为 1, 当前 token 没什么可看时, 多余的注意力会倒在开头几个 token 上 (sink)。StreamingLLM 永远保留开头 n_sink 个 + 最近 window 个; 位置按 cache 槽位重新编号, 所以 K 必须存未旋转的版本。',
      question: '为什么 SinkCache 里的 K 要存 pre-RoPE 的, 和普通 KV cache 正好相反?',
      code: 'llm_infer/m16_attention_sinks/sink_cache.py',
      points: [
        { title: 'sink 现象', body: '开头 token 对所有后续位置都可见, 训练中被学成"垃圾桶"。逐出它们 → 分母骤变 → 其余权重被迫重新分配, 输出分布漂移。' },
        { title: '有界 cache', body: 'cache 条目恒 ≤ n_sink + window, 超预算时逐出槽位 n_sink (窗口里最老的), sink 不动。' },
        { title: '位置重编号', body: '位置 = cache 槽位 0..L−1 而不是流里的绝对位置, 永远不超过训练长度; 代价是每步把 cache 里的 K 整体重转一遍。' },
      ],
      links: [
        { from: 'KV cache (m01)', to: 'SinkCache', body: '从 O(T) 增长变成 O(n_sink + window) 有界。' },
        { from: 'RoPE', to: 'positions = arange(L)', body: '相对位置只看槽位差, 逐出中间 token 后"距离"被压缩。' },
        { from: 'select_blocks forced', to: 'm22 稀疏 decode', body: '稀疏选块同样强制保留第 0 块 —— 同一个 sink。' },
      ],
      sourceRows: [
        { concept: '逐出规则', code: 'm16_attention_sinks/sink_cache.py:SinkCache.append', takeaway: 'keep = [0:n_sink] + [n_sink+1:], 永远去掉槽位 n_sink。' },
        { concept: '重编号', code: 'm16_attention_sinks/sink_cache.py:stream_step', takeaway: 'pos = arange(L); K 每步按当前槽位现转, query 在最后一个槽位。' },
        { concept: '不是长上下文', code: 'window 之外的 token', takeaway: '被逐出的内容彻底丢失; sink 只保证"不崩", 不保证"记得"。' },
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
      subtitle: '链式 draft 第一个猜错, 后面 K−1 个全废。两条改进路线: 猜得更准 (EAGLE), 一次验更多分支 (树)。',
      tldr: 'EAGLE 让 draft 吃 target 白送的 hidden state、共享 target 的 lm_head, 接受率更高。树形投机每个节点保留 draft 的 top-k 候选长成树, target 用 tree attention mask (只看上下文 + 祖先 + 自己) 一次验完, 接受与 target greedy 一致的最长路径。',
      question: '同一深度的兄弟节点为什么共享同一个 RoPE 位置?',
      code: 'llm_infer/m17_eagle_speculative/eagle.py · llm_infer/m19_tree_speculation/tree_spec.py',
      points: [
        { title: '特征级 draft', body: 'ĥ_{t+1} = Draft(h_t, emb(x_{t+1})), 再过 target 自己的 lm_head。第 1 步 h 是 target 的真特征, 之后是 draft 自己的 ĥ, 误差逐步累积。' },
        { title: 'tree mask', body: 'anc[i, j] = j 是 i 的祖先或自己。每个节点看到的恰好是"从根到自己这条链", 等价于把每条路径各跑一次顺序 decode。' },
        { title: '算力换延迟', body: 'widths=[3,2,1] → 16 个节点、1 次 target 调用、3 次 draft 调用。decode 带宽受限时多验几个 token 几乎免费; 大 batch 算力受限时不免费。' },
      ],
      links: [
        { from: 'm07 accept_greedy', to: 'accept_tree', body: '从"沿链比对"变成"从根往下走, 命中哪个孩子进哪个"。' },
        { from: 'TinyLM.forward(return_hidden)', to: 'EagleDrafter.propose', body: '验证那次 forward 白送了下一轮 draft 要的特征。' },
        { from: 'gather_kv', to: 'KV 回滚', body: '只留被接受路径的 KV 行; K 已是 post-RoPE, 挑行即可。' },
      ],
      sourceRows: [
        { concept: '树形状', code: 'm19_tree_speculation/tree_spec.py:tree_shape', takeaway: 'BFS 编号; [3,2,1] → 1+3+6+6 = 16 节点。' },
        { concept: '树 mask', code: 'm19_tree_speculation/tree_spec.py:tree_mask', takeaway: '上下文全可见, 树内只看祖先 + 自己; positions = n_ctx + depth。' },
        { concept: '最长路径', code: 'm19_tree_speculation/tree_spec.py:accept_tree', takeaway: 'top-k 互不相同 → 每层至多命中一个孩子。' },
        { concept: 'EAGLE drafter', code: 'm17_eagle_speculative/eagle.py:EagleDrafter', takeaway: '只换 drafter, 验证循环复用 m07.speculative_decode。' },
        { concept: '实测', code: 'm19 demo', takeaway: '同验证 token 数: chain K=15 每轮接受 1.40, tree [3,2,1] 1.80 (2.58 token/target 调用)。' },
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
      subtitle: '多轮对话每轮 prompt = 整段历史, GPU 装不下所有用户的历史。被挤出去的 KV 别扔, 降级存起来。',
      tldr: '搬 1 token KV (128 KiB) 走 PCIe 25 GB/s ≈ 5 µs, 重算 ≈ 125 µs —— 搬比算便宜 25 倍。但每次加载有固定延迟, 慢链路每 token 甚至比重算还慢, 所以每层都要判断"加载 vs 重算"。以下时间全部来自代价模型, 不是实测。',
      question: '为什么"多加一层缓存"有时反而让 TTFT 变差?',
      code: 'llm_infer/m20_kv_offload/tiered_cache.py',
      points: [
        { title: '降级而不是丢弃', body: '每层一个 LRU; GPU 层溢出的最旧 block 降到 CPU, CPU 溢出降到磁盘。再次命中时提升回 GPU 层。' },
        { title: '链式前缀', body: '沿 hash 链逐块查找, 第一个全层 miss 处停: 后面的块即使还在也不可用, 因为 KV 依赖整个前缀。' },
        { title: '交叉点', body: 'n* = latency / (每块重算 − 每块加载)。demo 里 disk 的 n* = 1.54 块: 只命中 1 块时重算更快。' },
      ],
      links: [
        { from: 'm04 PrefixCache', to: 'TieredKVCache', body: '同样的链式 block hash, 只是多了几层存储。' },
        { from: 'Tier.bandwidth_GBps', to: 'load_ms', body: '这里带宽单位是 GB/s (byte); 对照 m15 的 Gbps (bit)。' },
        { from: 'use_load', to: 'TTFT', body: '慢远端 0.5 GB/s: 命中就加载 68.9 ms, 取小 34.8 ms。' },
      ],
      sourceRows: [
        { concept: '代价模型', code: 'm20_kv_offload/tiered_cache.py:CostModel', takeaway: 'load = latency + bytes / bandwidth; recompute = tokens / 8000 tok/s。' },
        { concept: '降级', code: 'TieredKVCache._insert', takeaway: '溢出的最旧 block 递归插入下一层。' },
        { concept: '逐层决策', code: 'm20_kv_offload/tiered_cache.py:TieredKVCache.serve', takeaway: 'use_load = load ≤ recompute。' },
        { concept: 'demo 结果 (模拟)', code: '8 users × 6 turns', takeaway: 'mean TTFT: GPU only 51.5 → +CPU 34.8 → +disk 19.5 ms。' },
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
      subtitle: '专家分布在多张卡上, 每层都要同步 —— 一步的耗时等于最忙那张卡的耗时。',
      tldr: 'token 经 all-to-all 发到专家所在 rank (dispatch), 算完发回按 gate 加权 (combine)。路由倾斜时热专家所在 rank 负载可达均值 2.95×, 平均利用率只有 34%。EPLB 给热专家加冗余副本并拆分它的 token, 把 max/mean 压到 1.02×, 输出逐位不变。',
      question: '为什么只靠"重新摆放专家"不够, 必须复制热专家?',
      code: 'llm_infer/m21_moe_serving/moe.py',
      points: [
        { title: '看 max 不看 mean', body: 'EP 每层 combine 要等所有 rank; max/mean = 2.95 ⇒ 其余 rank 大部分时间在等最慢的那个。' },
        { title: '两步贪心', body: '① 反复给"每副本负载"最大的专家加一个副本; ② slot 按负载从大到小放到当前最轻且有空位的 rank。' },
        { title: '不可分的热点', body: '最热专家负载 / 平均 rank 负载 = 1.42 > 1: 一个专家就超过一张卡该分到的量, 怎么摆都不均衡, 只能复制拆分。' },
      ],
      links: [
        { from: '阶段 2 MoE 路由', to: 'MoELayer.route', body: '同一个 top-k 路由, 这里关心它在多卡上的负载。' },
        { from: '阶段 3 EP all-to-all', to: 'ep_forward', body: '训练用辅助 loss 压倾斜; 推理时路由已定, 只能靠放置。' },
        { from: 'eplb_placement', to: 'slot_of', body: '有副本的专家把自己的 token 轮流分给各副本。' },
      ],
      sourceRows: [
        { concept: '路由', code: 'm21_moe_serving/moe.py:MoELayer.route', takeaway: 'gate 只在被选中的 k 个 logit 上做 softmax。' },
        { concept: 'dispatch/combine', code: 'm21_moe_serving/moe.py:ep_forward', takeaway: 'rank_load[r] = 发到 rank r 的 (token, k) 分配数。' },
        { concept: 'EPLB', code: 'm21_moe_serving/moe.py:eplb_placement', takeaway: 'n_rep[argmax(load / n_rep)] += 1。' },
        { concept: 'demo 结果', code: '16 专家 / 4 rank / top-2', takeaway: 'max/mean: 连续 2.95× → 贪心无副本 1.48× → EPLB +4 副本 1.02×。' },
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
      tldr: '把 KV 切成 block, 每 block 常驻一份很小的摘要 (Quest: 逐维 min/max; NSA/DSA: 压缩 key / 低维 indexer)。用 q 给摘要打分选 top-k block, 只对它们做 attention。needle 负载上读 3.1% 的 KV 误差 0.023; 但随机权重模型注意力弥散, 稀疏化的前提不成立。',
      question: '为什么 Quest 用"上界"打分而不是直接用 block 均值?',
      code: 'llm_infer/m22_sparse_attention/sparse_attention.py',
      points: [
        { title: '廉价打分', body: 'Quest: Σ_i max(q_i·kmin_i, q_i·kmax_i) ≥ block 内任何 q·k。上界保证不漏掉"针", 代价是偏松 (demo 里平均 6.7×)。' },
        { title: '强制保留', body: '第 0 块 (attention sink) 和最后一块 (最近 token) 永远选中, 算在 k 内。' },
        { title: '前提', body: '收益来自注意力高度集中。训练过的 LLM 成立; 注意力均匀弥散时, 打分相对随机选块几乎没有优势。' },
      ],
      links: [
        { from: 'm02 KV block', to: 'block 摘要', body: '分页 KV 天然按 block 组织, 摘要 ≈ KV 的 1/16 ~ 1/8。' },
        { from: 'm16 sink', to: 'select_blocks forced', body: '首块必须留, 否则 softmax 分母崩。' },
        { from: '阶段 2 DSA', to: 'lightning indexer', body: 'DeepSeek-V3.2 用训练出来的低维 indexer 打分选 token; 这里只有免训练的选块。' },
      ],
      sourceRows: [
        { concept: 'Quest 上界', code: 'm22_sparse_attention/sparse_attention.py:quest_upper_bound', takeaway: 'q_i 正取 kmax, 负取 kmin, 逐维求和。' },
        { concept: '选块', code: 'm22_sparse_attention/sparse_attention.py:select_blocks', takeaway: 'forced = {0, nb−1}, 其余按分数取 top。' },
        { concept: '评测', code: 'm22_sparse_attention/sparse_attention.py:evaluate', takeaway: '相对 L2 误差 + 选中 block 覆盖的真实注意力质量 (recall)。' },
        { concept: 'demo 结果', code: 'T=4096, 256 blocks', takeaway: 'k=4 (1.6%) 误差 0.83 → k=8 (3.1%) 误差 0.023: 针全部进 top-k 的那一刻误差断崖下降。' },
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
