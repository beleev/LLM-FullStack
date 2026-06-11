// 模型元数据 — 与 llm_models/ 代码一一对应
// 三条主线: left (语言), eye (多模态理解), right (生成)

export const tracks = {
  left:  { label: '左脑 · 语言',      color: 'var(--left)',  cls: 'left'  },
  eye:   { label: '眼耳 · 多模态理解', color: 'var(--eye)',   cls: 'eye'   },
  right: { label: '右脑 · 生成',      color: 'var(--right)', cls: 'right' },
}

export const basicChapters = [
  { route: 'basic-data', label: '数据与 tokenizer', hint: 'input.txt → train.bin / val.bin / meta.npz' },
  { route: 'basic-forward', label: 'forward 与形状流', hint: 'embedding → block → logits → loss' },
  { route: 'basic-backward', label: '手写 backward', hint: 'cache、链式法则、gradcheck' },
  { route: 'basic-optim-sample', label: 'Adam 与采样', hint: '优化器状态与自回归生成' },
]

export const modelChapters = [
  { route: 'attention', label: '注意力演进',     hint: 'MHA → GQA → MLA → DSA' },
  { route: 'position',  label: '位置编码',       hint: 'Sin / Learnable → RoPE → M-RoPE' },
  { route: 'blocks',    label: 'Block 组装器',  hint: '4 个零件 × 几种组合 = 所有主流 LLM' },
  { route: 'moe',       label: 'MoE 路由',       hint: 'Mixtral 与 DeepSeek 的两套哲学' },
  { route: 'models-mtp', label: 'SWA · MTP · 混合线性', hint: 'Mistral 滑动窗口 + 多 token 预测 + Qwen3-Next DeltaNet' },
  { route: 'diffusion', label: '扩散生成',       hint: 'DDPM ε-pred → Flow Matching v-pred' },
]

export const trainChapters = [
  { route: 'train-batch-ddp', label: 'batch 与 DDP', hint: '梯度累积、数据并行、all-reduce' },
  { route: 'train-model-parallel', label: 'TP / PP 切模型', hint: '层内矩阵切分与层间流水线' },
  { route: 'train-memory', label: '状态与显存', hint: 'ZeRO/FSDP、activation checkpoint、resume' },
  { route: 'train-precision-stability', label: '精度与稳定性', hint: 'AMP、loss scaling、FP8 block scaling、clip、warmup' },
  { route: 'train-moe-seq', label: 'EP 与序列并行', hint: 'MoE all-to-all 路由 + Ring Attention' },
  { route: 'train-collectives-loop', label: '通信与 full_loop', hint: 'collectives 到完整训练主循环' },
]

export const finetuneChapters = [
  { route: 'finetune-sft', label: 'SFT 数据与 loss', hint: 'prompt mask 与 response-only CE' },
  { route: 'finetune-lora', label: 'LoRA 参数高效微调', hint: '低秩补丁、注入、merge、QLoRA NF4 量化基座' },
  { route: 'finetune-dpo', label: 'DPO 偏好对齐', hint: 'chosen/rejected 与 reference policy' },
  { route: 'finetune-rlhf', label: 'RM · GRPO · 蒸馏', hint: '奖励模型、组内相对优势、软标签蒸馏' },
  { route: 'finetune-runs', label: '训练脚本与落盘', hint: 'run_finetune、adapter、state_dict' },
]

export const inferChapters = [
  { route: 'infer-kv-memory', label: 'KV 与缓存内存', hint: 'KV cache、paged attention、prefix/radix cache' },
  { route: 'infer-scheduler', label: '调度与 prefill', hint: 'continuous batching、chunked prefill、P/D 分离' },
  { route: 'infer-decode-control', label: '解码加速与约束', hint: 'speculative/EAGLE、sampling、structured output' },
  { route: 'infer-compute', label: '算子与压缩', hint: 'quantization、FlashAttention、CUDA Graph、TP' },
  { route: 'infer-engine', label: 'mini-vLLM 引擎', hint: 'full_engine 主循环与 Multi-LoRA serving' },
]

export const agentChapters = [
  { route: 'agent-loop', label: 'Agent loop', hint: 'messages → model action → tool result → final' },
  { route: 'agent-tools-permissions', label: '工具与权限', hint: 'tool schema、execute、deny-first、auto approval' },
  { route: 'agent-context-memory', label: '上下文与记忆', hint: 'file memory、retrieval、compaction' },
  { route: 'agent-extensibility', label: 'Hooks / Skills / MCP', hint: '三类扩展点与上下文成本' },
  { route: 'agent-state-subagents', label: '持久化与子智能体', hint: 'JSONL resume、summary-only delegation' },
  { route: 'agent-full-loop', label: 'mini Agent harness', hint: '组合成应用层最小闭环' },
]

// ─────────────────────────────────────────────────────────────────────────
// 全书主线 — 一个 LLM 工程师从原理到应用层的六个阶段。
// 每个阶段对应仓库根目录下的一个子项目, Web 章节直接对照原始代码。
// ─────────────────────────────────────────────────────────────────────────
export const stages = [
  {
    id: 'basic',
    idx: 1,
    code: 'llm_basic/',
    title: '最小可跑闭环',
    oneliner: 'numpy 手写 forward / backward / Adam / 采样 — 看清梯度怎么流过 Transformer。',
    status: 'ready',
    route: 'basic',
    chapters: basicChapters,
    files: ['model.py', 'train.py', 'optim.py', 'sample.py', 'gradcheck.py'],
  },
  {
    id: 'models',
    idx: 2,
    code: 'llm_models/',
    title: '常见模型结构',
    oneliner: '把零件 (attn / ffn / norm / pos) 装进 Pre-LN Block, 堆出 20 种主流模型。',
    status: 'ready',
    route: 'models',
    // 这一阶段拆成 5 个章节, 用 chapters 映射;  Compare 合到终章。
    chapters: modelChapters,
  },
  {
    id: 'train',
    idx: 3,
    code: 'llm_train/',
    title: '规模化训练',
    oneliner: '把单机训练循环拆成 batch / 层 / 状态 / 精度 / 容错五条工程主线。',
    status: 'ready',
    route: 'train',
    chapters: trainChapters,
    files: ['m01..m13/demo.py', 'core/collectives.py', 'full_loop/demo.py'],
  },
  {
    id: 'finetune',
    idx: 4,
    code: 'llm_finetune/',
    title: '微调与对齐',
    oneliner: '冻结大部分参数, 用极少量数据把 base model 拨到具体任务/偏好上。',
    status: 'ready',
    route: 'finetune',
    chapters: finetuneChapters,
    files: ['methods/{sft,lora,qlora}.py', 'methods/dpo.py', 'methods/grpo.py', 'methods/distill.py'],
  },
  {
    id: 'infer',
    idx: 5,
    code: 'llm_infer/',
    title: '推理与部署优化',
    oneliner: '从逐 token 重算到 mini-vLLM: 缓存、分页、连续批、前缀复用与采样约束。',
    status: 'ready',
    route: 'infer',
    chapters: inferChapters,
    files: ['m01..m17/demo.py', 'full_engine/engine.py', 'core/tiny_model.py'],
  },
  {
    id: 'agent',
    idx: 6,
    code: 'llm_agent/',
    title: 'Agent 应用层',
    oneliner: '把推理服务接成能行动的系统: 工具、权限、上下文、持久化、扩展与子智能体。',
    status: 'ready',
    route: 'agent',
    chapters: agentChapters,
    files: ['core/agent.py', 'core/tools.py', 'core/permissions.py', 'full_loop/demo.py'],
  },
]

export const stageBy = Object.fromEntries(stages.map(s => [s.id, s]))

export const learningPath = [
  { route: 'home', label: '主线总览' },
  ...stages.flatMap(s => [
    ...(s.route ? [{ route: s.route, label: `阶段 ${s.idx}.0 · 阶段总览` }] : []),
    ...(s.chapters || []).map(c => ({ route: c.route, label: c.label })),
  ]),
  { route: 'compare', label: '总览对照表' },
]

export const trainModules = [
  {
    id: 'm01',
    name: 'Gradient Accumulation',
    concept: '大 batch 放不下时, 拆 micro-batch 累梯度',
    link: '等价于 full-batch mean reduction, 关键是按样本数缩放',
    file: 'llm_train/m01_gradient_accumulation/demo.py',
  },
  {
    id: 'm02',
    name: 'Data Parallel / DDP',
    concept: '每卡一份模型, 数据按 batch 维切开',
    link: 'local_grad → all_reduce_mean → 所有副本同一步更新',
    file: 'llm_train/m02_data_parallel/demo.py',
  },
  {
    id: 'm03',
    name: 'Tensor Parallel',
    concept: '把一个大矩阵切给多个 rank',
    link: 'column parallel 产 hidden shard, row parallel 后 all-reduce',
    file: 'llm_train/m03_tensor_parallel/demo.py',
  },
  {
    id: 'm04',
    name: 'Pipeline Parallel',
    concept: '按层切 stage, 用 micro-batch 填流水线',
    link: 'micro-batch 越多, bubble 越小, 激活驻留越复杂',
    file: 'llm_train/m04_pipeline_parallel/demo.py',
  },
  {
    id: 'm05',
    name: 'ZeRO / FSDP',
    concept: '参数 / 梯度 / 优化器状态分片',
    link: 'reduce-scatter 梯度, 本地更新 shard, all-gather 参数',
    file: 'llm_train/m05_zero_fsdp/demo.py',
  },
  {
    id: 'm06',
    name: 'Mixed Precision',
    concept: 'fp16/bf16 算得快, fp32 master 保更新精度',
    link: 'loss scaling 防止小梯度 cast 到 fp16 后归零',
    file: 'llm_train/m06_mixed_precision/demo.py',
  },
  {
    id: 'm07',
    name: 'Activation Checkpointing',
    concept: '少存激活, 反向时重算 forward',
    link: '用额外计算换峰值显存',
    file: 'llm_train/m07_activation_checkpointing/demo.py',
  },
  {
    id: 'm08',
    name: 'Checkpoint / Resume',
    concept: '保存模型、优化器、数据游标与随机状态',
    link: '恢复后 loss 曲线必须和未中断训练对齐',
    file: 'llm_train/m08_checkpoint_resume/demo.py',
  },
  {
    id: 'm09',
    name: 'Collectives',
    concept: '分布式训练的四个通信原语',
    link: 'all-reduce / reduce-scatter / all-gather / all-to-all',
    file: 'llm_train/m09_collectives/demo.py',
  },
  {
    id: 'm10',
    name: 'Training Stability',
    concept: 'warmup、cosine、grad clip、NaN guard',
    link: '不是提高上限, 而是让坏 step 不毁掉长训练',
    file: 'llm_train/m10_training_stability/demo.py',
  },
  {
    id: 'm11',
    name: 'Expert Parallel',
    concept: 'MoE 专家切卡, token 经 all-to-all 找专家',
    link: 'dispatch → 本地专家算 → combine; 路由倾斜 = 热点 + 丢 token',
    file: 'llm_train/m11_expert_parallel/demo.py',
  },
  {
    id: 'm12',
    name: 'Sequence Parallel',
    concept: '序列切卡, KV 块沿环传递 (Ring Attention)',
    link: 'online softmax 跨卡增量合并, 任何时刻只持有 1/D 的 KV',
    file: 'llm_train/m12_sequence_parallel/demo.py',
  },
  {
    id: 'm13',
    name: 'FP8 Training',
    concept: 'E4M3 管精度 / E5M2 管范围, block-wise scaling 防 outlier',
    link: '乘法省一半, 但 scaling 粒度 + FP32 master 一个都不能少',
    file: 'llm_train/m13_fp8_training/demo.py',
  },
  {
    id: 'full',
    name: 'Full Loop',
    concept: '把 DDP、累积、AMP、裁剪、ZeRO 和 checkpoint 串起来',
    link: '控制流对应真实训练脚本主干',
    file: 'llm_train/full_loop/demo.py',
  },
]

export const inferModules = [
  {
    id: 'm01',
    name: 'KV Cache',
    concept: 'prefill 一次, decode 只算新 token',
    link: '消掉每步重复的 K/V 计算, 是一切推理优化的起点',
    file: 'llm_infer/m01_kv_cache/demo.py',
  },
  {
    id: 'm02',
    name: 'Paged Attention',
    concept: 'KV cache 像虚拟内存一样分页',
    link: 'block table 解耦逻辑 token 与物理 KV block',
    file: 'llm_infer/m02_paged_attention/block_manager.py',
  },
  {
    id: 'm03',
    name: 'Continuous Batching',
    concept: '请求动态进出 batch, prefill/decode 分阶段调度',
    link: '吞吐来自让 GPU 每步都有活干',
    file: 'llm_infer/m03_continuous_batching/scheduler.py',
  },
  {
    id: 'm04',
    name: 'Prefix Cache',
    concept: '相同 system prompt 的完整 block 复用',
    link: 'hash(parent, token_block) → physical block',
    file: 'llm_infer/m04_prefix_cache/prefix_cache.py',
  },
  {
    id: 'm05',
    name: 'Radix Cache',
    concept: '任意长度公共前缀共享',
    link: '比 block hash 更细, 适合 SGLang 式 prompt 复用',
    file: 'llm_infer/m05_radix_cache/radix_tree.py',
  },
  {
    id: 'm06',
    name: 'Chunked Prefill',
    concept: '长 prompt 切片进调度器',
    link: '降低长输入对 decode 请求的阻塞',
    file: 'llm_infer/m06_chunked_prefill/demo.py',
  },
  {
    id: 'm07',
    name: 'Speculative Decoding',
    concept: 'draft 一次猜多个, target 批量验证',
    link: '接受率越高, target calls 越少',
    file: 'llm_infer/m07_speculative_decoding/demo.py',
  },
  {
    id: 'm08',
    name: 'Quantization',
    concept: '权重 / KV 用低比特存储',
    link: '带宽和显存下降, 误差靠 per-channel / per-token scale 控制',
    file: 'llm_infer/m08_quantization/{int8_weight.py,kv_quant.py}',
  },
  {
    id: 'm09',
    name: 'Tensor Parallel',
    concept: '推理时把大矩阵和权重切到多卡',
    link: '与训练 TP 同源, 但更关注 decode latency',
    file: 'llm_infer/m09_tensor_parallel/parallel_linear.py',
  },
  {
    id: 'm10',
    name: 'Sampling',
    concept: 'greedy / temperature / top-k / top-p / min-p',
    link: '最后一层 logits policy, 决定可控性与多样性',
    file: 'llm_infer/m10_sampling/samplers.py',
  },
  {
    id: 'm11',
    name: 'FlashAttention',
    concept: 'online softmax, 不落完整 T x T attention',
    link: '把显存峰值从 O(T²) 压到 block 级',
    file: 'llm_infer/m11_flash_attention/flash_attention.py',
  },
  {
    id: 'm12',
    name: 'CUDA Graph',
    concept: 'capture 固定形状计算图, replay 去 launch 开销',
    link: '小 batch decode 尤其吃 kernel launch',
    file: 'llm_infer/m12_cuda_graph/demo.py',
  },
  {
    id: 'm13',
    name: 'Multi-LoRA Serving',
    concept: '同一 batch 里混跑不同 LoRA adapter',
    link: '共享基座, adapter 增量按请求选择',
    file: 'llm_infer/m13_lora_serving/demo.py',
  },
  {
    id: 'm14',
    name: 'Structured Output',
    concept: 'grammar/FSM 把非法 token logit 置 -inf',
    link: '模型负责偏好, 约束器负责合法性',
    file: 'llm_infer/m14_structured_output/demo.py',
  },
  {
    id: 'm15',
    name: 'P/D Disaggregation',
    concept: 'prefill 与 decode 分到不同节点',
    link: '长输入/短输出场景用 KV transport 换集群利用率',
    file: 'llm_infer/m15_pd_disaggregation/demo.py',
  },
  {
    id: 'm16',
    name: 'Attention Sinks',
    concept: 'sink + 滑动窗口, KV cache 有界化 (StreamingLLM)',
    link: '保住 softmax 的"下水道", 流式无限输入不爆显存',
    file: 'llm_infer/m16_attention_sinks/demo.py',
  },
  {
    id: 'm17',
    name: 'EAGLE Speculative',
    concept: 'draft 吃 target 的 hidden state, 在特征空间自回归',
    link: '特征比 token 信息多得多 → 接受率显著高于 token-only draft',
    file: 'llm_infer/m17_eagle_speculative/demo.py',
  },
  {
    id: 'full',
    name: 'Full Engine',
    concept: 'mini-vLLM: 分页、连续批、前缀缓存、采样集成',
    link: '从单模块算法变成服务主循环',
    file: 'llm_infer/full_engine/engine.py',
  },
]

export const agentModules = [
  {
    id: 'm01',
    name: 'Agent Loop',
    concept: '最小 ReAct 风格闭环: 组装上下文、选工具、执行、回填结果',
    link: 'while-loop 很薄, 复杂度在 harness 周边',
    file: 'llm_agent/m01_agent_loop/demo.py',
  },
  {
    id: 'm02',
    name: 'Tool Use',
    concept: 'schema 给模型看, execute 由确定性代码执行',
    link: '模型决定意图, harness 负责动作边界和结果回填',
    file: 'llm_agent/m02_tool_use/demo.py',
  },
  {
    id: 'm03',
    name: 'Permissions',
    concept: 'deny-first、default ask、auto 风险分类',
    link: 'Agent 能动性越强, 权限门越应该显式化',
    file: 'llm_agent/m03_permissions/demo.py',
  },
  {
    id: 'm04',
    name: 'Context & Memory',
    concept: '文件记忆、相关检索、头尾保留 + 中间摘要压缩',
    link: '上下文是应用层最稀缺的资源',
    file: 'llm_agent/m04_context_memory/demo.py',
  },
  {
    id: 'm05',
    name: 'Extensibility',
    concept: 'hooks、skills、MCP-like tools 三类扩展点',
    link: '按上下文成本分层, 不把所有能力塞进 prompt',
    file: 'llm_agent/m05_extensibility/demo.py',
  },
  {
    id: 'm06',
    name: 'Persistence',
    concept: 'append-only JSONL transcript 与 resume',
    link: '可恢复上下文不等于自动恢复权限',
    file: 'llm_agent/m06_persistence_resume/demo.py',
  },
  {
    id: 'm07',
    name: 'Subagents',
    concept: '隔离子上下文, 父级只接收 summary',
    link: '避免子任务细节污染父级上下文',
    file: 'llm_agent/m07_subagents/demo.py',
  },
  {
    id: 'm08',
    name: 'Retrieval',
    concept: 'TF-IDF 向量检索: idf 压高频词, 余弦给连续分级',
    link: '关键词 → TF-IDF → 神经 embedding, 工具接口不变',
    file: 'llm_agent/m08_retrieval/demo.py',
  },
  {
    id: 'full',
    name: 'Full Loop',
    concept: '工具、权限、Hook、记忆、持久化和子智能体组合',
    link: '一个 mini-Claude-Code-style harness',
    file: 'llm_agent/full_loop/demo.py',
  },
]

export const topicPages = {
  'basic-data': {
    widgets: ['BpeLab'],
    title: '数据与 tokenizer · 把文本变成可训练张量',
    subtitle: '先别看 Transformer。训练闭环的第一个问题是: 字符串如何稳定地变成 ids, 再变成 batch。',
    tldr: 'prepare.py 负责把原始文本固化成 train.bin / val.bin / meta.npz; tokenizer.py 定义 encode/decode 的可逆映射。',
    question: '如果 vocab、stoi/itos 或数据切分不可复现, 后面的 loss 曲线还能被信任吗?',
    code: 'llm_basic/{prepare.py,tokenizer.py,input.txt,train.bin,val.bin,meta.npz}',
    points: [
      { title: '可复现数据', body: '原始文本只处理一次, 之后训练直接 mmap / np.fromfile 读二进制 ids, 避免每次运行重新构造 vocab。' },
      { title: '字符级 tokenizer', body: '字符级方案牺牲语义压缩率, 但把 BPE 合并规则先拿掉, 让注意力、loss、梯度成为唯一主角。' },
      { title: 'batch 是连续片段', body: 'get_batch 从长 token 序列中切 [T] 片段, y 是 x 右移一位; 这就是 next-token prediction。' },
    ],
    links: [
      { from: 'input.txt', to: 'prepare.py', body: '文本 → 字符表 → token ids' },
      { from: 'train.bin', to: 'train.py:get_batch', body: '二进制 ids → x/y batch' },
      { from: 'meta.npz', to: 'sample.py', body: 'itos/stoi → 把生成 ids 还原成文本' },
    ],
    sourceRows: [
      { concept: 'vocab 构造', code: 'prepare.py', takeaway: '把 unique chars 排序, 得到稳定 stoi/itos。' },
      { concept: '训练/验证切分', code: 'prepare.py', takeaway: '先切数据, 再保存 train.bin / val.bin, 避免验证集泄漏。' },
      { concept: 'batch 对齐', code: 'train.py:get_batch', takeaway: 'x = data[i:i+T], y = data[i+1:i+T+1]。' },
    ],
    snippetTitle: 'next-token 数据形态',
    snippet: `text:  "hello"
ids:   [h, e, l, l, o]

x:     [h, e, l, l]
y:     [e, l, l, o]

# 训练目标: 看到当前位置及之前 token, 预测下一个 token`,
    run: 'python llm_basic/prepare.py',
  },
  'basic-forward': {
    title: 'forward 与形状流 · 从 ids 到 logits',
    subtitle: '这一章只追踪张量形状和 cache。先看清前向, 反向才有落点。',
    tldr: 'Transformer forward = token/position embedding + Pre-LN block + final norm + lm_head。cache 保存反向需要的中间量。',
    question: '为什么 forward 不能只返回 logits, 还要把一长串 cache 带回去?',
    code: 'llm_basic/model.py:{embedding_forward,rmsnorm_forward,attention_forward,block_forward,transformer_forward}',
    points: [
      { title: '结构极简但完整', body: '单层、单头、ReLU MLP 是教学简化; 残差、norm、causal attention、lm_head 都保留。' },
      { title: '形状先行', body: 'B/T/D/V 四个维度贯穿全章。读代码时先检查矩阵乘法两边形状, 再看数值细节。' },
      { title: 'cache 是手写 autograd tape', body: '每个 forward 返回反向需要的输入、权重、归一化统计或 softmax 概率。' },
    ],
    links: [
      { from: 'llm_basic', to: 'llm_models', body: '单头 attention 后续会替换成 MHA/GQA/MLA。' },
      { from: 'learned pos_emb', to: 'RoPE', body: '阶段 2 把位置从加法 embedding 改成旋转 Q/K。' },
      { from: 'ReLU MLP', to: 'SwiGLU / MoE', body: 'FFN 槽位后续会换成门控或专家路由。' },
    ],
    sourceRows: [
      { concept: 'embedding', code: 'embedding_forward', takeaway: 'tok_emb[x] + pos_emb[:T] 得到 [B,T,D]。' },
      { concept: 'causal attention', code: 'attention_forward', takeaway: 'QK^T 加 causal mask 后 softmax, 再乘 V。' },
      { concept: '语言模型头', code: 'transformer_forward', takeaway: '最后投影到 vocab 维, logits 形状 [B,T,V]。' },
    ],
    snippetTitle: '主干形状',
    snippet: `ids[B,T]
  → tok_emb + pos_emb        [B,T,D]
  → RMSNorm → Attention      [B,T,D]
  → residual
  → RMSNorm → MLP            [B,T,D]
  → final RMSNorm
  → lm_head                  [B,T,V]`,
    run: 'python llm_basic/train.py',
  },
  'basic-backward': {
    title: '手写 backward · cache、链式法则与 gradcheck',
    subtitle: 'autograd 做的事在这里全部摊开。重点不是公式多, 而是每条梯度路径都能回到对应 forward。',
    tldr: '每个 forward 都有配套 backward; 残差梯度相加, softmax/RMSNorm 有耦合项, embedding 要累加重复 token。',
    question: '为什么一个 np.add.at 或 sum 维度写错, loss 仍能跑但模型永远学不对?',
    code: 'llm_basic/model.py:*_backward · llm_basic/gradcheck.py',
    points: [
      { title: '反向按图倒走', body: '从 cross-entropy 给出的 dlogits 开始, 按 forward 的逆序逐层传播。' },
      { title: '重复索引要累加', body: 'embedding 的同一个 token 可能出现多次, 梯度必须加到同一行参数上。' },
      { title: 'gradcheck 是保险丝', body: '解析梯度和中心差分逐元素对比, 能定位 transpose、broadcast、mask 的细小错误。' },
    ],
    links: [
      { from: 'softmax backward', to: 'attention 演进', body: '后续所有注意力变体仍要处理概率归一化的耦合梯度。' },
      { from: 'RMSNorm backward', to: '训练稳定性', body: '标准化层是深层 Transformer 稳定训练的基础。' },
      { from: 'gradcheck', to: '分布式等价测试', body: 'llm_train 里 dense vs parallel 的 assert 是同一种验证思想。' },
    ],
    sourceRows: [
      { concept: 'embedding backward', code: 'np.add.at(dW, ids, dout)', takeaway: '处理重复 token id 的梯度累加。' },
      { concept: 'softmax backward', code: 'attention_backward', takeaway: 'ds = a * (da - sum(a * da))。' },
      { concept: '数值梯度', code: 'gradcheck.py', takeaway: '(loss+ - loss-) / (2eps) 对照解析梯度。' },
    ],
    snippetTitle: '反向入口',
    snippet: `loss, dlogits = cross_entropy_forward_backward(logits, y)
grads = transformer_backward(dlogits, cache)

# gradcheck:
g_numeric = (loss_plus - loss_minus) / (2 * eps)
assert close(g_analytic, g_numeric)`,
    run: 'python llm_basic/gradcheck.py',
  },
  'basic-optim-sample': {
    title: 'Adam 与采样 · 训练后如何生成文本',
    subtitle: 'forward/backward 给出梯度, Adam 决定怎么走一步; sample.py 则把模型放回自回归使用方式。',
    tldr: 'Adam 用一阶/二阶矩自适应缩放梯度; 采样阶段不再算 loss, 而是循环取最后位置 logits 生成下一个 token。',
    question: '为什么训练时并行预测所有位置, 生成时却只能一个 token 一个 token 来?',
    code: 'llm_basic/{optim.py,sample.py,train.py}',
    points: [
      { title: 'Adam 是带状态的更新器', body: 'm 记录动量, v 记录平方梯度尺度, bias correction 修正训练早期估计偏小。' },
      { title: '训练并行, 生成串行', body: '训练有完整答案 y, 所有位置可同时算 CE; 生成时下一个 token 依赖刚采出的 token。' },
      { title: 'temperature/top-k 是 logits 后处理', body: '它不改变模型参数, 只改变从概率分布取样的方式。' },
    ],
    links: [
      { from: 'Adam', to: 'llm_train stability', body: '后续会加 warmup、cosine、grad clip 和混合精度保护。' },
      { from: 'sample.py', to: 'llm_infer sampling', body: '推理章节会把采样扩成 top-p、min-p、repetition penalty。' },
      { from: 'ckpt.npz', to: '部署权重', body: '保存参数是从训练切到推理的边界。' },
    ],
    sourceRows: [
      { concept: '一阶矩', code: 'optim.py: m = beta1*m + ...', takeaway: '平滑梯度方向。' },
      { concept: '二阶矩', code: 'optim.py: v = beta2*v + ...', takeaway: '按参数维度缩放学习率。' },
      { concept: '自回归循环', code: 'sample.py', takeaway: '取最后 token logits, 采样, append, 再 forward。' },
    ],
    snippetTitle: 'Adam 核心',
    snippet: `m = beta1 * m + (1 - beta1) * g
v = beta2 * v + (1 - beta2) * (g * g)

m_hat = m / (1 - beta1 ** t)
v_hat = v / (1 - beta2 ** t)
W -= lr * m_hat / (sqrt(v_hat) + eps)`,
    run: 'python llm_basic/sample.py',
  },

  'train-batch-ddp': {
    title: 'batch 与 DDP · 先切数据, 再同步梯度',
    subtitle: '规模化训练第一步不是切模型, 而是让多张卡处理不同样本并得到同一个全局梯度。',
    tldr: '梯度累积解决单卡显存, DDP 解决多卡吞吐; 两者都必须对齐 full-batch mean reduction。',
    question: 'micro-batch 和 data-parallel rank 都在切 batch, 它们的梯度缩放有什么不同?',
    code: 'llm_train/m01_gradient_accumulation/demo.py · llm_train/m02_data_parallel/demo.py',
    points: [
      { title: '梯度累积', body: 'rank 内拆 micro-batch, 每个 micro 反向后暂不更新, 累够再 step。' },
      { title: '数据并行', body: 'rank 间各算本地 batch, all-reduce mean 后每个 replica 用同一份梯度更新。' },
      { title: '验证方式', body: 'demo 都先算 dense baseline, 再 assert parallel path 与 baseline 一致。' },
    ],
    links: [
      { from: 'llm_basic train loop', to: 'gradient accumulation', body: 'update 频率从每个 batch 一次变成每 N 个 micro 一次。' },
      { from: 'local grads', to: 'all_reduce_mean', body: '本地梯度只是全局 batch 的一片, 必须同步。' },
      { from: 'DDP', to: 'ZeRO', body: 'DDP 复制完整状态; 状态放不下时才需要分片。' },
    ],
    sourceRows: [
      { concept: 'micro 权重', code: 'add_inplace(accum, grads, scale=micro_size / len(x))', takeaway: '匹配 full-batch mean。' },
      { concept: '梯度同步', code: 'all_reduce_mean([g[name] for g in local_grads])', takeaway: '每个 rank 得到相同 synced_grad。' },
      { concept: '通信成本', code: 'gradient all-reduce payload', takeaway: 'DDP 每步同步全模型梯度。' },
    ],
    snippetTitle: 'DDP 一步',
    snippet: `for rank in ranks:
    xb = split(x, world)[rank]
    _, grads = replica.loss_and_grads(xb, yb)
    local_grads.append(grads)

synced = all_reduce_mean(local_grads)
for replica in replicas:
    replica.apply_grads(synced, lr)`,
    run: 'python -m llm_train.m02_data_parallel.demo',
  },
  'train-model-parallel': {
    title: 'TP / PP 切模型 · 层内切矩阵, 层间切流水线',
    subtitle: '当模型本身放不进一张卡, 才开始切参数和层。TP 与 PP 的通信位置完全不同。',
    tldr: 'Tensor Parallel 切单层矩阵宽度; Pipeline Parallel 切层并用 micro-batch 填流水线 bubble。',
    question: '为什么 TP 需要层内 all-reduce, PP 却主要在 stage 边界传激活?',
    code: 'llm_train/m03_tensor_parallel/demo.py · llm_train/m04_pipeline_parallel/demo.py',
    points: [
      { title: 'TP 切层内矩阵', body: 'W1 column-parallel 产 hidden shard, W2 row-parallel 产 partial output 后求和。' },
      { title: 'PP 切层间执行', body: '不同 stage 持有不同层, micro-batch 交错通过 forward/backward wave。' },
      { title: '吞吐换复杂度', body: '并行越细, 通信、排程、激活驻留和重算策略越重要。' },
    ],
    links: [
      { from: 'llm_models Block', to: 'TP', body: 'TP 通常落在 attention/MLP 的 linear 矩阵上。' },
      { from: 'num_layers', to: 'PP', body: '层越深, 越自然按连续层切 stage。' },
      { from: 'micro-batch', to: 'pipeline bubble', body: 'micro 越多, bubble 占比越低。' },
    ],
    sourceRows: [
      { concept: 'column parallel', code: 'W1_shards = split(W1, axis=1)', takeaway: '每个 rank 计算 hidden 的一段。' },
      { concept: 'row parallel', code: 'tp_out = all_reduce_sum(partial_outs)[0] + b2', takeaway: 'partial output 汇总成 dense 输出。' },
      { concept: 'GPipe 排程', code: 'gpipe_schedule(stages, micro_batches)', takeaway: '离散时间表直接显示 bubble。' },
    ],
    snippetTitle: 'TP MLP 前向',
    snippet: `z_shards = [x @ W1_s + b1_s for W1_s, b1_s in shards]
h_shards = [relu(z) for z in z_shards]
partial_outs = [h @ W2_s for h, W2_s in zip(h_shards, W2_shards)]
tp_out = all_reduce_sum(partial_outs)[0] + b2`,
    run: 'python -m llm_train.m03_tensor_parallel.demo',
  },
  'train-memory': {
    title: '状态与显存 · ZeRO、checkpoint 与恢复',
    subtitle: '大模型训练的显存不只被参数占用, 梯度、优化器状态、激活和 checkpoint 状态同样关键。',
    tldr: 'ZeRO/FSDP 分片训练状态; activation checkpoint 用重算换激活显存; resume 要恢复模型、优化器、数据游标和随机状态。',
    question: '为什么只保存 model weights 不能称为可恢复训练?',
    code: 'llm_train/m05_zero_fsdp/demo.py · m07_activation_checkpointing/demo.py · m08_checkpoint_resume/demo.py',
    points: [
      { title: '状态分片', body: 'DDP 每卡保存 param/grad/Adam m/Adam v; ZeRO 逐步把这些状态切到不同 rank。' },
      { title: '激活重算', body: '不保存所有 forward activation, backward 前重跑部分 forward 得到中间量。' },
      { title: '完整恢复', body: '训练恢复后必须继续同一条数据流和随机轨迹, 否则 loss 曲线不可对照。' },
    ],
    links: [
      { from: 'DDP full state', to: 'ZeRO-3/FSDP shards', body: '复制换成分片, 显存下降但通信增加。' },
      { from: 'cache in backward', to: 'activation checkpoint', body: 'basic 里 cache 全存; 大模型里部分 cache 选择重算。' },
      { from: 'ToyDataStream.cursor', to: 'resume', body: '数据游标也是 checkpoint 的一部分。' },
    ],
    sourceRows: [
      { concept: 'reduce-scatter', code: 'grad_shards = reduce_scatter_sum(...)', takeaway: '求和后只保留本 rank 梯度 shard。' },
      { concept: 'all-gather', code: 'gathered = all_gather(updated_shards)[0]', takeaway: '需要完整参数时再聚合。' },
      { concept: 'resume payload', code: 'm08_checkpoint_resume/demo.py', takeaway: '保存 step、model、optimizer、data/RNG 状态。' },
    ],
    snippetTitle: 'ZeRO-style update',
    snippet: `grad_shards = reduce_scatter_sum(grad_per_rank, axis=0)
param_shards = split(param, world, axis=0)
updated = [p - lr * g for p, g in zip(param_shards, grad_shards)]
full_param = all_gather(updated, axis=0)[0]`,
    run: 'python -m llm_train.m05_zero_fsdp.demo',
  },
  'train-precision-stability': {
    title: '精度与稳定性 · AMP 让训练更快, guard 让训练不炸',
    subtitle: '混合精度降低显存和带宽, 但数值范围变窄; 稳定性组件负责把坏 step 拦住。',
    tldr: 'FP16/BF16 用于计算, FP32 master 保存更新精度; loss scaling、grad clip、warmup/cosine、NaN guard 共同保护长训练。',
    question: '为什么 FP16 训练不是简单把所有数组 astype(np.float16)?',
    code: 'llm_train/m06_mixed_precision/demo.py · llm_train/m10_training_stability/demo.py',
    points: [
      { title: '小梯度下溢', body: 'FP16 最小可表示范围有限, 小梯度直接 cast 可能变 0。' },
      { title: 'loss scaling', body: '反向前放大 loss/grad, 同步或更新前再除回去, 保住小梯度信号。' },
      { title: '坏 step 防护', body: 'warmup 降低训练早期冲击, grad clip 限制全局范数, NaN guard 阻止污染权重。' },
    ],
    links: [
      { from: 'Adam state', to: 'fp32 master', body: '参数更新仍要保留足够精度。' },
      { from: 'global grad norm', to: 'DDP/FSDP', body: '裁剪应发生在全局同步梯度之后。' },
      { from: 'long run', to: 'checkpoint', body: '检测到坏 step 后要能跳过或恢复。' },
    ],
    sourceRows: [
      { concept: 'loss scale', code: 'scaled_fp16 = (tiny_grad * scale).astype(np.float16)', takeaway: '放大后再 cast, 避免直接归零。' },
      { concept: 'master weight', code: 'master_w -= lr * update_grad', takeaway: '更新在 fp32 master 上发生。' },
      { concept: 'dynamic scale', code: 'LossScaler.update(has_overflow)', takeaway: '好 step 增大 scale, 溢出时回退。' },
    ],
    snippetTitle: 'AMP 思路',
    snippet: `scaled_grad = grad * loss_scale
fp16_grad = scaled_grad.astype(np.float16)
unscaled = fp16_grad.astype(np.float32) / loss_scale

if has_overflow:
    loss_scale *= 0.5`,
    run: 'python -m llm_train.m06_mixed_precision.demo',
  },
  'train-collectives-loop': {
    title: '通信与 full_loop · 从原语到完整训练主循环',
    subtitle: '所有并行策略最后都会落到少数通信原语。full_loop 把前面模块串成可运行控制流。',
    tldr: 'all-reduce、reduce-scatter、all-gather、all-to-all 分别支撑 DDP、ZeRO/FSDP、参数聚合和 MoE 路由。',
    question: '看一个训练框架时, 能否先把它的通信路径还原成这四个原语?',
    code: 'llm_train/core/collectives.py · llm_train/m09_collectives/demo.py · llm_train/full_loop/demo.py',
    points: [
      { title: '通信原语是成本中心', body: '并行策略的主要差别常常不是数学公式, 而是每一步要搬什么、搬多少、何时搬。' },
      { title: 'full_loop 是合成章', body: '它把 rank split、micro accumulation、AMP scale、clip、ZeRO update、checkpoint 放进同一步。' },
      { title: '读真实框架的入口', body: '先找 batch 怎么切、状态怎么切、通信怎么走, 再看框架封装。' },
    ],
    links: [
      { from: 'all-reduce', to: 'DDP', body: '每个 rank 得到完整平均梯度。' },
      { from: 'reduce-scatter + all-gather', to: 'FSDP', body: '梯度/参数在 shard 与 full view 之间切换。' },
      { from: 'all-to-all', to: 'MoE', body: 'token 按专家路由跨 rank 交换。' },
    ],
    sourceRows: [
      { concept: '原语实现', code: 'core/collectives.py', takeaway: '用 Python list 模拟 rank, 数据移动完全可见。' },
      { concept: '完整一步', code: 'distributed_step(...)', takeaway: '组合 DDP、累积、AMP、clip、ZeRO-style update。' },
      { concept: '最终验证', code: 'assert end_loss < start_loss', takeaway: '组合系统仍能训练下降。' },
    ],
    snippetTitle: 'full_loop 主干',
    snippet: `for rank, replica in enumerate(replicas):
    accum = zeros_like(replica.params())
    for xb, yb in micro_batches(rank):
        loss, grads = replica.loss_and_grads(xb, yb)
        accum += unscale(scale(grads))

synced = average_grad_trees(local_grads)
clipped = clip_by_global_norm(synced)
zero_style_sgd_step(replicas[0], clipped)`,
    run: 'python -m llm_train.full_loop.demo',
  },

  'finetune-sft': {
    title: 'SFT 数据与 loss · 只教模型回答, 不教模型提问',
    subtitle: 'SFT 与预训练都用 next-token CE, 真正的差别在数据格式和 labels mask。',
    tldr: 'instruction/prompt 区域 labels 置 -100, cross_entropy 只在 response token 上计算。',
    question: '为什么 prompt token 不应该贡献 loss?',
    code: 'llm_finetune/methods/sft.py · llm_finetune/data/instruction_data.py',
    points: [
      { title: '数据目标改变', body: '预训练学接龙, SFT 学在用户指令后输出符合任务的 response。' },
      { title: 'mask 是关键代码', body: 'labels 中 prompt/pad 位置为 -100, PyTorch CE 会 ignore。' },
      { title: 'SFT 是对齐起点', body: 'LoRA 可以承载 SFT, DPO 通常以 SFT 模型作为 policy/ref 起点。' },
    ],
    links: [
      { from: 'Standard LM loss', to: 'SFTLoss', body: 'loss 形式相同, labels 构造不同。' },
      { from: 'instruction_data', to: 'train_sft.py', body: '样本格式决定哪些 token 被优化。' },
      { from: 'SFT model', to: 'DPO reference', body: '偏好对齐通常从 SFT checkpoint 复制 ref。' },
    ],
    sourceRows: [
      { concept: 'ignore_index', code: 'SFTLoss.compute(... ignore_index=-100)', takeaway: '跳过 prompt 和 pad。' },
      { concept: 'response-only', code: 'instruction_data.py', takeaway: 'labels 只保留 answer 区域。' },
      { concept: '训练入口', code: 'run_finetune/sft/train_sft.py', takeaway: '把 dataset、loss、optimizer 接到一起。' },
    ],
    snippetTitle: 'SFT labels',
    snippet: `input  = [BOS, prompt..., response..., EOS]
labels = [-100, -100..., response..., EOS]

loss = cross_entropy(logits.reshape(-1, V),
                     labels.reshape(-1),
                     ignore_index=-100)`,
    run: 'python -m llm_finetune.run_finetune.sft.train_sft',
  },
  'finetune-lora': {
    title: 'LoRA 参数高效微调 · 冻结 W, 只学低秩 ΔW',
    subtitle: 'LoRA 不改变任务 loss, 改的是哪些参数允许更新。',
    tldr: '把线性层 y=Wx 替换成 y=Wx+(alpha/r)BAx; W 冻结, A/B 训练, 推理时可 merge 回 W。',
    question: '为什么 B 初始化为 0 是 LoRA 的关键安全设计?',
    code: 'llm_finetune/methods/lora.py · llm_finetune/utils/param_utils.py',
    points: [
      { title: '低秩增量', body: 'ΔW 用 B@A 表达, 参数量从 d_in*d_out 降到 r*(d_in+d_out)。' },
      { title: '无害启动', body: 'B=0 让训练第 1 步输出等于原模型, 不会一开始破坏 base 能力。' },
      { title: '部署灵活', body: 'adapter 可单独保存, 推理时 merge 成普通 Linear 或按请求动态加载。' },
    ],
    links: [
      { from: 'SFT loss', to: 'LoRA SFT', body: 'loss 不变, optimizer 只看到 A/B。' },
      { from: 'target_modules', to: '模型结构', body: '通常注入 q/k/v/o 或 FFN linear。' },
      { from: 'LoRA adapter', to: 'infer Multi-LoRA', body: '推理服务可以让多个 adapter 共享同一基座。' },
    ],
    sourceRows: [
      { concept: 'LoRALinear', code: 'methods/lora.py:LoRALinear', takeaway: 'base 分支 + adapter 分支。' },
      { concept: '注入', code: 'apply_lora(...)', takeaway: '按模块名替换目标 Linear。' },
      { concept: '落盘', code: 'get_lora_state_dict(...)', takeaway: '只保存 lora_A / lora_B。' },
    ],
    snippetTitle: 'LoRA forward',
    snippet: `base = linear(x, W, b)              # W frozen
delta = lora_B(lora_A(dropout(x))) * (alpha / r)
y = base + delta

# merge:
W <- W + (alpha / r) * (B @ A)`,
    run: 'python -m llm_finetune.run_finetune.lora.train_lora',
  },
  'finetune-dpo': {
    title: 'DPO 偏好对齐 · 用 chosen/rejected 直接优化策略',
    subtitle: 'DPO 把 RLHF 的 reward model + PPO 简化成一个基于偏好对的分类 loss。',
    tldr: '比较 policy 相对 reference 在 chosen 和 rejected 上的 log-prob 差, 用 logsigmoid 推高 chosen。',
    question: '为什么 DPO 需要冻结 reference model?',
    code: 'llm_finetune/methods/dpo.py · llm_finetune/data/preference_data.py',
    points: [
      { title: '偏好数据', body: '样本形态是 prompt + chosen + rejected, 不是单条标准答案。' },
      { title: '相对 ref', body: '优化 policy 的变化, 但用 ref 约束不要偏离 SFT 起点太远。' },
      { title: '序列级 log-prob', body: 'DPO 需要每个样本的 sum log p, 不能只要 batch 平均 CE。' },
    ],
    links: [
      { from: 'SFT checkpoint', to: 'policy/ref', body: 'policy 可训练, ref 冻结 eval + no_grad。' },
      { from: 'response-only labels', to: 'sequence_logprobs', body: '同样要忽略 prompt/pad。' },
      { from: 'DPO loss', to: '偏好边界', body: 'beta 控制贴近 ref 的强度。' },
    ],
    sourceRows: [
      { concept: 'logprob 汇总', code: 'compute_sequence_logprobs', takeaway: '返回 [B], 保留逐样本粒度。' },
      { concept: 'DPO logit', code: 'beta * ((p_c-p_r) - (ref_c-ref_r))', takeaway: 'chosen 与 rejected 的相对优势。' },
      { concept: '训练步', code: 'DPOTrainer.train_step', takeaway: 'policy 两次前向, ref 两次 no_grad 前向。' },
    ],
    snippetTitle: 'DPO 核心',
    snippet: `pi_logratios  = policy_chosen - policy_rejected
ref_logratios = ref_chosen - ref_rejected
logits = beta * (pi_logratios - ref_logratios)

loss = -F.logsigmoid(logits).mean()`,
    run: 'python -m llm_finetune.run_finetune.dpo.train_dpo',
  },
  'finetune-runs': {
    title: '训练脚本与落盘 · 从方法函数到可复现实验',
    subtitle: 'SFT/LoRA/DPO 的方法文件只定义算法, run_finetune 负责把数据、模型、优化器和保存策略接起来。',
    tldr: '读 run_finetune 时重点看四件事: 数据如何 batch, 哪些参数 requires_grad, loss 输入是什么, checkpoint 保存什么。',
    question: '一个 LoRA 实验如果只保存完整模型而不保存 adapter, 部署灵活性会丢掉什么?',
    code: 'llm_finetune/run_finetune/{sft,lora,dpo}/train_*.py',
    points: [
      { title: '方法与脚本分层', body: 'methods/ 放可复用算法, run_finetune/ 放一次实验的编排。' },
      { title: '参数过滤', body: 'LoRA 训练脚本应确认只有 adapter 参数可训练, base 参数冻结。' },
      { title: '产物边界', body: '全参 SFT/DPO 保存完整权重; LoRA 更适合保存 adapter state_dict。' },
    ],
    links: [
      { from: 'methods/*.py', to: 'train_*.py', body: '算法类被训练脚本实例化并喂入 batch。' },
      { from: 'param_utils', to: '可训练参数统计', body: '确认 PEFT 是否真的只训练少量参数。' },
      { from: 'adapter state', to: 'llm_infer/m13', body: 'Multi-LoRA serving 依赖 adapter 可独立切换。' },
    ],
    sourceRows: [
      { concept: 'SFT run', code: 'run_finetune/sft/train_sft.py', takeaway: 'instruction batch + SFTLoss。' },
      { concept: 'LoRA run', code: 'run_finetune/lora/train_lora.py', takeaway: 'apply_lora 后只优化 adapter。' },
      { concept: 'DPO run', code: 'run_finetune/dpo/train_dpo.py', takeaway: 'policy/ref 双模型路径。' },
    ],
    snippetTitle: '实验阅读清单',
    snippet: `1. dataset / collate_fn 产出哪些字段?
2. labels 中哪些位置是 -100?
3. 哪些参数 requires_grad=True?
4. loss 需要 policy/ref/chosen/rejected 哪些输入?
5. checkpoint 保存 full model 还是 adapter?`,
    run: 'python -m llm_finetune.run_finetune.lora.train_lora',
  },

  'infer-kv-memory': {
    widgets: ['AttnMaskLab'],
    title: 'KV 与缓存内存 · 推理优化从少算开始',
    subtitle: '训练关注参数和激活, 推理服务还要把每个请求的 KV cache 当成可分配资源管理。',
    tldr: 'KV cache 避免重复算旧 token; PagedAttention 把 KV 切成 block; prefix/radix cache 复用请求之间的公共前缀。',
    question: '为什么长上下文推理的瓶颈经常不是模型权重, 而是 KV cache?',
    code: 'llm_infer/m01_kv_cache · m02_paged_attention · m04_prefix_cache · m05_radix_cache',
    points: [
      { title: 'KV cache', body: 'prefill 保存每层 K/V, decode 只追加新 token 的 K/V。' },
      { title: 'PagedAttention', body: '逻辑 token 页通过 block_table 指向物理 KV block, 减少碎片。' },
      { title: '前缀复用', body: '相同 system prompt 或共享前缀不必重复 prefill, 只增加 block ref_count。' },
    ],
    links: [
      { from: 'sample.py', to: 'KV cache', body: '同样自回归, 但不再每步重算整段。' },
      { from: 'BlockManager', to: 'Scheduler', body: '调度器先问显存块够不够, 再决定能否接请求。' },
      { from: 'prefix cache', to: 'full_engine', body: 'Engine._step_prefill 会先查前缀命中。' },
    ],
    sourceRows: [
      { concept: 'cache 对照', code: 'm01_kv_cache/demo.py', takeaway: 'no_cache 与 with_cache 输出 ids 必须一致。' },
      { concept: '页表', code: 'BlockManager.block_tables', takeaway: 'seq_id → physical block ids。' },
      { concept: '链式 hash', code: 'PrefixCache.match_prefix', takeaway: 'parent_hash + token block 命中物理 block。' },
    ],
    snippetTitle: 'KV cache 路径',
    snippet: `logits, kv_cache = lm.prefill(prompt_ids)
next_id = argmax(logits[-1])

for _ in range(max_new - 1):
    logits, kv_cache = lm.decode_step(next_id, kv_cache)
    next_id = argmax(logits)`,
    run: 'python -m llm_infer.m01_kv_cache.demo',
  },
  'infer-scheduler': {
    title: '调度与 prefill · 让请求动态组成 batch',
    subtitle: '真实服务里请求长度不同、到达时间不同, 静态 batch 很快浪费算力。',
    tldr: 'continuous batching 维护 waiting/running 队列; chunked prefill 避免长 prompt 独占; P/D 分离把 prefill 和 decode 放到不同资源池。',
    question: '为什么 prefill 和 decode 最好分开调度?',
    code: 'llm_infer/m03_continuous_batching · m06_chunked_prefill · m15_pd_disaggregation',
    points: [
      { title: 'prefill 与 decode 负载不同', body: 'prefill 是大矩阵、大 token 数; decode 是小 batch、强内存带宽和低延迟。' },
      { title: '动态 batch', body: 'running 中每条请求每步 decode 一个 token, 新请求在 waiting 里等待 prefill。' },
      { title: '资源解耦', body: 'P/D disaggregation 让长 prompt 的 prefill 不阻塞 decode 节点。' },
    ],
    links: [
      { from: 'BlockManager.can_allocate', to: 'Scheduler.schedule', body: '显存容量决定本步能接哪些请求。' },
      { from: 'chunked prefill', to: '首 token 延迟', body: '长输入拆块后, decode 请求有机会插队。' },
      { from: 'KVTransport', to: 'P/D 分离', body: 'prefill 节点把 KV 搬给 decode 节点。' },
    ],
    sourceRows: [
      { concept: '队列', code: 'Scheduler.waiting / running', takeaway: '请求生命周期显式可见。' },
      { concept: 'prefill-priority', code: 'schedule()', takeaway: 'waiting 非空先组 prefill batch。' },
      { concept: '跨节点 KV', code: 'm15_pd_disaggregation:KVTransport', takeaway: '收益来自负载解耦, 代价是 KV 传输。' },
    ],
    snippetTitle: '调度骨架',
    snippet: `if self.waiting:
    picked = pick_prefill_batch(token_budget, block_budget)
    return picked, Stage.PREFILL

picked = schedule_decode_running()
return picked, Stage.DECODE`,
    run: 'python -m llm_infer.m03_continuous_batching.demo',
  },
  'infer-decode-control': {
    widgets: ['SoftmaxTempLab'],
    title: '解码加速与约束 · 更少 target calls, 更可控输出',
    subtitle: 'decode 阶段每步只出少量 token, 所以减少 target 前向次数和控制 logits 都很重要。',
    tldr: 'Speculative decoding 用 draft 猜多个 token, target 批量验证; sampling/structured output 在 logits 层控制分布和合法性。',
    question: '为什么结构化输出不应该只靠 prompt 约束?',
    code: 'llm_infer/m07_speculative_decoding · m10_sampling · m14_structured_output',
    points: [
      { title: '投机解码', body: 'draft 便宜地产生 K 个候选, target 一次验证, 接受率越高越省 target calls。' },
      { title: '采样策略', body: 'temperature/top-k/top-p/min-p/repetition penalty 都是 logits 后处理。' },
      { title: '语法约束', body: 'FSM/grammar 把非法 token 置 -inf, 合法性由解码器保证, 不靠模型自觉。' },
    ],
    links: [
      { from: 'sample.py temperature/top-k', to: 'm10_sampling', body: '基础采样扩展为服务端参数。' },
      { from: 'draft model', to: 'target model', body: '加速来自一次 target forward 确认多个 token。' },
      { from: 'FSM legal_chars', to: 'logits mask', body: '控制输出格式的最后一道硬约束。' },
    ],
    sourceRows: [
      { concept: '接受规则', code: 'spec_decode(...)', takeaway: 'target.argmax 与 draft token 匹配就接受。' },
      { concept: '采样顺序', code: 'sample(...)', takeaway: 'rep penalty → temp → top-k/top-p/min-p。' },
      { concept: 'JSON FSM', code: 'JsonFSM.legal_chars', takeaway: '每一步只允许合法字符。' },
    ],
    snippetTitle: 'logits mask',
    snippet: `legal = fsm.legal_chars()
mask = full(vocab_size, -inf)
for ch in legal:
    mask[stoi[ch]] = 0

next_id = argmax(logits + mask)`,
    run: 'python -m llm_infer.m14_structured_output.demo',
  },
  'infer-compute': {
    title: '算子与压缩 · 降低每步计算和内存带宽',
    subtitle: '服务端吞吐经常被显存带宽、attention 中间矩阵和 kernel launch 开销限制。',
    tldr: 'Quantization 降带宽和显存; FlashAttention 不落完整 T×T; CUDA Graph 复用固定形状执行图; 推理 TP 切大矩阵。',
    question: '为什么 FlashAttention 的核心收益是显存峰值, 不只是 FLOPs?',
    code: 'llm_infer/m08_quantization · m09_tensor_parallel · m11_flash_attention · m12_cuda_graph',
    points: [
      { title: '量化', body: '权重/KV 用 int8 等低比特存储, scale 控制反量化误差。' },
      { title: 'FlashAttention', body: 'online softmax 分块累计 max/sum/O, 不保存完整 attention matrix。' },
      { title: 'CUDA Graph', body: '固定 shape decode 捕获后 replay, 减少小 batch kernel launch 开销。' },
    ],
    links: [
      { from: 'GQA/MLA', to: 'KV 规模', body: '阶段 2 的注意力结构会直接影响推理 KV 内存。' },
      { from: 'int8_weight', to: 'matmul bandwidth', body: 'W_q + scale 代替 fp32 W 读取。' },
      { from: 'TP', to: '多卡推理', body: '与训练 TP 同源, 但目标偏向 latency/throughput。' },
    ],
    sourceRows: [
      { concept: 'per-channel INT8', code: 'quantize_int8(W)', takeaway: '每个输出通道独立 scale。' },
      { concept: 'online softmax', code: 'flash_attention.py', takeaway: 'm/l/O 随 KV block 增量更新。' },
      { concept: '图捕获', code: 'm12_cuda_graph/demo.py', takeaway: 'capture 固定计算, replay 降调度开销。' },
    ],
    snippetTitle: 'FlashAttention 状态',
    snippet: `m_new = maximum(m, max(S_block))
P_b = exp(S_block - m_new)
l = exp(m - m_new) * l + sum(P_b)
O = exp(m - m_new) * O + P_b @ V_block
m = m_new`,
    run: 'python -m llm_infer.m11_flash_attention.demo',
  },
  'infer-engine': {
    title: 'mini-vLLM 引擎 · 把模块接成服务主循环',
    subtitle: '单个 demo 讲算法, full_engine 讲接口、状态和资源账本如何一起工作。',
    tldr: 'Engine.add_request 接请求, step 在 prefill/decode 间切换, BlockManager 管 KV block, PrefixCache 复用前缀, SamplingParams 控制输出。',
    question: '为什么一个推理引擎首先是调度器和资源管理器, 其次才是 model.forward 包装?',
    code: 'llm_infer/full_engine/engine.py · llm_infer/m13_lora_serving/demo.py',
    points: [
      { title: '请求生命周期', body: 'waiting → prefill → running/decode → finished, 每一步都更新 KV 和 block table。' },
      { title: '模块集成', body: 'full_engine 集成 m02/m03/m04/m10, 用同一条控制流解释 mini-vLLM。' },
      { title: '服务增强', body: 'Multi-LoRA serving 让不同请求共享基座权重但使用不同 adapter。' },
    ],
    links: [
      { from: 'add_request', to: 'waiting queue', body: 'prompt encode 后进入调度系统。' },
      { from: '_step_prefill', to: 'prefix cache + first token', body: '命中复用 KV, 未命中分配 block 并 prefill。' },
      { from: '_step_decode', to: 'continuous batching', body: 'running 请求每步追加一个 token。' },
    ],
    sourceRows: [
      { concept: '入口 API', code: 'Engine.generate(...)', takeaway: '类似 vLLM 的离线接口。' },
      { concept: 'prefill 资源账本', code: 'Engine._step_prefill', takeaway: 'prefix hits、block allocation、cache register。' },
      { concept: 'decode 完成释放', code: 'Engine._step_decode', takeaway: 'is_finished 后 bm.free(seq_id)。' },
    ],
    snippetTitle: 'Engine.step',
    snippet: `def step(self):
    self.stats_step += 1
    if self.waiting:
        return self._step_prefill()
    return self._step_decode()`,
    run: 'python -m llm_infer.full_engine.demo',
  },

  'agent-loop': {
    title: 'Agent loop · 从纯生成到可行动闭环',
    subtitle: 'Agent 的最小单元不是一次 completion, 而是一轮“模型决定下一步 → harness 执行 → 结果回填”。',
    tldr: 'core/agent.py 里只有一条很薄的 loop: user message 进入 transcript, RuleBasedLLM 产出 tool/final, permission gate 拦截, tool result 再进入上下文。',
    question: '为什么复制 while-loop 很容易, 但复制一个可靠 Agent 很难?',
    code: 'llm_agent/core/agent.py · llm_agent/m01_agent_loop/demo.py',
    points: [
      { title: '模型只选动作', body: '教学版 RuleBasedLLM 不接真实 API, 只负责返回 ModelAction: tool 或 final。' },
      { title: 'harness 执行动作', body: 'Agent.run 负责上下文组装、权限评估、Hook 调度、工具执行和 transcript 写入。' },
      { title: '结果进入下一轮', body: 'tool result 作为 role=tool 的 Message 回到消息流, 模型再决定继续行动还是最终回答。' },
    ],
    links: [
      { from: 'llm_infer.generate', to: 'Agent.run', body: '推理只生成 token; Agent 把生成结果解释成外部动作。' },
      { from: 'ModelAction.tool', to: 'ToolRegistry.execute', body: '模型输出不直接执行, 先变成结构化 ToolCall。' },
      { from: 'ToolResult', to: 'Message(role=tool)', body: '观察结果回填后, 下一轮才有证据继续推理。' },
    ],
    sourceRows: [
      { concept: '主循环', code: 'Agent.run(... for turn in range(max_turns))', takeaway: '每轮只处理一个 action, 直到 final 或 max_turns。' },
      { concept: '上下文组装', code: '_assemble_context(...)', takeaway: 'system prompt、memory、history 统一进入模型视野。' },
      { concept: '停止条件', code: 'action.kind == "final"', takeaway: '没有工具调用时直接产出最终回答。' },
    ],
    snippetTitle: 'Agent loop 骨架',
    snippet: `messages.append(user_prompt)
while not stopped:
    context = assemble(system, memory, messages)
    action = model.next(context, tools)
    if action.kind == "final":
        return action.content

    if permission_gate.allows(action.tool_call):
        result = tool_registry.execute(action.tool_call)
        messages.append(tool_result_message(result))`,
    run: 'python -m llm_agent.m01_agent_loop.demo',
  },
  'agent-tools-permissions': {
    title: '工具与权限 · 能做什么, 必须先能拦什么',
    subtitle: 'Tool calling 把模型从“会说”扩展到“会做”, 权限系统则决定哪些动作能被执行。',
    tldr: 'm02 展示 schema/execute/result; m03 展示 deny-first、default ask 和 auto 风险分类。二者必须一起读。',
    question: '如果工具执行层完全信任模型输出, prompt injection 会落到哪里?',
    code: 'llm_agent/core/{tools.py,permissions.py} · m02_tool_use · m03_permissions',
    points: [
      { title: '工具是确定性边界', body: '每个 Tool 暴露 name/description/risk, execute 用普通 Python 完成实际动作。' },
      { title: '拒绝优先', body: 'PermissionGate 先匹配 deny 规则, 再看 allow, 最后按 mode 决定 ask/auto。' },
      { title: '风险按可逆性分层', body: 'calculator/search/read 是低风险; write_note 是 bounded write; shell 属高风险。' },
    ],
    links: [
      { from: 'Tool.schema()', to: 'model tool pool', body: '模型看到能力描述, 但不能绕过执行层。' },
      { from: 'PermissionRule', to: 'ToolCall', body: '规则按 tool 名和参数模式匹配。' },
      { from: 'auto classifier', to: 'human ask', body: '低风险自动放行, 不确定动作回退到人工策略。' },
    ],
    sourceRows: [
      { concept: '工具注册', code: 'ToolRegistry.register(tool)', takeaway: '工具池是模型可行动作集合。' },
      { concept: 'safe eval', code: '_safe_eval_arithmetic(expr)', takeaway: '即使是 calculator 也不用 Python eval。' },
      { concept: 'deny-first', code: 'PermissionGate.evaluate', takeaway: '广义拒绝规则先于狭义允许规则。' },
    ],
    snippetTitle: '权限评估顺序',
    snippet: `for rule in rules:
    if rule.decision == DENY and matches(rule, call):
        return deny()

for rule in rules:
    if rule.decision == ALLOW and matches(rule, call):
        return allow()

return mode_fallback(call)   # plan/default/auto/dont_ask`,
    run: 'python -m llm_agent.m03_permissions.demo',
  },
  'agent-context-memory': {
    widgets: ['RetrievalLab'],
    title: '上下文与记忆 · 模型到底看见什么',
    subtitle: 'Agent 的长期表现常常取决于上下文工程: 哪些信息进窗口, 什么时候压缩, 哪些状态留在文件里。',
    tldr: 'FileMemory 用 Markdown 文件做透明记忆; compact_messages 保留头尾, 摘要中间; Agent 在每轮前组装 system/memory/history。',
    question: '为什么“把所有历史都塞进 prompt”不是一个可扩展方案?',
    code: 'llm_agent/core/memory.py · llm_agent/m04_context_memory/demo.py',
    points: [
      { title: '文件记忆透明', body: '记忆是普通 .md 文件, 可读、可改、可版本控制, 不依赖黑盒向量库。' },
      { title: '检索只取相关片段', body: 'FileMemory.search 用关键词打分选少量文件, 演示 retrieval 的基本入口。' },
      { title: '压缩是渐进降级', body: '超预算时不丢全部历史, 而是保留开头/最近消息, 中间变 summary。' },
    ],
    links: [
      { from: 'session history', to: 'context window', body: '历史不是全部原样进入模型, 需要预算控制。' },
      { from: 'FileMemory.search', to: 'memory_messages', body: '相关文件变成 role=system 的 memory message。' },
      { from: 'compact summary', to: 'resume', body: '压缩后的 transcript 仍保持可解释。' },
    ],
    sourceRows: [
      { concept: '文件记忆', code: 'FileMemory.add(title, body)', takeaway: '记忆落到 Markdown 文件。' },
      { concept: '相关检索', code: 'FileMemory.search(query)', takeaway: '用词交集模拟最小 retrieval。' },
      { concept: '压缩', code: 'compact_messages(messages, max_chars)', takeaway: '保头尾, 压中间, 且控制预算。' },
    ],
    snippetTitle: '上下文组装',
    snippet: `base = [Message("system", system_prompt)]
base.extend(memory_messages(memory, current_prompt))

all_messages = base + transcript
context = compact_messages(all_messages, max_chars=budget)`,
    run: 'python -m llm_agent.m04_context_memory.demo',
  },
  'agent-extensibility': {
    title: 'Hooks / Skills / MCP · 扩展点按上下文成本分层',
    subtitle: '不是所有扩展都应该变成 prompt。能在执行前后用确定性代码解决的, 就不必消耗上下文窗口。',
    tldr: 'm05 用 HookManager 演示 session_start、UserPromptSubmit、PreToolUse、PostToolUse, 用 SkillRegistry 注入低成本策略, 用 FakeMCPServer 暴露外部工具。',
    question: '什么时候应该写 Hook, 什么时候应该写 Skill, 什么时候才需要 MCP?',
    code: 'llm_agent/core/hooks.py · llm_agent/m05_extensibility/demo.py',
    points: [
      { title: 'Hook 零上下文成本', body: 'pre_tool_use 可拦截危险参数, post_tool_use 可附加审计信息。' },
      { title: 'Skill 是轻量方法注入', body: '命中特定任务时把方法论追加到用户 prompt, 演示“按需加载”。' },
      { title: 'MCP-like 工具扩展能力面', body: 'FakeMCPServer 返回 WeatherTool, 模拟外部服务接入工具池。' },
    ],
    links: [
      { from: 'user_prompt_submit', to: 'skill instruction', body: '用户输入进入模型前可被补充上下文。' },
      { from: 'pre_tool_use', to: 'permission gate', body: '权限和 hook 是两层边界, 一个策略化, 一个可编程。' },
      { from: 'FakeMCPServer.list_tools', to: 'ToolRegistry.register', body: '外部能力最终仍以 Tool 进入统一执行面。' },
    ],
    sourceRows: [
      { concept: '事件注册', code: 'hooks.register(event, fn)', takeaway: '扩展点显式绑定生命周期事件。' },
      { concept: '阻断', code: 'HookResult(block=True, reason=...)', takeaway: 'Hook 可在工具执行前截停。' },
      { concept: '外部工具', code: 'FakeMCPServer().list_tools()', takeaway: '模拟 MCP server 暴露工具。' },
    ],
    snippetTitle: 'Hook 执行点',
    snippet: `submitted = hooks.on_user_prompt_submit(prompt)
action = model.next(context, tools)

pre = hooks.on_pre_tool_use(action.tool_call)
if pre.block:
    result = blocked_result(pre.reason)
else:
    result = tools.execute(pre.updated_call)

extra = hooks.on_post_tool_use(result)`,
    run: 'python -m llm_agent.m05_extensibility.demo',
  },
  'agent-state-subagents': {
    title: '持久化与子智能体 · 状态可恢复, 上下文要隔离',
    subtitle: 'Agent 系统需要能恢复、能审计, 同时不能让子任务的冗长轨迹污染父级上下文。',
    tldr: 'JsonlSessionStore 仅追加保存 transcript; resume 可加载旧消息, 但权限由新会话重新建立。DelegateTool 创建隔离 child agent, 父级只收到 summary。',
    question: '为什么恢复 transcript 不应该等于恢复 bypass 权限?',
    code: 'llm_agent/core/{persistence.py,subagents.py} · m06_persistence_resume · m07_subagents',
    points: [
      { title: 'append-only 审计', body: '每条 Message 单独写入 JSONL, 不就地改写历史。' },
      { title: 'resume 不继承信任', body: 'load_history=True 只恢复消息, PermissionGate 仍由新 Agent 配置。' },
      { title: '子智能体隔离', body: 'child 有自己的工具池、权限和 transcript, parent 只拿 summary。' },
    ],
    links: [
      { from: 'JsonlSessionStore.load', to: 'Agent(load_history=True)', body: '旧 transcript 作为新会话上下文。' },
      { from: 'PermissionGate(mode="default")', to: 'resume', body: '恢复状态和恢复权限是两回事。' },
      { from: 'DelegateTool.execute', to: 'child Agent', body: '子任务在独立上下文中运行。' },
    ],
    sourceRows: [
      { concept: 'JSONL append', code: 'JsonlSessionStore.append(message)', takeaway: '一行一条消息, 便于审计。' },
      { concept: '恢复', code: 'store.load()', takeaway: '从磁盘重建 Message 列表。' },
      { concept: '隔离委托', code: 'DelegateTool.execute', takeaway: 'child_transcripts 不进入 parent messages。' },
    ],
    snippetTitle: 'resume 与 delegation',
    snippet: `store = JsonlSessionStore(path)
agent1 = Agent(..., store=store)
agent1.run("搜索 agent loop")

agent2 = Agent(..., store=store, load_history=True,
               permissions=PermissionGate(mode="default"))

delegate = DelegateTool(docs)
parent = Agent(tools=ToolRegistry([delegate]))`,
    run: 'python -m llm_agent.m07_subagents.demo',
  },
  'agent-full-loop': {
    title: 'mini Agent harness · 把应用层机制串起来',
    subtitle: '单模块讲局部机制, full_loop 展示一个最小可运行的应用层系统。',
    tldr: 'full_loop/demo.py 同时接入 SearchDocsTool、WriteNoteTool、WeatherTool、ShellTool、DelegateTool、deny-first 权限、Hook、FileMemory 和 JSONL transcript。',
    question: '一个 Agent 产品的工程复杂度, 到底有多少在模型之外?',
    code: 'llm_agent/full_loop/demo.py · llm_agent/core/',
    points: [
      { title: '统一工具面', body: '内置工具、MCP-like 工具、子智能体都以 Tool 接入 ToolRegistry。' },
      { title: '多层边界', body: 'PermissionGate、Hook、ShellTool 自身模拟执行, 共同避免任意本地命令执行。' },
      { title: '状态闭环', body: 'memory 提供长期偏好, store 提供 transcript, child transcript 保持隔离。' },
    ],
    links: [
      { from: 'llm_infer/full_engine', to: 'llm_agent/full_loop', body: '前者服务 token, 后者编排行动。' },
      { from: 'Agent.run', to: 'full_loop/demo.py', body: '同一 loop 在多种工具和扩展下保持不变。' },
      { from: 'run_all.py', to: 'README 学习路径', body: '模块可独立运行, 也可组合整体运行。' },
    ],
    sourceRows: [
      { concept: '组合入口', code: 'full_loop/demo.py:main', takeaway: '所有 core 机制在一处组装。' },
      { concept: '安全规则', code: 'PermissionRule("shell", "*rm -rf*", DENY)', takeaway: '高风险动作被 deterministic guard 拦截。' },
      { concept: '运行统计', code: 'store.count(), child_transcripts', takeaway: '验证状态确实落盘和隔离。' },
    ],
    snippetTitle: 'full_loop 组装',
    snippet: `tools = ToolRegistry([
    SearchDocsTool(DOCS), WriteNoteTool(notes),
    WeatherTool(), ShellTool(), DelegateTool(DOCS),
])
permissions = PermissionGate(
    mode="auto",
    rules=[PermissionRule("shell", "*rm -rf*", DENY)],
)
agent = Agent(tools=tools, permissions=permissions,
              hooks=build_hooks(), memory=memory, store=store)
agent.run("排查 agent loop，并写入笔记")`,
    run: 'python -m llm_agent.full_loop.demo',
  },

  'models-mtp': {
    widgets: ['AttnMaskLab', 'MtpLab'],
    title: 'SWA · MTP · 混合线性 — 三种降本增效的改法',
    subtitle: 'Mistral 把注意力裁成带状; MTP 让每个位置一次预测多个 token; Qwen3-Next 干脆用固定大小状态替掉 75% 的 KV cache。',
    tldr: 'SWA 只是换一张 mask (注意力本体不变), KV cache 从 O(T) 封顶到 O(W); MTP 用共享 head 的级联模块把监督信号加密 K+1 倍, 推理时还白送投机解码草稿。',
    question: '局部注意力会不会掐断长程信息? 一次预测多个 token 为什么不破坏因果性?',
    code: 'llm_models/models/language_models/{mistral.py,mtp.py,qwen3_next.py} · llm_models/layers/sparse/linear_attention.py',
    points: [
      { title: '深度替代宽度', body: 'SWA 单层只看 W 个位置, 但信息跨层接力: L 层理论感受野 ≈ L·W (Mistral: 32×4096 ≈ 131K)。' },
      { title: 'mask 即架构', body: '全因果=LLaMA, 带状=Mistral, 带状+sink=GPT-OSS, top-k=DSA。QKV 投影一行不改, 谁可见谁全由 mask 决定。' },
      { title: '级联保因果', body: 'MTP-k 在位置 i 拼接真实 t_{i+k} 的 embedding 再过一个 Block — 每深一级多看一个 token, 因果链完整。' },
      { title: '状态替代缓存', body: 'Gated DeltaNet 用 Dh×Dh 状态矩阵替代逐 token 的 KV: delta rule 精准覆写 + α 门整体衰减, O(1) "cache"; 混 25% 全注意力兜底召回。' },
    ],
    links: [
      { from: 'build_sliding_window_mask', to: 'Mistral.forward', body: '与 LLaMA 唯一的结构性差异就是这张带状 mask。' },
      { from: 'MTPModule', to: 'lm_head (共享)', body: '每级只新增拼接投影 + 1 个 Block, embedding 与输出头全部共享。' },
      { from: 'mtp_logits', to: 'speculative decoding', body: 'MTP 对 t+2 的预测可直接当 draft, 对应 llm_infer/m07。' },
    ],
    sourceRows: [
      { concept: '带状掩码', code: 'masks.py:build_sliding_window_mask', takeaway: '(j <= i) 且 (j > i-W), 可选保留开头 S 个 sink。' },
      { concept: 'KV 上限', code: 'mistral.py:kv_cache_entries', takeaway: 'rolling buffer: min(T, W), 与流长度无关。' },
      { concept: '级联输入', code: 'mtp.py:MTPModule.forward', takeaway: '各自 RMSNorm 后拼接 [h; emb], 投影回 d 再过 Block。' },
      { concept: '标签对齐', code: 'mtp.py:MTPLoss.compute', takeaway: '第 k 级目标 = labels 左移 k 位, 末尾 -100 屏蔽。' },
      { concept: 'delta rule', code: 'layers/sparse/linear_attention.py', takeaway: 'S ← α(S - βk(kᵀS)) + βkvᵀ: 先擦 k 方向旧值再写新值。' },
      { concept: '混合排布', code: 'qwen3_next.py:layer_types', takeaway: '[Δ,Δ,Δ,A,...] 周期排布, mask/rope 对 Δ 层是 no-op。' },
    ],
    snippetTitle: 'SWA 与 MTP 的核心差异行',
    snippet: `# Mistral = LLaMA + 一张带状 mask
visible = (j <= i) & (j > i - window_size)      # 谁能看见谁

# MTP 级联: 位置 i 的第 k 级多看一个真实 token
shifted[:, :-k] = idx[:, k:]                     # teacher forcing
h = mtp_block(cat([norm(h), norm(emb(shifted))]))
loss = ce_main + lam * mean(ce_mtp_k)            # 联合训练`,
    run: 'python -m llm_models.run_models.language_models.mistral.infer_mistral',
  },

  'train-moe-seq': {
    widgets: ['MoeRouteLab', 'RingAttnLab'],
    title: 'EP 与序列并行 · 切专家, 切序列',
    subtitle: '模型并行切的是"层和矩阵"; MoE 时代还要切专家 (EP), 长上下文时代还要切序列本身 (CP)。',
    tldr: 'EP 把专家分卡, token 经两次 all-to-all 找专家再回家, 路由均衡是生死线; 序列并行把一条超长序列切给多张卡, KV 块沿环传递, online softmax 增量合并保证数值精确等价。',
    question: 'all-to-all 的通信量为什么取决于数据 (路由结果) 而不是模型结构? Ring Attention 为什么能做到数值上与完整注意力完全一致?',
    code: 'llm_train/m11_expert_parallel/demo.py · llm_train/m12_sequence_parallel/demo.py',
    points: [
      { title: '路由决定通信', body: 'DDP 的 all-reduce 通信量固定; EP 的 all-to-all 由 gating 结果决定 — 倾斜路由 = 热点卡 + 容量溢出丢 token。' },
      { title: '均衡是训练目标', body: 'Switch/Mixtral 用 aux loss, DeepSeek-V3 用 bias 调节 — 殊途同归: 把热门专家的路由概率压平。' },
      { title: '同一个 online softmax', body: '分块算注意力再增量合并: 单卡内做是 FlashAttention, 跨卡传块就是 Ring Attention。' },
    ],
    links: [
      { from: 'gating top-k', to: 'all-to-all dispatch', body: 'token 在哪张卡 ≠ 它的专家在哪张卡, 必须重排。' },
      { from: 'capacity factor', to: 'token dropping', body: '每个专家最多收 capacity 个, 溢出 token 走残差。' },
      { from: 'KV 环传递', to: 'online softmax (m, l, acc)', body: '收到新块就增量合并, 任何时刻只持有 1/D 的 KV。' },
    ],
    sourceRows: [
      { concept: 'dispatch 矩阵', code: 'm11:moe_forward_ep', takeaway: '行=源卡 列=目标卡, 这张表就是 all-to-all 的发货单。' },
      { concept: 'aux loss 梯度', code: 'm11:balance_router', takeaway: 'L = E·Σ f_e·P_e, f 视为常数, 对 P 求导推平路由。' },
      { concept: '环上一步', code: 'm12:ring_attention', takeaway: '卡 r 在 step s 处理来自卡 (r-s)%D 的 KV 块。' },
      { concept: '增量合并', code: 'm12: m_new / scale', takeaway: '修正旧累计量的指数缩放, 数值与一次性 softmax 等价 (~1e-16)。' },
    ],
    snippetTitle: 'EP 与 Ring Attention 的控制流',
    snippet: `# EP: 两次 all-to-all 夹一段本地专家计算
recv = all_to_all(tokens_by_dst)      # dispatch
out_local = expert(recv)              # 只算自己持有的专家
out = all_to_all(out_local)           # combine 回原卡

# Ring: D 步之后每张卡都见过完整序列
for step in range(D):
    src = (rank - step) % D           # 本步处理谁的 KV 块
    m, l, acc = online_merge(Q_local @ K[src].T, V[src])
    send_to_next(K[src], V[src])      # 环传, 可与计算重叠`,
    run: 'python -m llm_train.m11_expert_parallel.demo',
  },

  'finetune-rlhf': {
    widgets: ['GrpoLab', 'SoftmaxTempLab'],
    title: 'RM · GRPO · 蒸馏 — 从偏好到能力迁移',
    subtitle: 'DPO 之外的另一半对齐版图: 显式奖励模型、在线 RL (R1 配方)、以及把能力压进小模型的蒸馏。',
    tldr: 'RM 把"A 比 B 好"的序关系学成标量分; GRPO 用组内相对优势替代 critic, 配合可验证奖励 (RLVR) 连 RM 都能省; 蒸馏用温度软化的 teacher 分布给 student 提供比硬标签密得多的监督。',
    question: 'GRPO 砍掉 critic 之后, baseline 从哪里来? 为什么蒸馏的 KL 项要乘 T²?',
    code: 'llm_finetune/methods/{reward_model.py,grpo.py,distill.py}',
    points: [
      { title: '序关系 → 标量', body: 'Bradley-Terry: L = -log σ(r_chosen - r_rejected)。RM 的分数只有序意义, 没有量纲 — 它是 RL 阶段的罗盘。' },
      { title: '组内排名替代 critic', body: '同 prompt 采 G 条, Â = (r-μ)/σ。比组里平均好就强化 — PPO 的 value 网络整个省掉 (R1 配方)。' },
      { title: '暗知识在分布里', body: '硬标签只有 1 个 token 的信息; teacher 软化分布把相似 token 的相对排序全部交给 student。' },
    ],
    links: [
      { from: 'PreferenceDataGenerator', to: 'RewardModel', body: '与 DPO 同源的数据, 不同用法: RM 学打分, DPO 直接学策略。' },
      { from: 'policy.generate', to: 'reward_fn (RLVR)', body: '在线采样 + 规则验证, 数据分布随 policy 漂移 (on-policy)。' },
      { from: 'teacher logits / T', to: 'student KL', body: 'T 放大暗知识, T² 补偿 softmax 梯度的 1/T² 缩放。' },
    ],
    sourceRows: [
      { concept: 'value head', code: 'reward_model.py:RewardModel', takeaway: '复用 LLaMA 骨架, lm_head 换成 [D]→[1], 取最后位置。' },
      { concept: '组内优势', code: 'grpo.py:GRPOTrainer.step', takeaway: 'adv = (r - mean) / (std + 1e-4), GRPO 的全部精髓。' },
      { concept: 'KL k3 估计', code: 'grpo.py: log_ratio', takeaway: 'exp(q-p)-(q-p)-1 ≥ 0, 逐 token, 方差小。' },
      { concept: '采样多样性', code: 'run grpo: init std=0.02', takeaway: '初始 logits 太尖 → 组内零方差 → RL 没有梯度。' },
      { concept: '蒸馏损失', code: 'distill.py:DistillLoss', takeaway: 'α·CE + (1-α)·T²·KL(p_t^T ‖ p_s^T)。' },
    ],
    snippetTitle: 'GRPO 单步的完整控制流',
    snippet: `seqs = policy.generate(prompts × G)        # 1. 组内采样
rewards = reward_fn(seqs)                  # 2. 规则验证打分 (RLVR)
adv = (r - r.mean(group)) / r.std(group)   # 3. 组内相对优势
loss = -(adv * logp.mean(-1)).mean()       # 4. 策略梯度
     + beta * kl(policy, ref)              #    + KL 锚定
loss.backward(); opt.step()                # 5. 一次更新`,
    run: 'python -m llm_finetune.run_finetune.grpo.train_grpo',
  },
}

// 时间线数据, year 用于排序与定位
export const timeline = [
  { id: 'transformer', year: 2017, track: 'left', name: 'Transformer',
    kind: 'Encoder-Decoder',
    parts: { attn: 'MHA', ffn: 'ReLU', norm: 'LayerNorm (Post-LN)', pos: 'Sinusoidal' },
    blurb: '原始 attention is all you need, 开启 Transformer 时代',
    file: 'models/foundation/transformer.py' },
  { id: 'bert', year: 2018, track: 'left', name: 'BERT',
    kind: 'Encoder-only (MLM)',
    parts: { attn: 'MHA (双向)', ffn: 'GELU', norm: 'LayerNorm', pos: 'Learnable' },
    blurb: '双向注意力 + MLM 预训练, 理解任务的里程碑',
    file: 'models/language_models/bert.py' },
  { id: 'gpt3', year: 2020, track: 'left', name: 'GPT-3',
    kind: 'Decoder-only',
    parts: { attn: 'MHA', ffn: 'GELU', norm: 'LayerNorm', pos: 'Sinusoidal' },
    blurb: '大规模 decoder-only + few-shot, 开启生成式 LLM 范式',
    file: 'models/language_models/gpt3.py' },
  { id: 'clip', year: 2021, track: 'eye', name: 'CLIP',
    kind: '对比学习双塔',
    parts: { attn: 'MHA', ffn: 'GELU', norm: 'LayerNorm', pos: 'Learnable' },
    blurb: '图像-文本对比学习, 零样本 SOTA, 现代 VLM 的视觉基座',
    file: 'models/multimodal/clip.py' },
  { id: 'whisper', year: 2022, track: 'eye', name: 'Whisper',
    kind: '音频 Encoder-Decoder',
    parts: { attn: 'MHA', ffn: 'GELU', norm: 'LayerNorm', pos: 'Sinusoidal' },
    blurb: 'mel 频谱 + ASR 编解码, 大规模弱监督音频理解',
    file: 'models/multimodal/whisper.py' },
  { id: 'llama', year: 2023, track: 'left', name: 'LLaMA',
    kind: 'Decoder-only (现代)',
    parts: { attn: 'GQA', ffn: 'SwiGLU', norm: 'RMSNorm', pos: 'RoPE' },
    blurb: '现代开源 LLM 的事实模板: GQA + SwiGLU + RMSNorm + RoPE',
    file: 'models/language_models/llama.py' },
  { id: 'mistral', year: 2023, track: 'left', name: 'Mistral',
    kind: 'Decoder-only (SWA)',
    parts: { attn: 'GQA + 滑动窗口 mask', ffn: 'SwiGLU', norm: 'RMSNorm', pos: 'RoPE' },
    blurb: 'LLaMA + 带状因果 mask: 计算 O(T·W), KV cache 封顶 O(W)',
    file: 'models/language_models/mistral.py' },
  { id: 'mamba', year: 2023, track: 'left', name: 'Mamba',
    kind: '非注意力 SSM',
    parts: { attn: 'SelectiveSSM', ffn: '融合进 SSM 层', norm: 'RMSNorm', pos: '无 (时序 scan)' },
    blurb: 'S6 选择性状态空间, O(T) 线性复杂度, 注意力的另一条主线',
    file: 'models/language_models/mamba.py' },
  { id: 'vae', year: 2022, track: 'right', name: 'ImageVAE',
    kind: '潜空间压缩',
    parts: { attn: '—', ffn: 'Conv-GN-SiLU', norm: 'GroupNorm', pos: '—' },
    blurb: 'Latent Diffusion 前置: 把像素压到 latent, 扩散算力 ÷64',
    file: 'models/generative/vae.py' },
  { id: 'dit', year: 2023, track: 'right', name: 'DiT',
    kind: '扩散 Transformer',
    parts: { attn: 'MHA', ffn: 'GELU', norm: 'adaLN-Zero', pos: '2D Learnable' },
    blurb: '取代 UNet 成为 SD3/FLUX/Sora 的共同骨架',
    file: 'models/generative/dit.py' },
  { id: 'mixtral', year: 2024, track: 'left', name: 'Mixtral',
    kind: 'Sparse MoE',
    parts: { attn: 'GQA', ffn: 'MixtralMoE (softmax)', norm: 'RMSNorm', pos: 'RoPE' },
    blurb: 'LLaMA 骨架 + 8 专家 top-2 softmax MoE, 经典稀疏模板',
    file: 'models/moe/mixtral.py' },
  { id: 'deepseek_v3', year: 2024, track: 'left', name: 'DeepSeek-V3',
    kind: 'MLA + 细粒度 MoE',
    parts: { attn: 'MLA', ffn: 'DeepSeekMoE (sigmoid + shared)', norm: 'RMSNorm', pos: 'RoPE (decoupled)' },
    blurb: 'KV cache -93% + aux-loss-free bias + 共享专家, 671B/37B 激活',
    file: 'models/moe/deepseekV3.py' },
  { id: 'qwen3_next', year: 2025, track: 'left', name: 'Qwen3-Next',
    kind: '混合线性注意力',
    parts: { attn: 'GatedDeltaNet 3:1 GQA', ffn: 'SwiGLU', norm: 'RMSNorm', pos: 'RoPE (仅注意力层)' },
    blurb: '75% 层用 O(1) 状态的线性注意力, 25% 全注意力兜底长程检索',
    file: 'models/language_models/qwen3_next.py' },
  { id: 'qwen2_vl', year: 2024, track: 'eye', name: 'Qwen2-VL',
    kind: '早融合 VLM',
    parts: { attn: 'GQA', ffn: 'SwiGLU', norm: 'RMSNorm', pos: 'M-RoPE (三轴)' },
    blurb: '动态分辨率 + M-RoPE, 视觉 token 拼到 LLM 前缀',
    file: 'models/multimodal/qwen2_vl.py' },
  { id: 'mmdit', year: 2024, track: 'right', name: 'MM-DiT',
    kind: '双流扩散 (SD3/FLUX)',
    parts: { attn: 'MHA (跨流)', ffn: 'GELU (每流独立)', norm: 'adaLN-Zero', pos: '2D + 文本流' },
    blurb: '参数分、注意力合, Rectified Flow + v-pred',
    file: 'models/generative/mmdit.py' },
  { id: 'video_dit', year: 2024, track: 'right', name: 'Video DiT',
    kind: 'Sora-lite',
    parts: { attn: 'MHA', ffn: 'GELU', norm: 'adaLN-Zero', pos: '3D Spacetime Patches' },
    blurb: 'Spacetime patches + DiT, 视频作为世界模拟器',
    file: 'models/generative/video_dit.py' },
  { id: 'var', year: 2024, track: 'right', name: 'VAR',
    kind: '自回归图像',
    parts: { attn: 'MHA', ffn: 'GELU', norm: 'LayerNorm', pos: 'RoPE' },
    blurb: 'VQ 离散 token + next-token, 直接复用 GPT 框架',
    file: 'models/generative/var.py' },
  { id: 'deepseek_v32', year: 2025, track: 'left', name: 'DeepSeek-V3.2',
    kind: 'DSA 长上下文',
    parts: { attn: 'MLA + LightningIndexer (DSA)', ffn: 'DeepSeekMoE', norm: 'RMSNorm', pos: 'RoPE (decoupled)' },
    blurb: '稀疏 top-k 注意力, O(T²) → O(T·k), 主攻长上下文成本',
    file: 'models/moe/deepseekV3.py' },
  { id: 'omni', year: 2025, track: 'eye', name: 'Qwen2.5-Omni',
    kind: 'Thinker-Talker 全模态',
    parts: { attn: 'GQA', ffn: 'SwiGLU', norm: 'RMSNorm', pos: 'M-RoPE (时间对齐)' },
    blurb: '双脑: Thinker 理解, Talker cross-attn 流式语音生成',
    file: 'models/multimodal/qwen2_5_omni.py' },
]

export const years = Array.from(new Set(timeline.map(m => m.year))).sort()

export function findModel(id) {
  return timeline.find(m => m.id === id)
}
