const node = (id, x, y, label, category, summary, options = {}) => ({
  id,
  x,
  y,
  label,
  category,
  summary,
  ...options,
})

const edge = (from, to, label = '', options = {}) => ({ from, to, label, ...options })

const decoderRuntime = ({ attention, state, active = '全部参数', cache = 'K/V 随 T 增长' }) => ({
  structure: {
    headline: `Embedding → N × Block(${attention}) → Norm → LM Head`,
    note: '先沿主干读输入输出，再展开 Block 看 attention、FFN、norm 和残差如何组合。',
    stats: [
      { label: '主干', value: 'Decoder-only' },
      { label: 'Attention', value: attention },
      { label: '持久状态', value: state },
    ],
  },
  training: {
    headline: '整段 token 并行前向，反向沿计算图逆序返回',
    note: '权重、梯度和优化器状态长期驻留；为 backward 保存的激活通常是随 batch 与序列长度变化的显存大头。',
    stats: [
      { label: '参与更新', value: active, tone: 'weight' },
      { label: '激活', value: '[B,T,D] × 层数', tone: 'activation' },
      { label: '反向', value: 'dlogits → embedding', tone: 'gradient' },
    ],
  },
  prefill: {
    headline: 'Prompt 一次并行通过全部层，并为每层建立推理状态',
    note: '权重只读；大矩阵激活在算子结束后可释放，注意力层产生的 KV cache 会跨 decode step 保留。',
    stats: [
      { label: '输入', value: '[B,T] prompt', tone: 'activation' },
      { label: '权重', value: '只读 / 可量化', tone: 'weight' },
      { label: '缓存', value: cache, tone: 'cache' },
    ],
  },
  decode: {
    headline: '每步只输入 1 个新 token，复用历史状态生成下一 token',
    note: '临时激活从 [B,T,D] 缩成 [B,1,D]；模型参数不变，KV 或 SSM state 是跨步保留的关键状态。',
    stats: [
      { label: '步进输入', value: '[B,1]', tone: 'activation' },
      { label: '权重', value: '每步读取', tone: 'weight' },
      { label: '持久状态', value: state, tone: 'cache' },
    ],
  },
})

const commonDecoderNodes = ({ attentionLabel, attentionSummary, attentionSource, ffnLabel, ffnSummary, ffnSource }) => [
  node('tokens', 40, 126, 'Token IDs', 'input', '离散输入，训练时是完整序列，decode 时通常只有新 token。', {
    shape: '[B,T] / [B,1]',
    runtime: {
      training: 'teacher forcing 让所有位置并行预测下一 token。',
      prefill: '完整 prompt 一次进入模型。',
      decode: '只追加本轮采样得到的 1 个 token。',
    },
  }),
  node('embedding', 270, 126, 'Token Embedding', 'embedding', '查表把 token id 映射到模型隐藏维。', {
    shape: '[B,T] → [B,T,D]',
    weights: [{ name: 'embedding.weight', shape: '[V,D]', note: '通常与 lm_head 共享权重。' }],
    activations: 'embedding 输出在训练时需要为反向保存；推理时用完即可释放。',
    source: 'llm_models/layers/core/',
  }),
  node('blocks', 520, 126, 'Decoder Block × N', 'stack', '同一组结构重复 N 层；点击展开可查看单层内部。', {
    shape: '[B,T,D] → [B,T,D]',
    detail: 'Pre-Norm 子层 + 残差连接保持主干形状不变。',
  }),
  node('final_norm', 790, 126, 'Final Norm', 'norm', '最后一次归一化稳定输出尺度。', {
    shape: '[B,T,D]',
    weights: [{ name: 'norm.weight', shape: '[D]', note: '逐通道缩放参数。' }],
    source: 'llm_models/layers/core/normalization.py',
  }),
  node('lm_head', 1010, 126, 'LM Head', 'output', '把隐藏状态投影到词表 logits。', {
    shape: '[B,T,D] → [B,T,V]',
    weights: [{ name: 'lm_head.weight', shape: '[V,D]', note: '常与 token embedding weight tying。' }],
  }),
  node('logits', 1235, 126, 'Logits / Next Token', 'result', '训练进入交叉熵；推理只读取最后位置并采样。', {
    shape: '[B,T,V] / [B,V]',
    runtime: {
      training: '与 labels 计算 cross-entropy，得到反向起点 dlogits。',
      prefill: '通常只消费最后位置 logits，其余位置主要用于建立 cache。',
      decode: 'temperature / top-k / top-p 后采样下一个 token。',
    },
  }),
  node('norm1', 330, 340, 'RMSNorm', 'norm', 'Attention 前的 Pre-Norm。', {
    parent: 'blocks', shape: '[B,T,D]', source: 'llm_models/layers/core/normalization.py',
  }),
  node('attention', 555, 340, attentionLabel, 'attention', attentionSummary, {
    parent: 'blocks',
    shape: '[B,T,D] → Q/K/V → [B,T,D]',
    source: attentionSource,
    weights: [
      { name: 'Wq', shape: '[D,Hq·Dh]', note: '查询投影。' },
      { name: 'Wk / Wv', shape: '方案相关', note: '决定 KV cache 每 token 的宽度。' },
      { name: 'Wo', shape: '[Hq·Dh,D]', note: '多头输出合并。' },
    ],
    activations: '训练保存 Q/K/V、softmax 概率或重算所需输入；推理持久化 K/V。',
    runtime: {
      training: '所有位置并行计算因果 attention，并为 Q/K/V 投影累积梯度。',
      prefill: '一次生成整段 prompt 的 K/V 并写入每层 cache。',
      decode: '只生成新 token 的 Q/K/V；Q 查询全部可见历史 cache。',
    },
  }),
  node('norm2', 790, 340, 'RMSNorm', 'norm', 'FFN 前的 Pre-Norm。', {
    parent: 'blocks', shape: '[B,T,D]', source: 'llm_models/layers/core/normalization.py',
  }),
  node('ffn', 1010, 340, ffnLabel, 'ffn', ffnSummary, {
    parent: 'blocks',
    shape: '[B,T,D] → [B,T,Dff] → [B,T,D]',
    source: ffnSource,
    weights: [
      { name: 'gate/up projection', shape: '[D,Dff]', note: '扩维并形成门控分支。' },
      { name: 'down projection', shape: '[Dff,D]', note: '投回残差主干。' },
    ],
    activations: '训练时中间 Dff 激活通常比主干更宽；推理时是短生命周期临时张量。',
  }),
  node('residual', 1235, 340, 'Residual Add', 'merge', 'Attention 与 FFN 的输出分别加回残差主干。', {
    parent: 'blocks', shape: '[B,T,D]', formula: 'x ← x + Attention(Norm(x)); x ← x + FFN(Norm(x))',
  }),
]

const commonDecoderEdges = [
  edge('tokens', 'embedding', 'lookup'),
  edge('embedding', 'blocks', '[B,T,D]'),
  edge('blocks', 'final_norm'),
  edge('final_norm', 'lm_head'),
  edge('lm_head', 'logits'),
  edge('blocks', 'norm1', '展开单层', { detail: true }),
  edge('norm1', 'attention'),
  edge('attention', 'norm2', 'residual'),
  edge('norm2', 'ffn'),
  edge('ffn', 'residual'),
]

const llamaNodes = commonDecoderNodes({
  attentionLabel: 'GQA + RoPE',
  attentionSummary: '多个 Q 头共享较少的 K/V 头，RoPE 在每层旋转 Q/K。',
  attentionSource: 'llm_models/layers/core/attention.py',
  ffnLabel: 'SwiGLU FFN',
  ffnSummary: 'gate 与 up 两路投影逐元素相乘，再由 down projection 返回主干。',
  ffnSource: 'llm_models/layers/core/feedforward.py',
})

const mixtralNodes = commonDecoderNodes({
  attentionLabel: 'GQA + RoPE',
  attentionSummary: 'Attention 与 LLaMA 同类，稀疏化发生在 FFN 槽位。',
  attentionSource: 'llm_models/layers/core/attention.py',
  ffnLabel: 'Sparse MoE',
  ffnSummary: 'Router 为每个 token 选择 top-2 SwiGLU 专家。',
  ffnSource: 'llm_models/layers/sparse/moe.py',
})

mixtralNodes.push(
  node('router', 670, 555, 'Softmax Router', 'router', '为每个 token 计算 8 个专家分数并取 top-2。', {
    parent: 'ffn', shape: '[B·T,D] → [B·T,E]',
    weights: [{ name: 'router.weight', shape: '[E,D]', note: '很小但决定负载分布。' }],
    activations: '训练保存 router logits 与 top-k 权重，用于主 loss 和负载均衡 aux loss。',
  }),
  node('experts', 930, 555, 'Selected Experts 2 / 8', 'expert', '只有被选中的两个 SwiGLU 专家处理当前 token。', {
    parent: 'ffn', shape: 'token shards → 2 × SwiGLU',
    weights: [{ name: 'experts.*', shape: '8 × FFN weights', note: '总参数很大，每 token 只激活其中两份。' }],
    runtime: {
      training: '专家权重仅接收被路由 token 的梯度；跨卡时伴随 all-to-all。',
      prefill: 'prompt token 可落到不同专家，形成不规则 batch。',
      decode: '单步 token 数少，更容易出现专家负载不均。',
    },
  }),
  node('combine', 1190, 555, 'Weighted Combine', 'merge', '按 router 权重合并两个专家输出。', {
    parent: 'ffn', shape: '2 × [tokens,D] → [B,T,D]', formula: 'y = Σ p(e|x) · Expert_e(x)',
  }),
)

const mixtralEdges = [
  ...commonDecoderEdges,
  edge('ffn', 'router', '展开 MoE', { detail: true }),
  edge('router', 'experts', 'top-2'),
  edge('experts', 'combine', '加权'),
]

const deepseekNodes = commonDecoderNodes({
  attentionLabel: 'MLA',
  attentionSummary: '先把 KV 压成低秩 latent，只缓存可还原的压缩表示。',
  attentionSource: 'llm_models/layers/core/attention.py',
  ffnLabel: 'DeepSeek MoE',
  ffnSummary: '细粒度 routed experts + shared experts，sigmoid top-k 路由。',
  ffnSource: 'llm_models/models/moe/deepseekV3.py',
})

deepseekNodes.push(
  node('kv_down', 350, 555, 'KV Down Projection', 'weight', '把输入压缩成 c_kv latent，替代完整 K/V 进入 cache。', {
    parent: 'attention', shape: '[B,T,D] → [B,T,Rkv]',
    weights: [{ name: 'W_dkv', shape: '[D,Rkv]', note: 'KV 低秩下投影。' }],
  }),
  node('rope_side', 570, 555, 'Decoupled RoPE', 'position', '单独保留较小的位置 K 分量，避免位置旋转破坏低秩吸收。', {
    parent: 'attention', shape: '[B,T,Dr]', formula: 'cache = [c_kv, k_rope]',
  }),
  node('kv_up', 790, 555, 'KV Up Projection', 'weight', '计算时把 latent 升维还原为各头 K/V。', {
    parent: 'attention', shape: '[B,T,Rkv] → K/V heads',
    weights: [{ name: 'W_uk / W_uv', shape: '[Rkv,H·Dh]', note: '可在推理内核中与后续矩阵吸收融合。' }],
  }),
  node('ds_router', 900, 730, 'Sigmoid Top-k Router', 'router', '对 routed experts 独立打分，选择 top-k 后归一化。', {
    parent: 'ffn', shape: '[B·T,D] → [B·T,E]',
    weights: [{ name: 'router.weight', shape: '[E,D]', note: '训练时还会更新 aux-loss-free bias。' }],
  }),
  node('shared_experts', 1130, 680, 'Shared Experts', 'expert', '每个 token 都经过，承载通用知识。', {
    parent: 'ffn', shape: '[B,T,D] → [B,T,D]',
  }),
  node('routed_experts', 1130, 790, 'Routed Experts top-k', 'expert', '只激活少量细粒度专家，承载专业化能力。', {
    parent: 'ffn', shape: 'selected tokens → expert FFN',
  }),
  node('ds_combine', 1360, 730, 'Shared + Routed', 'merge', '共享专家输出与路由专家加权输出相加。', {
    parent: 'ffn', shape: '[B,T,D]', formula: 'y = shared(x) + Σ p_e · routed_e(x)',
  }),
)

const deepseekEdges = [
  ...commonDecoderEdges,
  edge('attention', 'kv_down', '缓存路径', { detail: true }),
  edge('kv_down', 'rope_side'),
  edge('rope_side', 'kv_up'),
  edge('ffn', 'ds_router', '路由路径', { detail: true }),
  edge('ds_router', 'shared_experts'),
  edge('ds_router', 'routed_experts'),
  edge('shared_experts', 'ds_combine'),
  edge('routed_experts', 'ds_combine'),
]

export const modelArchitectures = [
  {
    id: 'transformer',
    name: 'Transformer',
    badge: 'Encoder–Decoder',
    description: '原始序列到序列骨架：Encoder 建立源序列 memory，Decoder 通过 cross-attention 条件生成。',
    source: 'llm_models/models/foundation/transformer.py',
    canvas: { width: 1510, height: 760 },
    defaultExpanded: ['encoder', 'decoder'],
    runtime: {
      structure: {
        headline: 'Source → Encoder memory → Cross-Attention → Target Decoder',
        note: '源端与目标端是两条支路；Cross-Attention 是它与 decoder-only LLM 最关键的结构差异。',
        stats: [
          { label: '源端', value: '双向 Self-Attn' },
          { label: '目标端', value: '因果 Self-Attn' },
          { label: '桥接', value: 'Cross-Attn' },
        ],
      },
      training: {
        headline: '源句和右移后的目标句并行前向，CE 梯度同时更新两侧',
        note: 'Decoder 的 cross-attention 梯度会回到 encoder memory，因此源端和目标端不是两个独立模型。',
        stats: [
          { label: '激活', value: 'Encoder + Decoder', tone: 'activation' },
          { label: '权重', value: '两套 Embedding', tone: 'weight' },
          { label: '反向桥', value: 'Cross-Attn → Encoder', tone: 'gradient' },
        ],
      },
      prefill: {
        headline: 'Encoder 一次编码源序列，Decoder 对已有目标前缀并行计算',
        note: 'Encoder memory 可复用；目标端 self-attention 与 cross-attention 都可建立 K/V cache。',
        stats: [
          { label: 'Encoder', value: '一次执行', tone: 'activation' },
          { label: 'Memory', value: '[B,S,D] 保留', tone: 'cache' },
          { label: '权重', value: '只读', tone: 'weight' },
        ],
      },
      decode: {
        headline: 'Decoder 每步消费一个目标 token，并重复查询同一份 Encoder memory',
        note: '源端不再重算；目标 self-KV 逐步增长，cross-KV 可从 encoder memory 预先投影并复用。',
        stats: [
          { label: '新输入', value: '[B,1]', tone: 'activation' },
          { label: '源状态', value: 'Encoder memory', tone: 'cache' },
          { label: '目标状态', value: 'Self-KV 追加', tone: 'cache' },
        ],
      },
    },
    nodes: [
      node('src', 35, 80, 'Source Tokens', 'input', '待编码的源序列，例如机器翻译中的原文。', { shape: '[B,S]' }),
      node('src_emb', 265, 80, 'Source Embed + PE', 'embedding', '源词嵌入叠加正弦位置编码。', {
        shape: '[B,S,D]', weights: [{ name: 'src_embedding', shape: '[Vsrc,D]', note: '源词表参数。' }],
      }),
      node('encoder', 510, 80, 'Encoder Layer × N', 'stack', '双向 self-attention 编码完整源序列。', {
        shape: '[B,S,D]', detail: 'Self-Attn → Add → FFN → Add',
      }),
      node('memory', 770, 80, 'Encoder Memory', 'state', '供所有 Decoder 层 cross-attention 重复读取。', {
        shape: '[B,S,D]', activations: '训练时保存以反向到 Encoder；推理时在整个输出生成期间驻留。',
      }),
      node('tgt', 35, 325, 'Target Tokens', 'input', '训练时是右移后的答案；推理时是已生成前缀。', { shape: '[B,T]' }),
      node('tgt_emb', 265, 325, 'Target Embed + PE', 'embedding', '目标词嵌入与位置编码。', {
        shape: '[B,T,D]', weights: [{ name: 'tgt_embedding', shape: '[Vtgt,D]', note: '与源词表可不同。' }],
      }),
      node('decoder', 770, 325, 'Decoder Layer × N', 'stack', '因果 self-attention 后查询 Encoder memory。', {
        shape: '[B,T,D]', detail: 'Masked Self-Attn → Cross-Attn → FFN',
      }),
      node('head', 1040, 325, 'Final Norm + FC', 'output', '投影到目标词表。', {
        shape: '[B,T,D] → [B,T,Vtgt]', weights: [{ name: 'fc_out.weight', shape: '[Vtgt,D]', note: '目标词表分类器。' }],
      }),
      node('prediction', 1280, 325, 'Target Logits', 'result', '训练计算 CE，推理选择下一 token。', { shape: '[B,T,Vtgt]' }),
      node('enc_attn', 350, 555, 'Bidirectional MHA', 'attention', '源 token 可互相看见，没有因果上三角屏蔽。', {
        parent: 'encoder', shape: '[B,S,D]', source: 'llm_models/layers/core/attention.py',
      }),
      node('enc_ffn', 590, 555, 'ReLU FFN', 'ffn', '逐 token 的两层前馈网络。', {
        parent: 'encoder', shape: '[B,S,D] → [B,S,4D] → [B,S,D]', source: 'llm_models/layers/core/feedforward.py',
      }),
      node('dec_self', 830, 555, 'Masked MHA', 'attention', '目标端只能看当前位置及之前 token。', {
        parent: 'decoder', shape: '[B,T,D]',
      }),
      node('cross_attn', 1060, 555, 'Cross-Attention', 'attention', 'Q 来自 Decoder，K/V 来自 Encoder memory。', {
        parent: 'decoder', shape: 'Q:[B,T,D], K/V:[B,S,D]', formula: 'softmax(Q_dec K_encᵀ / √d) V_enc',
      }),
      node('dec_ffn', 1290, 555, 'ReLU FFN', 'ffn', 'Cross-Attention 后的逐 token 非线性变换。', {
        parent: 'decoder', shape: '[B,T,D] → [B,T,4D] → [B,T,D]',
      }),
    ],
    edges: [
      edge('src', 'src_emb'), edge('src_emb', 'encoder'), edge('encoder', 'memory'),
      edge('tgt', 'tgt_emb'), edge('tgt_emb', 'decoder'), edge('memory', 'decoder', 'K / V'),
      edge('decoder', 'head'), edge('head', 'prediction'),
      edge('encoder', 'enc_attn', '展开', { detail: true }), edge('enc_attn', 'enc_ffn'),
      edge('decoder', 'dec_self', '展开', { detail: true }), edge('dec_self', 'cross_attn'), edge('cross_attn', 'dec_ffn'),
    ],
  },
  {
    id: 'llama',
    name: 'LLaMA',
    badge: 'Modern Decoder',
    description: '现代开源 LLM 模板：GQA + SwiGLU + RMSNorm + RoPE，LM Head 与 Embedding 权重共享。',
    source: 'llm_models/models/language_models/llama.py',
    canvas: { width: 1480, height: 660 },
    defaultExpanded: ['blocks'],
    runtime: decoderRuntime({ attention: 'GQA + RoPE', state: '每层 KV cache', cache: 'GQA KV cache' }),
    nodes: llamaNodes,
    edges: commonDecoderEdges,
  },
  {
    id: 'mamba',
    name: 'Mamba',
    badge: 'Selective SSM',
    description: '不用 Attention：卷积负责局部模式，Selective SSM 以 O(T) scan 维护固定大小状态。',
    source: 'llm_models/models/language_models/mamba.py',
    canvas: { width: 1510, height: 700 },
    defaultExpanded: ['mamba_blocks'],
    runtime: decoderRuntime({
      attention: 'Selective SSM',
      state: '每层 hₜ 固定状态',
      cache: '无 KV；建立最终 SSM state',
    }),
    nodes: [
      node('tokens', 40, 126, 'Token IDs', 'input', 'Mamba 同样接收离散 token 序列。', { shape: '[B,T] / [B,1]' }),
      node('embedding', 270, 126, 'Token Embedding', 'embedding', '查表得到隐藏向量，不做 sqrt(D) 放大。', {
        shape: '[B,T,D]', weights: [{ name: 'embedding.weight', shape: '[V,D]', note: '与 lm_head 共享。' }],
      }),
      node('mamba_blocks', 520, 126, 'Mamba Block × N', 'stack', '每层只有 Pre-RMSNorm + MambaLayer + Residual。', { shape: '[B,T,D]' }),
      node('final_norm', 790, 126, 'Final RMSNorm', 'norm', '稳定最终隐藏状态。', { shape: '[B,T,D]' }),
      node('lm_head', 1010, 126, 'Tied LM Head', 'output', '共享 embedding 权重投影到词表。', { shape: '[B,T,D] → [B,T,V]' }),
      node('logits', 1235, 126, 'Next Token', 'result', '训练算 CE；推理采样下一 token。', { shape: '[B,V]' }),
      node('in_proj', 260, 350, 'Input Projection', 'weight', '一次投影后切成 main 与 gate 两支。', {
        parent: 'mamba_blocks', shape: '[B,T,D] → 2 × [B,T,Dinner]',
        weights: [{ name: 'in_proj.weight', shape: '[2Dinner,D]', note: '主分支和门控分支共用一次 GEMM。' }],
      }),
      node('conv', 500, 350, 'Causal DW-Conv1D', 'ffn', '短窗口因果卷积注入局部 token 混合。', {
        parent: 'mamba_blocks', shape: '[B,Dinner,T]', weights: [{ name: 'conv1d', shape: '[Dinner,1,K]', note: '逐通道卷积。' }],
      }),
      node('ssm', 740, 350, 'Selective SSM Scan', 'state', '按 token 更新状态 hₜ，并由输入动态产生 Δ/B/C。', {
        parent: 'mamba_blocks', shape: '[B,T,Dinner] ↔ h:[B,Dinner,N]',
        activations: '训练需保存或重算 scan 中间量；decode 只保留最新 hₜ。',
        runtime: {
          training: '并行 scan 的反向会穿过整段状态递推。',
          prefill: '顺序或并行 scan prompt，最终只需保留末状态。',
          decode: '每层用旧 hₜ₋₁ 和新 token 更新一次 hₜ，状态大小不随 T 增长。',
        },
      }),
      node('gate', 980, 350, 'SiLU Gate ⊙', 'merge', 'SSM 输出乘 gate 分支，选择保留哪些通道。', {
        parent: 'mamba_blocks', shape: '[B,T,Dinner]', formula: 'y = SSM(SiLU(Conv(x))) ⊙ SiLU(gate)',
      }),
      node('out_proj', 1220, 350, 'Output Projection', 'weight', '投回 D 后加残差。', {
        parent: 'mamba_blocks', shape: '[B,T,Dinner] → [B,T,D]',
        weights: [{ name: 'out_proj.weight', shape: '[D,Dinner]', note: '返回残差宽度。' }],
      }),
    ],
    edges: [
      edge('tokens', 'embedding'), edge('embedding', 'mamba_blocks'), edge('mamba_blocks', 'final_norm'),
      edge('final_norm', 'lm_head'), edge('lm_head', 'logits'),
      edge('mamba_blocks', 'in_proj', '展开单层', { detail: true }), edge('in_proj', 'conv'),
      edge('conv', 'ssm'), edge('ssm', 'gate'), edge('gate', 'out_proj'),
    ],
  },
  {
    id: 'mixtral',
    name: 'Mixtral',
    badge: 'Sparse MoE',
    description: '保留 LLaMA 风格 GQA 主干，把 dense SwiGLU FFN 换成每 token 激活 top-2 专家的 Sparse MoE。',
    source: 'llm_models/models/moe/mixtral.py',
    canvas: { width: 1480, height: 780 },
    defaultExpanded: ['blocks', 'ffn'],
    runtime: decoderRuntime({ attention: 'GQA + top-2 MoE', state: 'KV cache + routing', active: '每 token 2 / 8 experts', cache: 'GQA KV cache' }),
    nodes: mixtralNodes,
    edges: mixtralEdges,
  },
  {
    id: 'deepseek_v3',
    name: 'DeepSeek‑V3',
    badge: 'MLA + Fine-grained MoE',
    description: 'MLA 压缩每层 KV，细粒度 routed experts 与 shared experts 同时承担稀疏和通用计算。',
    source: 'llm_models/models/moe/deepseekV3.py',
    canvas: { width: 1600, height: 980 },
    defaultExpanded: ['blocks', 'attention', 'ffn'],
    runtime: decoderRuntime({
      attention: 'MLA + DeepSeekMoE',
      state: 'c_kv + k_rope',
      active: 'shared + routed top-k',
      cache: '压缩 latent KV cache',
    }),
    nodes: deepseekNodes,
    edges: deepseekEdges,
  },
  {
    id: 'qwen2_vl',
    name: 'Qwen2‑VL',
    badge: 'Vision–Language',
    description: '图像先变成视觉 token，再与文本 token 早融合；统一 Decoder 用 M‑RoPE 理解时间、高度和宽度。',
    source: 'llm_models/models/multimodal/qwen2_vl.py',
    canvas: { width: 1710, height: 920 },
    defaultExpanded: ['vision_encoder', 'text_decoder'],
    runtime: {
      structure: {
        headline: 'Image patches → Vision Encoder → Projector ↘ 与 Text Embedding 合并 → LLM',
        note: '关键不是额外加一个“看图工具”，而是把视觉内容对齐为 LLM 可以消费的 token 前缀。',
        stats: [
          { label: '视觉入口', value: 'ViT patches' },
          { label: '融合', value: 'Prefix tokens' },
          { label: '位置', value: 'M-RoPE 三轴' },
        ],
      },
      training: {
        headline: '语言 loss 可穿过 LLM、Projector 回传到 Vision Encoder',
        note: '实际训练可选择冻结视觉塔或 LLM；图中展示端到端可训练路径以及视觉/文本两类激活。',
        stats: [
          { label: '视觉激活', value: '[B,Nvis,Dv]', tone: 'activation' },
          { label: '融合激活', value: '[B,Nvis+T,D]', tone: 'activation' },
          { label: '权重域', value: 'Vision + Projector + LLM', tone: 'weight' },
        ],
      },
      prefill: {
        headline: '图像只编码一次，视觉 token 与文本 prompt 一起建立 LLM KV cache',
        note: '视觉前缀的 K/V 在后续生成期间持续复用，因此视觉 token 数直接影响首 token 延迟与 cache。',
        stats: [
          { label: '视觉塔', value: '一次执行', tone: 'activation' },
          { label: '融合序列', value: 'Nvis + T', tone: 'activation' },
          { label: '缓存', value: '视觉 + 文本 KV', tone: 'cache' },
        ],
      },
      decode: {
        headline: '每步只追加文本 token，但它可以查询已缓存的全部视觉前缀',
        note: 'Vision Encoder 与 Projector 不重算；新 token 经 LLM 后把自己的 K/V 追加到统一 cache。',
        stats: [
          { label: '新输入', value: '1 text token', tone: 'activation' },
          { label: '视觉状态', value: 'prefix KV 复用', tone: 'cache' },
          { label: '输出', value: 'next-token logits' },
        ],
      },
    },
    nodes: [
      node('image', 35, 90, 'Image / Frames', 'input', '像素输入，动态分辨率会改变 patch token 数。', { shape: '[B,3,H,W]' }),
      node('patch', 260, 90, 'Patch Embed', 'embedding', 'Conv2d 把图像切块并投影成视觉 token。', {
        shape: '[B,3,H,W] → [B,Nvis,Dv]', weights: [{ name: 'patch_embed.proj', shape: '[Dv,3,P,P]', note: '卷积式 patchify。' }],
      }),
      node('vision_encoder', 500, 90, 'Vision Transformer × N', 'stack', '视觉 token 通过双向 self-attention 建模。', { shape: '[B,Nvis,Dv]' }),
      node('resampler', 755, 90, 'Perceiver Resampler', 'attention', '可选地把大量视觉 token 压成固定数量 latent。', {
        shape: '[B,Nvis,Dv] → [B,L,Dv]', source: 'llm_models/layers/multimodal/multimodal.py',
      }),
      node('projector', 1010, 90, 'Vision Projector', 'weight', '把视觉维度对齐到语言模型隐藏维。', {
        shape: '[B,L,Dv] → [B,L,D]', weights: [{ name: 'projector', shape: '[Dv,D]', note: '视觉域到语言域的桥。' }],
      }),
      node('text', 260, 330, 'Text Tokens', 'input', '用户 prompt 与历史文本 token。', { shape: '[B,T]' }),
      node('text_embed', 500, 330, 'Text Embedding', 'embedding', '文本 token 查表得到语言隐藏向量。', {
        shape: '[B,T,D]', weights: [{ name: 'embed_tokens', shape: '[V,D]', note: '语言词表。' }],
      }),
      node('merge', 1010, 330, 'Prefix Merge + M-RoPE', 'merge', '视觉 token 放在文本前缀，并分配时间/高度/宽度三轴位置。', {
        shape: '[B,L+T,D]', formula: '[vision_tokens ; text_tokens] + modality embedding',
      }),
      node('text_decoder', 1260, 215, 'LLM Decoder × N', 'stack', '统一因果 Decoder 同时消费视觉和文本 token。', { shape: '[B,L+T,D]' }),
      node('head', 1500, 215, 'LM Head', 'output', '只在文本词表上输出 logits。', { shape: '[B,L+T,V]' }),
      node('vision_attn', 480, 565, 'Vision MHA', 'attention', '图像 patch 之间做双向注意力。', {
        parent: 'vision_encoder', shape: '[B,Nvis,Dv]', source: 'llm_models/layers/multimodal/multimodal.py',
      }),
      node('vision_ffn', 720, 565, 'Vision FFN', 'ffn', '视觉塔内的逐 patch 前馈网络。', { parent: 'vision_encoder', shape: '[B,Nvis,Dv]' }),
      node('mrope', 1010, 675, 'M-RoPE', 'position', '文本使用时间轴；图像 token 同时编码时间、高度、宽度。', {
        parent: 'text_decoder', shape: 'position_ids: [3,B,L+T]', source: 'llm_models/layers/core/position_encoding.py',
      }),
      node('gqa', 1260, 675, 'GQA', 'attention', '统一序列上的因果注意力，视觉前缀可被后续文本读取。', {
        parent: 'text_decoder', shape: '[B,L+T,D]', activations: '推理时视觉与文本 K/V 存在同一 cache。',
      }),
      node('swiglu', 1500, 675, 'SwiGLU', 'ffn', '语言 Decoder 的门控 FFN。', { parent: 'text_decoder', shape: '[B,L+T,D]' }),
    ],
    edges: [
      edge('image', 'patch'), edge('patch', 'vision_encoder'), edge('vision_encoder', 'resampler'),
      edge('resampler', 'projector'), edge('projector', 'merge', 'vision prefix'),
      edge('text', 'text_embed'), edge('text_embed', 'merge', 'text'), edge('merge', 'text_decoder'), edge('text_decoder', 'head'),
      edge('vision_encoder', 'vision_attn', '展开 ViT', { detail: true }), edge('vision_attn', 'vision_ffn'),
      edge('text_decoder', 'mrope', '展开 LLM', { detail: true }), edge('mrope', 'gqa'), edge('gqa', 'swiglu'),
    ],
  },
  {
    id: 'dit',
    name: 'DiT',
    badge: 'Diffusion Transformer',
    description: '把含噪 VAE latent 切成 patch token，用 timestep 条件调制 Transformer，输出噪声或 velocity。',
    source: 'llm_models/models/generative/dit.py',
    canvas: { width: 1540, height: 760 },
    defaultExpanded: ['dit_blocks'],
    runtime: {
      structure: {
        headline: 'Noisy latent + timestep condition → AdaLN-Zero Blocks → denoising prediction',
        note: 'DiT 与语言模型共享 Transformer 骨架，但输入输出是连续 latent，位置是二维 patch，条件通过 AdaLN 注入。',
        stats: [
          { label: 'Token', value: 'VAE latent patches' },
          { label: '条件', value: 'timestep / class' },
          { label: '目标', value: 'ε 或 velocity' },
        ],
      },
      training: {
        headline: '随机时间步加噪后一次前向，MSE 梯度更新全部 DiT 权重',
        note: '每个 batch 只训练一个随机 t；AdaLN 的调制激活、Attention/FFN 中间量都参与 backward。',
        stats: [
          { label: '输入', value: 'xₜ + t + condition', tone: 'activation' },
          { label: '目标', value: 'noise / velocity' },
          { label: 'Loss', value: 'MSE', tone: 'gradient' },
        ],
      },
      prefill: {
        headline: '一次去噪 step：整张 latent 的所有 patch 并行通过 DiT',
        note: 'DiT 没有跨 step KV cache；权重只读，当前 step 的激活在输出后释放。',
        stats: [
          { label: '输入', value: 'xₜ patches', tone: 'activation' },
          { label: '权重', value: '每 step 重读', tone: 'weight' },
          { label: '跨步缓存', value: '无 KV cache' },
        ],
      },
      decode: {
        headline: 'Scheduler 反复调用同一个 DiT，把 xₜ 迭代更新到 x₀',
        note: '迭代状态是整张 latent，而不是 token KV；CFG 会以有条件/无条件两路预测组合方向。',
        stats: [
          { label: '循环', value: 'N denoise steps' },
          { label: '持久状态', value: '当前 latent xₜ', tone: 'cache' },
          { label: '输出', value: 'VAE decode → image' },
        ],
      },
    },
    nodes: [
      node('latent', 35, 100, 'Noisy Latent xₜ', 'input', 'VAE 潜空间中按 scheduler 加噪后的连续张量。', { shape: '[B,C,H,W]' }),
      node('patchify', 270, 100, 'Patchify + 2D PE', 'embedding', 'Conv2d 切 patch，并加可学习二维位置。', {
        shape: '[B,C,H,W] → [B,N,D]', weights: [{ name: 'patchify.proj', shape: '[D,C,P,P]', note: 'stride=P 的卷积。' }],
      }),
      node('condition', 270, 330, 't / Class Condition', 'condition', '时间步正弦嵌入经 MLP，并可叠加类别或文本条件。', {
        shape: '[B] → [B,Cdim]', weights: [{ name: 't_embed.mlp', shape: '2-layer MLP', note: '把标量时间映射到调制空间。' }],
      }),
      node('dit_blocks', 570, 180, 'AdaLN-Zero Block × N', 'stack', '条件向量为 Attention 和 FFN 生成 shift、scale、gate。', { shape: '[B,N,D]' }),
      node('final', 860, 180, 'Final AdaLN + Linear', 'output', '再用条件调制后投影到每个 patch 的像素通道。', { shape: '[B,N,P²·C]' }),
      node('unpatchify', 1110, 180, 'Unpatchify', 'merge', '把 patch 输出重新排列成空间 latent。', { shape: '[B,C,H,W]' }),
      node('prediction', 1350, 180, 'ε / v Prediction', 'result', '训练与 target 做 MSE；采样时交给 scheduler 更新 xₜ。', { shape: '[B,C,H,W]' }),
      node('adaln', 390, 555, 'AdaLN Modulation', 'condition', '由 condition 产生 6 组 shift / scale / gate。', {
        parent: 'dit_blocks', shape: '[B,Cdim] → 6 × [B,D]', formula: 'h = gate ⊙ F((1+scale)·LN(x)+shift)',
      }),
      node('attn', 660, 555, 'Patch Self-Attention', 'attention', '同一 latent 的所有 patch 交换全局信息。', {
        parent: 'dit_blocks',
        shape: '[B,N,D]',
        runtime: {
          training: '保存或重算 Q/K/V 与 attention 中间量，梯度回到 patch token 和投影权重。',
          prefill: '当前去噪 step 内做全 patch attention；结束后释放，不建立跨 step KV cache。',
          decode: '每个 scheduler step 都重新计算整张 latent 的 patch attention。',
        },
      }),
      node('ffn', 920, 555, 'GeLU FFN', 'ffn', '逐 patch 非线性变换。', { parent: 'dit_blocks', shape: '[B,N,D] → [B,N,4D] → [B,N,D]' }),
      node('zero_gate', 1180, 555, 'Zero-init Residual Gate', 'merge', '训练开始时分支贡献接近 0，逐步学习有效调制。', { parent: 'dit_blocks', shape: '[B,N,D]' }),
    ],
    edges: [
      edge('latent', 'patchify'), edge('patchify', 'dit_blocks'), edge('condition', 'dit_blocks', 'c'),
      edge('dit_blocks', 'final'), edge('condition', 'final', 'c'), edge('final', 'unpatchify'), edge('unpatchify', 'prediction'),
      edge('dit_blocks', 'adaln', '展开 Block', { detail: true }), edge('adaln', 'attn'), edge('attn', 'ffn'), edge('ffn', 'zero_gate'),
    ],
  },
]

export const architectureById = Object.fromEntries(modelArchitectures.map((model) => [model.id, model]))
