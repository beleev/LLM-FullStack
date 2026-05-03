// 零件内部的矩阵变换 spec
//
// 每个 variant 定义:
//   - params: 权重矩阵列表, 形状用符号表达式 (如 'D', 'H*D_h')
//   - flow:   计算步骤列表, 每步标注操作与输出形状
//
// FlowStep 类型:
//   { type: 'input', label, shape, note? }           — 起点
//   { type: 'op',    op, out, shape, kind?, note?, highlight? }  — 单步
//   { type: 'branch',items: [{op,out,shape}...], note? }         — 并行分支 (e.g. 同时投 Q/K/V)
//   { type: 'output',label, shape }                  — 终点
//
// shape 用数组, 每个元素是符号表达式字符串, 会用 evalExpr 动态计算为数值
//
// kind 用于颜色分类: 'matmul' | 'reshape' | 'activation' | 'attn' | 'cond' | 'route'

// ---------------------------------------------------------------------------
// 注意力变体
// ---------------------------------------------------------------------------

export const attnSpecs = {
  mha: {
    title: 'Multi-Head Attention (MHA)',
    subtitle: '原始, 每头一对独立的 K/V',
    params: [
      { name: 'W_q', shape: ['D', 'D'], note: 'query projection' },
      { name: 'W_k', shape: ['D', 'D'], note: 'key projection — 与 Q 同维, cache 最大' },
      { name: 'W_v', shape: ['D', 'D'], note: 'value projection' },
      { name: 'W_o', shape: ['D', 'D'], note: 'output projection' },
    ],
    flow: [
      { type: 'input', label: 'x', shape: ['B', 'T', 'D'] },
      { type: 'branch', note: '三条独立线性投影, 原地并行', items: [
        { op: 'W_q · x', out: 'Q', shape: ['B', 'T', 'D'], kind: 'matmul' },
        { op: 'W_k · x', out: 'K', shape: ['B', 'T', 'D'], kind: 'matmul' },
        { op: 'W_v · x', out: 'V', shape: ['B', 'T', 'D'], kind: 'matmul' },
      ]},
      { type: 'op', op: 'view(B,T,H,D_h).transpose(1,2)', out: 'Q,K,V', shape: ['B', 'H', 'T', 'D_h'], kind: 'reshape',
        note: '把 head 维放到 batch 后面, 便于批量 matmul' },
      { type: 'op', op: 'Q @ K^T / √D_h', out: 'scores', shape: ['B', 'H', 'T', 'T'], kind: 'attn',
        highlight: true, note: 'O(T²) 算力 — 长上下文的真正瓶颈' },
      { type: 'op', op: 'softmax(-1) + causal mask', out: 'attn', shape: ['B', 'H', 'T', 'T'], kind: 'activation' },
      { type: 'op', op: 'attn @ V', out: 'out', shape: ['B', 'H', 'T', 'D_h'], kind: 'attn' },
      { type: 'op', op: 'transpose(1,2).reshape(B,T,D)', out: 'out', shape: ['B', 'T', 'D'], kind: 'reshape' },
      { type: 'op', op: 'W_o · out', out: 'y', shape: ['B', 'T', 'D'], kind: 'matmul' },
      { type: 'output', label: 'y', shape: ['B', 'T', 'D'] },
    ],
    cachePerToken: { expr: '2 * D', note: 'K + V 各 D 维, fp16 每元素 2 B' },
  },

  gqa: {
    title: 'Grouped-Query Attention (GQA)',
    subtitle: '多头 Q 共享少量 K/V — LLaMA-2/3 · Qwen2 标配',
    params: [
      { name: 'W_q', shape: ['D', 'H*D_h'],     note: 'Q 保留 H 头' },
      { name: 'W_k', shape: ['D', 'H_kv*D_h'],  note: 'K 只有 H_kv 头 — cache 源头收缩' },
      { name: 'W_v', shape: ['D', 'H_kv*D_h'],  note: 'V 只有 H_kv 头' },
      { name: 'W_o', shape: ['H*D_h', 'D'],     note: 'output projection' },
    ],
    flow: [
      { type: 'input', label: 'x', shape: ['B', 'T', 'D'] },
      { type: 'branch', note: 'K/V 头数更少, 直接投影出来就省了', items: [
        { op: 'W_q · x', out: 'Q', shape: ['B', 'T', 'H*D_h'], kind: 'matmul' },
        { op: 'W_k · x', out: 'K', shape: ['B', 'T', 'H_kv*D_h'], kind: 'matmul' },
        { op: 'W_v · x', out: 'V', shape: ['B', 'T', 'H_kv*D_h'], kind: 'matmul' },
      ]},
      { type: 'branch', items: [
        { op: 'view + transpose', out: 'Q', shape: ['B', 'H', 'T', 'D_h'], kind: 'reshape' },
        { op: 'view + transpose', out: 'K', shape: ['B', 'H_kv', 'T', 'D_h'], kind: 'reshape' },
        { op: 'view + transpose', out: 'V', shape: ['B', 'H_kv', 'T', 'D_h'], kind: 'reshape' },
      ]},
      { type: 'op', op: 'rope(Q), rope(K)', out: 'Q,K', shape: ['...', 'D_h'], kind: 'cond',
        note: 'RoPE 只作用在 Q/K — V 不含位置信息' },
      { type: 'op', op: 'K.repeat_interleave(H/H_kv, dim=1)', out: 'K', shape: ['B', 'H', 'T', 'D_h'], kind: 'reshape',
        note: '把 K/V 头复制 num_groups 份对齐 Q (FlashAttention 会跳过这步)' },
      { type: 'op', op: 'Q @ K^T / √D_h', out: 'scores', shape: ['B', 'H', 'T', 'T'], kind: 'attn', highlight: true },
      { type: 'op', op: 'softmax + causal mask', out: 'attn', shape: ['B', 'H', 'T', 'T'], kind: 'activation' },
      { type: 'op', op: 'attn @ V', out: 'out', shape: ['B', 'H', 'T', 'D_h'], kind: 'attn' },
      { type: 'op', op: 'transpose + reshape', out: 'out', shape: ['B', 'T', 'H*D_h'], kind: 'reshape' },
      { type: 'op', op: 'W_o · out', out: 'y', shape: ['B', 'T', 'D'], kind: 'matmul' },
      { type: 'output', label: 'y', shape: ['B', 'T', 'D'] },
    ],
    cachePerToken: { expr: '2 * H_kv * D_h', note: 'K/V 只缓存 H_kv 头, cache ÷ (H/H_kv)' },
  },

  mla: {
    title: 'Multi-Head Latent Attention (MLA)',
    subtitle: 'DeepSeek-V2/V3: KV 低秩压缩 + 解耦 RoPE',
    params: [
      { name: 'W_q_down', shape: ['D', 'r_q'],                        note: '(可选) Q 低秩 down-proj' },
      { name: 'W_q_up',   shape: ['r_q', 'H*(D_nope+D_rope)'],        note: 'Q 升维并切分 nope/rope 两段' },
      { name: 'W_kv_down',shape: ['D', 'r_kv+D_rope'],                note: 'KV 压成 latent + 共享 k_rope' },
      { name: 'W_k_up',   shape: ['r_kv', 'H*D_nope'],                note: '从 c_kv 升维 K-nope' },
      { name: 'W_v_up',   shape: ['r_kv', 'H*D_nope'],                note: '从 c_kv 升维 V' },
      { name: 'W_o',      shape: ['H*D_nope', 'D'],                   note: '输出投影' },
    ],
    flow: [
      { type: 'input', label: 'x', shape: ['B', 'T', 'D'] },
      { type: 'op', op: 'W_q_down · x', out: 'q_c', shape: ['B', 'T', 'r_q'], kind: 'matmul' },
      { type: 'op', op: 'W_q_up · q_c', out: 'q_proj', shape: ['B', 'T', 'H*(D_nope+D_rope)'], kind: 'matmul' },
      { type: 'op', op: 'view + transpose + split', out: '(q_nope, q_rope)',
        shape: ['B', 'H', 'T', 'D_nope or D_rope'], kind: 'reshape',
        note: 'q_nope 走 latent 路径, q_rope 单独走旋转' },
      { type: 'op', op: 'W_kv_down · x', out: 'kv_mix', shape: ['B', 'T', 'r_kv+D_rope'], kind: 'matmul', highlight: true,
        note: '只有这一组低秩投影需要被缓存 (KV cache = c_kv + k_rope)' },
      { type: 'op', op: 'split', out: '(c_kv, k_rope)', shape: ['B', 'T', 'r_kv or D_rope'], kind: 'reshape' },
      { type: 'branch', note: '训练路径: 从 c_kv 升维出 K-nope 与 V', items: [
        { op: 'W_k_up · c_kv → view',  out: 'k_nope', shape: ['B', 'H', 'T', 'D_nope'], kind: 'matmul' },
        { op: 'W_v_up · c_kv → view',  out: 'v',      shape: ['B', 'H', 'T', 'D_nope'], kind: 'matmul' },
      ]},
      { type: 'op', op: 'rope(q_rope), rope(k_rope)', out: '', shape: ['...', 'D_rope'], kind: 'cond',
        note: 'k_rope 所有 head 共享 → broadcast' },
      { type: 'op', op: 'concat nope & rope along head_dim', out: '(Q, K)',
        shape: ['B', 'H', 'T', 'D_nope+D_rope'], kind: 'reshape' },
      { type: 'op', op: 'Q @ K^T / √(D_nope+D_rope)', out: 'scores', shape: ['B', 'H', 'T', 'T'], kind: 'attn', highlight: true },
      { type: 'op', op: 'softmax + mask', out: 'attn', shape: ['B', 'H', 'T', 'T'], kind: 'activation' },
      { type: 'op', op: 'attn @ v', out: 'out', shape: ['B', 'H', 'T', 'D_nope'], kind: 'attn' },
      { type: 'op', op: 'transpose + reshape', out: 'out', shape: ['B', 'T', 'H*D_nope'], kind: 'reshape' },
      { type: 'op', op: 'W_o · out', out: 'y', shape: ['B', 'T', 'D'], kind: 'matmul' },
      { type: 'output', label: 'y', shape: ['B', 'T', 'D'] },
    ],
    cachePerToken: { expr: 'r_kv + D_rope',
      note: 'cache 只是 latent c_kv + 共享 k_rope — 相对 MHA 可降 93%' },
  },

  dsa: {
    title: 'MLA + Deepseek Sparse Attention (DSA)',
    subtitle: 'V3.2: MLA + Lightning Indexer 的稀疏 top-k',
    params: [
      { name: 'W_q_idx', shape: ['D', 'H_idx*D_h_idx'], note: 'indexer Q (小头)' },
      { name: 'W_k_idx', shape: ['D', 'H_idx*D_h_idx'], note: 'indexer K' },
      { name: '— (MLA 权重)', shape: ['复用', '上方 MLA'], note: '主体仍走 MLA, 上面是额外 indexer 头' },
    ],
    flow: [
      { type: 'input', label: 'x', shape: ['B', 'T', 'D'] },
      { type: 'op', op: 'Lightning Indexer: W_q_idx · x, W_k_idx · x',
        out: '(q_idx, k_idx)', shape: ['B', 'H_idx', 'T', 'D_h_idx'], kind: 'route',
        note: 'indexer 头小、计算量远小于主 attention' },
      { type: 'op', op: 'score = Σ_h ReLU(q_idx · k_idx^T / √D_h_idx)',
        out: 'idx_scores', shape: ['B', 'T', 'T'], kind: 'route',
        note: '用 ReLU 而非 softmax, 保留单调性即可排序' },
      { type: 'op', op: 'mask_fill(not_visible, -inf) + topk(k)',
        out: 'sparse_mask', shape: ['B', 'T', 'T'], kind: 'attn', highlight: true,
        note: '必须先 mask 再 topk, 否则 topk 可能选中未来 token (数据泄漏)' },
      { type: 'op', op: '↓  MLA(..., mask=sparse_mask) 仅在 top-k 位置算 softmax · V',
        out: 'y', shape: ['B', 'T', 'D'], kind: 'attn',
        note: '算力: O(T · k) 而非 O(T²) — 主攻 128K+ 上下文' },
      { type: 'output', label: 'y', shape: ['B', 'T', 'D'] },
    ],
    cachePerToken: { expr: 'r_kv + D_rope', note: '与 MLA 相同, 稀疏的是算力而非 cache' },
  },

  ssm: {
    title: 'Selective SSM (Mamba)',
    subtitle: '非注意力分支, 线性 O(T)',
    params: [
      { name: 'A_log', shape: ['D', 'N'], note: 'log 化的状态转移矩阵 (-exp 保证稳定)' },
      { name: 'D',     shape: ['D'],      note: '每通道直通增益 (skip)' },
      { name: 'x_proj',shape: ['D', 'dt_rank+2N'], note: 'Δ/B/C 的共享投影' },
      { name: 'dt_proj',shape: ['dt_rank', 'D'],  note: 'Δ 从低秩升到每通道' },
    ],
    flow: [
      { type: 'input', label: 'x', shape: ['B', 'T', 'D'] },
      { type: 'op', op: 'x_proj · x → split', out: '(dt_low, B_t, C_t)',
        shape: ['B', 'T', 'dt_rank or N'], kind: 'matmul' },
      { type: 'op', op: 'softplus(dt_proj · dt_low)', out: 'Δ',
        shape: ['B', 'T', 'D'], kind: 'activation',
        note: 'Δ 是"selective"的灵魂: 依输入决定记/忘强度' },
      { type: 'op', op: 'Ā = exp(Δ · A),  B̄ = Δ · B_t', out: '(Ā, B̄)',
        shape: ['B', 'T', 'D', 'N'], kind: 'cond', note: '零阶保持离散化' },
      { type: 'op', op: 'for t: h_t = Ā · h_{t-1} + B̄ · x_t', out: 'h',
        shape: ['B', 'D', 'N'], kind: 'attn', highlight: true,
        note: '顺序 scan (教学版); 生产用 CUDA 并行 scan' },
      { type: 'op', op: 'y_t = (h · C_t).sum(-1) + D · x_t', out: 'y',
        shape: ['B', 'T', 'D'], kind: 'matmul' },
      { type: 'output', label: 'y', shape: ['B', 'T', 'D'] },
    ],
    cachePerToken: { expr: 'D * N', note: '隐状态而非 KV — 每 token 滚动更新' },
  },
}

// ---------------------------------------------------------------------------
// FFN 变体
// ---------------------------------------------------------------------------

export const ffnSpecs = {
  relu: {
    title: 'ReLU FFN',
    subtitle: '原始 Transformer (2017)',
    params: [
      { name: 'W_1', shape: ['D', 'd_ff'], note: '升维, 经验 d_ff = 4·D' },
      { name: 'W_2', shape: ['d_ff', 'D'], note: '降维' },
    ],
    flow: [
      { type: 'input', label: 'x', shape: ['B', 'T', 'D'] },
      { type: 'op', op: 'W_1 · x', out: 'h', shape: ['B', 'T', 'd_ff'], kind: 'matmul', highlight: true,
        note: '通常 > 70% 的 LLM 参数量在这里' },
      { type: 'op', op: 'ReLU(h)', out: 'h', shape: ['B', 'T', 'd_ff'], kind: 'activation',
        note: '负半轴梯度恒 0 — 存在 "dying neuron"' },
      { type: 'op', op: 'W_2 · h', out: 'y', shape: ['B', 'T', 'D'], kind: 'matmul' },
      { type: 'output', label: 'y', shape: ['B', 'T', 'D'] },
    ],
    paramBreakdown: '2 · D · d_ff',
  },

  gelu: {
    title: 'GELU FFN',
    subtitle: 'BERT / GPT-2/3 风',
    params: [
      { name: 'W_1', shape: ['D', 'd_ff'], note: '升维' },
      { name: 'W_2', shape: ['d_ff', 'D'], note: '降维' },
    ],
    flow: [
      { type: 'input', label: 'x', shape: ['B', 'T', 'D'] },
      { type: 'op', op: 'W_1 · x', out: 'h', shape: ['B', 'T', 'd_ff'], kind: 'matmul', highlight: true },
      { type: 'op', op: 'GELU(h) = h · Φ(h)', out: 'h', shape: ['B', 'T', 'd_ff'], kind: 'activation',
        note: 'ReLU 的平滑版, 避免硬截断导致的梯度死亡' },
      { type: 'op', op: 'W_2 · h', out: 'y', shape: ['B', 'T', 'D'], kind: 'matmul' },
      { type: 'output', label: 'y', shape: ['B', 'T', 'D'] },
    ],
    paramBreakdown: '2 · D · d_ff',
  },

  swiglu: {
    title: 'SwiGLU FFN',
    subtitle: '门控激活 — LLaMA/Qwen/DeepSeek 标配',
    params: [
      { name: 'W_gate', shape: ['D', 'd_ff'], note: '门控分支, 决定哪些通道通过' },
      { name: 'W_up',   shape: ['D', 'd_ff'], note: '内容分支, 提供"通过的是什么"' },
      { name: 'W_down', shape: ['d_ff', 'D'], note: '降维回 D' },
    ],
    flow: [
      { type: 'input', label: 'x', shape: ['B', 'T', 'D'] },
      { type: 'branch', note: '两条并行升维路径', items: [
        { op: 'SiLU(W_gate · x)', out: 'gate', shape: ['B', 'T', 'd_ff'], kind: 'activation' },
        { op: 'W_up · x',         out: 'up',   shape: ['B', 'T', 'd_ff'], kind: 'matmul' },
      ]},
      { type: 'op', op: 'gate ⊙ up (逐元素相乘)', out: 'h', shape: ['B', 'T', 'd_ff'], kind: 'activation', highlight: true,
        note: '让网络自学"哪些通道开多大"' },
      { type: 'op', op: 'W_down · h', out: 'y', shape: ['B', 'T', 'D'], kind: 'matmul' },
      { type: 'output', label: 'y', shape: ['B', 'T', 'D'] },
    ],
    paramBreakdown: '3 · D · d_ff  (d_ff 按 2/3 缩放以保总参)',
  },

  moe_mx: {
    title: 'Mixtral MoE',
    subtitle: 'softmax 路由 + top-k + 外部 aux loss',
    params: [
      { name: 'router', shape: ['D', 'E'], note: 'expert 选择头 (无 bias)' },
      { name: 'experts', shape: ['E × SwiGLU(D, d_ff)', ''], note: '每个专家是一个完整 SwiGLU FFN' },
    ],
    flow: [
      { type: 'input', label: 'x', shape: ['B', 'T', 'D'] },
      { type: 'op', op: 'reshape(-1, D)', out: 'x', shape: ['N=B·T', 'D'], kind: 'reshape' },
      { type: 'op', op: 'router · x', out: 'router_logits', shape: ['N', 'E'], kind: 'route' },
      { type: 'op', op: 'softmax(-1)', out: 'probs', shape: ['N', 'E'], kind: 'activation' },
      { type: 'op', op: 'topk(k, -1)', out: '(top_probs, top_idx)', shape: ['N', 'K'], kind: 'route', highlight: true,
        note: '每 token 只激活 k 个专家, 算力 × k/E' },
      { type: 'op', op: 'weights = top_probs / sum', out: 'weights', shape: ['N', 'K'], kind: 'activation' },
      { type: 'op', op: 'for i: out[sel==i] += weights · expert_i(x[sel==i])',
        out: 'out', shape: ['N', 'D'], kind: 'route',
        note: '教学实现: 按专家 for-loop; 工业用 grouped GEMM / 专家并行' },
      { type: 'op', op: 'reshape(B, T, D)', out: 'y', shape: ['B', 'T', 'D'], kind: 'reshape' },
      { type: 'output', label: 'y', shape: ['B', 'T', 'D'] },
    ],
    paramBreakdown: 'D · E  +  E · (3 · D · d_ff)   — 激活比 K/E',
  },

  moe_ds: {
    title: 'DeepSeek MoE',
    subtitle: 'sigmoid + shared experts + aux-loss-free bias',
    params: [
      { name: 'router',       shape: ['D', 'E'], note: '路由头' },
      { name: 'routing_bias', shape: ['E'],      note: '不参与梯度, 外部按负载调节 (aux-free)' },
      { name: 'routed experts', shape: ['E × SwiGLU(D, d_ff_small)', ''], note: '细粒度专家 (E 大, d_ff 小)' },
      { name: 'shared experts', shape: ['S × SwiGLU(D, d_ff_small)', ''], note: '始终激活, 托底通用能力' },
    ],
    flow: [
      { type: 'input', label: 'x', shape: ['B', 'T', 'D'] },
      { type: 'op', op: 'reshape(-1, D)', out: 'x', shape: ['N=B·T', 'D'], kind: 'reshape' },
      { type: 'branch', note: '两路并行: 共享 + 路由', items: [
        { op: 'Σ_s expert_s^shared(x)', out: 'shared_out', shape: ['N', 'D'], kind: 'route' },
        { op: 'sigmoid(router · x)',    out: 's_i', shape: ['N', 'E'], kind: 'activation' },
      ]},
      { type: 'op', op: 's_i + bias (仅用于选择, 不污染加权)', out: 'select_scores', shape: ['N', 'E'], kind: 'route',
        note: 'bias 不求梯度 — 负载均衡与主 loss 解耦' },
      { type: 'op', op: 'topk(select_scores, K)', out: 'top_idx', shape: ['N', 'K'], kind: 'route', highlight: true },
      { type: 'op', op: 'weights = s_i[top_idx] (原始 sigmoid) → renorm', out: 'weights', shape: ['N', 'K'], kind: 'activation',
        note: '用"无 bias 的原始分"做加权, 保持梯度纯净' },
      { type: 'op', op: 'routed_out[sel==i] += w · expert_i^routed(x[sel==i])', out: 'routed_out',
        shape: ['N', 'D'], kind: 'route' },
      { type: 'op', op: 'dropout(shared_out + routed_out)', out: 'y', shape: ['N', 'D'], kind: 'activation' },
      { type: 'op', op: 'reshape(B, T, D)', out: 'y', shape: ['B', 'T', 'D'], kind: 'reshape' },
      { type: 'output', label: 'y', shape: ['B', 'T', 'D'] },
    ],
    paramBreakdown: 'D · E  +  (E + S) · (3 · D · d_ff_small)',
  },
}

// ---------------------------------------------------------------------------
// 归一化变体
// ---------------------------------------------------------------------------

export const normSpecs = {
  post_ln: {
    title: 'LayerNorm (Post-LN)',
    subtitle: 'Vaswani 2017 — 需 warmup',
    params: [
      { name: 'γ', shape: ['D'], note: '缩放' },
      { name: 'β', shape: ['D'], note: '偏移' },
    ],
    flow: [
      { type: 'input', label: 'x_in', shape: ['B', 'T', 'D'] },
      { type: 'op', op: 'y = sublayer(x_in)', out: 'y', shape: ['B', 'T', 'D'], kind: 'attn',
        note: 'Post-LN: LN 包住整个残差后结果' },
      { type: 'op', op: 'r = x_in + y', out: 'r', shape: ['B', 'T', 'D'], kind: 'activation' },
      { type: 'op', op: 'μ = mean(r, -1)', out: 'μ', shape: ['B', 'T', '1'], kind: 'activation' },
      { type: 'op', op: 'σ² = var(r, -1)', out: 'σ²', shape: ['B', 'T', '1'], kind: 'activation' },
      { type: 'op', op: '(r - μ) / √(σ² + ε) · γ + β', out: 'x_out', shape: ['B', 'T', 'D'], kind: 'activation',
        highlight: true, note: '主残差路径上有 LN → 梯度不直接, 深网络难训' },
      { type: 'output', label: 'x_out', shape: ['B', 'T', 'D'] },
    ],
  },

  pre_ln: {
    title: 'LayerNorm (Pre-LN)',
    subtitle: 'GPT-3 风: 残差路径无 LN, 梯度直通',
    params: [
      { name: 'γ', shape: ['D'], note: '缩放' },
      { name: 'β', shape: ['D'], note: '偏移' },
    ],
    flow: [
      { type: 'input', label: 'x', shape: ['B', 'T', 'D'] },
      { type: 'op', op: 'μ = mean(x, -1), σ² = var(x, -1)', out: '(μ, σ²)', shape: ['B', 'T', '1'], kind: 'activation' },
      { type: 'op', op: 'h = (x - μ)/√(σ²+ε) · γ + β', out: 'h', shape: ['B', 'T', 'D'], kind: 'activation' },
      { type: 'op', op: 'y = sublayer(h)', out: 'y', shape: ['B', 'T', 'D'], kind: 'attn',
        note: 'LN 只发生在子层分支内' },
      { type: 'op', op: 'return x + y  (残差主路径无 LN)', out: 'x_out', shape: ['B', 'T', 'D'], kind: 'activation', highlight: true },
      { type: 'output', label: 'x_out', shape: ['B', 'T', 'D'] },
    ],
  },

  pre_rms: {
    title: 'RMSNorm (Pre-Norm)',
    subtitle: 'LLaMA/DeepSeek: 去均值 + 无 bias',
    params: [
      { name: 'γ', shape: ['D'], note: '缩放 — 无 β, 少一组参数' },
    ],
    flow: [
      { type: 'input', label: 'x', shape: ['B', 'T', 'D'] },
      { type: 'op', op: 'rms = √(mean(x², -1) + ε)', out: 'rms', shape: ['B', 'T', '1'], kind: 'activation', highlight: true,
        note: '只算二阶矩 — 比 LayerNorm 省 ~25% 计算' },
      { type: 'op', op: '(x / rms) · γ', out: 'x_out', shape: ['B', 'T', 'D'], kind: 'activation',
        note: '无中心化, 无偏移 — 大模型上效果几乎等同 LayerNorm' },
      { type: 'output', label: 'x_out', shape: ['B', 'T', 'D'] },
    ],
  },

  ada_ln: {
    title: 'adaLN-Zero',
    subtitle: 'DiT: 用条件 c 调制每层的 (γ, β, α) · 初始化为 0',
    params: [
      { name: 'adaLN(c)', shape: ['c_dim', '6·D'], note: '把条件切成 6 段: γ₁,β₁,α₁,γ₂,β₂,α₂' },
      { name: 'LayerNorm', shape: ['—'], note: '无仿射 (affine=False), 仿射由 adaLN 给' },
    ],
    flow: [
      { type: 'input', label: 'x', shape: ['B', 'T', 'D'] },
      { type: 'input', label: 'c', shape: ['B', 'c_dim'] },
      { type: 'op', op: 'adaLN(c).chunk(6, -1)', out: '(γ,β,α) × 2',
        shape: ['B', 'D'], kind: 'cond',
        note: '初始化为 0 → 每个 block 起点是恒等映射 (identity)' },
      { type: 'op', op: 'h = LN(x)  (no affine)', out: 'h', shape: ['B', 'T', 'D'], kind: 'activation' },
      { type: 'op', op: 'modulate: h = (1+γ) · h + β', out: 'h', shape: ['B', 'T', 'D'], kind: 'cond', highlight: true },
      { type: 'op', op: 'sub = sublayer(h)', out: 'sub', shape: ['B', 'T', 'D'], kind: 'attn' },
      { type: 'op', op: 'x_out = x + α · sub  (α 作为 gate)', out: 'x_out',
        shape: ['B', 'T', 'D'], kind: 'activation' },
      { type: 'output', label: 'x_out', shape: ['B', 'T', 'D'] },
    ],
  },
}

// ---------------------------------------------------------------------------
// 位置编码变体
// ---------------------------------------------------------------------------

export const posSpecs = {
  sin: {
    title: 'Sinusoidal PE',
    subtitle: 'Vaswani 2017: 加在 embedding 上, 绝对位置',
    params: [
      { name: 'pe (buffer)', shape: ['max_len', 'D'], note: '预计算, 不参与训练 (persistent=False)' },
    ],
    flow: [
      { type: 'input', label: 'x', shape: ['B', 'T', 'D'] },
      { type: 'op', op: 'pe[:, :T]', out: 'pe_slice', shape: ['1', 'T', 'D'], kind: 'reshape',
        note: '每个位置 pos 的 2i 维 = sin(pos/10000^(2i/D)), 2i+1 维 = cos(...)' },
      { type: 'op', op: 'y = x + pe_slice', out: 'y', shape: ['B', 'T', 'D'], kind: 'activation', highlight: true },
      { type: 'output', label: 'y', shape: ['B', 'T', 'D'] },
    ],
  },

  learn: {
    title: 'Learnable Positional Embedding',
    subtitle: 'BERT / ViT: 每个位置一个可学习向量',
    params: [
      { name: 'pos_embed', shape: ['max_len', 'D'], note: '可学, 训练数据中的位置分布会被烙印进来' },
    ],
    flow: [
      { type: 'input', label: 'x', shape: ['B', 'T', 'D'] },
      { type: 'op', op: 'pos_embed[:T]', out: 'p', shape: ['T', 'D'], kind: 'reshape' },
      { type: 'op', op: 'y = x + p', out: 'y', shape: ['B', 'T', 'D'], kind: 'activation' },
      { type: 'output', label: 'y', shape: ['B', 'T', 'D'] },
    ],
  },

  rope: {
    title: 'Rotary Position Embedding (RoPE)',
    subtitle: '旋转 Q/K · 编码相对位置 · 无参数',
    params: [
      { name: 'cos, sin (buffer)', shape: ['max_len', 'D_h'], note: '预计算表 (buffer)' },
    ],
    flow: [
      { type: 'input', label: 'q or k', shape: ['B', 'H', 'T', 'D_h'] },
      { type: 'op', op: 'lookup(cos, sin, position_ids)', out: '(cos, sin)',
        shape: ['T', 'D_h'], kind: 'cond' },
      { type: 'op', op: 'rotate_half(x) = [-x2, x1]',
        out: 'x_rot', shape: ['B', 'H', 'T', 'D_h'], kind: 'reshape',
        note: '前半段与后半段对调并取负' },
      { type: 'op', op: 'out = x · cos + rotate_half(x) · sin',
        out: 'out', shape: ['B', 'H', 'T', 'D_h'], kind: 'activation', highlight: true,
        note: '等价于把相邻两维当复数乘 e^(imθ)' },
      { type: 'output', label: 'out', shape: ['B', 'H', 'T', 'D_h'] },
    ],
  },

  mrope: {
    title: 'Multimodal RoPE (M-RoPE)',
    subtitle: 'Qwen2-VL: head_dim 切三段, 各轴独立 RoPE',
    params: [
      { name: '3 × cos, sin', shape: ['max_len', 'D_t|D_h|D_w'],
        note: '每段独立的 cos/sin 表, 不共享' },
    ],
    flow: [
      { type: 'input', label: 'x', shape: ['B', 'H', 'T', 'D_h'] },
      { type: 'input', label: 'position_ids', shape: ['3', 'B', 'T'], note: '三行分别是 (t, h, w)' },
      { type: 'op', op: 'split(x, [D_t, D_h_ax, D_w], dim=-1)', out: '(xt, xh, xw)',
        shape: ['B', 'H', 'T', 'D_t | D_h_ax | D_w'], kind: 'reshape',
        note: '三段之和 = D_h; 每段必须为偶数 (RoPE 要成对旋转)' },
      { type: 'branch', note: '每段用对应轴的 position 独立 RoPE', items: [
        { op: 'rope_t(xt, pos[0])', out: 'xt', shape: ['B', 'H', 'T', 'D_t'], kind: 'cond' },
        { op: 'rope_h(xh, pos[1])', out: 'xh', shape: ['B', 'H', 'T', 'D_h_ax'], kind: 'cond' },
        { op: 'rope_w(xw, pos[2])', out: 'xw', shape: ['B', 'H', 'T', 'D_w'], kind: 'cond' },
      ]},
      { type: 'op', op: 'concat([xt, xh, xw], dim=-1)', out: 'out',
        shape: ['B', 'H', 'T', 'D_h'], kind: 'reshape', highlight: true,
        note: '文本 token 的 (t, t, t) 三轴相同 → 退化为 1D RoPE' },
      { type: 'output', label: 'out', shape: ['B', 'H', 'T', 'D_h'] },
    ],
  },
}

// ---------------------------------------------------------------------------
// Shape 表达式求值
// ---------------------------------------------------------------------------

export function evalExpr(expr, ctx) {
  if (typeof expr !== 'string') return expr
  // 简化常见分隔符号, 统一成可求值的 JS
  const cleaned = expr
    .replace(/·/g, '*')
    .replace(/×/g, '*')
    .replace(/\s/g, '')
  if (!/^[0-9A-Za-z_+\-*/()]+$/.test(cleaned)) return expr
  try {
    const keys = Object.keys(ctx)
    const vals = Object.values(ctx)
    // eslint-disable-next-line no-new-func
    const fn = new Function(...keys, `return (${cleaned})`)
    const result = fn(...vals)
    return Number.isFinite(result) ? result : expr
  } catch {
    return expr
  }
}

export function formatShape(shape, ctx) {
  return shape.map(s => {
    const v = evalExpr(s, ctx)
    if (typeof v === 'number') return v.toLocaleString()
    return v // 符号, 如 '...' 或 '—'
  })
}

export function formatParams(count) {
  if (count < 1e3) return count.toLocaleString()
  if (count < 1e6) return (count / 1e3).toFixed(1) + 'K'
  if (count < 1e9) return (count / 1e6).toFixed(2) + 'M'
  return (count / 1e9).toFixed(2) + 'B'
}

export function computeParamCount(param, ctx) {
  const dims = param.shape.map(s => evalExpr(s, ctx))
  if (dims.some(d => typeof d !== 'number')) return null
  return dims.reduce((a, b) => a * b, 1)
}
