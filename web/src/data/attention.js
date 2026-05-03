// Attention 演进谱 — 与 layers/attention.py 对应

export const variants = [
  {
    id: 'mha',
    name: 'MHA',
    fullName: 'Multi-Head Attention',
    year: 2017,
    paper: 'Attention Is All You Need',
    usedIn: ['Transformer', 'BERT', 'GPT-3', 'CLIP', 'DiT'],
    // cacheBytes(B, T, d_model, n_heads, {num_kv_heads, rank, rope_dim})
    cache: ({ T, d_model, n_heads }) => 2 * T * d_model, // K + V, full d_model
    formula: 'Attention(Q,K,V) = softmax(QK^T / √d_k) · V',
    color: '#9ca3af',
    pros: '表达力满, 每头独立 KV',
    cons: 'KV cache 最大 (长上下文的显存瓶颈)',
    description: '每头一对独立的 K/V, 是注意力的原始设计。推理时每个新 token 都要把所有层的 K, V 写进 cache, 随 seq_len 与 n_heads 线性增长。'
  },
  {
    id: 'gqa',
    name: 'GQA',
    fullName: 'Grouped-Query Attention',
    year: 2023,
    paper: 'Ainslie et al. — LLaMA-2 70B / Qwen2',
    usedIn: ['LLaMA-2/3', 'Qwen2', 'Mixtral', 'Qwen2-VL', 'Omni'],
    cache: ({ T, d_model, n_heads, num_kv_heads }) => 2 * T * (d_model / n_heads) * num_kv_heads,
    formula: 'K, V 只有 num_kv_heads 组, 多头 Q 共享它们',
    color: '#60a5fa',
    pros: 'KV cache ÷ (n_heads / num_kv_heads), 效果 ≈ MHA',
    cons: '仍需缓存 num_kv_heads × head_dim',
    description: '把 Q 的 N 个头分成 G 组, 每组共享同一对 K/V head。num_kv_heads=1 退化为 MQA (极小 cache 但掉点), num_kv_heads=n_heads 退化为 MHA。现代开源 LLM 事实标准。'
  },
  {
    id: 'mla',
    name: 'MLA',
    fullName: 'Multi-Head Latent Attention',
    year: 2024,
    paper: 'DeepSeek-V2/V3',
    usedIn: ['DeepSeek-V2', 'DeepSeek-V3'],
    // cache = latent c_kv + shared rope dim
    cache: ({ T, kv_lora_rank, qk_rope_head_dim }) => T * (kv_lora_rank + qk_rope_head_dim),
    formula: 'c_kv = W_DKV·x,  K/V 由 c_kv 升维;  解耦 RoPE 段独立处理',
    color: '#34d399',
    pros: 'KV cache -93%, 生产推理只缓存 c_kv + 共享 k_rope',
    cons: '需要解耦 RoPE + up-projection 矩阵吸收',
    description: 'KV 低秩投到一个小维度 latent c_kv (e.g. 512), 运行时升维还原。单独一段 rope_dim 全头共享, 承载位置信息, 其余 nope 段参与吸收 trick。DeepSeek 报告 671B 模型在长上下文下 KV 压力几乎可忽略。'
  },
  {
    id: 'dsa',
    name: 'MLA + DSA',
    fullName: 'DeepSeek Sparse Attention',
    year: 2025,
    paper: 'DeepSeek-V3.2 Tech Report',
    usedIn: ['DeepSeek-V3.2'],
    cache: ({ T, kv_lora_rank, qk_rope_head_dim }) => T * (kv_lora_rank + qk_rope_head_dim),
    formula: 'LightningIndexer 选 top-k; MLA 仅在 k 个位置算 softmax · V',
    color: '#f472b6',
    pros: '算力 O(T²) → O(T·k), 主攻 128K+ 超长上下文',
    cons: '稀疏选择需避免数据泄漏 (先 mask 再 topk)',
    description: '在 MLA 之上叠 Lightning Indexer (几个小头 + ReLU 打分) 预选 top-k 关键位置。MLA 解决 cache, DSA 解决算力, 两者解耦组合。'
  },
]

// 复杂度与 cache 计算工具
export function computeMetrics(variant, params) {
  const bytesPerToken = variant.cache(params)  // 以 fp16 = 2 bytes/element 估算
  const totalBytes = bytesPerToken * 2         // fp16
  const flopsPerToken = variant.id === 'dsa'
    ? params.T * Math.min(params.sparse_top_k, params.T)
    : params.T * params.T
  return { bytesPerToken, totalBytes, flopsPerToken }
}
