// 注意力的四代演进 —— 与 llm_models/layers/core/attention.py 对应。
// 每一代都是在解上一代留下的账: MHA 的 cache 太大 → GQA 共享 K/V → MLA 压成 latent → DSA 再砍算力。

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
    pros: '每个头有自己的 K/V, 表达力上限最高',
    cons: 'cache 也最大 —— LLaMA-2-7B 一个 token 要存 512 KiB',
    description: '每个头配一对自己的 K、V。这是注意力最初的样子, 也是 cache 最贵的样子: 每生成一个 token, 32 层 × 32 个头的 K 和 V 都要写进显存, 之后每一步都要整个读一遍。长上下文推理卡在这里。'
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
    pros: 'cache 除以 n_heads/num_kv_heads, 质量几乎不掉',
    cons: '还是要按 num_kv_heads × head_dim 存一份',
    description: '让几个 Q 头共用一对 K/V。LLaMA-3-8B 用 8 组代替 32 个头, 一个 token 的 KV 从 512 KiB 降到 128 KiB, 评测基本没动。两头是极端: num_kv_heads=1 就是 MQA, cache 最小但会掉点; =n_heads 就退回 MHA。现在开源模型基本都用它。'
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
    pros: '只缓存 latent + 共享 rope 段, 每 token 68.6 KiB',
    cons: '要把 RoPE 解耦出来, 升维矩阵才能被吸收进 Q',
    description: 'K 和 V 不直接存, 先压成一个低维 latent (比如 512 维) 存起来, 用的时候再升回去。位置信息单独走一小段所有头共享的 RoPE —— 这一步必须解耦, 否则旋转会挡住升维矩阵被吸收进 Q, 解码时就省不下来。DeepSeek-V3 一个 token 68.6 KiB, 比同规模的 MHA 少 56.9 倍。'
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
    pros: '算力 O(T²) → O(T·k), 冲的是 128K 以上的长上下文',
    cons: 'top-k 不可导, 必须另给 indexer 一个对齐损失, 否则它拿不到梯度',
    description: 'MLA 省的是显存, 它省的是算力: 先用几个小头快速给所有位置打个分, 只挑 top-k 个位置做真正的注意力。要先 mask 再 top-k, 否则会选到未来的 token。本仓库实测: 只用语言模型损失时 indexer 的梯度是 None, 加上 KL 对齐后 top-8 召回从 0.450 (等于瞎猜) 升到 0.922。'
  },
]

