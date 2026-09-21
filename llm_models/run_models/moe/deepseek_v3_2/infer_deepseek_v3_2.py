#!/usr/bin/env python
"""
DeepSeek-V3.2 推理示例: DSA = MLA + Lightning Indexer

验证四件事 (全是 assert):
  1. 构造不浪费: 每层的 MLA 只建一次; 参数量 = 同配置 V3 + indexer 参数
  2. 稀疏 mask: 每个 query 最多看 sparse_top_k 个 key, 且绝不看未来
  3. 稀疏确实生效: 序列长于 top-k 时, V3.2 的输出与 "同权重走稠密注意力" 不同
  4. decode 时 indexer 也走 cache (多缓存一份 indexer key), 贪心输出与无 cache 完全相同
"""

import torch

from llm_models.layers.core.attention import MultiHeadLatentAttention
from llm_models.models.moe.deepseekV3 import DeepSeekV3, DeepSeekV3_2
from llm_models.utils.generation import KVCache

SPARSE_TOP_K, IDX_HEADS, IDX_DIM, R, ROPE = 8, 2, 32, 64, 32


def cached_logits(model, idx: torch.Tensor, prefill: int) -> torch.Tensor:
    """prefill 前 `prefill` 个 token, 其余逐个 decode, 拼回 [B, T, V] —— 应与一次性 forward 相同。"""
    cache = KVCache(len(model.layers))
    chunks = [idx[:, :prefill]] + list(idx[:, prefill:].split(1, dim=1))
    with torch.inference_mode():
        return torch.cat([model(c, cache=cache)[0] for c in chunks], dim=1)


def main():
    torch.manual_seed(42)
    base = dict(
        vocab_size=1000, d_model=256, n_heads=4, num_layers=2, max_len=128,
        num_routed_experts=4, num_shared_experts=1, top_k=2, latent_dim=R, qk_rope_head_dim=ROPE,
    )
    dsa = dict(sparse_top_k=SPARSE_TOP_K, indexer_heads=IDX_HEADS, indexer_head_dim=IDX_DIM)

    # ---- 1) 每层只建一次 MLA (以前: V3 的层先建一遍再整体丢弃, Block 里的 MLA 又建了再替换) ----
    built, orig_init = [], MultiHeadLatentAttention.__init__
    MultiHeadLatentAttention.__init__ = lambda self, *a, **k: (built.append(1), orig_init(self, *a, **k))[1]
    try:
        model = DeepSeekV3_2(**base, **dsa).eval()
    finally:
        MultiHeadLatentAttention.__init__ = orig_init
    assert len(built) == base["num_layers"], f"MLA 被构造了 {len(built)} 次"

    count = lambda m: sum(p.numel() for p in m.parameters())
    indexer_params = sum(count(layer.attn.indexer) for layer in model.layers)
    assert indexer_params == base["num_layers"] * 2 * base["d_model"] * IDX_HEADS * IDX_DIM  # w_q + w_k
    assert count(model) == count(DeepSeekV3(**base)) + indexer_params
    print(f"MLA 构造次数 {len(built)} (= 层数) | 总参数 {count(model):,} = V3 + indexer {indexer_params:,}")

    # ---- 2) 稀疏 mask 的两条不变量 ----
    T = 24
    attn = model.layers[0].attn
    causal = model._causal_mask(T)  # [1, T, T]
    with torch.inference_mode():
        scores = attn.indexer(torch.randn(2, T, base["d_model"]), mask=causal)  # [2, T, T]
        sparse = attn._sparse_mask_from_topk(scores, causal)
    assert not (sparse & ~causal).any(), "选中了未来 token"
    per_row = sparse.sum(-1)  # [2, T]
    assert per_row.max() == SPARSE_TOP_K and (per_row[:, :SPARSE_TOP_K] == torch.arange(1, SPARSE_TOP_K + 1)).all()
    print(f"T={T}: 每个 query 看到的 key 数 = {per_row[0].tolist()}  (上限 {SPARSE_TOP_K}, 稠密时是 1..{T})")

    # ---- 3) 稀疏真的改变了计算 ----
    idx = torch.randint(0, base["vocab_size"], (1, T))
    with torch.inference_mode():
        sparse_logits = model(idx)[0]
        model.set_dense_warmup(True)
        dense_logits = model(idx)[0]
        model.set_dense_warmup(False)
    assert torch.allclose(sparse_logits[:, :SPARSE_TOP_K], dense_logits[:, :SPARSE_TOP_K], atol=1e-5)  # 前 k 个位置本来就全看
    assert not torch.allclose(sparse_logits[:, -1], dense_logits[:, -1], atol=1e-5)
    print(f"稀疏 vs 稠密 最后位置 logits 最大差 {(sparse_logits[:, -1] - dense_logits[:, -1]).abs().max():.4f} (> 0)")

    # ---- 4) 带 cache 的 decode: indexer key 也缓存 ----
    full = torch.randint(0, base["vocab_size"], (2, 30))
    with torch.inference_mode():
        diff = (model(full)[0] - cached_logits(model, full, prefill=10)).abs().max().item()
    print(f"prefill 10 + 逐 token decode 20 步 vs 一次性 forward: logits 最大差 {diff:.2e}")
    assert diff < 1e-4, "cache 路径与无 cache 路径的 logits 不一致"
    prompt = idx[:, :12]
    with_cache = model.generate(prompt, max_new_tokens=20, temperature=0, use_cache=True)
    no_cache = model.generate(prompt, max_new_tokens=20, temperature=0, use_cache=False)
    assert torch.equal(with_cache, no_cache), "cache 与无 cache 的贪心输出必须完全相同"

    cache = KVCache(len(model.layers))
    with torch.inference_mode():
        model(prompt, cache=cache)
    per_token = {k: v.numel() // prompt.size(1) for k, v in cache.layers[0].items()}
    assert per_token == {"idx_k": IDX_HEADS * IDX_DIM, "c_kv": R, "k_rope": ROPE}
    print(f"贪心生成 20 token, cache 与无 cache 一致 | 每 token 每层 cache: {per_token} (idx_k 是 DSA 多付的代价)")


if __name__ == "__main__":
    main()
