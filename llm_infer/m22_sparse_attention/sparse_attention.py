"""
sparse_attention.py — 稀疏 attention decode: 先给 KV block 廉价打分, 只读 top-k 个 block

是什么: 长上下文 decode 时 attention 质量集中在少数 token 上。把 KV 切成 block, 每 block 存一份
        很小的摘要 (逐维 min/max 或均值), 用 q 和摘要算分数选 top-k block, 只对选中的 block 做 attention。
瓶颈:   decode 延迟 / 带宽。decode attention 是访存瓶颈, 每步要读全部 KV; 只读 1/8 的 block ≈ 访存降 8 倍。
关键数字: 摘要开销 = 每 block 2 个向量 (Quest) 或 1 个 (mean), block=16 时 ≈ KV 的 1/16 ~ 1/8。
盯住:   quest_upper_bound 的逐维 max(q·kmin, q·kmax) (保证 ≥ block 内真实 max q·k) 和 select_blocks 的 forced。
真实系统: Quest (min/max 界) / DeepSeek NSA (压缩 key 打分 + 选块) / DeepSeek-V3.2 DSA lightning indexer
        (低维 indexer 打分 + top-k token); 这里只有 decode 选块, 没有训练出来的 indexer。
"""
from __future__ import annotations
import numpy as np

from llm_infer.core import dense_attention, softmax


def block_view(K, bs: int):
    """K (T, d) → (nb, bs, d)。要求 T 整除 bs (demo 保证)。"""
    return K.reshape(-1, bs, K.shape[-1])


def quest_upper_bound(q, K, bs: int):
    """Quest: 每 block 只存逐维 kmin/kmax (nb, d)。对任意 block 内的 k:
    q_i·k_i ≤ max(q_i·kmin_i, q_i·kmax_i) (q_i 正取 kmax, 负取 kmin) → 逐维求和即 q·k 的上界。"""
    Kb = block_view(K, bs)
    kmin, kmax = Kb.min(1), Kb.max(1)                       # (nb, d) ← 这就是常驻显存的 "摘要"
    return np.maximum(q * kmin, q * kmax).sum(-1)           # (nb,)


def mean_score(q, K, bs: int):
    """均值池化 key 打分 (NSA 压缩分支 / 低维 indexer 的最简替身): q·mean(K_block)。"""
    return block_view(K, bs).mean(1) @ q                    # (nb,)


def select_blocks(scores, k: int):
    """top-k block。永远保留第 0 块 (attention sink, 见 m16) 和最后一块 (最近 token), 它们算在 k 内。"""
    nb = len(scores)
    forced = {0, nb - 1}
    rest = [b for b in np.argsort(-scores, kind="stable") if b not in forced]
    return np.sort(np.array(list(forced) + rest[:max(k - len(forced), 0)]))


def random_blocks(nb: int, k: int, rs):
    """对照组: 同样强制首/尾块, 其余随机。"""
    return select_blocks(rs.rand(nb), k)


def sparse_decode(q, K, V, blocks, bs: int):
    """只 gather 选中 block 的 KV 再做 attention。q (d,), K/V (T, d) → (d,)"""
    idx = (blocks[:, None] * bs + np.arange(bs)).reshape(-1)     # (k·bs,) token 下标
    return dense_attention(q[None], K[idx], V[idx])[0]           # Tq=1: 因果 mask 全可见


def evaluate(q, K, V, blocks, bs: int):
    """→ (相对 L2 误差, 选中 block 覆盖的真实 attention 质量)。"""
    full = dense_attention(q[None], K, V)[0]
    err = np.linalg.norm(sparse_decode(q, K, V, blocks, bs) - full) / np.linalg.norm(full)
    probs = softmax(K @ q / np.sqrt(len(q)))                     # (T,) 真实 attention 分布, 只用于评测
    return err, block_view(probs[:, None], bs).sum((1, 2))[blocks].sum()


def needle_context(T=4096, d=64, bs=16, n_needle_blocks=4, per_block=2, strength=10.0, seed=0):
    """合成长上下文: 噪声 key + 少数 "针" block, 针的 key 与 q 对齐 (logit ≈ strength, 噪声 logit ~ N(0,1))。"""
    rs = np.random.RandomState(seed)
    q = rs.randn(d)
    K, V = rs.randn(T, d), rs.randn(T, d)
    nb = T // bs
    needles = rs.choice(np.arange(1, nb - 1), n_needle_blocks, replace=False)
    for b in needles:
        pos = b * bs + rs.choice(bs, per_block, replace=False)
        K[pos] += q * (strength * np.sqrt(d) / (q @ q))          # 使 q·k/√d 增加 strength
    return q, K, V, np.sort(needles)
