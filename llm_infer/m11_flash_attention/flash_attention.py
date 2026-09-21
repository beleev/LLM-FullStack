"""
flash_attention.py — 分块 (tiling) + online softmax 的 attention, numpy 版

是什么: 不落地 (Tq,Tk) 分数矩阵, 每次只算一块 (b_q, b_k), 用 logsumexp 增量合并。
解决的瓶颈: 显存 O(T²) → 工作集 O(b_q·b_k); 真实 GPU 上更关键的是少读写 HBM (延迟)。
关键数字: T=4096, b=64 → 工作集 4,096 个元素 vs 16,777,216 (4096×); causal 下约一半的块整块跳过。
循环顺序: 外层 Q 块, 内层 K/V 块 —— FlashAttention-2 的顺序 (FA-1 相反)。
    好处: 一个 Q 块的 (O_i, lse_i) 在内层循环里常驻 SRAM, 只写回 HBM 一次; Q 块之间无依赖, 可并行。
读代码盯住: `lse` (每个 query 行的 logsumexp), 它是 softmax 分母的 log。有了它:
    - 两段部分 attention 可以精确合并 (merge_attention) → ring attention / chunked prefill 的原语
    - 反向传播不用存 P (T,T), 由 lse 重算 P = exp(S - lse)
对应真实系统: flash-attn 库返回的 softmax_lse; vLLM / SGLang 的 FlashAttention / FlashInfer backend。
简化: 单头 (T,D), 无 batch; 要求 Tq <= Tk (query 对齐 K 的尾部, 与 core.dense_attention 同约定)。
"""
from __future__ import annotations
import numpy as np


def _logsumexp(S: np.ndarray) -> np.ndarray:
    """按行 logsumexp, (bq,bk) → (bq,); 整行 -inf (本块对该行全被 mask) 时返回 -inf 而不是 nan。"""
    m = np.max(S, axis=-1)
    m_safe = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):                    # log(0) = -inf 是预期结果
        return m_safe + np.log(np.sum(np.exp(S - m_safe[:, None]), axis=-1))


def merge_attention(O1, lse1, O2, lse2):
    """把同一批 query 在两段不相交 K/V 上的部分结果, 精确合并成"在全部 K/V 上"的结果。

    O_i (Tq,dv) 是各自归一化过的输出, lse_i (Tq,) 是各自分母的 log。
    全局分母 = e^lse1 + e^lse2, 所以 O = (e^lse1·O1 + e^lse2·O2) / (e^lse1 + e^lse2)。
    """
    lse = np.logaddexp(lse1, lse2)                        # (Tq,)
    w1 = np.exp(lse1 - lse)[:, None]                      # (Tq,1) 第 1 段占全局 softmax 质量的比例
    w2 = np.exp(lse2 - lse)[:, None]
    return w1 * O1 + w2 * O2, lse


def flash_attention(Q, K, V, block_q: int = 16, block_k: int = 16, causal: bool = True):
    """Q (Tq,d), K (Tk,d), V (Tk,dv) → (O (Tq,dv), lse (Tq,), stats)。

    常驻内存: O 与 lse 是 O(Tq) 的输出; 临时工作集只有一块 S (b_q, b_k), 与 T 无关。
    stats: full / partial (跨对角线, 需逐元素 mask) / skipped (整块在对角线上方, 连 matmul 都不做)
           三类块数, 以及 peak_elems = 见过的最大 S.size。
    """
    Tq, d = Q.shape
    Tk = K.shape[0]
    assert Tq <= Tk, "causal 约定: query 对齐 K 的尾部"
    offset = Tk - Tq                                      # query 行 i 的绝对位置 = i + offset
    O = np.zeros((Tq, V.shape[1]), dtype=np.float32)
    lse = np.full(Tq, -np.inf, dtype=np.float32)
    stats = dict(full=0, partial=0, skipped=0, peak_elems=0)
    n_kblocks = -(-Tk // block_k)

    for qs in range(0, Tq, block_q):                      # 外层: Q 块 (FA-2 顺序)
        qe = min(Tq, qs + block_q)
        Qb = Q[qs:qe]                                     # (bq, d)
        Ob, lb = O[qs:qe], lse[qs:qe]                     # 视图: 本 Q 块的累加器, 内层循环里原地更新
        for kb, ks in enumerate(range(0, Tk, block_k)):   # 内层: K/V 块
            ke = min(Tk, ks + block_k)
            if causal and ks > qe - 1 + offset:           # 块内最早的 key 也晚于块内最晚的 query
                stats["skipped"] += n_kblocks - kb        # 后面的 K 块更晚, 一起跳过
                break
            S = Qb @ K[ks:ke].T / np.sqrt(d)              # (bq, bk) ← 唯一的 "attention 矩阵", 只有一块
            stats["peak_elems"] = max(stats["peak_elems"], S.size)
            if causal and ke - 1 > qs + offset:           # 块跨过对角线 → 逐元素 mask
                i_abs = np.arange(qs, qe)[:, None] + offset        # (bq,1)
                S = np.where(np.arange(ks, ke)[None, :] > i_abs, -np.inf, S)
                stats["partial"] += 1
            else:
                stats["full"] += 1
            # online softmax: 与 merge_attention 同一个公式, 只是新块不先归一化
            # (省一次除法, 也避开 "整行被 mask → 0/0")
            new_l = np.logaddexp(lb, _logsumexp(S))       # (bq,)
            Ob[:] = np.exp(lb - new_l)[:, None] * Ob + np.exp(S - new_l[:, None]) @ V[ks:ke]  # (bq,bk)@(bk,dv)
            lb[:] = new_l
    return O, lse, stats
