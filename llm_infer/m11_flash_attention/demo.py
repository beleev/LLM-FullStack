"""
m11 demo — FlashAttention: 分块结果 == 朴素 attention, 工作集与 T 无关

四件事, 全部带 assert:
    1) O 与 core.dense_attention 对拍, lse 与朴素 logsumexp 对拍 (< 1e-5)
    2) causal 整块跳过: 统计 full / partial / skipped 块数
    3) 峰值工作集 b_q×b_k vs T² (由 stats 实测, 不是公式估算)
    4) 用 lse 合并两段部分 attention == 全量 attention (ring attention / chunked prefill 原语)
"""
from __future__ import annotations
import numpy as np

from llm_infer.core import dense_attention, causal_mask
from llm_infer.core.utils import banner
from llm_infer.m11_flash_attention.flash_attention import flash_attention, merge_attention

TOL = 1e-5


def naive_lse(Q, K, mask):
    """朴素 logsumexp: 先落地整张 (Tq,Tk) 分数矩阵。"""
    S = (Q @ K.T / np.sqrt(Q.shape[-1]) + mask).astype(np.float64)
    m = S.max(-1)
    return m + np.log(np.exp(S - m[:, None]).sum(-1))


def main():
    banner("M11 - FlashAttention (tiling + online softmax + LSE)")
    rs = np.random.RandomState(0)
    D = 32
    rand = lambda *s: rs.randn(*s).astype(np.float32)

    print("\n[1] 数值对拍 vs core.dense_attention (Tq<Tk 即 chunked prefill / decode 形态)")
    print(f"  {'Tq':>5} {'Tk':>5} {'b_q':>4} {'b_k':>4} {'causal':>6}  {'max|ΔO|':>10}  {'max|Δlse|':>10}")
    worst = 0.0
    for Tq, Tk in [(16, 16), (64, 64), (256, 256), (8, 64), (1, 100)]:
        Q, K, V = rand(Tq, D), rand(Tk, D), rand(Tk, D)
        for bq, bk in [(4, 4), (16, 8), (64, 32)]:
            for causal in (True, False):
                mask = causal_mask(Tq, Tk) if causal else np.zeros((Tq, Tk), np.float32)
                O, lse, _ = flash_attention(Q, K, V, bq, bk, causal)
                dO = np.max(np.abs(O - dense_attention(Q, K, V, mask)))
                dl = np.max(np.abs(lse - naive_lse(Q, K, mask)))
                worst = max(worst, dO, dl)
                if (bq, bk) == (16, 8):
                    print(f"  {Tq:>5} {Tk:>5} {bq:>4} {bk:>4} {str(causal):>6}  {dO:>10.2e}  {dl:>10.2e}")
    print(f"  30 组配置的最坏误差 = {worst:.2e}")
    assert worst < TOL

    print("\n[2] causal 整块跳过 + [3] 峰值工作集 (b_q=b_k=64, stats 实测)")
    print(f"  {'T':>5} {'full':>6} {'partial':>8} {'skipped':>8} {'跳过比例':>8}  {'peak b_q×b_k':>12} {'T²':>12} {'省':>7}")
    for T in [256, 1024, 4096]:
        Q, K, V = rand(T, D), rand(T, D), rand(T, D)
        O, _, st = flash_attention(Q, K, V, 64, 64, causal=True)
        n = T // 64
        total = st["full"] + st["partial"] + st["skipped"]
        print(f"  {T:>5} {st['full']:>6} {st['partial']:>8} {st['skipped']:>8} {st['skipped'] / total:>8.1%}"
              f"  {st['peak_elems']:>12,} {T * T:>12,} {T * T // st['peak_elems']:>6}x")
        assert total == n * n and st["skipped"] == n * (n - 1) // 2 and st["partial"] == n
        assert st["peak_elems"] == 64 * 64                # 与 T 无关
        if T <= 1024:                                     # T=4096 的朴素基线要 16M 元素, 不跑
            assert np.max(np.abs(O - dense_attention(Q, K, V))) < TOL

    print("\n[4] 用 lse 合并两段部分 attention (ring attention / chunked prefill 原语)")
    Tk, Tq = 128, 32                                      # 最后 32 个 query, K/V 前后两半各 64 (想象在两张卡上)
    Q, K, V = rand(Tq, D), rand(Tk, D), rand(Tk, D)
    h = Tk // 2
    O1, l1, _ = flash_attention(Q, K[:h], V[:h], causal=False)   # 前半段全在过去 → 全可见
    O2, l2, _ = flash_attention(Q, K[h:], V[h:], causal=True)    # 后半段含 query 自己 → causal
    O_m, l_m = merge_attention(O1, l1, O2, l2)
    dO = np.max(np.abs(O_m - dense_attention(Q, K, V)))
    dl = np.max(np.abs(l_m - naive_lse(Q, K, causal_mask(Tq, Tk))))
    d_wrong = np.max(np.abs((O1 + O2) / 2 - dense_attention(Q, K, V)))
    print(f"  merge(O1,lse1,O2,lse2) vs 全量: max|ΔO| = {dO:.2e}, max|Δlse| = {dl:.2e}")
    print(f"  反例: 不用 lse, 直接 (O1+O2)/2  : max|ΔO| = {d_wrong:.2e}  ← 两段的 softmax 质量不相等")
    assert dO < TOL and dl < TOL and d_wrong > 100 * TOL

    print("\n  ✓ 分块输出与 lse 均与朴素实现一致 (< 1e-5)")
    print("  ✓ causal 下 (n-1)/2n ≈ 一半的块整块跳过; 工作集恒为 b_q×b_k = 4,096 个元素")
    print("  ✓ 只凭 (O, lse) 就能跨设备 / 跨 chunk 精确合并 attention")


if __name__ == "__main__":
    main()
