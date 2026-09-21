"""
M16 — DeepSpeed-Ulysses 序列并行

是什么: 注意力之外, 每卡持有序列的 1/P: [T/P, H, d]; 进注意力前做一次 all-to-all, 把
        "按序列切" 换成 "按头切": [T, H/P, d] —— 每卡看到 **完整序列** 但只负责 H/P 个头; 算完再换回来。
解决的瓶颈: 长序列激活显存, 与 Ring Attention (m12) 同一个目标, 通信方式不同:
    Ulysses: 4 次 all-to-all (Q, K, V, 输出), 每 rank 发 4·(P-1)/P · (T/P·H·d)   → 随 P 增大而 **下降**
    Ring   : P-1 轮 P2P,               每 rank 发 (P-1) · 2·(T/P·H·d)           → 随 P 基本 **不变**
代价: 头数必须能被 P 整除 (P ≤ H; GQA/MQA 下 KV 头更少, 限制更紧); all-to-all 吃对分带宽, 跨机不如 ring 的邻居 P2P 友好。
      注意力核本身不用改 —— 每卡就是一次普通的 (Flash) attention, 因果 mask 也天然均衡, 不需要 zigzag。
读代码盯住: `seq_to_head` 的 shape 变化 [T/P, H, d] → [T, H/P, d], 以及它的逆 `head_to_seq`。
"""
from __future__ import annotations

import numpy as np

from llm_train.core import all_to_all, banner, comm, kv, make_rng, max_abs_diff, softmax
from llm_train.m12_sequence_parallel.demo import ring_attention


def causal_mha(q, k, v):
    """q/k/v: [T, H, d] → [T, H, d]。各头独立的因果注意力。"""
    t, _, d = q.shape
    scores = np.einsum("thd,shd->hts", q, k) / np.sqrt(d)                # [H, T, T]
    scores[:, np.triu(np.ones((t, t), dtype=bool), k=1)] = -np.inf
    return np.einsum("hts,shd->thd", softmax(scores), v)


def seq_to_head(shards):
    """P × [T/P, H, d] → P × [T, H/P, d]。rank s 把自己的第 j 组头发给 rank j。"""
    P = len(shards)
    send = [np.split(x, P, axis=1) for x in shards]                      # send[s][j]: [T/P, H/P, d]
    recv = all_to_all(send)                                              # recv[j][s]: 来自 rank s (即第 s 段序列)
    return [np.concatenate(row, axis=0) for row in recv]                 # 按 s 顺序沿序列维拼 → [T, H/P, d]


def head_to_seq(shards):
    """逆变换: P × [T, H/P, d] → P × [T/P, H, d]。"""
    P = len(shards)
    send = [np.split(x, P, axis=0) for x in shards]                      # send[j][s]: 第 s 段序列, 第 j 组头
    recv = all_to_all(send)                                              # recv[s][j]
    return [np.concatenate(row, axis=1) for row in recv]                 # 按 j 顺序沿头维拼 → [T/P, H, d]


def ulysses_attention(q, k, v, P: int):
    H = q.shape[1]
    assert H % P == 0, f"Ulysses 要求头数能被并行度整除 (H={H}, P={P})"
    qs, ks, vs = (seq_to_head(np.split(a, P, axis=0)) for a in (q, k, v))     # 3 次 all-to-all
    outs = [causal_mha(qs[r], ks[r], vs[r]) for r in range(P)]           # 每卡: 完整序列 × H/P 个头, 普通注意力
    return np.concatenate(head_to_seq(outs), axis=0)                     # 第 4 次 all-to-all, 再拼回 [T, H, d] 供比对


def main() -> None:
    banner("M16 - Ulysses Sequence Parallel")

    rs = make_rng(16)
    T, H, d = 32, 8, 8
    q, k, v = (rs.randn(T, H, d) for _ in range(3))
    base = causal_mha(q, k, v)

    print(f"\n  T={T}, H={H}, d_head={d}; 每 rank 发送字节:")
    print(f"  {'P':>4}{'Ulysses':>12}{'公式':>10}{'Ring (m12)':>14}{'公式':>10}{'max|Δ| Ulysses':>18}")
    rows = {}
    for P in (2, 4, 8):
        comm.reset()
        out = ulysses_attention(q, k, v, P)
        uly = comm.total
        comm.reset()
        ring = np.stack([ring_attention(q[:, h], k[:, h], v[:, h], P, zigzag=True)[0] for h in range(H)], axis=1)
        ring_bytes = comm.total
        shard_bytes = T // P * H * d * 8                                  # 一个 rank 的 Q (或 K, 或 V) 的字节数
        f_uly, f_ring = 4 * (P - 1) / P * shard_bytes, (P - 1) * 2 * shard_bytes
        rows[P] = (uly, ring_bytes)
        print(f"  {P:>4}{uly:>12.0f}{f_uly:>10.0f}{ring_bytes:>14.0f}{f_ring:>10.0f}{max_abs_diff(base, out):>18.1e}")
        assert max_abs_diff(base, out) < 1e-12, "只是换了切法, 数学上就是同一个注意力"
        assert max_abs_diff(base, ring) < 1e-12
        assert uly == f_uly and ring_bytes == f_ring

    assert rows[8][0] < rows[4][0] < rows[2][0], "Ulysses: P 越大每卡通信越少"
    assert rows[8][1] > rows[2][1], "Ring: 每卡通信随 P 不降反略升"
    print()
    kv("P=8 时 Ring / Ulysses 通信量", f"{rows[8][1] / rows[8][0]:.1f}x")

    try:
        ulysses_attention(q, k, v, P=16)                                  # 8 个头切不成 16 份
        raise SystemExit("应当报错")
    except AssertionError as e:
        kv("P=16 > H=8", f"AssertionError: {e}")

    print("\n  OK: Ulysses == 完整注意力; 通信随 P 下降但 P ≤ 头数。实战常混用: 机内 Ulysses × 机间 Ring (USP / 2D-CP)。")


if __name__ == "__main__":
    main()
