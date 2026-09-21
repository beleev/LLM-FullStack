"""
M07 — 激活重算 (Activation / Gradient Checkpointing)

是什么: 前向只保存每段 (segment) 的入口激活; 反向走到哪一段, 就从入口重跑那段前向, 用完即弃。
解决的瓶颈: 显存。L 层全存要 O(L) 份激活; 分成长度 k 的段后:
    峰值 ≈ L/k (段边界, 常驻) + k (当前段重算出来的, 瞬时)   →   k = √L 时最小, O(√L)
代价: 前向几乎多算一遍 → 计算 +25%~33% (fwd : bwd ≈ 1 : 2, 总量 3 → 接近 4), 梯度逐位不变。
读代码盯住: `live` 集合 —— 每保存/释放一个激活都经过它, `peak` 是它的历史最大值 (含段内瞬时激活)。
"""
from __future__ import annotations

import numpy as np

from llm_train.core import banner, kv, max_abs_diff


class Tracker:
    """激活账本: 同时数 '此刻活着几份激活' 和 '前向层调用了几次'。"""

    def __init__(self) -> None:
        self.live, self.peak, self.fwd_calls = set(), 0, 0

    def keep(self, i: int) -> None:
        self.live.add(i)
        self.peak = max(self.peak, len(self.live))

    def free(self, i: int) -> None:
        self.live.discard(i)


def layer(h, w, tr: Tracker):
    tr.fwd_calls += 1
    return np.tanh(h @ w)                                  # [B, H] -> [B, H]


def backward_layer(d_out, h_in, h_out, w):
    d_pre = d_out * (1 - h_out * h_out)                    # 需要 h_out (tanh') 和 h_in (dW) → 两者都得活着
    return h_in.T @ d_pre, d_pre @ w.T


def run(x, y, weights, segment: int):
    """segment=1 即 '每层都存' 的普通反向; segment=k 即每 k 层设一个 checkpoint。"""
    L, tr = len(weights), Tracker()
    acts = {0: x}                                          # acts[i] = 第 i 层的输入 (= 第 i-1 层的输出)
    tr.keep(0)

    # ---- 前向: 只在段边界保存 ----
    h = x
    for i in range(L):
        h = layer(h, weights[i], tr)
        if (i + 1) % segment == 0 or i + 1 == L:
            acts[i + 1] = h
            tr.keep(i + 1)

    d = 2.0 * (h - y) / y.size
    grads = [None] * L

    # ---- 反向: 逐段 "重算 → 反向 → 释放" ----
    for start in reversed(range(0, L, segment)):
        end = min(start + segment, L)
        h = acts[start]
        for i in range(start, end - 1):                    # 重算段内部激活; 段出口 acts[end] 本来就存着
            h = layer(h, weights[i], tr)
            acts[i + 1] = h
            tr.keep(i + 1)                                 # 瞬时激活也占显存, 必须计入峰值
        for i in reversed(range(start, end)):
            grads[i], d = backward_layer(d, acts[i], acts[i + 1], weights[i])
            tr.free(i + 1)                                 # 第 i 层反向完, 它的输出就没用了
            del acts[i + 1]
    return grads, tr


def main() -> None:
    banner("M07 - Activation Checkpointing")

    rs = np.random.RandomState(7)
    L, H = 16, 32
    x = rs.randn(4, H).astype(np.float32)
    y = rs.randn(4, H).astype(np.float32)
    weights = [(rs.randn(H, H) / np.sqrt(H)).astype(np.float32) for _ in range(L)]

    base_grads, base = run(x, y, weights, segment=1)

    print(f"  L = {L} 层; 每份激活 {x.nbytes} B\n")
    print(f"  {'segment':>8}{'峰值激活份数':>10}{'前向调用':>8}{'额外前向':>8}   max|Δgrad|")
    peaks = {}
    for k in (1, 2, 4, 8, 16):
        grads, tr = run(x, y, weights, segment=k)
        diff = max(max_abs_diff(a, b) for a, b in zip(grads, base_grads))
        peaks[k] = tr.peak
        note = "  ← 全存基线" if k == 1 else "  ← √L" if k * k == L else ""
        print(f"  {k:>8}{tr.peak:>14}{tr.fwd_calls:>12}{tr.fwd_calls - L:>11}   {diff:.1e}{note}")
        assert diff == 0.0, "重算的是同一串浮点运算, 梯度必须逐位相同"
        assert tr.peak == L // k + k, "峰值 = 段边界数 (L/k + 输入) + 段内瞬时 (k - 1)"
        assert tr.fwd_calls == L + (L - L // k), "每段除出口外的层都重算一次"

    print()
    kv("峰值显存 (k=1 → k=4)", f"{peaks[1] * x.nbytes} → {peaks[4] * x.nbytes} B  ({peaks[1] / peaks[4]:.1f}x)")
    assert min(peaks, key=peaks.get) == 4, "k = √L 最省"
    assert base.peak == L + 1

    print("\n  OK: 梯度与全存基线逐位相同; 峰值激活 17 → 8 份, 代价是 12 次额外前向 (总计算 48 → 60 个前向当量, +25%)。")


if __name__ == "__main__":
    main()
