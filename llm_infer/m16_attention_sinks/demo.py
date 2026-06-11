"""
m16 demo — Attention Sinks / StreamingLLM: 无限流式输入下的有界 KV cache

问题:
    KV cache 随上下文线性增长 (m01), 流式场景 (长对话/实时字幕) 迟早爆显存。
    最朴素的办法是只保留最近 W 个 token 的滑动窗口 —— 但实测一滑就崩:
    开头几个 token 被逐出后, 困惑度瞬间飙升。

StreamingLLM (Xiao et al., 2023) 的发现 — attention sink:
    softmax 要求注意力权重和为 1, 当一个 query 与谁都不相关时, 多余的
    注意力必须有处可去。训练后的模型习惯把这部分"垃圾注意力"倾倒在
    **开头几个 token** 上 (它们对所有后续位置可见, 是天然的锚点)。
    开头 token 一旦被逐出, softmax 分母失去了最大的一项, 剩余权重被
    强行放大重排 —— 输出分布整体畸变。

解法 (本 demo 验证):
    cache = [开头 S 个 sink token] + [最近 W 个 token]
    显存 O(S+W) 有界, 输出与完整 cache 几乎一致。
    (工程细节: 窗口滑动后, RoPE 位置按 cache 内的相对位置重新编号)

后续演化:
    GPT-OSS (2025) 把 sink 做成每个 head 一个**可学习的 logit**, 参与
    softmax 分母但不输出 value —— 模型自己学会"把多余注意力丢进下水道"。

运行:
    python -m llm_infer.m16_attention_sinks.demo

说明: 真实模型的 sink 现象来自预训练; 本 demo 用"给第一个 token 的 key
      加上与平均 query 对齐的分量"来复现这一统计特征, 机制完全相同。
"""
from __future__ import annotations

import numpy as np

from llm_infer.core.utils import banner, kv, softmax


def attention_out(q: np.ndarray, ks: np.ndarray, vs: np.ndarray) -> np.ndarray:
    """单 query 对一组 KV 的注意力输出: softmax(q·K^T/√d)·V。"""
    scores = ks @ q / np.sqrt(len(q))
    w = softmax(scores)
    return w @ vs


def sink_mass(q: np.ndarray, ks: np.ndarray, n_sink: int) -> float:
    """完整 cache 下, 开头 n_sink 个位置吸收的注意力权重之和。"""
    w = softmax(ks @ q / np.sqrt(len(q)))
    return float(w[:n_sink].sum())


def main() -> None:
    banner("M16 - Attention Sinks / StreamingLLM")

    rs = np.random.RandomState(0)
    d, t_max, window, n_sink = 32, 256, 32, 4

    # 平均 query 方向 u; 第一个 token 的 key 与 u 强对齐 → 它吸走大部分注意力。
    # (复现真实模型里 "开头 token 是注意力下水道" 的统计特征: StreamingLLM 实测
    #  许多层 >70% 的注意力质量落在开头几个 token 上)
    u = rs.randn(d)
    u /= np.linalg.norm(u)
    ks = rs.randn(t_max, d)
    vs = rs.randn(t_max, d)
    ks[0] = u * 16.0

    # 所有 query 都带一个稳定朝向 sink 的分量 (真实模型中这是训练出来的习惯)
    def new_query() -> np.ndarray:
        return u * 3.0 + rs.randn(d) * 0.6

    # ---- 1) sink 现象: 第一个 token 吸收的注意力占比 ----
    q_probe = new_query()
    print("\n[1] attention sink 现象 (完整 cache 下开头 4 个位置的注意力占比)")
    for t in (32, 64, 128, 256):
        mass = sink_mass(q_probe, ks[:t], n_sink)
        bar = "#" * int(mass * 40)
        kv(f"上下文 T={t:>3}", f"{mass:5.1%}  {bar}")

    # ---- 2) 两种有界 cache 策略的输出误差 (相对完整 cache, 16 个 query 平均) ----
    def mean_errors(t: int, n_queries: int = 16) -> tuple[float, float]:
        keep_a = list(range(n_sink)) + list(range(t - window, t))   # sink + 窗口
        keep_b = list(range(t - window - n_sink, t))                # 纯窗口 (同预算)
        errs_a, errs_b = [], []
        for _ in range(n_queries):
            q = new_query()
            full = attention_out(q, ks[:t], vs[:t])
            scale = np.linalg.norm(full)
            errs_a.append(np.linalg.norm(attention_out(q, ks[keep_a], vs[keep_a]) - full) / scale)
            errs_b.append(np.linalg.norm(attention_out(q, ks[keep_b], vs[keep_b]) - full) / scale)
        return float(np.mean(errs_a)), float(np.mean(errs_b))

    print("\n[2] 最新 token 的注意力输出平均误差 (vs 完整 cache, 16 个 query)")
    print(f"  {'T':>5} | {'窗口+sink (S=4+W=32)':>22} | {'纯窗口 (W=36)':>16} | cache 条目")
    last_errs = None
    for t in (64, 128, 192, 256):
        err_a, err_b = mean_errors(t)
        last_errs = (err_a, err_b)
        print(f"  {t:>5} | {err_a:>21.2%} | {err_b:>15.2%} | "
              f"{n_sink + window} (有界) vs {t} (线性)")

    # ---- 3) 结论性断言 ----
    err_sink, err_win = last_errs
    assert err_sink < err_win / 3, "保留 sink 应当显著优于纯滑动窗口"

    print("\n[3] 显存对比")
    kv("完整 cache", f"O(T) — T={t_max} 时 {t_max} 条, 持续增长")
    kv("sink + window", f"O(S+W) — 恒定 {n_sink + window} 条, 与流长度无关")

    print("\n结论: 多花 4 个 token 的显存保住 softmax 的'下水道', 误差缩小约一个量级;")
    print("      纯滑动窗口丢掉 sink 后, 剩余注意力被迫重新归一化, 输出整体畸变。")


if __name__ == "__main__":
    main()
