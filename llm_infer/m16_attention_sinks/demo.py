"""
m16 demo — Attention Sinks / StreamingLLM: 无限流下的有界 KV cache

是什么: cache = [开头 S 个 sink token] + [最近 W 个 token], 位置按 cache 槽位重编号。
瓶颈  : 显存 O(T) → O(S+W); 且位置永不超过 max_seq_len, 流可以无限长。
为什么要留开头: softmax 权重和必须为 1, 训练后的模型把"无处可去"的注意力倒在开头几个 token 上
        (attention sink); 逐出它们 = softmax 分母丢掉最大项, 剩余权重被整体放大, 输出畸变。
本 demo:
    [A] 真实 TinyLM + SinkCache (sink_cache.py): 1024 token 的流, 完整 / 纯窗口 / sink+窗口 三种策略的
        PPL 与 KL。**TinyLM 是随机权重、没训练过, sink 现象不会出现** —— 如实打印, 只断言机制性质:
        显存有界、流长 ≤ S+W 时与完整 cache 逐位相等、位置重编号后用 64 行的 RoPE 表跑 1024 步。
    [B] 合成注意力分数 + **人工植入**的 sink: 演示 softmax 分母论证 (真实模型里这个 sink 是训练出来的)。
盯住  : sink_cache.py 里的 `positions = arange(L)`; 本文件 [B] 里 `ks[0] = u * 16`。
对应  : StreamingLLM / HF SinkCache / TensorRT-LLM sink_token_length。

运行: python -m llm_infer.m16_attention_sinks.demo
"""
from __future__ import annotations

import numpy as np

from llm_infer.core import ModelConfig, TinyLM, dense_attention, softmax
from llm_infer.core.tiny_model import apply_rope, init_weights
from llm_infer.core.utils import banner, kv
from llm_infer.m16_attention_sinks.sink_cache import SinkCache, stream_step

T, N_SINK, WINDOW = 1024, 4, 60          # 流长 = 16 × cache 预算
BUDGET = N_SINK + WINDOW


def log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def run_tinylm(seed: int, check_full: bool):
    """一个权重 seed 上跑三种策略 → {策略: (PPL, KL→full, mean|Δlogit|)}, 只统计 t ≥ BUDGET (已开始逐出) 的步。"""
    cfg = ModelConfig(max_seq_len=T)
    lm = TinyLM(cfg, init_weights(cfg, seed))                       # 完整 cache 需要 T 行 RoPE 表
    lm_small = TinyLM(ModelConfig(max_seq_len=BUDGET), lm.w)        # 同一份权重, RoPE 表只有 64 行

    # 基线: core 的标准 KV cache (K 存 post-RoPE, 绝对位置)。流由基线模型自己采样 → 它的 PPL 最低
    rng = np.random.RandomState(0)
    stream, full, cache = [1], [], None
    for _ in range(T):
        logits, cache = lm.decode_step(stream[-1], cache) if cache else lm.forward(stream)
        full.append(logits.reshape(-1))
        stream.append(int(rng.choice(cfg.vocab_size, p=softmax(full[-1]))))
    full, targets = np.array(full), np.array(stream[1:])            # (T, V), (T,)
    lp_full = log_softmax(full)

    def evaluate(out: np.ndarray):
        lp = log_softmax(out)
        nll = -lp[np.arange(T), targets]                            # teacher-forced: 真实下一个 token 的 NLL
        kl = (np.exp(lp_full) * (lp_full - lp)).sum(-1)             # KL(full ‖ policy), (T,)
        late = slice(BUDGET, T)
        return float(np.exp(nll[late].mean())), float(kl[late].mean()), float(np.abs(out - full)[late].mean())

    results = {"完整 cache": evaluate(full)}
    for name, n_sink, window in (("纯窗口", 0, BUDGET), ("sink+窗口", N_SINK, WINDOW)):   # 同预算
        c, out, max_len = SinkCache(cfg.n_layer, cfg.d_model, n_sink, window), [], 0
        for tok in stream[:-1]:
            out.append(stream_step(lm_small, c, tok))
            max_len = max(max_len, len(c))
        out = np.array(out)
        assert max_len == BUDGET, "显存有界: cache 条目永不超过 n_sink + window"
        assert np.abs(out[:BUDGET] - full[:BUDGET]).max() < 1e-4, "还没逐出时应与完整 cache 一致"
        results[name] = evaluate(out)

    mass = {}
    if check_full:   # window=∞ 的 SinkCache: 槽位 == 绝对位置 → 必须复现 core 的标准 cache (验证"存未旋转 K + 每步重转")
        c, out = SinkCache(cfg.n_layer, cfg.d_model, 0, T), []
        for t, tok in enumerate(stream[:-1]):
            w = [] if t + 1 in (64, 256, 1024) else None
            out.append(stream_step(lm, c, tok, w))
            if w:
                mass[t + 1] = [float(x[:N_SINK].sum()) for x in w]  # 每层: 开头 4 个 token 吃掉的注意力
        diff = float(np.abs(np.array(out) - full).max())
        assert diff < 1e-4
        kv("SinkCache(window=∞) vs core 标准 cache, max|Δlogit|", f"{diff:.1e}")

        k = np.ones((1, cfg.d_model), dtype=np.float32)
        try:
            apply_rope(k, lm_small.cos, lm_small.sin, positions=np.array([T - 1]))
            raise AssertionError("绝对位置本应越界")
        except IndexError:
            kv(f"RoPE 表仅 {BUDGET} 行: 绝对位置 {T - 1}", "IndexError (跑不了); 槽位重编号 → 上面两种有界策略全程正常")
    return results, mass


def main() -> None:
    banner("M16 - Attention Sinks / StreamingLLM")

    # ================= [A] 真实 TinyLM + SinkCache =================
    print(f"\n[A] TinyLM (随机权重) 流长 T={T}, cache 预算 {BUDGET} = sink {N_SINK} + 窗口 {WINDOW}; 纯窗口同预算 W={BUDGET}")
    wins = 0
    for i, seed in enumerate((42, 1, 2)):
        results, mass = run_tinylm(seed, check_full=(i == 0))
        if mass:
            print("\n  完整 cache 下开头 4 个 token 的注意力占比 (逐层) vs 均匀注意力 4/T:")
            for t, m in mass.items():
                print(f"    T={t:>4}: " + "  ".join(f"{x:.4f}" for x in m) + f"   | 均匀 = {4 / t:.4f}")
            print(f"\n  {'权重 seed':>8} | {'策略':<10} | {'PPL':>8} | {'KL→完整':>8} | {'mean|Δlogit|':>12} | cache 条目")
        for name, (ppl, kl, drift) in results.items():
            n = T if name == "完整 cache" else BUDGET
            print(f"  {seed:>9} | {name:<10} | {ppl:>8.2f} | {kl:>8.4f} | {drift:>12.4f} | {n}")
        wins += results["sink+窗口"][1] < results["纯窗口"][1]
    print(f"\n  sink+窗口 的 KL 优于纯窗口: {wins}/3 个权重 seed  ← 如实报告, 不做断言")
    print("  诚实结论: 随机权重的 TinyLM 没有 attention sink (开头 token 的注意力 ≈ 均匀水平), 注意力也没有局部性,")
    print("  所以逐出任何 token 都伤, 留不留开头 4 个没有稳定差别。sink 是**训练**出来的统计特征, 见 [B]。")
    print("  已断言 (机制性质): cache ≤ 64 条; 前 64 步与完整 cache 一致; window=∞ 复现标准 cache; 重编号后不越界。")

    # ================= [B] 合成分数 + 人工植入的 sink =================
    print("\n[B] softmax 分母论证 (合成 K/V, sink 是**人工植入**的: ks[0] 与平均 query 方向对齐)")
    rs = np.random.RandomState(0)
    d, t_max, window, n_sink = 32, 256, 32, 4
    u = rs.randn(d)
    u /= np.linalg.norm(u)
    ks, vs = rs.randn(t_max, d), rs.randn(t_max, d)                 # (T, d)
    ks[0] = u * 16.0                                                # 植入: 第 0 个 key 与所有 query 强对齐

    def new_query() -> np.ndarray:
        return (u * 3.0 + rs.randn(d) * 0.6)[None]                  # (1, d) 稳定朝向 sink + 噪声

    q_probe = new_query()
    for t in (32, 64, 128, 256):
        w = dense_attention(q_probe, ks[:t], np.eye(t))[0]          # V=I → 输出即注意力权重 (t,)
        kv(f"T={t:>3} 开头 {n_sink} 个位置的注意力占比", f"{w[:n_sink].sum():5.1%}  {'#' * int(w[:n_sink].sum() * 40)}")

    print(f"\n  最新 token 的注意力输出相对误差 (vs 完整 cache, 16 个 query 平均)")
    print(f"  {'T':>5} | {'sink+窗口 (4+32)':>18} | {'纯窗口 (36)':>12}")
    for t in (64, 128, 192, 256):
        keep_sink = np.r_[0:n_sink, t - window:t]
        keep_win = np.r_[t - window - n_sink:t]                     # 同预算
        errs = []
        for _ in range(16):
            q = new_query()
            ref = dense_attention(q, ks[:t], vs[:t])                # (1, d)
            errs.append([np.linalg.norm(dense_attention(q, ks[keep], vs[keep]) - ref) / np.linalg.norm(ref)
                         for keep in (keep_sink, keep_win)])
        err_sink, err_win = np.mean(errs, axis=0)
        print(f"  {t:>5} | {err_sink:>17.2%} | {err_win:>11.2%}")
    assert err_sink < err_win / 3, "有 sink 的注意力分布下, 保留 sink 应显著优于纯滑动窗口"
    print("\n  纯窗口丢掉 sink 后 softmax 分母失去最大项, 剩余权重被迫放大重排 (误差 ≈100%); 多留 4 个 token 降到 4%~19%。")


if __name__ == "__main__":
    main()
