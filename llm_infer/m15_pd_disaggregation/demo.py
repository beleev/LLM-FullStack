"""
m15 demo — Prefill/Decode 分离: 真的在两个 "节点" (各自一份 TinyLM 权重) 之间搬 KV。

[1] P 节点 prefill → KV 序列化成 bytes → D 节点反序列化后 decode; 输出与单机 generate_greedy 逐 token 相同
[2] KV 字节数: 公式 2·L·T·D·itemsize == 实际 nbytes == 线上 payload 长度
[3] 传输时间代价模型 (Gbps → bytes/s 要除以 8): LLaMA-7B 4k 上下文在几种链路上的耗时
[4] 为什么要分离: 实测 TinyLM 一次长 prefill ≈ 多少个 decode step — 混跑时这就是 decode 请求被卡住的时间

运行: python -m llm_infer.m15_pd_disaggregation.demo
"""
from __future__ import annotations
import numpy as np

from llm_infer.core import TinyLM, ModelConfig, Timer
from llm_infer.core.utils import banner, kv
from llm_infer.m15_pd_disaggregation.pd import PrefillNode, DecodeNode, KVLink, kv_nbytes_formula, deserialize_kv

# 链路标称带宽, 单位 Gbps (bit!)。NVLink 厂商按 GB/s 标: H100 NVLink4 单向 450 GB/s = 3600 Gbps
LINKS_GBPS = {"100GbE / RoCE": 100, "IB NDR 400G": 400, "8×400G (DGX H100 节点间)": 3200, "NVLink4 单向 (节点内)": 3600}


def main():
    banner("M15 - Prefill/Decode Disaggregation")
    cfg = ModelConfig(d_model=64, d_mlp=128, n_layer=4, vocab_size=128, max_seq_len=512)
    prompt = np.random.RandomState(0).randint(0, cfg.vocab_size, size=48)
    max_new = 16

    print("\n[1] 两个节点, 各自一份权重; 中间只传 bytes")
    p_node, d_node = PrefillNode(cfg), DecodeNode(cfg)
    assert not np.shares_memory(p_node.lm.w.tok_emb, d_node.lm.w.tok_emb)   # 真的是两份权重, 不是同一个对象
    link = KVLink(link_gbps=400, latency_ms=0.05)

    first_token, wire_kv = p_node.run(prompt)               # ← TTFT 在这里结束: 首 token 由 P 节点给出
    payload, xfer_ms = link.send(wire_kv[2])
    assert isinstance(payload, bytes)
    gen = d_node.run(first_token, (wire_kv[0], wire_kv[1], payload), max_new)

    baseline = TinyLM(cfg).generate_greedy(prompt, max_new=max_new)
    kv("单机 baseline 生成", baseline[len(prompt):])
    kv("P→D 分离 生成", gen)
    assert list(prompt) + gen == baseline, "分离后输出必须与单机逐 token 相同"
    print("  ✓ 逐 token 相同 (首 token 来自 P 节点, 其余 15 个来自 D 节点)")

    print("\n[2] KV 字节数")
    shape, dtype, _ = wire_kv
    L, _, T, D = shape
    itemsize = np.dtype(dtype).itemsize
    formula = kv_nbytes_formula(L, T, D, itemsize)
    actual = sum(K.nbytes + V.nbytes for K, V in deserialize_kv(*wire_kv))
    kv(f"公式 2·L·T·D·itemsize = 2·{L}·{T}·{D}·{itemsize}", f"{formula} B")
    kv("实际 nbytes / 线上 payload", f"{actual} B / {len(payload)} B")
    assert formula == actual == len(payload) == link.bytes_sent
    kv("该 KV 过 400 Gbps 链路 (模型值)", f"{xfer_ms:.4f} ms  (其中固定延迟 0.05 ms)")

    print("\n[3] 代价模型: LLaMA-7B (32 层, D=4096, fp16), T=4096 的 KV 传输时间  [公式计算, 非实测]")
    big = kv_nbytes_formula(32, 4096, 4096, 2)
    kv("KV 大小", f"{big / 1e9:.3f} GB ({big / 2**30:.2f} GiB)")
    print(f"  {'链路':<28} {'Gbps':>6} {'GB/s':>7} {'传输 ms':>9}")
    for name, gbps in LINKS_GBPS.items():
        ln = KVLink(gbps)
        print(f"  {name:<28} {gbps:>6} {ln.bytes_per_s / 1e9:>7.1f} {ln.transfer_ms(big):>9.1f}")
    ms_400 = KVLink(400).transfer_ms(big)
    wrong_ms = big / (400 * 1e9) * 1e3                       # 常见错误: 把 Gbps 当 GB/s, 少除了 8
    kv("400 Gbps: 正确 / 把 Gbps 当 GB/s", f"{ms_400:.1f} ms / {wrong_ms:.1f} ms  (低估 {ms_400 / wrong_ms:.0f}x)")
    assert KVLink(400).bytes_per_s == 50e9 and abs(ms_400 / wrong_ms - 8) < 1e-9
    assert abs(ms_400 - 2147483648 / 50e9 * 1e3) < 1e-9

    print("\n[4] 为什么分离: prefill 与 decode 混跑时的干扰 (TinyLM 实测, 3 次取最快)")
    lm = p_node.lm
    long_prompt = np.random.RandomState(1).randint(0, cfg.vocab_size, size=400)
    _, kv_cache = lm.prefill(prompt)

    def best_ms(fn):
        best = float("inf")
        for _ in range(3):
            with Timer() as t:
                fn()
            best = min(best, t.ms)
        return best
    t_prefill = best_ms(lambda: lm.prefill(long_prompt))
    t_decode = best_ms(lambda: lm.decode_step(first_token, kv_cache))
    kv("400-token prefill", f"{t_prefill:.2f} ms")
    kv("1 个 decode step", f"{t_decode:.3f} ms")
    kv("同卡混跑: 被插队的那一步 ITL", f"{t_decode + t_prefill:.2f} ms  ({(t_decode + t_prefill) / t_decode:.0f}x 抖动)")
    kv("分离后: ITL", f"{t_decode:.3f} ms  (prefill 在别的节点上, 互不影响)")
    assert t_prefill > 5 * t_decode


if __name__ == "__main__":
    main()
