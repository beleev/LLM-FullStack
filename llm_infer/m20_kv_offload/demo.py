"""
m20 demo — 分层 KV 卸载 (GPU → CPU → 磁盘), 所有时间来自代价模型而非实测

    [1] 多轮对话负载 (8 用户轮流发言, GPU 层装不下 → 抖动): GPU-only / +CPU / +CPU+disk 的逐层命中率与 TTFT
    [2] 交叉点: 命中前缀多短时, 从磁盘加载不如重算
    [3] 慢链路: "命中就加载" 反而拖慢 TTFT; 带 "加载 vs 重算" 判断的策略永远不差于全量重算
"""
from __future__ import annotations

from llm_infer.core.utils import banner, kv
from llm_infer.m20_kv_offload.tiered_cache import Tier, CostModel, TieredKVCache, chat_workload

COST = CostModel(block_size=16, kv_bytes_per_token=131072, prefill_tok_per_s=8000.0)
GPU = Tier("GPU", capacity_blocks=64, bandwidth_GBps=float("inf"), latency_ms=0.0)
CPU = Tier("CPU", capacity_blocks=192, bandwidth_GBps=25.0, latency_ms=0.2)      # PCIe4 x16
DISK = Tier("disk", capacity_blocks=4096, bandwidth_GBps=3.0, latency_ms=2.0)    # NVMe
SLOW = Tier("remote", capacity_blocks=4096, bandwidth_GBps=0.5, latency_ms=20.0)  # 跨机房 / 机械盘


def run(tiers, reqs, **kw):
    cache = TieredKVCache(tiers, COST, **kw)
    ttfts = []
    for _, prompt, with_reply in reqs:
        ttfts.append(cache.serve(prompt))
        cache.store(with_reply)                  # decode 产生的回复 KV 也登记进 cache
    s = cache.stats
    assert sum(s.hit) + s.miss == s.total        # token 守恒: 每个 prompt token 要么命中某层, 要么 miss
    assert all(len(t) <= cfg.capacity_blocks for t, cfg in zip(cache.tiers, tiers))
    return s, ttfts


def main():
    banner("M20 - Hierarchical KV offload (GPU→CPU→disk)")
    print("  注意: 以下时间全部来自代价模型 (cost model), 不是实测。模型参数:")
    kv("block_size / KV bytes per token", f"{COST.block_size} / {COST.kv_bytes_per_token:,} (LLaMA-3-8B fp16)")
    kv("recompute (prefill) speed", f"{COST.prefill_tok_per_s:.0f} tok/s → {COST.recompute_ms(COST.block_size):.2f} ms/block")
    for t in (GPU, CPU, DISK, SLOW):
        kv(f"tier {t.name}", f"cap={t.capacity_blocks} blocks, bw={t.bandwidth_GBps} GB/s, "
                              f"lat={t.latency_ms} ms → {COST.load_ms(t, 1) - t.latency_ms:.3f} ms/block")

    reqs = chat_workload(n_users=8, n_turns=6)
    total_blocks = sum(len(r[2]) for r in reqs[-8:]) // COST.block_size
    print(f"\n[1] 多轮对话: 8 users × 6 turns, 最终历史共 ≈{total_blocks} blocks (GPU 只放得下 {GPU.capacity_blocks})")
    print(f"  {'config':<16}{'hit GPU':>9}{'hit CPU':>9}{'hit disk':>9}{'miss':>8}{'mean TTFT(模拟)':>18}")
    results = {}
    for name, tiers in (("GPU only", [GPU]), ("GPU+CPU", [GPU, CPU]), ("GPU+CPU+disk", [GPU, CPU, DISK])):
        s, _ = run(tiers, reqs)
        rates = [h / s.total for h in s.hit] + [0.0] * (3 - len(s.hit))
        results[name] = s.ttft_ms / s.n_req
        print(f"  {name:<16}{rates[0]:>9.1%}{rates[1]:>9.1%}{rates[2]:>9.1%}{s.miss / s.total:>8.1%}{results[name]:>15.2f} ms")
    no_cache = sum(COST.recompute_ms(len(r[1])) for r in reqs) / len(reqs)
    kv("no cache (全量重算)", f"{no_cache:.2f} ms")
    assert results["GPU+CPU+disk"] < results["GPU+CPU"] < results["GPU only"] < no_cache

    print("\n[2] 交叉点: 从 disk 加载 n 个命中 block vs 重算它们")
    per_blk_load = COST.load_ms(DISK, 1) - DISK.latency_ms
    n_star = DISK.latency_ms / (COST.recompute_ms(COST.block_size) - per_blk_load)
    for n in (1, 2, 4, 8, 32):
        l, r = COST.load_ms(DISK, n), COST.recompute_ms(n * COST.block_size)
        print(f"  n={n:<3} load={l:6.2f} ms  recompute={r:6.2f} ms  → {'load' if l <= r else 'recompute'}")
        assert (l <= r) == (n >= n_star)
    kv("n* = latency / (recompute − load per block)", f"{n_star:.2f} blocks")

    print("\n[3] 慢链路 (remote 0.5 GB/s: 每 block 加载比重算还慢)")
    base, _ = run([GPU, CPU], reqs)
    naive, _ = run([GPU, CPU, SLOW], reqs, always_load=True)
    smart, smart_ttfts = run([GPU, CPU, SLOW], reqs)
    kv("GPU+CPU (无慢层)", f"{base.ttft_ms / base.n_req:.2f} ms")
    kv("+remote, 命中就加载", f"{naive.ttft_ms / naive.n_req:.2f} ms   ← 加了一层缓存反而更慢")
    kv("+remote, 加载/重算取小", f"{smart.ttft_ms / smart.n_req:.2f} ms  (命中但选择重算 {smart.recomputed_hit} tokens)")
    assert naive.ttft_ms > base.ttft_ms                       # 盲目加载有害
    assert smart.ttft_ms <= base.ttft_ms + 1e-9 and smart.recomputed_hit > 0
    for (_, prompt, _), t in zip(reqs, smart_ttfts):          # 逐请求: 永不差于全量重算
        assert t <= COST.recompute_ms(len(prompt)) + 1e-9

    print("\n  all asserts passed")


if __name__ == "__main__":
    main()
