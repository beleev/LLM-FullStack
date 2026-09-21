"""Run all llm_infer demos in learning-path order (每个 demo 都以 assert 收尾, 任何一个失败即非 0 退出)。

    python -m llm_infer.run_all
"""
from __future__ import annotations

import importlib
import time

DEMOS = [
    "m01_kv_cache", "m02_paged_attention", "m03_continuous_batching", "m04_prefix_cache",
    "m05_radix_cache", "m06_chunked_prefill", "m07_speculative_decoding", "m08_quantization",
    "m09_tensor_parallel", "m10_sampling", "m11_flash_attention", "m12_cuda_graph",
    "m13_lora_serving", "m14_structured_output", "m15_pd_disaggregation", "m16_attention_sinks",
    "m17_eagle_speculative", "m18_kv_attention_variants", "m19_tree_speculation", "m20_kv_offload",
    "m21_moe_serving", "m22_sparse_attention", "full_engine",
]


def main() -> None:
    times = []
    for name in DEMOS:
        t0 = time.perf_counter()
        importlib.import_module(f"llm_infer.{name}.demo").main()
        times.append((name, time.perf_counter() - t0))
    print("\n" + "=" * 70)
    for name, dt in times:
        print(f"  ✓ {name:<28} {dt:6.2f}s")
    print(f"  {len(times)} demos passed, total {sum(dt for _, dt in times):.1f}s")


if __name__ == "__main__":
    main()
