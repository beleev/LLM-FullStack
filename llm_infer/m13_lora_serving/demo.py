"""
m13 demo — Multi-LoRA Serving 接口

3 个用户, 3 套 LoRA, 同一个底模, 一次 forward 同时算完。
对比:
  - naive: 每请求单独 forward 后拼起来
  - batched: 一次 forward 内联 LoRA 应用
两者数值相同, batched 节省底模 matmul 次数。
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np

from llm_infer.core.utils import banner, kv, Timer


@dataclass
class LoRAAdapter:
    """单个 adapter: A (r, D_out)  B (D_in, r), 一般 r << min(D_in, D_out)"""
    A: np.ndarray
    B: np.ndarray
    alpha: float = 16.0

    @property
    def r(self) -> int:
        return self.A.shape[0]

    @property
    def scale(self) -> float:
        return self.alpha / self.r


def make_adapter(D_in: int, D_out: int, r: int, seed: int) -> LoRAAdapter:
    rs = np.random.RandomState(seed)
    return LoRAAdapter(
        A=rs.randn(r, D_out).astype(np.float32) * 0.01,
        B=rs.randn(D_in, r).astype(np.float32) * 0.01,
    )


def naive_forward(
    x_batch: List[np.ndarray],          # list of (B_i, D_in)
    adapter_ids: List[int],
    W: np.ndarray,
    adapters: Dict[int, LoRAAdapter],
) -> List[np.ndarray]:
    """每条请求单独走一遍 base + lora, 串起来。"""
    outs = []
    for x, aid in zip(x_batch, adapter_ids):
        base = x @ W
        ad = adapters[aid]
        lora = (x @ ad.B) @ ad.A
        outs.append(base + ad.scale * lora)
    return outs


def batched_forward(
    x_batch: List[np.ndarray],
    adapter_ids: List[int],
    W: np.ndarray,
    adapters: Dict[int, LoRAAdapter],
) -> List[np.ndarray]:
    """优化: 把 base matmul 合并 (concat → 一次 W); LoRA 仍按 adapter 分。

    真实 SGMV 把 LoRA 也按 adapter 分桶后, 用 segmented gemm 一次跑完。
    本演示用循环展示语义, 关键是: base 部分一次 matmul 而不是 N 次。
    """
    # ---- 1) base: concat 一次 matmul ---------------------------- #
    sizes = [x.shape[0] for x in x_batch]
    x_cat = np.concatenate(x_batch, axis=0)
    base_cat = x_cat @ W                                  # (sum_B, D_out)

    # 拆回每条请求
    base_split = np.split(base_cat, np.cumsum(sizes)[:-1], axis=0)

    # ---- 2) LoRA: 按 adapter 分桶, 每桶内一次 matmul ----------- #
    bucket: Dict[int, List[int]] = {}
    for i, aid in enumerate(adapter_ids):
        bucket.setdefault(aid, []).append(i)

    outs = [None] * len(x_batch)
    for aid, idxs in bucket.items():
        ad = adapters[aid]
        x_bucket = np.concatenate([x_batch[i] for i in idxs], axis=0)
        lora_bucket = (x_bucket @ ad.B) @ ad.A * ad.scale
        bucket_sizes = [x_batch[i].shape[0] for i in idxs]
        lora_split = np.split(lora_bucket, np.cumsum(bucket_sizes)[:-1], axis=0)
        for i, l in zip(idxs, lora_split):
            outs[i] = base_split[i] + l
    return outs


def main():
    banner("M13 - Multi-LoRA Serving")

    rs = np.random.RandomState(0)
    D_in, D_out, r = 64, 64, 4
    W = rs.randn(D_in, D_out).astype(np.float32) * 0.05

    # 3 个 adapter
    adapters = {
        0: make_adapter(D_in, D_out, r, seed=1),
        1: make_adapter(D_in, D_out, r, seed=2),
        2: make_adapter(D_in, D_out, r, seed=3),
    }

    # 5 条请求, 不同长度, 不同 adapter
    x_batch = [
        rs.randn(3, D_in).astype(np.float32),
        rs.randn(5, D_in).astype(np.float32),
        rs.randn(2, D_in).astype(np.float32),
        rs.randn(4, D_in).astype(np.float32),
        rs.randn(1, D_in).astype(np.float32),
    ]
    adapter_ids = [0, 1, 0, 2, 1]

    print(f"\n[1] 5 个请求, adapter_ids = {adapter_ids}")

    # ---- 数值等价 ----------------------------------------------- #
    out_naive = naive_forward(x_batch, adapter_ids, W, adapters)
    out_batched = batched_forward(x_batch, adapter_ids, W, adapters)
    diffs = [np.max(np.abs(a - b)) for a, b in zip(out_naive, out_batched)]
    kv("max diff per req", [f"{d:.2e}" for d in diffs])
    assert max(diffs) < 1e-5

    # ---- 性能对比 ----------------------------------------------- #
    print("\n[2] 性能 (放大请求数模拟真实)")
    big_batch = x_batch * 100
    big_ids = adapter_ids * 100
    with Timer() as ta:
        for _ in range(5):
            naive_forward(big_batch, big_ids, W, adapters)
    with Timer() as tb:
        for _ in range(5):
            batched_forward(big_batch, big_ids, W, adapters)
    kv("naive   (ms)", f"{ta.ms:.1f}")
    kv("batched (ms)", f"{tb.ms:.1f}")
    kv("加速", f"{ta.ms / tb.ms:.2f}x")

    print("\n  ✓ batched 把 N 个 base matmul 合并成 1 个, 显著减少调度开销")
    print("  ✓ 真实 S-LoRA: 一卡服 1000+ adapter, 主要靠 SGMV kernel")


if __name__ == "__main__":
    main()
