"""
m15 demo — Prefill / Decode Disaggregation

两个"节点" (单进程模拟):
    PrefillEngine: 接 prompt → 出最后 logits + KV cache
    KVTransport:   把 KV 从 prefill 节点搬到 decode 节点
    DecodeEngine:  接 (KV, last_token) → 流式 decode 出文本

数值正确性: 与单引擎 prefill+decode 完全一致。
延迟模拟: KV 传输耗时按"显存大小 / 带宽"估算。
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import time
import numpy as np

from llm_infer.core import TinyLM, ModelConfig
from llm_infer.core.utils import banner, kv, Timer


SIM_BANDWIDTH_GBPS = 200.0          # 模拟 NVLink 带宽


# --------------------------------------------------------------------- #
# 三个组件                                                              #
# --------------------------------------------------------------------- #

class PrefillEngine:
    """专门跑 prefill。"""
    def __init__(self, lm: TinyLM):
        self.lm = lm
        self.calls = 0

    def prefill(self, prompt_ids: np.ndarray) -> Tuple[int, list]:
        logits, kv_cache = self.lm.prefill(prompt_ids)
        first_token = int(np.argmax(logits[-1]))
        self.calls += 1
        return first_token, kv_cache


class DecodeEngine:
    """专门跑 decode。"""
    def __init__(self, lm: TinyLM):
        self.lm = lm
        self.calls = 0

    def decode_loop(
        self, first_token: int, kv_cache: list, max_new: int
    ) -> List[int]:
        out = [first_token]
        nxt = first_token
        for _ in range(max_new - 1):
            logits, kv_cache = self.lm.decode_step(nxt, kv_cache)
            self.calls += 1
            nxt = int(np.argmax(logits))
            out.append(nxt)
        return out


@dataclass
class KVTransport:
    """模拟跨节点 KV 传输; 单进程内是 dict 拷贝, 但能算时间。"""
    bandwidth_gbps: float = SIM_BANDWIDTH_GBPS
    bytes_transferred: int = 0
    time_ms: float = 0.0

    def send(self, kv_cache: list) -> list:
        """模拟 RDMA: 拷贝 + 按带宽算时间。"""
        # 算 KV 总大小
        total_bytes = 0
        for K, V in kv_cache:
            total_bytes += K.nbytes + V.nbytes
        # 按带宽估时
        sec = total_bytes / (self.bandwidth_gbps * 1e9)
        time.sleep(min(sec, 0.005))    # 截到 5ms 防 demo 太慢
        self.bytes_transferred += total_bytes
        self.time_ms += sec * 1000
        return [(K.copy(), V.copy()) for K, V in kv_cache]


# --------------------------------------------------------------------- #
# main                                                                  #
# --------------------------------------------------------------------- #

def main():
    banner("M15 - Prefill/Decode Disaggregation")

    cfg = ModelConfig(d_model=64, d_mlp=128, n_layer=4, vocab_size=128, max_seq_len=512)
    lm = TinyLM(cfg)
    prompt = np.array([1, 5, 10, 15, 20, 25, 30, 35], dtype=np.int64)
    max_new = 16

    # ---- 1) baseline: 单引擎 ----------------------------------- #
    print("\n[1] baseline (单引擎)")
    base_out = lm.generate_greedy(prompt, max_new=max_new)
    print(f"  output ids = {base_out}")

    # ---- 2) 分离架构 ------------------------------------------- #
    print("\n[2] disaggregated: PrefillEngine → KVTransport → DecodeEngine")
    pe = PrefillEngine(lm)
    de = DecodeEngine(lm)
    kv_xport = KVTransport(bandwidth_gbps=200.0)

    with Timer() as t_pre:
        first_token, kv_cache = pe.prefill(prompt)
    with Timer() as t_xfer:
        kv_remote = kv_xport.send(kv_cache)
    with Timer() as t_dec:
        gen_ids = de.decode_loop(first_token, kv_remote, max_new=max_new)
    full_out = list(prompt) + gen_ids

    print(f"  prefill cluster:  {t_pre.ms:.2f} ms")
    print(f"  KV transport:     {t_xfer.ms:.2f} ms ({kv_xport.bytes_transferred:,} bytes)")
    print(f"  decode cluster:   {t_dec.ms:.2f} ms")
    print(f"  output ids        = {full_out}")
    print(f"  与 baseline 一致: {full_out == base_out}")

    # ---- 3) 收益分析 ------------------------------------------ #
    print("\n[3] 真实场景收益")
    print(f"  baseline: prefill 与 decode 在同一卡, 长 prompt 阻塞 decode")
    print(f"  分离后:   两类负载并行, GPU 类型可异构 (prefill 用 H100, decode 用 H200)")
    print(f"  关键代价: KV transport (本例 LLaMA-7B 4k context ~2 GB, 200GBps NVLink ~10ms)")
    print(f"  对长输入 / 短输出场景 (RAG, 长 prompt), 分离收益最大")


if __name__ == "__main__":
    main()
