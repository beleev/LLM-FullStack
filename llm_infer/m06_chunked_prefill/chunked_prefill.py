"""
chunked_prefill.py — 分块 prefill + 与 decode 混批 (Sarathi-Serve)

是什么: 把长 prompt 切成 ≤ token 预算的 chunk, 每步 batch = 所有 decode (各 1 token) + 一个 prefill chunk。
解决什么: 延迟抖动。prefill 优先的调度器遇到 2K token 的新请求, 会让所有正在 decode 的用户卡一整个 prefill 的时间
        (TBT 尖刺); 分块后每步耗时被 token 预算封顶, 代价是长请求自己的 TTFT 变长。
关键数字: 每步耗时 ≈ 固定开销 + 每 token 开销 × batch 内 token 数 → 预算 B 决定了 TBT 上限。
盯住: (1) chunked_prefill() 里 kv 逐块变长而结果与整段 prefill 相同 —— 因果 attention 下, 前面 token 的 KV 不依赖后面的 token;
      (2) simulate() 里每步的 n_tokens 与各序列 token 到达时刻 t。
对应真实系统: vLLM `enable_chunked_prefill` + `max_num_batched_tokens`, SGLang `chunked_prefill_size`, TensorRT-LLM in-flight batching。
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from llm_infer.core import TinyLM
from llm_infer.m03_continuous_batching.scheduler import Scheduler, SchedulerConfig


def chunked_prefill(lm: TinyLM, prompt_ids: np.ndarray, chunk_size: int):
    """逐块 prefill → (最后一块的 logits (≤chunk, V), kv_cache)。"""
    kv, logits = None, None
    for s in range(0, len(prompt_ids), chunk_size):
        # 第 s 块的 Q (chunk, D) 对 已缓存+本块 的 K (s+chunk, D) 做因果 attention; 分数矩阵只有 chunk × (s+chunk)
        logits, kv = lm.forward(prompt_ids[s:s + chunk_size], kv)
    return logits, kv


@dataclass
class CostModel:
    """每步耗时的**代价模型** (不是实测): decode 访存受限 ≈ 固定开销, prefill 计算受限 ≈ 正比于 token 数。"""
    fixed_ms: float = 20.0          # 每步固定开销 (读一遍权重 / kernel launch)
    per_token_ms: float = 0.25      # batch 里每多 1 个 token 的计算开销

    def step_ms(self, n_tokens: int) -> float:
        return self.fixed_ms + self.per_token_ms * n_tokens


def simulate(chunked: bool, token_budget: int, arrivals: List[Tuple[int, int, int]],
             cost: CostModel = CostModel()) -> Dict:
    """用 m03 的真调度器 + 代价模型回放一段负载。arrivals: [(到达 step, prompt_len, max_new)]。
    → {"tbt": 每条序列相邻 token 的间隔 (ms), "ttft": 首 token 延迟, "step_tokens": 每步 batch token 数}"""
    sched = Scheduler(SchedulerConfig(max_batch_seqs=16, max_batch_tokens=token_budget,
                                      block_size=16, num_blocks=1024, chunked_prefill=chunked))
    arrivals = sorted(arrivals)
    now, step = 0.0, 0
    arrive_t: Dict[int, float] = {}
    token_t: Dict[int, List[float]] = {}
    step_tokens: List[int] = []
    while arrivals or sched.has_unfinished():
        while arrivals and arrivals[0][0] <= step:
            _, plen, max_new = arrivals.pop(0)
            seq = sched.add_request([7] * plen, max_new, eos_id=-1)
            arrive_t[seq.seq_id], token_t[seq.seq_id] = now, []
        batch = sched.schedule()
        n_tok = sum(n for _, n in batch)
        now += cost.step_ms(n_tok)
        step += 1
        step_tokens.append(n_tok)
        emits = [seq.num_computed + n == seq.num_tokens for seq, n in batch]   # 追平的序列本步出 1 个 token
        sched.postprocess(batch, [9 if e else None for e in emits])
        for (seq, _), e in zip(batch, emits):
            if e:
                token_t[seq.seq_id].append(now)
    return {
        "tbt": {sid: np.diff(ts) for sid, ts in token_t.items()},
        "ttft": {sid: ts[0] - arrive_t[sid] for sid, ts in token_t.items()},
        "step_tokens": step_tokens,
        "total_ms": now,
    }
