"""
engine.py — mini-vLLM: 把前面的模块接成一个能跑的离线推理引擎

是什么: 50 行胶水。每个 step:  scheduler.schedule() → model_runner.run() → sample() → scheduler.postprocess()
    调度 / 抢占 / 分块 prefill   m03.Scheduler (同一个类, 不是拷贝; chunked_prefill 开关见 m06)
    block 记账                   m02.BlockManager          物理 KV pool + 分页读写   model_runner.py + m02.paged_attention
    前缀复用                     m04.PrefixCache           采样                      m10.sample
盯住: 一步里 batch = [(seq, n)] —— n>1 的 prefill chunk 与 n=1 的 decode 混在同一步;
      以及 stats 里 prefill_tokens_saved 与 runner.tokens_computed 两个数必须对得上账。
正确性契约 (demo 里 assert): greedy 下, 无论是否发生前缀共享 / 分块 / 抢占, 每条输出与 TinyLM.generate_greedy 逐 token 相同。
对应真实系统: vLLM `LLMEngine.step()` / `EngineCore.step()`。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from llm_infer.core import TinyLM, ModelConfig, CharTokenizer, Sequence
from llm_infer.m02_paged_attention.block_manager import BlockManager
from llm_infer.m03_continuous_batching.scheduler import Scheduler, SchedulerConfig
from llm_infer.m04_prefix_cache.prefix_cache import PrefixCache
from llm_infer.m10_sampling.samplers import SamplingParams, sample
from llm_infer.full_engine.model_runner import ModelRunner


@dataclass(frozen=True)
class EngineConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    block_size: int = 16
    num_blocks: int = 64
    max_batch_seqs: int = 8
    max_batch_tokens: int = 256
    chunked_prefill: bool = True
    prefix_caching: bool = True


class Engine:
    def __init__(self, cfg: EngineConfig, lm: Optional[TinyLM] = None):
        self.cfg = cfg
        self.lm = lm or TinyLM(cfg.model)
        self.tok = CharTokenizer()
        bm = BlockManager(num_blocks=cfg.num_blocks, block_size=cfg.block_size)
        self.prefix_cache = PrefixCache(bm) if cfg.prefix_caching else None
        self.scheduler = Scheduler(
            SchedulerConfig(cfg.max_batch_seqs, cfg.max_batch_tokens, cfg.block_size,
                            cfg.num_blocks, cfg.chunked_prefill),
            prefix_cache=self.prefix_cache)
        self.runner = ModelRunner(self.lm, cfg.num_blocks, cfg.block_size)
        self.sampling_for: Dict[int, SamplingParams] = {}
        self.finished: Dict[int, Sequence] = {}
        self._rng = np.random.RandomState(0)
        self.steps = 0

    def add_request(self, prompt, sampling: Optional[SamplingParams] = None, max_new: int = 32) -> int:
        """prompt: str (走 CharTokenizer) 或现成的 token id 列表。"""
        ids = self.tok.encode(prompt, add_bos=True) if isinstance(prompt, str) else list(prompt)
        seq = self.scheduler.add_request(ids, max_new, eos_id=self.tok.EOS_ID)
        self.sampling_for[seq.seq_id] = sampling or SamplingParams(temperature=0.0)
        return seq.seq_id

    def has_unfinished(self) -> bool:
        return self.scheduler.has_unfinished()

    def step(self) -> List[Sequence]:
        """跑一步, 返回本步完成的序列。self.last_batch 留给 demo 打印。"""
        self.steps += 1
        bm = self.scheduler.bm
        batch = self.last_batch = self.scheduler.schedule()
        tokens: List[Optional[int]] = []
        for seq, n in batch:
            ids = seq.all_ids[:seq.num_computed + n]
            logits = self.runner.run(ids, bm.block_table(seq.seq_id), seq.num_computed)
            done_prefill = seq.num_computed + n == seq.num_tokens     # 追平了才有"下一个 token"可采
            tokens.append(sample(logits, self.sampling_for[seq.seq_id], history=seq.all_ids, rng=self._rng)
                          if done_prefill else None)
        finished = self.scheduler.postprocess(batch, tokens)
        for seq in finished:
            self.finished[seq.seq_id] = seq
        return finished

    def generate(self, prompts, sampling: Optional[SamplingParams] = None, max_new: int = 32) -> Dict[int, List[int]]:
        """一键跑完 (类似 vLLM `LLM.generate`), 返回 {seq_id: 全部 output token ids}。"""
        sids = [self.add_request(p, sampling, max_new) for p in prompts]
        while self.has_unfinished():
            self.step()
        return {i: self.finished[i].output_ids for i in sids}

    def report_stats(self) -> dict:
        return {
            "steps": self.steps,
            "tokens_computed": self.runner.tokens_computed,
            "prefix_hit_tokens": self.scheduler.prefix_hit_tokens,   # 命中 → 没做前向的 token
            "preempt": self.scheduler.preempt_count,
            "pool": self.scheduler.bm.stats(),
        }
