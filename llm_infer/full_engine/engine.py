"""
engine.py — mini-vLLM: 集成 paged attention + continuous batching + prefix cache + sampling

接口:
    engine = Engine(EngineConfig(...))
    engine.add_request(prompt: str, sampling: SamplingParams)
    while engine.has_unfinished():
        finished_outputs = engine.step()

注意:
    本实现复用 m02/m03/m04/m10 的代码, 但因为本教学项目的 KV 是按"层"
    保存 (List[Tuple[K,V]]), 不直接走 paged 的物理 block。这里我们用
    BlockManager 跟踪"逻辑容量与释放", KV 实际数据仍存在 Sequence 里。
    做法等价于"分页只管显存配额, 物理 KV 跟在序列上", 教学上更易看懂。

    真实 vLLM 是把 KV 物理放进 pool, 这里若要严格对齐, 需要在 attention 里
    实现分页 gather, 见 m02/paged_attention.py 已经实现, 可作为练习自己合上。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from llm_infer.core import TinyLM, ModelConfig, CharTokenizer
from llm_infer.m02_paged_attention.block_manager import BlockManager
from llm_infer.m03_continuous_batching.sequence import Sequence, SeqStatus, Stage
from llm_infer.m04_prefix_cache.prefix_cache import PrefixCache
from llm_infer.m10_sampling.samplers import SamplingParams, sample


# --------------------------------------------------------------------- #
# 配置                                                                  #
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class EngineConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    block_size: int = 16
    num_blocks: int = 64
    max_batch_seqs: int = 8
    max_batch_tokens: int = 256


# --------------------------------------------------------------------- #
# Engine                                                                #
# --------------------------------------------------------------------- #

class Engine:
    """mini-vLLM 风格的离线引擎。"""

    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self.lm = TinyLM(cfg.model)
        self.tok = CharTokenizer()
        self.bm = BlockManager(num_blocks=cfg.num_blocks, block_size=cfg.block_size)
        self.prefix_cache = PrefixCache(self.bm)

        self.waiting: List[Sequence] = []
        self.running: List[Sequence] = []
        self.sampling_for: Dict[int, SamplingParams] = {}
        self.finished_outputs: Dict[int, str] = {}
        self._next_id = 0
        self._rng = np.random.RandomState(0)

        # 简易统计
        self.stats_step = 0
        self.stats_prefix_hits = 0
        self.stats_prefill_tokens_saved = 0

    # ------------------------------------------------------------- #
    # 用户接口                                                      #
    # ------------------------------------------------------------- #

    def add_request(
        self,
        prompt: str,
        sampling: Optional[SamplingParams] = None,
        max_new: int = 32,
    ) -> int:
        sid = self._next_id
        self._next_id += 1
        ids = self.tok.encode(prompt, add_bos=True)
        seq = Sequence(seq_id=sid, prompt_ids=ids, max_new_tokens=max_new,
                       eos_id=self.tok.EOS_ID)
        self.waiting.append(seq)
        self.sampling_for[sid] = sampling or SamplingParams(temperature=0.0)
        return sid

    def has_unfinished(self) -> bool:
        return bool(self.waiting) or bool(self.running)

    # ------------------------------------------------------------- #
    # 主循环 step: prefill 优先, 否则 decode                        #
    # ------------------------------------------------------------- #

    def step(self) -> List[Tuple[int, str]]:
        """跑一步; 返回本步完成的 (seq_id, text)。"""
        self.stats_step += 1
        # ---- prefill 优先 ----------------------------------------- #
        if self.waiting:
            return self._step_prefill()
        # ---- decode ---------------------------------------------- #
        return self._step_decode()

    # ------------------------------------------------------------- #
    # prefill 阶段                                                  #
    # ------------------------------------------------------------- #

    def _step_prefill(self) -> List[Tuple[int, str]]:
        finished: List[Tuple[int, str]] = []
        token_budget = self.cfg.max_batch_tokens
        picked: List[Sequence] = []
        while self.waiting and len(picked) < self.cfg.max_batch_seqs and token_budget > 0:
            seq = self.waiting[0]
            n_blocks_needed = (len(seq.prompt_ids) + self.cfg.block_size - 1) // self.cfg.block_size

            # ---- prefix cache 查询 -------------------------------- #
            hits, n_hit_tokens = self.prefix_cache.match_prefix(seq.prompt_ids)
            new_blocks_needed = n_blocks_needed - len(hits)
            if not self.bm.can_allocate(max(new_blocks_needed, 0)):
                break
            if len(seq.prompt_ids) > token_budget:
                break

            # ---- 实际分配: 命中部分 share, 未命中 allocate -------- #
            self.waiting.pop(0)
            block_table: List[int] = []
            for blk in hits:
                self.bm.share_block(blk)
                block_table.append(blk)
            self.stats_prefix_hits += len(hits)
            self.stats_prefill_tokens_saved += n_hit_tokens
            for _ in range(new_blocks_needed):
                blk = self.bm.free_list.popleft()
                self.bm.ref_count[blk] = 1
                block_table.append(blk)
            self.bm.block_tables[seq.seq_id] = block_table

            # ---- 跑 prefill (若全命中, 也得跑一次拿 logits, 简化) - #
            ids_arr = np.array(seq.prompt_ids, dtype=np.int64)
            logits, kv_cache = self.lm.prefill(ids_arr)
            seq.kv_cache = kv_cache
            seq.status = SeqStatus.RUNNING
            seq.stage = Stage.DECODE

            # ---- 注册 prefix cache (对完整 block) ----------------- #
            parent = None
            for i in range(len(seq.prompt_ids) // self.cfg.block_size):
                chunk = seq.prompt_ids[i*self.cfg.block_size:(i+1)*self.cfg.block_size]
                h = self.prefix_cache.register_block(parent, chunk, block_table[i])
                parent = h

            # ---- 采样首个 token ---------------------------------- #
            params = self.sampling_for[seq.seq_id]
            tok_id = sample(logits[-1], params, history=seq.prompt_ids, rng=self._rng)
            seq.append_token(tok_id)
            picked.append(seq)
            token_budget -= len(seq.prompt_ids)

            if seq.is_finished():
                seq.status = SeqStatus.FINISHED
                self.bm.free(seq.seq_id)
                txt = self.tok.decode(seq.output_ids)
                self.finished_outputs[seq.seq_id] = txt
                finished.append((seq.seq_id, txt))
            else:
                self.running.append(seq)
        return finished

    # ------------------------------------------------------------- #
    # decode 阶段                                                   #
    # ------------------------------------------------------------- #

    def _step_decode(self) -> List[Tuple[int, str]]:
        finished: List[Tuple[int, str]] = []
        # 给每条 running 跑一步 decode
        survivors: List[Sequence] = []
        for seq in self.running:
            try:
                self.bm.append(seq.seq_id, seq.num_tokens)
            except MemoryError:
                # preempt
                self._preempt(seq)
                continue
            last_id = seq.output_ids[-1] if seq.output_ids else seq.prompt_ids[-1]
            logits, seq.kv_cache = self.lm.decode_step(last_id, seq.kv_cache)
            params = self.sampling_for[seq.seq_id]
            tok_id = sample(logits, params, history=seq.all_ids, rng=self._rng)
            seq.append_token(tok_id)
            if seq.is_finished():
                seq.status = SeqStatus.FINISHED
                self.bm.free(seq.seq_id)
                txt = self.tok.decode(seq.output_ids)
                self.finished_outputs[seq.seq_id] = txt
                finished.append((seq.seq_id, txt))
            else:
                survivors.append(seq)
        self.running = survivors
        return finished

    def _preempt(self, seq: Sequence) -> None:
        self.bm.free(seq.seq_id)
        seq.kv_cache = None
        seq.prompt_ids = seq.prompt_ids + seq.output_ids
        seq.output_ids = []
        seq.status = SeqStatus.WAITING
        seq.stage = Stage.PREFILL
        self.waiting.insert(0, seq)

    # ------------------------------------------------------------- #
    # 一键 generate (类似 vLLM API)                                 #
    # ------------------------------------------------------------- #

    def generate(
        self, prompts: List[str], sampling: Optional[SamplingParams] = None,
        max_new: int = 32,
    ) -> Dict[int, str]:
        ids = [self.add_request(p, sampling, max_new=max_new) for p in prompts]
        while self.has_unfinished():
            self.step()
        return {i: self.finished_outputs[i] for i in ids}

    def report_stats(self) -> dict:
        return {
            "steps": self.stats_step,
            "prefix_block_hits": self.stats_prefix_hits,
            "prefill_tokens_saved": self.stats_prefill_tokens_saved,
            "pool": self.bm.stats(),
        }
