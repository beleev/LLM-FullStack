"""
scheduler.py — 连续批调度器 (iteration-level scheduling), full_engine 直接用的就是这个类

是什么: 每个 step 重新决定 "哪些序列、各算几个 token"。返回 batch = [(seq, n_tokens), ...]:
        n>1 是 prefill (或一个 chunk), n=1 是 decode。请求随到随进、随完随出, 不用等整批结束。
解决什么: 吞吐。静态 batch 要等最长的那条; 连续批让 GPU 每步都满载 (Orca / vLLM 的核心)。
两种策略 (cfg.chunked_prefill):
    False  prefill 优先 (vLLM V0): 有新请求能进就整段 prefill, 这一步 decode 全体停顿; 进不了就**落到 decode**。
    True   decode 优先 + 分块 prefill (Sarathi / vLLM V1): 先给每条 running 1 个 token,
           剩下的 token 预算切给 prefill chunk → 同一个 batch 里 prefill 与 decode 混跑, 见 m06。
盯住: budget (每步 token 预算 max_batch_tokens) 与 seq.num_computed; 抢占 = 还 block + num_computed 归 0。
对应真实系统: vLLM `Scheduler.schedule()`, SGLang `PrefillAdder`, nano-vllm `engine/scheduler.py`。
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

from llm_infer.core.sequence import Sequence, SeqStatus
from llm_infer.m02_paged_attention.block_manager import BlockManager

Batch = List[Tuple[Sequence, int]]      # (序列, 本步要算的 token 数)


@dataclass
class SchedulerConfig:
    max_batch_seqs: int = 8             # 同时 running 的序列上限
    max_batch_tokens: int = 256         # 每步 token 预算 (prefill + decode 合计)
    block_size: int = 16
    num_blocks: int = 64
    chunked_prefill: bool = False


class Scheduler:
    def __init__(self, cfg: SchedulerConfig, prefix_cache=None):
        """prefix_cache: 可选, 鸭子类型 (match_prefix / register), 见 m04; 为 None 则不做前缀复用。"""
        self.cfg = cfg
        self.bm = prefix_cache.bm if prefix_cache is not None else \
            BlockManager(num_blocks=cfg.num_blocks, block_size=cfg.block_size)
        self.prefix_cache = prefix_cache
        self.waiting: deque[Sequence] = deque()
        self.running: List[Sequence] = []          # 按进入顺序, 越靠后越年轻
        self._next_id = 0
        self.preempt_count = 0
        self.prefix_hit_tokens = 0                 # 真正被跳过前向的 token 数 (只在成功接入时累计)
        self._just_preempted = False

    def add_request(self, prompt_ids: List[int], max_new: int = 32, eos_id: int = 2) -> Sequence:
        # 单条序列跑满也装不下 → 永远调度不了, 早失败好过死循环
        assert self.bm.blocks_needed(len(prompt_ids) + max_new) <= self.bm.num_blocks, \
            "prompt + max_new 超过整个 KV pool"
        seq = Sequence(self._next_id, list(prompt_ids), max_new, eos_id)
        self._next_id += 1
        self.waiting.append(seq)
        return seq

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    # ---- 调度 ---- #

    def schedule(self) -> Batch:
        budget = self.cfg.max_batch_tokens
        if not self.cfg.chunked_prefill:
            # prefill 优先; 队首拿不到 block 时 batch 为空 → 必须落到 decode,
            # 否则 running 永远不前进、block 永远不释放 = 活锁
            return self._admit(budget) or self._schedule_running(budget)
        batch = self._schedule_running(budget)
        budget -= sum(n for _, n in batch)
        # 刚因为没 block 抢占过, 这一步就别再接新请求, 免得立刻把腾出的 block 又占掉
        return batch + ([] if self._just_preempted else self._admit(budget))

    def _schedule_running(self, budget: int) -> Batch:
        """给 running 里每条序列它还欠的 token (decode 欠 1 个, 未完成的 prefill 欠多个)。
        block 不够就抢占最年轻的; 轮到自己最年轻就抢占自己。最老的序列永远能前进 → 不会活锁。"""
        batch: Batch = []
        self._just_preempted = False
        todo = list(self.running)
        while todo and budget > 0:
            seq = todo.pop(0)
            n = min(seq.num_tokens - seq.num_computed, budget)
            while not self.bm.can_append(seq.seq_id, seq.num_computed + n):
                victim = todo.pop() if todo else seq
                self._preempt(victim)
                if victim is seq:
                    break
            else:
                self.bm.ensure_capacity(seq.seq_id, seq.num_computed + n)
                batch.append((seq, n))
                budget -= n
        return batch

    def _admit(self, budget: int) -> Batch:
        """从 waiting 队首接新请求 (FCFS)。命中前缀的 token 不占预算、不用算。"""
        batch: Batch = []
        while self.waiting and len(self.running) < self.cfg.max_batch_seqs and budget > 0:
            seq = self.waiting[0]
            ids = seq.all_ids                       # 被抢占过的序列要连同已生成的 token 一起重算
            hits, n_hit = self.prefix_cache.match_prefix(ids) if self.prefix_cache else ([], 0)
            n = len(ids) - n_hit
            if self.cfg.chunked_prefill:
                n = min(n, budget)                  # 只吃预算内的一个 chunk, 剩下的下一步接着算
            elif n > budget and batch:
                break                               # 整段放不进本步预算 (batch 为空时放行, 否则超长 prompt 会饿死)
            if not self.bm.can_allocate(len(ids), hits):
                break
            self.waiting.popleft()
            self.bm.allocate(seq.seq_id, len(ids), hits)
            seq.num_computed = n_hit
            self.prefix_hit_tokens += n_hit
            seq.status = SeqStatus.RUNNING
            self.running.append(seq)
            batch.append((seq, n))
            budget -= n
        return batch

    def _preempt(self, seq: Sequence) -> None:
        """recompute 式抢占: block 全还, 回 waiting 队首。output_ids 原样保留 ——
        既不丢已生成的文本, 也不会让 max_new_tokens 重新计数。"""
        self.bm.free(seq.seq_id)
        self.running.remove(seq)
        seq.reset_for_recompute()
        self.waiting.appendleft(seq)
        self.preempt_count += 1
        self._just_preempted = True

    # ---- 后处理 ---- #

    def postprocess(self, batch: Batch, new_tokens: List[Optional[int]]) -> List[Sequence]:
        """new_tokens[i]: batch[i] 本步采样出的 token; 还在 prefill 中途的 chunk 没有 logits 可采 → None。"""
        finished: List[Sequence] = []
        for (seq, n), tok in zip(batch, new_tokens):
            seq.num_computed += n
            if self.prefix_cache is not None:
                self.prefix_cache.register(seq.all_ids, self.bm.block_table(seq.seq_id), seq.num_computed)
            if seq.num_computed < seq.num_tokens:
                continue                            # prefill 还没追平
            seq.append_token(tok)
            if seq.is_finished():
                seq.status = SeqStatus.FINISHED
                self.bm.free(seq.seq_id)
                self.running.remove(seq)
                finished.append(seq)
        return finished

    def stats(self) -> dict:
        return {
            "waiting": len(self.waiting),
            "running": len(self.running),
            "pool": self.bm.stats(),
            "preempt": self.preempt_count,
        }
