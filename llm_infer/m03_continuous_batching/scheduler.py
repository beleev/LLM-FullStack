"""
scheduler.py — 连续批调度器 (prefill-priority + preempt)

仅依赖:
    - llm_infer.m02_paged_attention.block_manager.BlockManager
    - llm_infer.m03_continuous_batching.sequence.{Sequence, SeqStatus, Stage}

调度循环 (在 demo / engine 里):
    while still_have_seq:
        batch, stage = scheduler.schedule()
        outputs = model_runner.run(batch, stage)
        scheduler.postprocess(batch, outputs)
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple

from llm_infer.m02_paged_attention.block_manager import BlockManager
from llm_infer.m03_continuous_batching.sequence import Sequence, SeqStatus, Stage


@dataclass
class SchedulerConfig:
    max_batch_seqs: int = 8         # 一个 batch 最多多少序列
    max_batch_tokens: int = 256     # prefill batch 最多多少 token
    block_size: int = 16
    num_blocks: int = 64


class Scheduler:
    """vLLM-style prefill-priority continuous batching scheduler。"""

    def __init__(self, cfg: SchedulerConfig):
        self.cfg = cfg
        self.bm = BlockManager(num_blocks=cfg.num_blocks, block_size=cfg.block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self._next_id = 0
        self.preempt_count = 0

    # ------------------------------------------------------------- #
    # 用户接口                                                      #
    # ------------------------------------------------------------- #

    def add_request(self, prompt_ids: List[int], max_new: int = 32) -> int:
        sid = self._next_id
        self._next_id += 1
        seq = Sequence(seq_id=sid, prompt_ids=list(prompt_ids), max_new_tokens=max_new)
        self.waiting.append(seq)
        return sid

    def has_unfinished(self) -> bool:
        return bool(self.waiting) or bool(self.running)

    # ------------------------------------------------------------- #
    # 调度: 决定本步跑什么                                          #
    # ------------------------------------------------------------- #

    def schedule(self) -> Tuple[List[Sequence], Stage]:
        """返回 (本步要跑的序列列表, 阶段)。

        策略: 只要 waiting 有人, 优先 prefill 一批; 否则 decode running。
        """
        # ---- prefill 优先 -------------------------------------- #
        if self.waiting:
            picked: List[Sequence] = []
            tokens_budget = self.cfg.max_batch_tokens
            while (self.waiting and
                   len(picked) < self.cfg.max_batch_seqs and
                   tokens_budget > 0):
                seq = self.waiting[0]
                # 检查 block 是否够 prefill 这条
                n_blocks = (len(seq.prompt_ids) + self.cfg.block_size - 1) // self.cfg.block_size
                if not self.bm.can_allocate(n_blocks):
                    break
                if len(seq.prompt_ids) > tokens_budget:
                    break
                # 通过, 真正分配
                self.waiting.popleft()
                self.bm.allocate(seq.seq_id, len(seq.prompt_ids))
                seq.status = SeqStatus.RUNNING
                seq.stage = Stage.PREFILL
                picked.append(seq)
                tokens_budget -= len(seq.prompt_ids)
            if picked:
                return picked, Stage.PREFILL

        # ---- decode running ------------------------------------ #
        return list(self.running) + [], Stage.DECODE  # 注意: 此时 picked 已经在下方处理

    def schedule_decode(self) -> List[Sequence]:
        """单独 decode 调度: 检查每个 running 是否能再 +1 token, 不行就 preempt。"""
        ok: List[Sequence] = []
        # 反向遍历, 优先抢占新加入的 (年轻的); deque 里越靠后越年轻
        # 我们尝试给每条 +1 token 分配 block, 失败就 preempt 队尾
        candidates = list(self.running)
        while candidates:
            seq = candidates.pop(0)
            # 尝试 append
            try:
                self.bm.append(seq.seq_id, seq.num_tokens)
            except MemoryError:
                # pool 满, 抢占队尾的 (最年轻 / 最后加入)
                if not ok and not candidates:
                    raise              # 只剩自己, 真没办法
                # 把 ok 列表里最后一个或 candidates 最后一个抢占
                victim = (candidates.pop() if candidates else ok.pop())
                self._preempt(victim)
                # 重新尝试当前 seq
                try:
                    self.bm.append(seq.seq_id, seq.num_tokens)
                except MemoryError:
                    self._preempt(seq)
                    continue
            seq.stage = Stage.DECODE
            ok.append(seq)
        # 更新 running
        self.running = deque(ok)
        return ok

    def _preempt(self, seq: Sequence) -> None:
        """把序列踢回 waiting, 释放它占的 block。"""
        self.bm.free(seq.seq_id)
        seq.reset_for_recompute()
        # 保留 output_ids, 让下次重新 prefill prompt+output 一起
        seq.prompt_ids = seq.prompt_ids + seq.output_ids
        seq.output_ids = []
        self.waiting.appendleft(seq)
        self.preempt_count += 1

    # ------------------------------------------------------------- #
    # 后处理: 把生成的 token 写回, 检查终止                         #
    # ------------------------------------------------------------- #

    def postprocess(
        self, picked: List[Sequence], stage: Stage, new_token_ids: List[int]
    ) -> List[Sequence]:
        """new_token_ids[i] 是 picked[i] 本步生成的 token。"""
        finished: List[Sequence] = []
        for seq, tok in zip(picked, new_token_ids):
            seq.append_token(tok)
            if seq.is_finished():
                seq.status = SeqStatus.FINISHED
                self.bm.free(seq.seq_id)
                finished.append(seq)
                # decode 阶段下, 还要把它从 running 中摘掉
                if stage == Stage.DECODE and seq in self.running:
                    self.running.remove(seq)
            else:
                if stage == Stage.PREFILL:
                    # prefill 完成, 进入 decode 阶段
                    seq.stage = Stage.DECODE
                    self.running.append(seq)
                # decode 阶段下, 已经在 running 里, 无需再加
        return finished

    # ------------------------------------------------------------- #
    # 诊断                                                          #
    # ------------------------------------------------------------- #

    def stats(self) -> dict:
        return {
            "waiting": len(self.waiting),
            "running": len(self.running),
            "pool": self.bm.stats(),
            "preempt": self.preempt_count,
        }
