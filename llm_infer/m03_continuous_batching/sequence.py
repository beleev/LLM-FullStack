"""
sequence.py — 单个推理请求的状态机

状态转换:
        ┌──────────┐  schedule  ┌──────────┐  finish  ┌──────────┐
        │ WAITING  │ ─────────→ │ RUNNING  │ ───────→ │ FINISHED │
        └──────────┘            └──────────┘          └──────────┘
              ↑                       │
              │                       │ preempt (pool full)
              └───────────────────────┘

阶段 (与状态正交):
    PREFILL  : 还在处理 prompt, 一次可吃多个 token
    DECODE   : 在生成新 token, 每步只产出 1 个
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple
import numpy as np


class SeqStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


class Stage(Enum):
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass
class Sequence:
    """单条请求的全部 mutable state。

    KV cache 是一个引用 (List[Tuple[K, V]]), prefill 后填; preempt 时清掉。
    """
    seq_id: int
    prompt_ids: List[int]
    max_new_tokens: int
    eos_id: int = 2

    # mutable 状态
    output_ids: List[int] = field(default_factory=list)
    status: SeqStatus = SeqStatus.WAITING
    stage: Stage = Stage.PREFILL
    kv_cache: Optional[list] = None             # [(K, V), ...]; 由 model_runner 填

    # ------- 派生属性 ------- #
    @property
    def all_ids(self) -> List[int]:
        return self.prompt_ids + self.output_ids

    @property
    def num_tokens(self) -> int:
        return len(self.all_ids)

    @property
    def num_output(self) -> int:
        return len(self.output_ids)

    def append_token(self, token_id: int) -> None:
        self.output_ids.append(token_id)

    def is_finished(self) -> bool:
        if self.num_output >= self.max_new_tokens:
            return True
        if self.output_ids and self.output_ids[-1] == self.eos_id:
            return True
        return False

    def reset_for_recompute(self) -> None:
        """preempt: 把已生成 token 留下, KV 丢弃, 下次 prefill 整段重算。"""
        self.kv_cache = None
        self.stage = Stage.PREFILL
        self.status = SeqStatus.WAITING
