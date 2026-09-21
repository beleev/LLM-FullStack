"""
block_manager.py — KV cache 的"页式分配器" (只管账, 不碰张量)

是什么: 把 KV 显存切成定长 block, 每条序列用 block_table 记 "逻辑第 i 页 → 物理 block id", 即 OS 页表。
解决什么: 显存碎片。连续预留 max_len 的方案浪费 60~80% 显存; 分页后只有每条序列最后一页有内碎片 (< block_size)。
盯住: ref_count (多序列共享同一前缀 block) 与 free_list 的顺序 —— 它同时就是 prefix cache 的 LRU 淘汰顺序:
    ref_count 归 0 的 block 进 free_list 队尾, 但内容与 hash 仍在; 被重新分配的那一刻才通过 on_evict 失效 (vLLM 做法)。
对应真实系统: vLLM `BlockPool` / `KVCacheManager`, nano-vllm `engine/block_manager.py`。
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence


@dataclass
class BlockManager:
    num_blocks: int
    block_size: int = 16

    ref_count: List[int] = field(init=False)             # 0 = 空闲 (可能仍被 prefix cache 索引)
    block_tables: Dict[int, List[int]] = field(init=False)  # seq_id → 物理 block id 列表
    free_list: deque = field(init=False)                 # 队首最先被复用 = 最久未用
    on_evict: Optional[Callable[[int], None]] = None     # 空闲 block 被改写前的回调 (prefix cache 清索引)

    def __post_init__(self):
        self.ref_count = [0] * self.num_blocks
        self.block_tables = {}
        self.free_list = deque(range(self.num_blocks))

    # ---- 查询 ---- #

    def blocks_needed(self, n_tokens: int) -> int:
        return -(-n_tokens // self.block_size)           # ceil

    def can_allocate(self, n_tokens: int, shared: Sequence[int] = ()) -> bool:
        """n_tokens 的序列放得下吗? shared 是前缀命中的 block: 不用新分配,
        但若它正躺在 free_list 里 (ref=0), 复活它同样会吃掉一个空闲名额。"""
        revive = sum(1 for b in shared if self.ref_count[b] == 0)
        return len(self.free_list) >= self.blocks_needed(n_tokens) - len(shared) + revive

    def can_append(self, seq_id: int, n_tokens: int) -> bool:
        need = self.blocks_needed(n_tokens) - len(self.block_tables[seq_id])
        return len(self.free_list) >= need

    def num_free_blocks(self) -> int:
        return len(self.free_list)

    def used_blocks(self) -> int:
        return self.num_blocks - len(self.free_list)

    def block_table(self, seq_id: int) -> List[int]:
        return self.block_tables[seq_id]

    # ---- 分配 / 增长 / 释放 ---- #

    def _pop_free(self) -> int:
        blk = self.free_list.popleft()
        if self.on_evict is not None:
            self.on_evict(blk)                           # 内容即将被覆盖 → 先让 prefix cache 忘掉它
        self.ref_count[blk] = 1
        return blk

    def share_block(self, blk: int) -> None:
        """引用 +1。ref=0 说明它是"已释放但还没被覆盖"的缓存 block → 从 free_list 捞回来。"""
        if self.ref_count[blk] == 0:
            self.free_list.remove(blk)   # ponytail: deque.remove 是 O(n); vLLM 用双向链表做 O(1)
        self.ref_count[blk] += 1

    def allocate(self, seq_id: int, n_tokens: int, shared: Sequence[int] = ()) -> List[int]:
        """为新序列建页表: 前 len(shared) 页复用已有 block, 其余新分配。"""
        assert seq_id not in self.block_tables, f"seq {seq_id} already allocated"
        if not self.can_allocate(n_tokens, shared):
            raise MemoryError(f"need {self.blocks_needed(n_tokens) - len(shared)} new blocks, "
                              f"only {len(self.free_list)} free")
        for blk in shared:               # 先 share 再 pop: 否则可能把自己要复用的 block 淘汰掉
            self.share_block(blk)
        n_new = self.blocks_needed(n_tokens) - len(shared)
        table = list(shared) + [self._pop_free() for _ in range(n_new)]
        self.block_tables[seq_id] = table
        return table

    def ensure_capacity(self, seq_id: int, n_tokens: int) -> List[int]:
        """让页表装得下 n_tokens 个 token, 返回新分配的 block (通常 0 或 1 个)。"""
        if not self.can_append(seq_id, n_tokens):
            raise MemoryError("no free block to append")
        table = self.block_tables[seq_id]
        new = [self._pop_free() for _ in range(self.blocks_needed(n_tokens) - len(table))]
        table.extend(new)
        return new

    def append(self, seq_id: int, current_len: int) -> int:
        """序列从 current_len 长到 current_len+1; 返回新 block id, 不需要新 block 则 -1。"""
        new = self.ensure_capacity(seq_id, current_len + 1)
        return new[0] if new else -1

    def free(self, seq_id: int) -> None:
        """引用 -1, 归零的进 free_list 队尾。倒序释放: 尾部 block 先被淘汰,
        越靠前的前缀 block (越可能被别人命中) 活得越久。"""
        for blk in reversed(self.block_tables.pop(seq_id)):
            self.ref_count[blk] -= 1
            if self.ref_count[blk] == 0:
                self.free_list.append(blk)

    def stats(self) -> dict:
        return {
            "total": self.num_blocks,
            "used": self.used_blocks(),
            "free": self.num_free_blocks(),
            "utilization": f"{100 * self.used_blocks() / self.num_blocks:.1f}%",
            "n_seqs": len(self.block_tables),
        }
