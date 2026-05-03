"""
block_manager.py — KV cache 的"页式分配器"

类比操作系统的虚拟内存:
    OS 虚拟页 ≡ 序列的逻辑 token 区间 (block_size 个 token 一页)
    OS 物理页 ≡ KV pool 中的物理 block_id
    OS 页表   ≡ 序列的 block_table[逻辑 page index] = 物理 block id

提供 4 个核心操作:
    allocate(seq_id, n_tokens)  : 首次为序列分配 block
    append(seq_id)              : 序列长度 +1, 必要时申请新 block
    free(seq_id)                : 序列结束, 引用计数 -1, 归还 block
    can_allocate(n_blocks)      : 显存是否足够 (调度器问)

引用计数 ref_count:
    多序列共享前缀时, 同一个物理 block 被多个序列指向 → ref_count > 1。
    free 时只在 ref_count 降到 0 才真正"归还到 free list"。
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class BlockManager:
    """页式 KV cache 分配器, 与具体 attention 实现解耦。"""

    num_blocks: int            # 池子总 block 数
    block_size: int = 16       # 每 block 装多少 token

    # 每个 block 的引用计数; ref_count=0 表示空闲
    ref_count: List[int] = field(init=False)
    # 每个序列的"页表": seq_id → list of physical block_id
    block_tables: Dict[int, List[int]] = field(init=False)
    # 空闲 block 队列, 用 FIFO 即可 (LRU 留给前缀缓存做)
    free_list: deque = field(init=False)

    def __post_init__(self):
        self.ref_count = [0] * self.num_blocks
        self.block_tables = {}
        self.free_list = deque(range(self.num_blocks))

    # ------------------------------------------------------------- #
    # 查询                                                          #
    # ------------------------------------------------------------- #

    def can_allocate(self, n_blocks: int) -> bool:
        return len(self.free_list) >= n_blocks

    def num_free_blocks(self) -> int:
        return len(self.free_list)

    def used_blocks(self) -> int:
        return self.num_blocks - len(self.free_list)

    def block_table(self, seq_id: int) -> List[int]:
        return self.block_tables[seq_id]

    # ------------------------------------------------------------- #
    # 分配 / 追加 / 释放                                            #
    # ------------------------------------------------------------- #

    def allocate(self, seq_id: int, n_tokens: int) -> List[int]:
        """首次为新序列分配 ceil(n_tokens / block_size) 个 block。"""
        assert seq_id not in self.block_tables, f"seq {seq_id} already allocated"
        n_blocks = (n_tokens + self.block_size - 1) // self.block_size
        if not self.can_allocate(n_blocks):
            raise MemoryError(
                f"need {n_blocks} blocks, only {len(self.free_list)} free"
            )
        table = []
        for _ in range(n_blocks):
            blk = self.free_list.popleft()
            self.ref_count[blk] = 1
            table.append(blk)
        self.block_tables[seq_id] = table
        return table

    def append(self, seq_id: int, current_len: int) -> int:
        """序列长度从 current_len 涨到 current_len+1, 看是否需要新 block。

        返回新分配的 block_id; 若不需要新分配, 返回 -1。
        """
        table = self.block_tables[seq_id]
        # 当前已分配 block 能装的 token 数
        capacity = len(table) * self.block_size
        if current_len + 1 <= capacity:
            return -1                                  # 还有空位, 不分配
        if not self.can_allocate(1):
            raise MemoryError("no free block to append")
        blk = self.free_list.popleft()
        self.ref_count[blk] = 1
        table.append(blk)
        return blk

    def free(self, seq_id: int) -> None:
        """序列结束, 所有 block 引用 -1; 归零的进 free_list。"""
        table = self.block_tables.pop(seq_id)
        for blk in table:
            self.ref_count[blk] -= 1
            if self.ref_count[blk] == 0:
                self.free_list.append(blk)

    # ------------------------------------------------------------- #
    # 共享: 让 src 与 dst 共享某个前缀 block (供 m04 prefix cache 用) #
    # ------------------------------------------------------------- #

    def share_block(self, blk: int) -> None:
        """显式 +1 引用; 调用方负责保证 blk 是合法的。"""
        assert self.ref_count[blk] >= 1, "cannot share a free block"
        self.ref_count[blk] += 1

    # ------------------------------------------------------------- #
    # 诊断                                                          #
    # ------------------------------------------------------------- #

    def stats(self) -> dict:
        return {
            "total": self.num_blocks,
            "used": self.used_blocks(),
            "free": self.num_free_blocks(),
            "utilization": f"{100 * self.used_blocks() / self.num_blocks:.1f}%",
            "n_seqs": len(self.block_tables),
        }
