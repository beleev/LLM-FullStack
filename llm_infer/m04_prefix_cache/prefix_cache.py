"""
prefix_cache.py — vLLM 风格的"按 block 哈希"前缀缓存

数据结构:
    hash_to_block: Dict[bytes, int]    一个 hash 对应一个物理 block_id
    block_hashes : Dict[int, bytes]    反向索引, 用于驱逐时清理

哈希算法:
    用 SHA-1 (够快, 16 字节, 演示用; 生产用 xxhash)
    chain_hash = SHA1(parent_hash || token_ids_bytes)
    None 父: parent_hash = b'<root>'

使用流程:
    1) 新请求来 → match_prefix(token_ids) 返回 (hits: List[block_id], hit_len: int)
    2) Scheduler 用 hits 去 BlockManager.share_block 增加引用
    3) 后续未命中部分照常分配 + 计算 + 写 cache
    4) 写新 block 后调 register_block(parent_hash, token_ids, block_id)
"""
from __future__ import annotations
import hashlib
from typing import Dict, List, Optional, Tuple

from llm_infer.m02_paged_attention.block_manager import BlockManager


def _block_hash(parent_hash: Optional[bytes], token_ids: List[int]) -> bytes:
    h = hashlib.sha1()
    h.update(parent_hash if parent_hash is not None else b"<root>")
    h.update(bytes(token_ids))
    return h.digest()


class PrefixCache:
    """挂在 BlockManager 之上, 索引"哪些 block 装了哪段 token"。"""

    def __init__(self, block_manager: BlockManager):
        self.bm = block_manager
        self.hash_to_block: Dict[bytes, int] = {}
        self.block_to_hash: Dict[int, bytes] = {}

    # ------------------------------------------------------------- #
    # 查询: 给一段 token, 返回能复用的 block 列表                    #
    # ------------------------------------------------------------- #

    def match_prefix(self, token_ids: List[int]) -> Tuple[List[int], int]:
        """返回 (命中的物理 block_id 列表, 命中的 token 数)。

        只会命中**完整 block**: 比如 block_size=4, prompt 长 7,
        最多命中第 0 块 (4 token), 第 1 块不完整就 miss。
        """
        block_size = self.bm.block_size
        n_full = len(token_ids) // block_size
        hits: List[int] = []
        parent: Optional[bytes] = None
        for i in range(n_full):
            chunk = token_ids[i * block_size:(i + 1) * block_size]
            h = _block_hash(parent, chunk)
            blk = self.hash_to_block.get(h)
            if blk is None:
                break
            hits.append(blk)
            parent = h
        hit_tokens = len(hits) * block_size
        return hits, hit_tokens

    # ------------------------------------------------------------- #
    # 注册: 新算完一个 block 之后调用                                #
    # ------------------------------------------------------------- #

    def register_block(
        self,
        parent_hash: Optional[bytes],
        token_ids: List[int],
        block_id: int,
    ) -> bytes:
        """记录 (hash → block) 与 (block → hash) 双向映射, 返回该 block 的 hash。"""
        h = _block_hash(parent_hash, token_ids)
        # 若已有同 hash, 优先复用旧的 (本 block 当作冗余被覆盖)
        if h in self.hash_to_block:
            return h
        self.hash_to_block[h] = block_id
        self.block_to_hash[block_id] = h
        return h

    def evict(self, block_id: int) -> None:
        """block 被回收时清掉索引 (避免 hash → 空 block)。"""
        h = self.block_to_hash.pop(block_id, None)
        if h is not None:
            self.hash_to_block.pop(h, None)

    def stats(self) -> dict:
        return {
            "indexed_blocks": len(self.hash_to_block),
            "pool": self.bm.stats(),
        }
