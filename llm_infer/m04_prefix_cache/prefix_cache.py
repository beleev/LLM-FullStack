"""
prefix_cache.py — vLLM 风格的"按 block 链式哈希"前缀缓存 (Automatic Prefix Caching)

是什么: 给每个**写满的** KV block 算 hash = H(父 block 的 hash ‖ 本 block 的 token), 建 hash → 物理 block 索引;
        新请求逐块查表, 命中的 block 直接挂进自己的页表, 这些 token 的 prefill 整段跳过。
解决什么: 共享 system prompt / 多轮对话的重复 prefill (TTFT 与算力)。命中 N 个 token ≈ 省 N 个 token 的前向。
盯住: 缓存项的寿命 = block 内容的寿命。序列结束后 block 回到 free_list 但内容还在 → 仍可命中 (ref 0→1 复活);
      直到分配器真的要覆盖它, BlockManager.on_evict → self.evict 才删索引。free_list 顺序就是 LRU。
为什么链式: KV 依赖整个前文, 同样的 16 个 token 出现在不同前文后 KV 不同 → hash 必须带上父 hash。
对应真实系统: vLLM `KVCacheManager.get_computed_blocks` + `BlockPool.cached_block_hash_to_block`。
"""
from __future__ import annotations
import hashlib
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from llm_infer.m02_paged_attention.block_manager import BlockManager


def _block_hash(parent_hash: Optional[bytes], token_ids: Sequence[int]) -> bytes:
    h = hashlib.sha1(parent_hash or b"<root>")
    h.update(np.asarray(token_ids, dtype=np.int64).tobytes())   # 不能用 bytes(list): id ≥ 256 会抛错
    return h.digest()


class PrefixCache:
    def __init__(self, block_manager: BlockManager):
        self.bm = block_manager
        self.hash_to_block: Dict[bytes, int] = {}
        self.block_to_hash: Dict[int, bytes] = {}
        block_manager.on_evict = self.evict          # 缓存项与 block 内容同生共死

    def match_prefix(self, token_ids: Sequence[int]) -> Tuple[List[int], int]:
        """→ (命中的物理 block 列表, 命中 token 数)。只命中完整 block, 且至少留 1 个 token 不命中:
        全命中时也得真算最后一个 token 才有 logits 可采样 (vLLM 同样处理)。"""
        bs = self.bm.block_size
        hits: List[int] = []
        parent: Optional[bytes] = None
        for i in range((len(token_ids) - 1) // bs):
            parent = _block_hash(parent, token_ids[i * bs:(i + 1) * bs])
            blk = self.hash_to_block.get(parent)
            if blk is None:
                break
            hits.append(blk)
        return hits, len(hits) * bs

    def register(self, token_ids: Sequence[int], block_table: Sequence[int], num_computed: int) -> None:
        """把 KV 已算完的完整 block (前 num_computed // bs 个) 登记进索引。
        ponytail: 每次从头重算 hash 链 O(T); 真实系统把每块 hash 存在 request 上增量算。"""
        bs = self.bm.block_size
        parent: Optional[bytes] = None
        for i in range(num_computed // bs):
            parent = _block_hash(parent, token_ids[i * bs:(i + 1) * bs])
            # 同 hash 已有别的 block (两条请求同时算了同一前缀): 留旧的, 自己这块保持私有
            if parent not in self.hash_to_block and block_table[i] not in self.block_to_hash:
                self.hash_to_block[parent] = block_table[i]
                self.block_to_hash[block_table[i]] = parent

    def evict(self, block_id: int) -> None:
        h = self.block_to_hash.pop(block_id, None)
        if h is not None:
            del self.hash_to_block[h]

    def stats(self) -> dict:
        return {
            "indexed_blocks": len(self.hash_to_block),
            "pool": self.bm.stats(),
        }
