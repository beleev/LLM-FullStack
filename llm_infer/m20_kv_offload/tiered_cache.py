"""
tiered_cache.py — 分层 KV cache 卸载: GPU → CPU → 磁盘

是什么: prefix cache 的 block 被 GPU 挤出去时不丢弃, 而是降级到 CPU 内存 / 磁盘; 再次命中时搬回来。
瓶颈:   TTFT。多轮对话每轮 prompt = 整段历史, GPU 装不下所有用户的历史 → 反复重算 prefill。
关键数字: 搬 1 token KV (128 KiB, LLaMA-3-8B fp16) 走 PCIe 25 GB/s ≈ 5 µs, 重算 ≈ 125 µs (8k tok/s)。
        但每次加载有固定延迟, 且慢链路每 token 可能比重算还慢 → 必须逐层判断 "加载 vs 重算"。
盯住:   TieredKVCache.tiers (每层一个 LRU OrderedDict: block_hash → True) 和 serve() 里的 use_load。
真实系统: LMCache / Mooncake Store / SGLang HiCache / vLLM KV connector。
注意:   这里没有真的搬数据, 所有时间来自 CostModel (可见参数的代价模型), 不是实测。
"""
from __future__ import annotations
import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class Tier:
    name: str
    capacity_blocks: int
    bandwidth_GBps: float      # 该层 → GPU 的带宽; GPU 层自身填 inf
    latency_ms: float          # 每次请求从该层加载的固定开销 (调度 / seek / RPC)


@dataclass
class CostModel:
    block_size: int = 16
    kv_bytes_per_token: int = 131072       # LLaMA-3-8B fp16: 2·8·128·32·2
    prefill_tok_per_s: float = 8000.0      # 重算速度 (线性近似, 忽略 attention 的 O(T²))

    def recompute_ms(self, n_tokens: int) -> float:
        return n_tokens / self.prefill_tok_per_s * 1e3

    def load_ms(self, tier: Tier, n_blocks: int) -> float:
        if n_blocks == 0 or np.isinf(tier.bandwidth_GBps):
            return 0.0
        nbytes = n_blocks * self.block_size * self.kv_bytes_per_token
        return tier.latency_ms + nbytes / (tier.bandwidth_GBps * 1e9) * 1e3


def block_hashes(ids, block_size: int) -> List[bytes]:
    """链式 hash: h_i = H(h_{i-1} ‖ block_i), 所以 h_i 唯一标识 "到第 i 块为止的整个前缀"。只算完整块。"""
    ids = np.asarray(ids, np.int64)
    out, prev = [], b""
    for i in range(len(ids) // block_size):
        prev = hashlib.sha256(prev + ids[i * block_size:(i + 1) * block_size].tobytes()).digest()
        out.append(prev)
    return out


@dataclass
class Stats:
    total: int = 0
    miss: int = 0
    hit: List[int] = field(default_factory=list)          # 每层命中的 token 数
    recomputed_hit: int = 0                               # 命中了但策略选择重算的 token 数
    ttft_ms: float = 0.0
    n_req: int = 0


class TieredKVCache:
    def __init__(self, tiers: List[Tier], cost: CostModel, always_load: bool = False):
        self.tier_cfg, self.cost, self.always_load = tiers, cost, always_load
        self.tiers = [OrderedDict() for _ in tiers]       # 每层 LRU: 队首最旧
        self.stats = Stats(hit=[0] * len(tiers))

    def _insert(self, level: int, h: bytes):
        """放进 level 层; 溢出的最旧 block 降级到下一层 (最后一层之后 = 丢弃)。"""
        if level >= len(self.tiers):
            return
        lru = self.tiers[level]
        lru[h] = True
        lru.move_to_end(h)
        while len(lru) > self.tier_cfg[level].capacity_blocks:
            old, _ = lru.popitem(last=False)
            self._insert(level + 1, old)

    def lookup(self, hashes: List[bytes]) -> List[int]:
        """沿链逐块查找所在层; 第一个全层 miss 处停 (后面的块即使在也不可用: KV 依赖整个前缀)。"""
        where = []
        for h in hashes:
            level = next((i for i, t in enumerate(self.tiers) if h in t), None)
            if level is None:
                break
            where.append(level)
        return where

    def store(self, ids):
        """把 ids 的所有完整块放到 GPU 层 MRU 端: 命中的 = 提升, 新算的 = 写入 (也用于登记 decode 生成的回复 KV)。"""
        hashes = block_hashes(ids, self.cost.block_size)
        # ponytail: 倒序 touch 让链尾比链头更 "旧", 近似真实系统的 "先驱逐叶子" (radix tree); 需要精确时换成树
        for h in reversed(hashes):
            for lower in self.tiers[1:]:
                lower.pop(h, None)                        # 提升 = 从下层移走, 保证一个 block 只在一层
            self._insert(0, h)

    def serve(self, prompt_ids) -> float:
        """处理一个请求的 prefill, 返回模拟 TTFT (ms)。"""
        c, bs = self.cost, self.cost.block_size
        hashes = block_hashes(prompt_ids, bs)
        where = self.lookup(hashes)
        n_hit = [where.count(i) for i in range(len(self.tiers))]          # 每层命中块数
        miss_tokens = len(prompt_ids) - len(where) * bs                   # 含不足一块的尾巴
        ttft = c.recompute_ms(miss_tokens)
        for tier, nb in zip(self.tier_cfg, n_hit):
            load, recompute = c.load_ms(tier, nb), c.recompute_ms(nb * bs)
            # 交叉点: 命中块太少 (固定延迟摊不开) 或链路太慢时, 加载比重算还慢 → 重算
            use_load = self.always_load or load <= recompute
            ttft += load if use_load else recompute
            if not use_load:
                self.stats.recomputed_hit += nb * bs
        self.store(prompt_ids)

        s = self.stats
        s.total += len(prompt_ids); s.miss += miss_tokens; s.ttft_ms += ttft; s.n_req += 1
        for i, nb in enumerate(n_hit):
            s.hit[i] += nb * bs
        return ttft


def chat_workload(n_users=8, n_turns=6, sys_len=64, seed=0):
    """多轮对话: 每轮 prompt = 共享 system prompt + 该用户全部历史 + 新消息; 用户轮流发言 (GPU 层抖动)。
    产出 [(user, prompt_ids, prompt+reply ids)], 按到达顺序。"""
    rs = np.random.RandomState(seed)
    system = rs.randint(0, 32000, sys_len)
    hist = [system.copy() for _ in range(n_users)]
    reqs = []
    for _ in range(n_turns):
        for u in range(n_users):
            hist[u] = np.concatenate([hist[u], rs.randint(0, 32000, rs.randint(40, 100))])   # 用户消息
            prompt = hist[u]
            hist[u] = np.concatenate([hist[u], rs.randint(0, 32000, rs.randint(40, 100))])   # 模型回复, KV 在 decode 时生成
            reqs.append((u, prompt, hist[u]))
    return reqs
