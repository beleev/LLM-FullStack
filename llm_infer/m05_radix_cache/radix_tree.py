"""
radix_tree.py — KV cache 的 Radix Tree (SGLang 风格)

每条边对应一段 token 序列 (而非单个 token), 可被 split。
节点持有这段 token 对应的 "kv_handle" — 教学起见, kv_handle 就是个
不透明 id (真实系统里指向 KV pool 中的 block 列表)。

LRU 驱逐:
    每次 match / lock / unlock 都会更新 timestamp。
    驱逐时挑 ref_count==0 且 timestamp 最小的叶节点。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import itertools

_clock = itertools.count()  # 全局逻辑时钟, 简化 LRU


@dataclass
class RadixNode:
    """edge_tokens: 从父到本节点的 token 串; root 节点 edge_tokens=[]"""
    edge_tokens: List[int]
    kv_handle: Optional[int] = None              # 该路径上 token 对应的 KV
    children: Dict[int, "RadixNode"] = field(default_factory=dict)  # key=首个 token
    parent: Optional["RadixNode"] = None
    ref_count: int = 0
    last_used: int = field(default_factory=lambda: next(_clock))

    def is_root(self) -> bool:
        return self.parent is None

    def is_leaf(self) -> bool:
        return not self.children

    def total_tokens_to_here(self) -> int:
        """从 root 走到这里, 累计的 token 数。"""
        n, cur = 0, self
        while cur is not None:
            n += len(cur.edge_tokens)
            cur = cur.parent
        return n


class RadixCache:
    """支持任意前缀长度共享的 KV cache 索引。"""

    def __init__(self) -> None:
        self.root = RadixNode(edge_tokens=[])

    # ------------------------------------------------------------- #
    # match: 沿树走, 返回 (停在的节点, 命中 token 数)                #
    # ------------------------------------------------------------- #

    def match(self, tokens: List[int]) -> Tuple[RadixNode, int]:
        node = self.root
        i = 0
        while i < len(tokens):
            child = node.children.get(tokens[i])
            if child is None:
                break
            # 看 child.edge_tokens 与 tokens[i:] 的最长公共前缀
            cmp_len = 0
            for a, b in zip(child.edge_tokens, tokens[i:]):
                if a != b:
                    break
                cmp_len += 1
            if cmp_len < len(child.edge_tokens):
                # 部分匹配 → 必须 split, 但本步不改树, 由 insert 做
                # 返回"split 后会停的"虚拟位置
                # 教学简化: 返回 (child, i+cmp_len), 调用方自己理解为
                #          "命中到 child 内部 cmp_len 处"
                if cmp_len > 0:
                    # 在树上"看到"了 cmp_len 个 token 的命中, 但节点本身还没拆
                    return child, i + cmp_len
                return node, i
            # 完全吃掉了 child 的 edge_tokens, 继续往下
            i += cmp_len
            node.last_used = next(_clock)
            node = child
        node.last_used = next(_clock)
        return node, i

    # ------------------------------------------------------------- #
    # insert: 把序列插进去, 必要时 split                             #
    # ------------------------------------------------------------- #

    def insert(self, tokens: List[int], kv_handle: int) -> RadixNode:
        node = self.root
        i = 0
        while i < len(tokens):
            child = node.children.get(tokens[i])
            if child is None:
                # 新分支
                new_node = RadixNode(
                    edge_tokens=tokens[i:],
                    kv_handle=kv_handle,
                    parent=node,
                )
                node.children[tokens[i]] = new_node
                return new_node
            # 算公共前缀长度
            cmp_len = 0
            for a, b in zip(child.edge_tokens, tokens[i:]):
                if a != b:
                    break
                cmp_len += 1
            if cmp_len == len(child.edge_tokens):
                # 完全吃掉 child, 继续向下
                i += cmp_len
                node = child
                continue
            # 部分匹配 → split child
            split_node = RadixNode(
                edge_tokens=child.edge_tokens[:cmp_len],
                kv_handle=None,            # split 出来的中间节点不持有 kv
                parent=node,
            )
            child.edge_tokens = child.edge_tokens[cmp_len:]
            child.parent = split_node
            split_node.children[child.edge_tokens[0]] = child
            node.children[tokens[i]] = split_node
            i += cmp_len
            node = split_node
            # 继续匹配 tokens 剩余部分
        return node

    # ------------------------------------------------------------- #
    # 引用 / LRU                                                    #
    # ------------------------------------------------------------- #

    def lock_path(self, node: RadixNode) -> None:
        """从 node 一路向 root, 每个节点 ref_count++。"""
        cur: Optional[RadixNode] = node
        while cur is not None and not cur.is_root():
            cur.ref_count += 1
            cur.last_used = next(_clock)
            cur = cur.parent

    def unlock_path(self, node: RadixNode) -> None:
        cur: Optional[RadixNode] = node
        while cur is not None and not cur.is_root():
            cur.ref_count -= 1
            cur = cur.parent

    def evict_lru(self, k: int = 1) -> List[int]:
        """驱逐 k 个 ref_count==0 的叶子, 返回它们的 kv_handle (调用方负责释放)。"""
        evicted = []
        for _ in range(k):
            leaf = self._pick_lru_leaf(self.root)
            if leaf is None:
                break
            kv = leaf.kv_handle
            parent = leaf.parent
            if parent is not None:
                # 从父节点 children 中移除
                key = leaf.edge_tokens[0] if leaf.edge_tokens else None
                if key is not None and key in parent.children:
                    del parent.children[key]
            if kv is not None:
                evicted.append(kv)
        return evicted

    def _pick_lru_leaf(self, node: RadixNode) -> Optional[RadixNode]:
        """递归找全树中 ref==0 且 last_used 最小的叶子。"""
        if node.is_leaf() and not node.is_root() and node.ref_count == 0:
            return node
        best: Optional[RadixNode] = None
        for child in node.children.values():
            cand = self._pick_lru_leaf(child)
            if cand is None:
                continue
            if best is None or cand.last_used < best.last_used:
                best = cand
        return best

    # ------------------------------------------------------------- #
    # 诊断: 打印整棵树                                              #
    # ------------------------------------------------------------- #

    def pretty(self) -> str:
        lines: List[str] = []
        def walk(node: RadixNode, depth: int) -> None:
            tag = (
                "ROOT" if node.is_root()
                else f"{node.edge_tokens}  ref={node.ref_count}  kv={node.kv_handle}"
            )
            lines.append("  " * depth + "└─ " + tag)
            for c in node.children.values():
                walk(c, depth + 1)
        walk(self.root, 0)
        return "\n".join(lines)
