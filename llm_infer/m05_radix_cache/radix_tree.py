"""
radix_tree.py — 用基数树索引 KV cache, 命中任意长度的公共前缀 (SGLang RadixAttention)。

瓶颈: 多轮对话 / few-shot / 共享 system prompt 的请求前缀大量重复, 重复 prefill 浪费算力和 TTFT。
做法: 边上存一段 token + 等长的 KV 槽位号 `slots`; 新请求沿树走到最长公共前缀, 这段 KV 直接复用。
关键数字: 命中 n 个 token → prefill 少算 n 个; m04 的 block hash 只能命中 ⌊n/block⌋·block 个。
读代码盯住: `_split` — 在边中间分叉时把 (edge_tokens, slots) 同步切两半, ref_count 要继承。
真实系统: SGLang `RadixCache` (match_prefix / insert / evict / inc_lock_ref), TreeNode.value 就是这里的 slots。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import itertools

_clock = itertools.count()  # 逻辑时钟: LRU 只需要先后顺序, 不需要真实时间


@dataclass(eq=False)
class RadixNode:
    edge_tokens: List[int]                       # 父 → 本节点这条边上的 token 串; root 为 []
    slots: List[int]                             # 与 edge_tokens 等长: 每个 token 的 KV 在 pool 里的槽位
    children: Dict[int, "RadixNode"] = field(default_factory=dict)  # key = 子边的首 token (兄弟边首 token 必不同)
    parent: Optional["RadixNode"] = None
    ref_count: int = 0                           # 有多少个在跑的请求正用着这段 KV; >0 不可驱逐
    last_used: int = field(default_factory=lambda: next(_clock))


def _lcp(a: List[int], b: List[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class RadixCache:
    def __init__(self) -> None:
        self.root = RadixNode(edge_tokens=[], slots=[])

    def _split(self, child: RadixNode, n: int) -> RadixNode:
        """把 child 的边在第 n 个 token 处切开: parent → mid(前 n 个) → child(剩余)。返回 mid。"""
        mid = RadixNode(child.edge_tokens[:n], child.slots[:n], parent=child.parent,
                        ref_count=child.ref_count,       # 锁着 child 的请求必然也经过 mid, 不继承的话 unlock 会减成负数
                        last_used=child.last_used)
        child.parent.children[mid.edge_tokens[0]] = mid  # 首 token 没变, 直接顶替 child 的位置
        child.edge_tokens, child.slots = child.edge_tokens[n:], child.slots[n:]
        child.parent = mid
        mid.children[child.edge_tokens[0]] = child
        return mid

    def match(self, tokens: List[int]) -> Tuple[RadixNode, List[int]]:
        """最长前缀匹配 → (停下的节点, 命中 token 的 slots); 命中长度 = len(slots)。

        停在边中间时就地 split, 保证返回的节点恰好代表命中的前缀 — 这样 lock_path 才不会多锁。
        """
        node, i, slots = self.root, 0, []
        while i < len(tokens):
            child = node.children.get(tokens[i])
            if child is None:
                break
            n = _lcp(child.edge_tokens, tokens[i:])      # ≥1, 因为首 token 已相等
            if n < len(child.edge_tokens):
                child = self._split(child, n)
            child.last_used = next(_clock)               # 沿途都刷新: 父节点永远不比子节点旧
            slots += child.slots
            i += n
            node = child                                 # 若刚 split 过, 下一轮 children.get 必落空 → 自然停下
        return node, slots

    def insert(self, tokens: List[int], slots: List[int]) -> int:
        """插入 (tokens, 它们的 KV slots)。返回树里已有的前缀长度 n:
        slots[:n] 与树上已有 KV 重复, 调用方应当释放它们; 只有 slots[n:] 挂到新叶子上。"""
        assert len(tokens) == len(slots)
        node, hit = self.match(tokens)
        n = len(hit)
        if n < len(tokens):
            leaf = RadixNode(tokens[n:], slots[n:], parent=node)
            node.children[tokens[n]] = leaf
        return n

    def lock_path(self, node: RadixNode) -> None:
        """请求开始使用 node 代表的前缀: node → root 整条路径 ref_count+1。"""
        while node.parent is not None:
            node.ref_count += 1
            node = node.parent

    def unlock_path(self, node: RadixNode) -> None:
        while node.parent is not None:
            node.ref_count -= 1
            assert node.ref_count >= 0
            node = node.parent

    def evict(self, n_tokens: int) -> List[int]:
        """按 LRU 驱逐未上锁的叶子, 直到释放 ≥ n_tokens 个槽位或无可驱逐。返回释放的 slots。

        只驱逐叶子: 砍掉中间节点会让子孙的 KV 失去前缀 (KV 依赖完整前缀, 单独留着没意义)。
        """
        freed: List[int] = []
        while len(freed) < n_tokens:
            # ponytail: 每次 O(N) 全树扫描; SGLang 用按 last_access_time 的小顶堆, 树大了再换
            leaves = [x for x in self._nodes() if not x.children and x.ref_count == 0]
            if not leaves:
                break
            leaf = min(leaves, key=lambda x: x.last_used)
            del leaf.parent.children[leaf.edge_tokens[0]]    # 父节点可能因此变成叶子, 下一轮成为候选
            freed += leaf.slots
        return freed

    def _nodes(self) -> List[RadixNode]:
        """除 root 外的全部节点。"""
        out, stack = [], list(self.root.children.values())
        while stack:
            x = stack.pop()
            out.append(x)
            stack += x.children.values()
        return out

    def total_tokens(self) -> int:
        return sum(len(x.edge_tokens) for x in self._nodes())

    def pretty(self) -> str:
        lines: List[str] = []

        def walk(node: RadixNode, depth: int) -> None:
            tag = "ROOT" if node.parent is None else f"{node.edge_tokens}  slots={node.slots}  ref={node.ref_count}"
            lines.append("  " * depth + "└─ " + tag)
            for c in node.children.values():
                walk(c, depth + 1)
        walk(self.root, 0)
        return "\n".join(lines)
