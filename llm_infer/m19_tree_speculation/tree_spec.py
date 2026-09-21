"""
m19 tree_spec.py — 树形投机 (Medusa / EAGLE-2 / SpecInfer 式): 一次 target forward 验一整棵 token 树

瓶颈: 链式 draft (m07) 第一个 token 猜错, 后面 K-1 个全废; target 那次 forward 的算力
      (decode 是带宽受限的, 多验几个 token 几乎不加延迟) 白白浪费。
做法: 每个节点保留 draft 的 top-k 候选, 长成一棵树; target 用 **tree attention mask**
      (节点只看 上下文 + 祖先 + 自己) 一次验完所有分支, 接受与 target greedy 一致的最长路径。
关键数字: widths=[3,2,1] → 1+3+6+6=16 个节点, 1 次 target 调用、3 次 draft 调用;
      首槽命中率从 top-1 提到 top-3。
读代码盯住: tree_mask 的 anc 矩阵、positions = n_ctx + depth (同深度的兄弟共享位置)、
      gather_kv (只留被接受路径的 KV 行)。
对应真实系统: SGLang EAGLE-2 的 build_tree / vLLM 的 tree attention; Medusa 的 medusa_mask。
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from llm_infer.core import TinyLM, truncate_kv


def tree_shape(widths: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """固定形状的树, BFS 编号, 节点 0 = 根 (= 已确定但还没喂给 target 的 out[-1])。

    widths[d] = 第 d 层每个节点的孩子数。返回 parents (n,), depth (n,); parents[0] = -1。
    [3,2,1] → parents = [-1, 0,0,0, 1,1,2,2,3,3, 4,5,6,7,8,9]
    """
    parents, depth, level = [-1], [0], [0]
    for d, w in enumerate(widths):
        nxt = []
        for p in level:
            for _ in range(w):
                nxt.append(len(parents))
                parents.append(p)
                depth.append(d + 1)
        level = nxt
    return np.array(parents), np.array(depth)


def ancestor_matrix(parents: np.ndarray) -> np.ndarray:
    """anc[i, j] = True ⇔ j 是 i 的祖先或 i 自己。(n, n) bool。BFS 序保证 parent 先于 child。"""
    n = len(parents)
    anc = np.eye(n, dtype=bool)
    for i in range(1, n):
        anc[i] |= anc[parents[i]]
    return anc


def tree_mask(anc: np.ndarray, n_ctx: int) -> np.ndarray:
    """加性 mask (n_rows, n_ctx + n_cols): 上下文全可见; 树内只看祖先 + 自己。"""
    tree = np.where(anc, 0.0, -np.inf).astype(np.float32)
    return np.concatenate([np.zeros((anc.shape[0], n_ctx), np.float32), tree], axis=1)


def gather_kv(kv_cache, idx: np.ndarray):
    """只保留 idx 指定的 KV 行。K 存的是 RoPE 之后的值, 位置信息已烙在行里, 所以挑行即可,
    被接受路径的 depth 连续 → 位置 n_ctx, n_ctx+1, ... 与顺序 decode 完全一致。"""
    return [(K[idx], V[idx]) for K, V in kv_cache]


class TreeDrafter:
    """用 draft LM 逐层长树: 第 d 层所有节点一次 forward (draft 自己也用 tree mask)。"""

    def __init__(self, lm: TinyLM, widths: List[int]):
        self.lm, self.widths, self.kv, self.calls = lm, widths, None, 0
        self.parents, self.depth = tree_shape(widths)
        self.anc = ancestor_matrix(self.parents)

    def propose(self, out: List[int]) -> np.ndarray:
        """返回树上每个节点的 token (n,), tokens[0] = out[-1]。"""
        n_fed = 0 if self.kv is None else self.kv[0][0].shape[0]
        # 第 0 层: 因果地追平上轮接受的 token, 最后一行就是根
        logits, kv = self.lm.forward(out[n_fed:], self.kv)
        self.calls += 1
        n_ctx = len(out) - 1                              # 根在 draft KV 里的行号
        tokens, logits, s = [out[-1]], logits[-1:], 1     # logits: (上一层节点数, V)
        for d, w in enumerate(self.widths):
            top = np.argsort(-logits, axis=-1)[:, :w]     # (n_prev, w) 每个节点的 top-w 孩子
            tokens += top.ravel().tolist()                # 行优先展开 = BFS 序, 与 tree_shape 对齐
            e = s + top.size
            if d + 1 < len(self.widths):                  # 叶子层不用再 forward
                mask = tree_mask(self.anc[s:e, :e], n_ctx)               # (n_level, n_ctx + e)
                logits, kv = self.lm.forward(tokens[s:e], kv, positions=n_ctx + self.depth[s:e], mask=mask)
                self.calls += 1
            s = e
        # ponytail: draft 侧不做 gather, 直接丢掉整棵树的 KV, 下轮把接受的 token 重喂一遍
        # (并进第 0 层那次 forward, 不多花调用); 真实系统里 draft 也 gather
        self.kv = truncate_kv(kv, len(out))
        return np.array(tokens)


def accept_tree(tokens: np.ndarray, parents: np.ndarray, t_logits: np.ndarray) -> Tuple[List[int], int]:
    """从根往下走: 当前节点的 target argmax 若等于某个孩子的 token 就进入该孩子。

    t_logits (n, V): 第 i 行 = target 看完 [ctx + i 的祖先 + i] 后对"i 的下一个 token"的预测。
    返回 (被接受路径的节点号 [0, ...], next_token); 同 m07: next_token 是纠错或 bonus。
    """
    t_pred = t_logits.argmax(-1)                          # (n,)
    path = [0]
    while True:
        kids = np.flatnonzero(parents == path[-1])
        hit = kids[tokens[kids] == t_pred[path[-1]]]      # top-k 互不相同 → 至多命中一个
        if len(hit) == 0:
            return path, int(t_pred[path[-1]])
        path.append(int(hit[0]))


def tree_speculative_decode(target: TinyLM, drafter: TreeDrafter, prompt, max_new: int):
    """返回 (prompt+新 token, target forward 次数, 每轮 trace)。不变量同 m07: kv 覆盖 out[:-1]。"""
    logits, kv = target.forward(prompt)                   # prefill
    target_calls = 1
    out = list(map(int, prompt)) + [int(np.argmax(logits[-1]))]
    parents, depth = drafter.parents, drafter.depth
    trace = []

    while len(out) - len(prompt) < max_new:
        n_ctx = len(out) - 1
        tokens = drafter.propose(out)                                        # (n,)
        logits, kv = target.forward(tokens, kv, positions=n_ctx + depth,     # 兄弟节点共享位置
                                    mask=tree_mask(drafter.anc, n_ctx))      # (n, V)
        target_calls += 1
        path, nxt = accept_tree(tokens, parents, logits)
        kv = gather_kv(kv, np.concatenate([np.arange(n_ctx), n_ctx + np.array(path)]))
        out += tokens[path[1:]].tolist() + [nxt]
        trace.append({"tokens": tokens, "path": path, "t_pred": logits.argmax(-1)})

    return out[: len(prompt) + max_new], target_calls, trace


def render_tree(tokens: np.ndarray, parents: np.ndarray, path: List[int]) -> str:
    """ASCII 画树, 被接受路径上的节点标 ✓。"""
    lines = [f"[根 {tokens[0]}] ✓  (= out[-1], 已确定)"]

    def walk(node: int, prefix: str) -> None:
        kids = np.flatnonzero(parents == node)
        for k in kids:
            last = k == kids[-1]
            lines.append(f"{prefix}{'└─ ' if last else '├─ '}{tokens[k]}{' ✓' if k in path else ''}")
            walk(k, prefix + ("   " if last else "│  "))

    walk(0, "")
    return "\n".join(lines)
