"""
M08 — Retrieval Memory (RAG-lite)

上下文窗口装不下全部知识。检索增强 (RAG) 的分工是:
    知识放外部索引 (文档库 / 代码库 / 笔记) → 按需取 top-k → 拼进上下文。

m02/m04 的 SearchDocsTool 与 FileMemory.search 用的是**关键词计数**:
高频词 ("how", "model") 和真正的关键词等权, 一篇凑满常见词的文档
就能挤掉正确答案。本模块实现最小 **TF-IDF 向量检索**:

    - 文档 → 稀疏词向量, 词权重 = tf × idf (越罕见的词信息量越大)
    - 查询 → 同一空间的向量, 余弦相似度排序
    - 把 embed() 换成神经网络 embedding, 就是工业级语义检索 —— 接口不变

工具接口不变 (仍叫 search_docs), 对 agent/模型完全透明 —— 这正是
"工具抽象"的价值: 检索质量升级不需要动 agent 循环一行代码。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Tuple

from llm_agent.core import Agent, PermissionGate, RuleBasedLLM, ToolRegistry
from llm_agent.core.schema import ToolResult
from llm_agent.core.tools import SearchDocsTool, Tool
from llm_agent.core.utils import banner, kv


def _words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class TfidfIndex:
    """最小 TF-IDF 索引: 无依赖, ~30 行, 但排序行为已经接近真实检索器。"""

    def __init__(self, docs: Dict[str, str]) -> None:
        self.docs = docs
        n = len(docs)
        df = Counter()
        self._doc_tf: Dict[str, Counter] = {}
        for title, body in docs.items():
            tokens = _words(f"{title} {body}")
            self._doc_tf[title] = Counter(tokens)
            df.update(set(tokens))
        # idf: 出现在越少文档里的词, 权重越高 (+1 平滑防止除零)
        self.idf = {w: math.log(n / (1 + c)) + 1.0 for w, c in df.items()}
        self._doc_vec = {t: self._vectorize(tf) for t, tf in self._doc_tf.items()}

    def _vectorize(self, tf: Counter) -> Dict[str, float]:
        vec = {w: c * self.idf.get(w, 1.0) for w, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {w: v / norm for w, v in vec.items()}

    def embed(self, text: str) -> Dict[str, float]:
        return self._vectorize(Counter(_words(text)))

    def search(self, query: str, k: int = 3) -> List[Tuple[float, str]]:
        q = self.embed(query)
        scored = []
        for title, vec in self._doc_vec.items():
            cos = sum(q[w] * vec.get(w, 0.0) for w in q)   # 稀疏点积 = 余弦
            scored.append((cos, title))
        scored.sort(reverse=True)
        return scored[:k]


class VectorSearchTool(Tool):
    """TF-IDF 版 search_docs — 与关键词版同名同 schema, 可直接热替换。"""

    name = "search_docs"
    description = "Search docs by TF-IDF cosine similarity (top-k with scores)."
    reversible = True
    risk = "low"

    def __init__(self, index: TfidfIndex) -> None:
        self.index = index

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        query = str(args.get("query", ""))
        hits = self.index.search(query, k=3)
        if not hits or hits[0][0] <= 0:
            return ToolResult(self.name, "no matches")
        lines = [f"[{score:.2f}] {title}: {self.index.docs[title]}" for score, title in hits]
        return ToolResult(self.name, "\n".join(lines))


DOCS = {
    "paged_attention": "paged attention fixes memory fragmentation in the kv cache "
                       "with block tables, like virtual memory pages for the model.",
    "kv_cache": "the kv cache is the model memory of past keys and values, "
                "decode becomes incremental.",
    "lora": "lora finetunes a model with low rank adapters, under one percent of weights.",
    "dpo": "dpo aligns a model with preference pairs, no reward model needed.",
    "ring_attention": "ring attention passes kv blocks between devices so long "
                      "context fits in device memory.",
    "sampling": "temperature and top k shape how a model samples the next token.",
}


def main() -> None:
    banner("M08 - Retrieval Memory (RAG-lite)")

    query = "model memory fragmentation"
    index = TfidfIndex(DOCS)

    # ---- 1) 关键词计数 vs TF-IDF 余弦 ----
    print("\n[1] 同一查询, 两种检索的排序")
    kv("query", query)

    # 关键词版: 整数命中数, 'model' 这种 6 篇里 5 篇都有的词与
    # 'fragmentation' 这种唯一关键词各算 1 分 → 大量并列, 排不出次序
    keyword_tool = SearchDocsTool(DOCS)
    print("\n  关键词计数 (m02 同款, 整数分):")
    print("    " + keyword_tool.execute({"query": query}).output.replace("\n", "\n    ")[:240])

    print("\n  TF-IDF 余弦 (连续分):")
    for score, title in index.search(query, k=4):
        print(f"    [{score:.2f}] {title}")

    top1 = index.search(query, k=1)[0][1]
    assert top1 == "paged_attention", f"TF-IDF 应命中 paged_attention, 实际 {top1}"
    print("\n  idf 把烂大街的 'model' 压到 ~0 权重, 罕见词 'fragmentation' 主导;")
    print("  连续分数把关键词版分不开的并列文档拉出了梯度。")

    # ---- 2) 热替换进 agent: 工具名不变, agent 循环零改动 ----
    print("\n[2] 把 TF-IDF 检索接进 agent (工具名仍是 search_docs)")
    agent = Agent(
        llm=RuleBasedLLM(),
        tools=ToolRegistry([VectorSearchTool(index)]),
        permissions=PermissionGate(mode="auto"),
        max_turns=4,
        name="m08",
    )
    final = agent.run(f"检索: {query}", verbose=True)
    kv("final", final.splitlines()[0] + " ...")

    print("\n  OK: 检索是 agent 的可扩展长期记忆; 关键词 → TF-IDF → 神经 embedding,")
    print("      接口不变, 质量逐级升级 —— 真实系统 (Claude Code 的 grep/语义检索) 同理。")


if __name__ == "__main__":
    main()
