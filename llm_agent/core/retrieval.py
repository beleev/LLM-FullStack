"""TF-IDF 向量检索: 稀疏向量 + 余弦相似度, 支持中文。

没有它: 关键词计数把 "the / model" 和 "fragmentation" 等权对待, 凑满常见词的文档能挤掉正确答案;
只认 [a-z0-9]+ 的分词器会让中文查询得分恒为 0。
关键设计:
  - idf 用 BM25 同款 ln(1 + (N-df+0.5)/(df+0.5)): 几乎每篇都有的词权重趋近 0, 罕见词主导排序。
  - 分词: 英文按词, 中文按字符 bigram (utils.tokenize)。
  - embed() 是唯一需要替换的点: 换成神经 embedding 就是稠密语义检索, search() 与工具接口不变。
对应: RAG 的检索层; Claude Code 的 grep / glob 走的是同一思路 (按需取回, 而非预先塞满上下文)。
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Tuple

from llm_agent.core.schema import ToolResult
from llm_agent.core.tools import Tool, _obj
from llm_agent.core.utils import tokenize


class TfidfIndex:
    def __init__(self, docs: Dict[str, str]) -> None:
        self.docs = docs
        n = len(docs)
        df: Counter = Counter()
        tfs = {}
        for title, body in docs.items():
            tokens = tokenize(f"{title} {body}")
            tfs[title] = Counter(tokens)
            df.update(set(tokens))
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self._doc_vec = {t: self._vectorize(tf) for t, tf in tfs.items()}

    def _vectorize(self, tf: Counter) -> Dict[str, float]:
        # 1+ln(tf): 同一个词重复 3 次不该有 3 倍话语权; 语料外的词没有方向, 直接丢
        vec = {w: (1 + math.log(c)) * self.idf[w] for w, c in tf.items() if w in self.idf}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {w: v / norm for w, v in vec.items()}

    def embed(self, text: str) -> Dict[str, float]:
        return self._vectorize(Counter(tokenize(text)))

    def search(self, query: str, k: int = 3) -> List[Tuple[float, str]]:
        q = self.embed(query)
        scored = [(sum(q[w] * vec.get(w, 0.0) for w in q), title) for title, vec in self._doc_vec.items()]
        scored.sort(reverse=True)  # 两边都已归一化, 稀疏点积 = 余弦
        return scored[:k]


class VectorSearchTool(Tool):
    """与关键词版 SearchDocsTool 同名同 schema: 升级检索质量不用动 agent loop 和模型侧任何东西。"""

    name = "search_docs"
    description = "Search docs by TF-IDF cosine similarity (top-k with scores)."
    parameters = _obj(["query"], query={"type": "string"})

    def __init__(self, index: TfidfIndex, k: int = 3) -> None:
        self.index, self.k = index, k

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        hits = [(s, t) for s, t in self.index.search(args["query"], self.k) if s > 0]
        if not hits:
            return ToolResult(self.name, "no matches")
        return ToolResult(self.name, "\n".join(f"[{s:.2f}] {t}: {self.index.docs[t]}" for s, t in hits))
