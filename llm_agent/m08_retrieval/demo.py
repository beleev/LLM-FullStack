"""M08 — Retrieval (RAG-lite): 知识放在上下文之外, 按需取 top-k。

没有它: 要么把整个知识库塞进上下文 (装不下、也贵), 要么模型凭记忆编。
关键设计:
  - 关键词计数把每个命中词算 1 分 —— 一篇凑满 "how / the / model" 的文档能挤掉正确答案。
  - TF-IDF: 词权重 = tf × idf, 罕见词主导; 余弦相似度给出连续分数。
  - 中文没有空格: 分词器只认 [a-z0-9]+ 时, 中文查询得分恒为 0 → 用字符 bigram。
  - 工具名和 schema 不变 (仍是 search_docs): 检索升级对 agent loop 和模型完全透明。
实现在 core/retrieval.py; 把 embed() 换成神经 embedding 就是稠密语义检索。
对应: RAG 检索层; Claude Code 的 grep/glob 式按需检索。
"""

from __future__ import annotations

import re

from llm_agent.core import Agent, PermissionGate, RuleBasedLLM, SearchDocsTool, TfidfIndex, ToolRegistry, VectorSearchTool
from llm_agent.core.utils import banner, kv, tokenize

DOCS = {
    "paged_attention": "paged attention fixes memory fragmentation in the kv cache with block tables, "
    "like virtual memory pages for the model.",
    "kv_cache": "how does decode stay fast: the kv cache is the model memory of past keys and values.",
    "faq": "how does the model work, how does the team manage the model, and how does the budget get set.",
    "lora": "how does the team manage finetuning cost: lora trains low rank adapters for the model.",
    "dpo": "how does the team manage alignment: dpo tunes the model on preference pairs, no reward model.",
    "sampling": "how does the model pick the next token: temperature and top k control randomness.",
    "分页注意力": "分页注意力用块表管理 KV 缓存, 解决显存碎片问题, 思路类似操作系统的虚拟内存分页。",
    "低秩微调": "LoRA 只训练低秩适配器, 可训练参数不到百分之一, 显存占用大幅下降。",
}


def main() -> None:
    banner("M08 - Retrieval (RAG-lite)")
    index = TfidfIndex(DOCS)
    query = "how does the model manage memory fragmentation"

    print("\n[1] 同一查询, 两种排序")
    kv("query", query)
    keyword_top = SearchDocsTool(DOCS).execute({"query": query}).output.split(":")[0]
    hits = {t: len(set(tokenize(query)) & set(tokenize(f"{t} {b}"))) for t, b in DOCS.items()}
    kv("关键词命中数", {t: n for t, n in hits.items() if n >= 4})
    kv("关键词计数 top-1", keyword_top)
    ranked = index.search(query, k=3)
    for score, title in ranked:
        print(f"    tf-idf [{score:.2f}] {title}")
    # 凑满 how/does/the/model/manage 的文档并列 5 分, 唯一提到 fragmentation 的正确答案只有 4 分
    assert hits["paged_attention"] < hits[keyword_top] and keyword_top != "paged_attention"
    assert ranked[0][1] == "paged_attention" and ranked[0][0] > 1.5 * ranked[1][0]

    print("\n[2] 为什么: idf 给每个词的权重")
    for word in ("the", "model", "how", "memory", "fragmentation"):
        print(f"    idf({word:<13}) = {index.idf[word]:.2f}")
    assert index.idf["fragmentation"] > 4 * index.idf["model"]  # 罕见词的话语权是烂大街词的数倍

    print("\n[3] 中文查询: bigram 分词")
    zh = "显存碎片怎么解决"
    kv("旧分词 [a-z0-9]+", re.findall(r"[a-z0-9]+", zh))
    kv("bigram", tokenize(zh))
    score, title = index.search(zh, k=1)[0]
    kv("top-1", f"[{score:.2f}] {title}")
    assert re.findall(r"[a-z0-9]+", zh) == [] and title == "分页注意力" and score > 0

    print("\n[4] 热替换进 agent: 工具名不变, loop 零改动")
    agent = Agent(RuleBasedLLM(), ToolRegistry([VectorSearchTool(index)]), PermissionGate("auto"), max_turns=4, name="m08")
    final = agent.run(f"检索: {query}")
    assert final.splitlines()[1].startswith("search_docs: [") and "paged_attention" in final.splitlines()[1]

    print("\n  OK: 关键词 → TF-IDF → 神经 embedding, 接口不变, 质量逐级升级。")


if __name__ == "__main__":
    main()
