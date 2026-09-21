"""文件记忆 + 上下文瘦身的三个档位。

没有它: 长会话必然撑爆窗口; 而"爆了再截断"会把最重要的早期目标一起截掉。
关键设计 —— 由便宜到贵, 逐级降级:
  1. clear_tool_results  旧工具结果换成占位符 (无损于对话结构, 零模型开销)
  2. summarize_with_llm  让模型写结构化摘要, 替换掉旧历史 (有损, 花一次模型调用)
  3. truncate_messages   头尾保留 + 中间硬截 (最差: 消息被拍平成文本, tool_use/tool_result 结构全部抹掉, 仅作反例)
对应: Claude API 的 context editing (clear_tool_uses) 与 compaction; Claude Code 的 /compact 和 CLAUDE.md。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Tuple

from llm_agent.core.schema import Message
from llm_agent.core.utils import tokenize

COMPACT_PROMPT = "[compact] 请把以上对话压缩成结构化摘要: 目标 / 已完成 / 关键结果。"


class FileMemory:
    """CLAUDE.md 的迷你版: 记忆就是 markdown 文件 —— 可读、可改、可进版本库。"""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def add(self, title: str, body: str) -> Path:
        safe = re.sub(r"[^a-zA-Z0-9_\-一-鿿]+", "_", title).strip("_")
        path = self.root / f"{safe or 'memory'}.md"
        path.write_text(f"# {title}\n\n{body.strip()}\n", encoding="utf-8")
        return path

    def search(self, query: str, limit: int = 3) -> List[Tuple[str, str]]:
        q = set(tokenize(query))
        hits = []
        for path in sorted(self.root.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            score = len(q & set(tokenize(text)))
            if score:
                hits.append((score, path.name, text.strip()))
        hits.sort(reverse=True)
        return [(name, text) for _, name, text in hits[:limit]]


def memory_messages(memory: FileMemory, query: str) -> List[Message]:
    # 每轮按当前 prompt 现查现拼, 不写进 transcript: 记忆文件改了, 下一轮立刻生效
    return [Message("system", f"Memory {name}:\n{text}", name="memory") for name, text in memory.search(query)]


def total_chars(messages: Iterable[Message]) -> int:
    return sum(len(m.text) for m in messages)


def clear_tool_results(messages: List[Message], keep_last: int = 1) -> List[Message]:
    """只保留最近 keep_last 个工具结果的正文, 其余换成占位符。返回新列表, 不改原消息。

    工具结果通常是上下文里最胖、也最快过时的部分; tool_use/tool_result 的配对结构原样保留。
    """
    ids = [b["tool_use_id"] for m in messages for b in m.tool_results()]
    stale = set(ids[:-keep_last] if keep_last else ids)
    out = []
    for m in messages:
        if not any(b["tool_use_id"] in stale for b in m.tool_results()):
            out.append(m)
            continue
        blocks = [
            {**b, "content": f"[cleared: {len(str(b['content']))} chars]"}
            if b["type"] == "tool_result" and b["tool_use_id"] in stale
            else b
            for b in m.blocks
        ]
        out.append(Message(m.role, blocks, m.name))
    return out


def summarize_with_llm(llm, messages: List[Message], keep: str = "") -> Message:
    """压缩 = 再调一次模型。keep 来自 pre_compact hook: 人指定"摘要里必须留下什么"。"""
    ask = Message("user", COMPACT_PROMPT + (f"\n必须保留: {keep}" if keep else ""))
    summary = llm.next(list(messages) + [ask], []).content
    return Message("system", f"[compact summary of {len(messages)} messages]\n{summary}", name="compact_summary")


def truncate_messages(messages: List[Message], max_chars: int) -> List[Message]:
    """反例基线: 头 2 条 + 尾 2 条, 中间每条只留 32 字符。便宜, 但语义和配对结构都会坏。"""
    if total_chars(messages) <= max_chars or len(messages) <= 4:
        return list(messages)
    edge = max(40, max_chars // 6)
    clip = lambda m: Message(m.role, m.text[:edge], m.name)  # noqa: E731
    head, tail, middle = [clip(m) for m in messages[:2]], [clip(m) for m in messages[-2:]], messages[2:-2]
    room = max(80, max_chars - total_chars(head) - total_chars(tail) - 80)
    gist = " | ".join(f"{m.role}:{m.text[:32]}" for m in middle)[:room]
    return head + [Message("system", f"[truncated {len(middle)} messages] {gist}", name="compact_summary")] + tail
