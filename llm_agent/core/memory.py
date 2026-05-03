"""Transparent file-based memory and context compaction."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Tuple

from llm_agent.core.schema import Message


def _words(text: str) -> set:
    return {w.lower() for w in re.findall(r"[a-zA-Z0-9_\u4e00-\u9fff]+", text)}


class FileMemory:
    """A tiny version of CLAUDE.md / markdown memory.

    Memory is plain files: inspectable, editable, and version-controllable.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def add(self, title: str, body: str) -> Path:
        safe = re.sub(r"[^a-zA-Z0-9_\-\u4e00-\u9fff]+", "_", title).strip("_")
        path = self.root / f"{safe or 'memory'}.md"
        path.write_text(f"# {title}\n\n{body.strip()}\n", encoding="utf-8")
        return path

    def search(self, query: str, limit: int = 3) -> List[Tuple[str, str]]:
        q = _words(query)
        hits = []
        for path in sorted(self.root.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            score = len(q & _words(text))
            if score:
                hits.append((score, path.name, text.strip()))
        hits.sort(reverse=True)
        return [(name, text) for _, name, text in hits[:limit]]


def total_chars(messages: Iterable[Message]) -> int:
    return sum(len(m.content) for m in messages)


def compact_messages(messages: List[Message], max_chars: int) -> List[Message]:
    """Keep head and tail, summarize the middle.

    This mirrors the production idea: do the cheapest lossy operation before
    giving up or calling an expensive model summarizer.
    """

    if total_chars(messages) <= max_chars or len(messages) <= 4:
        return list(messages)

    per_edge_message = max(40, max_chars // 6)
    head = [_clip_message(m, per_edge_message) for m in messages[:2]]
    tail = [_clip_message(m, per_edge_message) for m in messages[-2:]]
    middle = messages[2:-2]
    # Leave room for head/tail and make the summary itself obey the remaining
    # budget.  This keeps the demo honest: compaction should actually shrink.
    remaining = max(80, max_chars - total_chars(head) - total_chars(tail) - 80)
    summary = " | ".join(f"{m.role}:{m.content[:32]}" for m in middle)
    if len(summary) > remaining:
        summary = summary[: remaining - 3] + "..."
    compact = Message(
        role="system",
        name="compact_summary",
        content=f"[compact summary of {len(middle)} messages] {summary}",
    )
    return head + [compact] + tail


def _clip_message(message: Message, limit: int) -> Message:
    if len(message.content) <= limit:
        return message
    return Message(
        role=message.role,
        name=message.name,
        content=message.content[: max(0, limit - 3)] + "...",
    )


def memory_messages(memory: FileMemory, query: str) -> List[Message]:
    snippets = []
    for name, text in memory.search(query):
        snippets.append(Message(role="system", name="memory", content=f"Memory {name}:\n{text}"))
    return snippets
