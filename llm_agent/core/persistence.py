"""Append-only JSONL session transcripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from llm_agent.core.schema import Message


class JsonlSessionStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, message: Message) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(message.to_dict(), ensure_ascii=False) + "\n")

    def load(self) -> List[Message]:
        if not self.path.exists():
            return []
        messages = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    messages.append(Message.from_dict(json.loads(line)))
        return messages

    def count(self) -> int:
        return len(self.load())

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()

