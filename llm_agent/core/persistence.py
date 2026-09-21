"""append-only JSONL 会话日志。

没有它: 进程一退, 会话全丢; 出了事故也无从复盘"模型当时到底调了什么"。
关键设计:
  - 只追加不改写: 崩溃最多坏最后一行 (load 时跳过); 无 fsync / 文件锁, 这是教学版的上限。
  - 压缩不删历史, 而是追加一条 compact_boundary 记录;
    load() (给 resume 用) 只返回"最后一个 boundary 的摘要 + 保留的尾部 + 之后的消息",
    load_all() (给审计用) 返回全部。于是: 文件只增不减, 恢复出来的上下文却真的变小了。
对应: Claude Code ~/.claude/projects/*.jsonl 会话文件与其中的 compact boundary。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, List

from llm_agent.core.schema import Message


class JsonlSessionStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, record: dict) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def append(self, message: Message) -> None:
        self._write(message.to_dict())

    def append_compact(self, summary: Message, kept: int) -> None:
        """kept = boundary 之前有多少条尾部消息原样保留 (当前这一轮)。"""
        self._write({"type": "compact_boundary", "summary": summary.to_dict(), "kept": kept})

    def _records(self) -> Iterator[dict]:
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        if line.strip():
                            yield json.loads(line)
                    except json.JSONDecodeError:  # 崩溃时写了一半的行: 跳过, 不让整个会话无法恢复
                        continue

    def load_all(self) -> List[Message]:
        return [Message.from_dict(r) for r in self._records() if r.get("type") != "compact_boundary"]

    def load(self) -> List[Message]:
        view: List[Message] = []
        for r in self._records():
            if r.get("type") == "compact_boundary":
                tail = view[-r["kept"] :] if r["kept"] else []
                view = [Message.from_dict(r["summary"])] + tail
            else:
                view.append(Message.from_dict(r))
        return view
