"""Shared data structures.

The real products use richer protocol objects.  For teaching, four dataclasses
are enough to show the loop:

    user/assistant/tool messages -> model action -> tool result -> next turn
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Message:
    role: str
    content: str
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"role": self.role, "content": self.content, "name": self.name}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        return cls(role=data["role"], content=data["content"], name=data.get("name"))


@dataclass
class ToolCall:
    name: str
    args: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""


@dataclass
class ToolResult:
    name: str
    output: str
    ok: bool = True


@dataclass
class ModelAction:
    kind: str  # "tool" or "final"
    content: str = ""
    tool_call: Optional[ToolCall] = None

    @classmethod
    def tool(cls, call: ToolCall) -> "ModelAction":
        return cls(kind="tool", tool_call=call)

    @classmethod
    def final(cls, text: str) -> "ModelAction":
        return cls(kind="final", content=text)

