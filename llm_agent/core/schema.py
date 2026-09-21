"""共享数据结构: transcript 用 Claude Messages API 同款 content block。

没有它: 助手 "我要调用哪个工具" 这一步不进 transcript, JSONL 只剩孤零零的工具结果,
既没法审计, 也没法原样喂给真实 API (tool_result 必须紧跟同 id 的 tool_use)。
关键设计: content 要么是 str, 要么是 block 列表:
    assistant: {"type":"tool_use","id","name","input"}
    user     : {"type":"tool_result","tool_use_id","content","is_error"}
一个 assistant turn 可带多个 tool_use (并行工具调用), 其结果必须放进同一条 user 消息。
对应: Anthropic Messages API 的 tool use 消息格式。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

Block = Dict[str, Any]
Content = Union[str, List[Block]]


@dataclass
class Message:
    role: str  # system / user / assistant
    content: Content
    name: Optional[str] = None  # 仅标记 harness 注入的 system 消息来源 (memory / hook_context ...)

    @property
    def blocks(self) -> List[Block]:
        if isinstance(self.content, str):
            return [{"type": "text", "text": self.content}]
        return self.content

    @property
    def text(self) -> str:
        """拍平成纯文本, 供打印 / 计长度 / toy LLM 使用。"""
        parts = []
        for b in self.blocks:
            if b["type"] == "text":
                parts.append(b["text"])
            elif b["type"] == "tool_use":
                parts.append(f"[tool_use {b['name']} {json.dumps(b['input'], ensure_ascii=False)}]")
            elif b["type"] == "tool_result":
                parts.append(str(b["content"]))
        return "\n".join(parts)

    def tool_uses(self) -> List[Block]:
        return [b for b in self.blocks if b["type"] == "tool_use"]

    def tool_results(self) -> List[Block]:
        return [b for b in self.blocks if b["type"] == "tool_result"]

    @property
    def is_user_prompt(self) -> bool:
        """真正的用户输入; 携带 tool_result 的 user 消息不算。"""
        return self.role == "user" and not self.tool_results()

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
    id: str = ""  # 真实模型自带 id; toy LLM 留空, 由 agent 分配

    def to_block(self) -> Block:
        return {"type": "tool_use", "id": self.id, "name": self.name, "input": self.args}


@dataclass
class ToolResult:
    name: str
    output: str
    ok: bool = True
    tool_use_id: str = ""

    def to_block(self) -> Block:
        return {
            "type": "tool_result",
            "tool_use_id": self.tool_use_id,
            "content": self.output,
            "is_error": not self.ok,  # 失败也要回填, 不能丢: 模型靠它自我纠正
        }


@dataclass
class ModelAction:
    kind: str  # "tool" | "final"
    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    raw_content: Optional[List[Block]] = None  # 真实 API 的原始 blocks (含 thinking), 必须原样回传

    @property
    def tool_call(self) -> Optional[ToolCall]:
        return self.tool_calls[0] if self.tool_calls else None

    @classmethod
    def tool(cls, *calls: ToolCall, text: str = "") -> "ModelAction":
        return cls(kind="tool", content=text, tool_calls=list(calls))

    @classmethod
    def final(cls, text: str) -> "ModelAction":
        return cls(kind="final", content=text)


def validate_transcript(messages: List[Message]) -> List[str]:
    """检查 tool_use / tool_result 配对, 返回问题列表 (空 = 可直接发给真实 API)。"""
    problems = []
    for i, msg in enumerate(messages):
        uses = [b["id"] for b in msg.tool_uses()] if msg.role == "assistant" else []
        if uses:
            nxt = messages[i + 1] if i + 1 < len(messages) else None
            got = [b["tool_use_id"] for b in nxt.tool_results()] if nxt and nxt.role == "user" else []
            if sorted(uses) != sorted(got):
                problems.append(f"msg[{i}] tool_use {uses} 没有在下一条 user 消息里配齐 tool_result {got}")
        if msg.role == "user" and msg.tool_results():
            prev = messages[i - 1] if i else None
            prev_ids = [b["id"] for b in prev.tool_uses()] if prev and prev.role == "assistant" else []
            for b in msg.tool_results():
                if b["tool_use_id"] not in prev_ids:
                    problems.append(f"msg[{i}] 孤儿 tool_result {b['tool_use_id']}")
    return problems
