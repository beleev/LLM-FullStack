"""A deterministic toy LLM.

No API calls, no hidden model dependency.  The goal is to make the harness
visible: the same agent loop can run with this rule-based policy or a real LLM.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Set

from llm_agent.core.schema import Message, ModelAction, ToolCall


class RuleBasedLLM:
    def next(self, messages: List[Message], available_tools: Iterable[str]) -> ModelAction:
        tools = set(available_tools)
        prompt = self._latest_user(messages)
        used = self._used_tools(messages)
        lower = prompt.lower()
        last_tool = self._pending_tool_name(messages)

        # After a tool result, normally return a final answer.  The only
        # planned multi-step exception in this toy policy is "search -> note".
        if last_tool:
            if self._wants_note(lower) and "write_note" in tools and "write_note" not in used:
                text = self._last_tool_output(messages) or prompt
                return ModelAction.tool(ToolCall("write_note", {"text": text}, reason="persist a note"))
            outputs = self._tool_outputs(messages)
            return ModelAction.final("基于工具结果完成：\n" + outputs[-1])

        if self._wants_delegate(lower) and "delegate" in tools and "delegate" not in used:
            return ModelAction.tool(
                ToolCall("delegate", {"task": prompt}, reason="isolate a subtask in a child agent")
            )

        expr = self._extract_expr(prompt)
        if expr and "calculator" in tools and "calculator" not in used:
            return ModelAction.tool(ToolCall("calculator", {"expr": expr}, reason="need arithmetic"))

        if self._wants_search(lower) and "search_docs" in tools and "search_docs" not in used:
            return ModelAction.tool(
                ToolCall("search_docs", {"query": prompt}, reason="need external knowledge")
            )

        if self._wants_weather(lower) and "weather" in tools and "weather" not in used:
            city = "Shanghai"
            if "北京" in prompt or "beijing" in lower:
                city = "Beijing"
            return ModelAction.tool(ToolCall("weather", {"city": city}, reason="MCP-like tool"))

        if self._wants_shell(lower) and "shell" in tools and "shell" not in used:
            return ModelAction.tool(
                ToolCall("shell", {"command": self._extract_command(prompt)}, reason="user requested shell")
            )

        if self._wants_note(lower) and "write_note" in tools and "write_note" not in used:
            text = self._last_tool_output(messages) or prompt
            return ModelAction.tool(ToolCall("write_note", {"text": text}, reason="persist a note"))

        if self._wants_read_notes(lower) and "read_notes" in tools and "read_notes" not in used:
            return ModelAction.tool(ToolCall("read_notes", {}, reason="read notebook"))

        outputs = self._tool_outputs(messages)
        if outputs:
            return ModelAction.final("基于工具结果完成：\n" + outputs[-1])
        return ModelAction.final("这是一个无需工具的直接回答。")

    @staticmethod
    def _latest_user(messages: List[Message]) -> str:
        for msg in reversed(messages):
            if msg.role == "user":
                return msg.content
        return ""

    @staticmethod
    def _used_tools(messages: List[Message]) -> Set[str]:
        return {m.name for m in messages if m.role == "tool" and m.name}

    @staticmethod
    def _tool_outputs(messages: List[Message]) -> List[str]:
        return [f"{m.name}: {m.content}" for m in messages if m.role == "tool"]

    @staticmethod
    def _last_tool_output(messages: List[Message]) -> str:
        for msg in reversed(messages):
            if msg.role == "tool":
                return msg.content
        return ""

    @staticmethod
    def _pending_tool_name(messages: List[Message]) -> str:
        if messages and messages[-1].role == "tool":
            return messages[-1].name or ""
        return ""

    @staticmethod
    def _extract_expr(prompt: str) -> str:
        match = re.search(r"[-+]?\d+(?:\s*[-+*/]\s*[-+]?\d+)+", prompt)
        return match.group(0) if match else ""

    @staticmethod
    def _extract_command(prompt: str) -> str:
        match = re.search(r"(?:shell|运行|执行)[:：]?\s*(.+)", prompt, re.IGNORECASE)
        return match.group(1).strip() if match else "echo hello"

    @staticmethod
    def _wants_search(lower: str) -> bool:
        return any(x in lower for x in ["search", "搜索", "检索", "查找", "找文档", "排查", "调研"])

    @staticmethod
    def _wants_note(lower: str) -> bool:
        return any(x in lower for x in ["note", "笔记", "记住", "写入", "保存"])

    @staticmethod
    def _wants_read_notes(lower: str) -> bool:
        return any(x in lower for x in ["read note", "读取笔记", "查看笔记"])

    @staticmethod
    def _wants_weather(lower: str) -> bool:
        return any(x in lower for x in ["weather", "天气"])

    @staticmethod
    def _wants_shell(lower: str) -> bool:
        return any(x in lower for x in ["shell", "运行", "执行"])

    @staticmethod
    def _wants_delegate(lower: str) -> bool:
        return any(x in lower for x in ["delegate", "subagent", "子智能体", "委托"])
