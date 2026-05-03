"""Lifecycle hooks.

Hooks are deterministic code around the model loop.  They are useful when a
team wants policy or context shaping without spending model context tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from llm_agent.core.schema import ToolCall, ToolResult


@dataclass
class HookResult:
    block: bool = False
    reason: str = ""
    updated_call: Optional[ToolCall] = None
    additional_context: str = ""


class HookManager:
    def __init__(self) -> None:
        self._hooks: Dict[str, List[Callable]] = {
            "session_start": [],
            "user_prompt_submit": [],
            "pre_tool_use": [],
            "post_tool_use": [],
            "pre_compact": [],
        }

    def register(self, event: str, fn: Callable) -> None:
        if event not in self._hooks:
            raise ValueError(f"unknown hook event: {event}")
        self._hooks[event].append(fn)

    def on_session_start(self) -> List[str]:
        outputs = []
        for fn in self._hooks["session_start"]:
            out = fn()
            if out:
                outputs.append(str(out))
        return outputs

    def on_user_prompt_submit(self, prompt: str) -> HookResult:
        current = prompt
        extra = []
        for fn in self._hooks["user_prompt_submit"]:
            result = fn(current)
            if result.block:
                return result
            if result.additional_context:
                extra.append(result.additional_context)
        if extra:
            current = current + "\n" + "\n".join(extra)
        return HookResult(additional_context=current)

    def on_pre_tool_use(self, call: ToolCall) -> HookResult:
        current = call
        for fn in self._hooks["pre_tool_use"]:
            result = fn(current)
            if result.block:
                return result
            if result.updated_call:
                current = result.updated_call
        return HookResult(updated_call=current)

    def on_post_tool_use(self, result: ToolResult) -> str:
        extras = []
        for fn in self._hooks["post_tool_use"]:
            extra = fn(result)
            if extra:
                extras.append(str(extra))
        return "\n".join(extras)

    def on_pre_compact(self, text: str) -> str:
        current = text
        for fn in self._hooks["pre_compact"]:
            updated = fn(current)
            if updated:
                current = str(updated)
        return current

