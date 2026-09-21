"""生命周期 hooks: 包在模型循环外面的确定性代码。

没有它: 团队规范只能写进 prompt 里"求"模型遵守 —— 花 token 且不保证执行。
关键设计:
  - hook 只能 拦截 / 改写调用 / 追加上下文, 三者都不会污染工具数据本身:
    追加的文字作为独立 system 消息进 transcript, 不拼进用户 prompt 或 tool_result。
  - pre_tool_use 在权限门之前运行, 改写后的调用仍要过门 (hook 不是提权通道)。
对应: Claude Code hooks 的同名事件 SessionStart / UserPromptSubmit / PreToolUse /
PostToolUse / PreCompact / Stop / SubagentStop (真实的 Stop 还能阻止结束, 这里只做通知)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from llm_agent.core.schema import ToolCall, ToolResult

EVENTS = (
    "session_start",  # fn(source: "startup"|"resume") -> str | None
    "user_prompt_submit",  # fn(prompt) -> HookResult
    "pre_tool_use",  # fn(call) -> HookResult
    "post_tool_use",  # fn(result) -> str | None
    "pre_compact",  # fn(messages) -> str | None   压缩时必须保留的要点
    "stop",  # fn(final_text) -> None
    "subagent_stop",  # fn(agent_type, summary) -> None
)


@dataclass
class HookResult:
    block: bool = False
    reason: str = ""
    updated_call: Optional[ToolCall] = None
    additional_context: str = ""


class HookManager:
    def __init__(self) -> None:
        self._hooks: Dict[str, List[Callable]] = {event: [] for event in EVENTS}

    def register(self, event: str, fn: Callable) -> None:
        if event not in self._hooks:
            raise ValueError(f"unknown hook event: {event}")
        self._hooks[event].append(fn)

    def _collect(self, event: str, *args) -> List[str]:
        outputs = (fn(*args) for fn in self._hooks[event])
        return [str(out) for out in outputs if out]

    def on_session_start(self, source: str) -> List[str]:
        return self._collect("session_start", source)

    def on_user_prompt_submit(self, prompt: str) -> HookResult:
        extra = []
        for fn in self._hooks["user_prompt_submit"]:
            result = fn(prompt) or HookResult()  # 返回 None = 无意见
            if result.block:
                return result
            if result.additional_context:
                extra.append(result.additional_context)
        return HookResult(additional_context="\n".join(extra))

    def on_pre_tool_use(self, call: ToolCall) -> HookResult:
        current = call
        for fn in self._hooks["pre_tool_use"]:
            result = fn(current) or HookResult()
            if result.block:
                return result
            if result.updated_call:
                current = result.updated_call  # 链式改写: 后一个 hook 看到前一个的结果
        return HookResult(updated_call=current)

    def on_post_tool_use(self, result: ToolResult) -> str:
        return "\n".join(self._collect("post_tool_use", result))

    def on_pre_compact(self, messages: list) -> str:
        return "\n".join(self._collect("pre_compact", messages))

    def on_stop(self, final_text: str) -> None:
        self._collect("stop", final_text)

    def on_subagent_stop(self, agent_type: str, summary: str) -> None:
        self._collect("subagent_stop", agent_type, summary)
