"""
M05 — Extensibility

三类常见扩展点:
    hooks  : 不进模型上下文, 在执行前后改写/拦截
    skills : 低成本注入任务方法
    MCP    : 外部服务把工具暴露给 agent
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from llm_agent.core import (
    Agent,
    HookManager,
    HookResult,
    PermissionGate,
    RuleBasedLLM,
    SearchDocsTool,
    ToolCall,
    ToolRegistry,
    WeatherTool,
)
from llm_agent.core.utils import banner


DOCS = {"logs": "When debugging, search logs first, then summarize the smallest cause."}


@dataclass
class Skill:
    name: str
    trigger: str
    instruction: str


class SkillRegistry:
    def __init__(self, skills: List[Skill]) -> None:
        self.skills = skills

    def instructions_for(self, prompt: str) -> str:
        hits = [s.instruction for s in self.skills if s.trigger in prompt]
        return "\n".join(f"[skill] {x}" for x in hits)


class FakeMCPServer:
    """A local stand-in for an MCP server exposing tools."""

    def list_tools(self):
        return [WeatherTool()]


def main() -> None:
    banner("M05 - Hooks, Skills, MCP-like Tools")

    skills = SkillRegistry(
        [Skill("debug", "排查", "排查问题时先 search 日志或文档, 再给结论。")]
    )
    hooks = HookManager()

    def inject_skill(prompt: str) -> HookResult:
        extra = skills.instructions_for(prompt)
        return HookResult(additional_context=extra)

    def block_secret_shell(call: ToolCall) -> HookResult:
        if call.name == "shell" and "token" in str(call.args).lower():
            return HookResult(block=True, reason="secret-like shell command")
        return HookResult(updated_call=call)

    hooks.register("user_prompt_submit", inject_skill)
    hooks.register("pre_tool_use", block_secret_shell)

    tools = ToolRegistry([SearchDocsTool(DOCS)])
    for tool in FakeMCPServer().list_tools():
        tools.register(tool)

    print("\n[1] skill 注入让普通 prompt 变成 search 任务")
    agent = Agent(
        llm=RuleBasedLLM(),
        tools=tools,
        permissions=PermissionGate(mode="auto"),
        hooks=hooks,
        max_turns=4,
        name="m05",
    )
    agent.run("排查为什么 agent 没有结果", verbose=True)

    print("\n[2] fake MCP server 暴露 weather 工具")
    agent.run("查询北京天气", verbose=True)

    print("\n  OK: 扩展点按成本分层, 不必把所有能力都塞进 prompt。")


if __name__ == "__main__":
    main()

