"""Isolated subagent delegation."""

from __future__ import annotations

from typing import Any, Dict, List

from llm_agent.core.agent import Agent
from llm_agent.core.permissions import PermissionGate
from llm_agent.core.schema import ToolResult
from llm_agent.core.tools import CalculatorTool, SearchDocsTool, Tool, ToolRegistry
from llm_agent.core.toy_llm import RuleBasedLLM


class DelegateTool(Tool):
    name = "delegate"
    description = "Run an isolated child agent and return only its summary."
    reversible = True
    risk = "medium"

    def __init__(self, docs: Dict[str, str]) -> None:
        self.docs = docs
        self.child_transcripts: List[List[str]] = []

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        task = str(args.get("task", ""))
        child_tools = ToolRegistry([SearchDocsTool(self.docs), CalculatorTool()])
        child = Agent(
            llm=RuleBasedLLM(),
            tools=child_tools,
            permissions=PermissionGate(mode="auto"),
            system_prompt="You are an isolated research subagent. Return a short summary.",
            max_turns=4,
            name="child",
        )
        final = child.run(task, verbose=False)
        self.child_transcripts.append([f"{m.role}:{m.name or ''}:{m.content}" for m in child.messages])
        return ToolResult(self.name, f"subagent summary: {final}")

