"""
Full Loop — mini-Claude-Code-style harness

组合演示:
    tools          search_docs / write_note / weather / delegate
    permissions    deny-first + auto classifier
    hooks          session_start / user_prompt_submit / post_tool_use
    memory         transparent markdown snippets
    persistence    append-only JSONL transcript
    subagent       isolated child transcript, summary-only return
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from llm_agent.core import (
    Agent,
    DelegateTool,
    FileMemory,
    HookManager,
    HookResult,
    JsonlSessionStore,
    PermissionGate,
    PermissionRule,
    RuleBasedLLM,
    SearchDocsTool,
    ShellTool,
    ToolRegistry,
    WeatherTool,
    WriteNoteTool,
)
from llm_agent.core.permissions import Decision
from llm_agent.core.schema import ToolResult
from llm_agent.core.utils import banner, kv


DOCS = {
    "agent_loop": "Agent loop = assemble context, call model, dispatch tool, check permission, execute.",
    "permissions": "Deny-first policy: deny rules win, unknown actions ask, low-risk actions may be auto-approved.",
    "context": "Context is scarce; compact middle turns and retrieve only relevant memory.",
    "subagents": "Subagents keep isolated transcripts and return a compact summary to the parent.",
}


def build_hooks() -> HookManager:
    hooks = HookManager()

    hooks.register("session_start", lambda: "Session policy: prefer tools, keep answers short.")

    def teaching_skill(prompt: str) -> HookResult:
        if "排查" in prompt:
            return HookResult(additional_context="[skill] 排查类任务: 先 search_docs, 再写结论。")
        return HookResult()

    def post_tool_marker(result: ToolResult) -> str:
        if result.ok:
            return f"[post_tool] {result.name} finished"
        return ""

    hooks.register("user_prompt_submit", teaching_skill)
    hooks.register("post_tool_use", post_tool_marker)
    return hooks


def main() -> None:
    banner("Full Loop - mini Agent Harness")

    with tempfile.TemporaryDirectory(prefix="llm_agent_full_") as tmp:
        tmp_path = Path(tmp)
        memory = FileMemory(tmp_path / "memory")
        memory.add("project_style", "回答中文；先给结论，再给关键原因。")
        memory.add("agent_design", "Agent harness should own permissions, tools, context, and persistence.")

        notes = []
        delegate = DelegateTool(DOCS)
        tools = ToolRegistry(
            [
                SearchDocsTool(DOCS),
                WriteNoteTool(notes),
                WeatherTool(),
                ShellTool(),
                delegate,
            ]
        )
        permissions = PermissionGate(
            mode="auto",
            rules=[PermissionRule("shell", "*rm -rf*", Decision.DENY, "never allow destructive demo shell")],
        )
        store = JsonlSessionStore(tmp_path / "session.jsonl")

        agent = Agent(
            llm=RuleBasedLLM(),
            tools=tools,
            permissions=permissions,
            hooks=build_hooks(),
            memory=memory,
            store=store,
            context_budget_chars=900,
            max_turns=6,
            name="full",
        )

        print("\n[1] gather -> act -> persist")
        agent.run("排查 agent loop，并写入笔记", verbose=True)

        print("\n[2] delegate isolated research")
        agent.run("请委托子智能体调研 subagents", verbose=True)

        print("\n[3] external MCP-like tool")
        agent.run("查询上海天气", verbose=True)

        print("\n[4] denied dangerous action")
        agent.run("运行 rm -rf /tmp/demo", verbose=True)

        banner("Stats")
        kv("notes", notes)
        kv("jsonl path", store.path)
        kv("jsonl messages", store.count())
        kv("parent messages", len(agent.messages))
        kv("child transcripts", len(delegate.child_transcripts))
        if delegate.child_transcripts:
            kv("last child messages", len(delegate.child_transcripts[-1]))

    print("\n  OK: 简单 while-loop + 周边 harness = 可运行的 agent 系统原型。")


if __name__ == "__main__":
    main()

