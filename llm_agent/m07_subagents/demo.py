"""
M07 — Subagents

子智能体的教学重点:
    - 子任务有自己的上下文和工具集合
    - 父 agent 只收到 summary, 避免上下文爆炸
    - 权限也可单独配置
"""

from __future__ import annotations

from llm_agent.core import Agent, DelegateTool, PermissionGate, RuleBasedLLM, ToolRegistry
from llm_agent.core.utils import banner, kv


DOCS = {
    "agent_loop": "The loop is small; harness systems around it carry most complexity.",
    "context": "Subagent side transcripts should not flood the parent context.",
}


def main() -> None:
    banner("M07 - Isolated Subagents")

    delegate = DelegateTool(DOCS)
    parent = Agent(
        llm=RuleBasedLLM(),
        tools=ToolRegistry([delegate]),
        permissions=PermissionGate(mode="auto"),
        max_turns=4,
        name="parent",
    )

    parent.run("请委托子智能体调研 agent loop", verbose=True)

    print("\n[parent transcript]")
    for msg in parent.messages:
        print(f"  {msg.role:<9} {msg.name or '-':<10} {msg.content}")

    print("\n[child transcript kept out of parent context]")
    for line in delegate.child_transcripts[-1]:
        print(f"  {line}")

    kv("parent messages", len(parent.messages))
    kv("child messages", len(delegate.child_transcripts[-1]))
    print("\n  OK: 父级只拿摘要, 子级细节保存在隔离 transcript 中。")


if __name__ == "__main__":
    main()

