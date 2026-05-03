"""
M01 — Agent Loop

最小闭环:
    user prompt -> context assembly -> model chooses tool -> permission gate
    -> tool executes -> tool result returns -> model final answer

这里的 "LLM" 是 RuleBasedLLM, 目的是把 agent harness 看清楚。
"""

from __future__ import annotations

from llm_agent.core import Agent, CalculatorTool, PermissionGate, RuleBasedLLM, ToolRegistry
from llm_agent.core.utils import banner, kv


def main() -> None:
    banner("M01 - Minimal Agent Loop")

    tools = ToolRegistry([CalculatorTool()])
    agent = Agent(
        llm=RuleBasedLLM(),
        tools=tools,
        permissions=PermissionGate(mode="dont_ask"),
        max_turns=4,
        name="m01",
    )

    prompt = "请计算 2 + 3 * 4"
    kv("user prompt", prompt)
    final = agent.run(prompt, verbose=True)

    print("\n[transcript]")
    for msg in agent.messages:
        print(f"  {msg.role:<9} {msg.name or '-':<12} {msg.content}")

    print("\n  OK: while-loop 本身很薄, 但已经形成了可执行闭环。")
    kv("final", final)


if __name__ == "__main__":
    main()

