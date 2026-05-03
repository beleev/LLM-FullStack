"""
M02 — Tool Use

工具是模型能力的外部化:
    - schema 告诉模型有什么能力
    - execute 由确定性代码真正执行
    - tool result 再回到上下文, 供模型继续决策
"""

from __future__ import annotations

from llm_agent.core import (
    Agent,
    PermissionGate,
    RuleBasedLLM,
    SearchDocsTool,
    ToolRegistry,
    WriteNoteTool,
)
from llm_agent.core.utils import banner, kv


DOCS = {
    "agent_loop": "Agent loop = assemble context, call model, run tools, repeat.",
    "permissions": "Deny-first gates keep unknown or risky actions under human control.",
    "context": "Context windows are scarce, so agents compact history and retrieve memory.",
}


def main() -> None:
    banner("M02 - Tool Use")

    notes = []
    tools = ToolRegistry([SearchDocsTool(DOCS), WriteNoteTool(notes)])

    print("\n[1] tool schemas exposed to the model")
    for schema in tools.schemas():
        print(f"  - {schema['name']}: {schema['description']} risk={schema['risk']}")

    print("\n[2] run search -> write_note -> final")
    agent = Agent(
        llm=RuleBasedLLM(),
        tools=tools,
        permissions=PermissionGate(mode="auto"),
        max_turns=5,
        name="m02",
    )
    final = agent.run("搜索 agent loop，并写入笔记", verbose=True)

    print("\n[notes]")
    for i, note in enumerate(notes, 1):
        print(f"  {i}. {note}")

    kv("final", final)
    print("\n  OK: 工具调用把模型从纯文本回答扩展成 gather-act-verify。")


if __name__ == "__main__":
    main()

