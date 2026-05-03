"""
M06 — Session Persistence / Resume

会话持久化不只为了聊天记录, 还影响:
    - 审计: append-only JSONL 能看到每一步
    - resume: 新 agent 可读取旧 transcript
    - 权限: 恢复上下文不等于恢复权限
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from llm_agent.core import (
    Agent,
    JsonlSessionStore,
    PermissionGate,
    RuleBasedLLM,
    SearchDocsTool,
    ToolRegistry,
    WriteNoteTool,
)
from llm_agent.core.utils import banner, kv


DOCS = {"agent_loop": "Agent loop uses messages, tools, permissions and persistence."}


def main() -> None:
    banner("M06 - Append-only Session Persistence")

    with tempfile.TemporaryDirectory(prefix="llm_agent_session_") as tmp:
        path = Path(tmp) / "session.jsonl"
        store = JsonlSessionStore(path)
        notes = []

        tools = ToolRegistry([SearchDocsTool(DOCS), WriteNoteTool(notes)])
        agent1 = Agent(
            llm=RuleBasedLLM(),
            tools=tools,
            permissions=PermissionGate(mode="auto"),
            store=store,
            max_turns=4,
            name="session-A",
        )
        agent1.run("搜索 agent loop", verbose=True)

        print("\n[1] recreate agent and load transcript")
        agent2 = Agent(
            llm=RuleBasedLLM(),
            tools=tools,
            permissions=PermissionGate(
                mode="default",
                ask_policy=lambda call: call.name == "write_note",
            ),  # trust is re-established by the new session
            store=store,
            load_history=True,
            max_turns=4,
            name="session-B",
        )
        agent2.run("把刚才结果写入笔记", verbose=True)

        kv("jsonl path", path)
        kv("jsonl messages", store.count())
        kv("notes", notes)

        print("\n[raw jsonl]")
        for line in path.read_text(encoding="utf-8").splitlines()[:6]:
            print("  " + line)

    print("\n  OK: transcript 可恢复, 但权限策略由新会话重新决定。")


if __name__ == "__main__":
    main()
