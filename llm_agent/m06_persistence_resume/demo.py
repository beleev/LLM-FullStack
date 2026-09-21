"""M06 — Persistence & Resume: append-only JSONL 会话日志。

没有它: 进程一退会话全丢; 出事后也没法复盘"模型当时到底要求了什么、harness 放行了什么"。
关键设计:
  - 每条消息 (含助手的 tool_use 和回填的 tool_result) 立刻追加一行 —— 先记录"模型要什么", 再去执行。
  - resume = 读回 transcript。恢复的只有上下文: 权限要由新会话重新建立, SessionStart 注入不重复。
  - 落盘的 block 格式与 id 配对和 Messages API 一致; harness 注入的 system 消息发送前由 claude_llm.to_api_messages 转换。
对应: Claude Code 的 ~/.claude/projects/*.jsonl 与 `claude --resume`。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from llm_agent.core import (
    Agent,
    HookManager,
    JsonlSessionStore,
    PermissionGate,
    RuleBasedLLM,
    SearchDocsTool,
    ToolRegistry,
    WriteNoteTool,
    validate_transcript,
)
from llm_agent.core.utils import banner, kv

DOCS = {"agent_loop": "Agent loop uses messages, tools, permissions and persistence."}


def main() -> None:
    banner("M06 - Append-only Session Persistence")

    sources = []
    hooks = HookManager()
    hooks.register("session_start", lambda source: sources.append(source) or "Policy: answer in Chinese.")

    with tempfile.TemporaryDirectory(prefix="llm_agent_session_") as tmp:
        path = Path(tmp) / "session.jsonl"
        notes: list = []
        tools = ToolRegistry([SearchDocsTool(DOCS), WriteNoteTool(notes)])

        print("\n[1] session A: 搜索, 然后进程'退出'")
        Agent(RuleBasedLLM(), tools, PermissionGate("auto"), hooks=hooks, store=JsonlSessionStore(path), name="session-A").run(
            "搜索 agent loop"
        )

        print("\n[raw jsonl]")
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for record in lines:
            print("  " + json.dumps(record, ensure_ascii=False)[:150])
        kinds = [b["type"] for r in lines if isinstance(r["content"], list) for b in r["content"]]
        assert kinds == ["tool_use", "tool_result"], kinds  # 旧版只有工具结果, 看不到模型发起的调用

        print("\n[2] session B: 新进程读回 transcript 继续; 引用'刚才的结果'")
        agent_b = Agent(
            RuleBasedLLM(),
            tools,
            # 会话 A 是 auto 模式; B 退回 default, 写操作要重新问人 —— 恢复上下文 ≠ 恢复授权
            PermissionGate("default", ask_policy=lambda call: call.name == "write_note"),
            hooks=hooks,
            store=JsonlSessionStore(path),
            load_history=True,
            name="session-B",
        )
        assert len(agent_b.messages) == len(lines)
        agent_b.run("把刚才结果写入笔记")

        assert notes == ["agent_loop: " + DOCS["agent_loop"]], notes  # 写进笔记的是上一个会话的工具数据, 原样无杂质
        assert validate_transcript(agent_b.messages) == []
        assert sources == ["startup", "resume"]
        assert len([m for m in agent_b.messages if m.name == "session_start"]) == 1  # resume 不重复注入
        ids = [b["id"] for m in agent_b.messages for b in m.tool_uses()]
        assert ids == ["toolu_0001", "toolu_0002"], ids  # id 跨会话续号, 审计时不会撞

        kv("jsonl lines", len(JsonlSessionStore(path).load_all()))
        kv("notes", notes)

    print("\n  OK: transcript 可完整复盘、可恢复; 权限不随 transcript 恢复。")


if __name__ == "__main__":
    main()
