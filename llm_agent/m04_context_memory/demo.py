"""
M04 — Context & Memory

生产级 agent 的关键资源不是 prompt 字符串, 而是上下文窗口:
    - 文件记忆: 透明、可审计、可版本控制
    - 检索: 只把相关片段放进上下文
    - 压缩: 超预算时保留头尾, 摘要中间
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from llm_agent.core import FileMemory, Message, compact_messages
from llm_agent.core.memory import total_chars
from llm_agent.core.utils import banner, kv


def main() -> None:
    banner("M04 - Context, Memory, Compaction")

    with tempfile.TemporaryDirectory(prefix="llm_agent_memory_") as tmp:
        memory = FileMemory(Path(tmp))
        memory.add("agent_loop", "Agent loop is assemble -> model -> tool -> permission -> result.")
        memory.add("permissions", "Deny-first permission rules protect shell and file writes.")
        memory.add("subagents", "Subagents should keep separate context and return summaries.")

        print("\n[1] file memory search")
        for name, text in memory.search("权限 permission shell"):
            print(f"  hit {name}: {text.splitlines()[0]}")

        print("\n[2] context compaction")
        messages = [Message("system", "You are an agent.")]
        for i in range(8):
            messages.append(Message("user", f"user turn {i}: " + "long text " * 12))
            messages.append(Message("assistant", f"assistant turn {i}: " + "reasoning " * 12))

        before = total_chars(messages)
        compacted = compact_messages(messages, max_chars=420)
        after = total_chars(compacted)
        kv("messages before", len(messages))
        kv("chars before", before)
        kv("messages after", len(compacted))
        kv("chars after", after)
        print("\n[compacted transcript]")
        for msg in compacted:
            print(f"  {msg.role:<9} {msg.name or '-':<16} {msg.content[:90]}")

    print("\n  OK: 上下文管理要渐进降级, 不应等爆窗后才处理。")


if __name__ == "__main__":
    main()

