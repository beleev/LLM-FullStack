"""M07 — Subagents: 把一件脏活关进隔离的上下文, 只拿回摘要。

没有它: 调研过程中读到的每一篇文档都堆进主上下文, 主 agent 很快被细节淹没, 每轮还要为它们重复付费。
关键设计:
  - delegate 是一个普通工具; 子 agent 有自己的 messages、工具集、权限门。
  - 父级只收到一行摘要; 子级完整 transcript 落盘备查, 不回流。
  - 隔离是双向的: 子级也拿不到父级的工具 (这里父级有 shell, 子级没有)。
多个子级并行扇出 + token 记账见 m11。
对应: Claude Code 的 Task (Agent) 工具与 SubagentStop hook。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from llm_agent.core import (
    Agent,
    DelegateTool,
    HookManager,
    JsonlSessionStore,
    PermissionGate,
    RuleBasedLLM,
    SearchDocsTool,
    ShellTool,
    ToolRegistry,
)
from llm_agent.core.utils import banner, kv

DOCS = {
    "agent_loop": "The loop is small; harness systems around it carry most complexity. " + "detail " * 120,
    "context": "Subagent side transcripts should not flood the parent context.",
}


def main() -> None:
    banner("M07 - Isolated Subagents")

    stops = []
    hooks = HookManager()
    hooks.register("subagent_stop", lambda agent_type, summary: stops.append(agent_type))

    with tempfile.TemporaryDirectory(prefix="llm_agent_sub_") as tmp:
        delegate = DelegateTool(
            {"researcher": lambda: ToolRegistry([SearchDocsTool(DOCS)])}, hooks=hooks, transcript_dir=Path(tmp)
        )
        parent = Agent(
            RuleBasedLLM(), ToolRegistry([delegate, ShellTool()]), PermissionGate("auto"), hooks=hooks, name="parent"
        )
        parent.run("请委托子智能体调研 agent loop")

        child = delegate.children[0]
        print("\n[parent transcript]")
        for msg in parent.messages:
            print(f"  {msg.role:<9} {msg.text[:100]}")
        print("\n[child transcript — 只在磁盘上, 不在父上下文里]")
        for msg in child["messages"]:
            print(f"  {msg.role:<9} {msg.text[:100]}")

        parent_tools = [b["name"] for m in parent.messages for b in m.tool_uses()]
        child_tools = [b["name"] for m in child["messages"] for b in m.tool_uses()]
        assert parent_tools == ["delegate"] and child_tools == ["search_docs"]
        assert len(parent.messages) == 4  # user / tool_use / tool_result(摘要) / final —— 子级的 4 条消息一条都没进来
        assert stops == ["researcher"]
        assert len(JsonlSessionStore(child["path"]).load()) == len(child["messages"])  # 子 transcript 已落盘

        kv("parent peak context tokens", parent.usage["peak_context_tokens"])
        kv("child  peak context tokens", child["usage"]["peak_context_tokens"])
        assert parent.usage["peak_context_tokens"] < child["usage"]["peak_context_tokens"]  # 长文档只撑大了子级的上下文

    print("\n  OK: 父级拿结论, 子级留细节; 隔离的是上下文, 也是工具权限。")


if __name__ == "__main__":
    main()
