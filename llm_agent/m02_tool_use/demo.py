"""M02 — Tool Use: schema 给模型看, execute 由确定性代码跑, 结果回填上下文。

没有 JSON Schema: 模型只知道工具名, 参数靠猜; harness 也没法在执行前拦下畸形参数。
关键设计:
  - Tool.schema() 就是 Claude API 的 tool 定义 {name, description, input_schema}。
  - 模型输出是不可信输入: 先按 schema 校验, 不合格直接回一个 is_error 的 tool_result,
    让模型下一轮自己改 —— 而不是让工具带着坏参数崩溃。
  - 一个 assistant turn 可带多个 tool_use (并行工具调用), 结果必须合在同一条 user 消息里。
对应: Anthropic tool use (`tools=[...]`, `strict: true`, parallel tool use)。
"""

from __future__ import annotations

import json

from llm_agent.core import (
    Agent,
    CalculatorTool,
    PermissionGate,
    RuleBasedLLM,
    SearchDocsTool,
    ToolCall,
    ToolRegistry,
    WriteNoteTool,
    validate_transcript,
)
from llm_agent.core.utils import banner, kv

DOCS = {
    "agent_loop": "Agent loop = assemble context, call model, run tools, repeat.",
    "permissions": "Deny-first gates keep unknown or risky actions under human control.",
    "context": "Context windows are scarce, so agents compact history and retrieve memory.",
}


def main() -> None:
    banner("M02 - Tool Use")

    notes: list = []
    tools = ToolRegistry([CalculatorTool(), SearchDocsTool(DOCS), WriteNoteTool(notes)])

    print("\n[1] 发给模型的工具定义 (Claude API 格式)")
    for schema in tools.schemas():
        print("  " + json.dumps(schema, ensure_ascii=False))
        assert set(schema) == {"name", "description", "input_schema"}
        assert schema["input_schema"]["type"] == "object"

    print("\n[2] 执行前校验: 模型给错参数时, 得到的是可自我纠正的错误, 不是崩溃")
    for bad in (
        ToolCall("calculator", {"expression": "1+1"}),  # 参数名猜错
        ToolCall("calculator", {"expr": 42}),  # 类型不对
        ToolCall("calculator", {"expr": "__import__('os').system('id')"}),  # 合法字符串, 但不是算式
    ):
        result = tools.execute(bad)
        print(f"  {bad.args!s:<52} -> {result.output}")
        assert not result.ok
    assert tools.execute(ToolCall("calculator", {"expr": "1+1"})).output == "1+1 = 2"

    print("\n[3] 并行 gather → 串行 act")
    agent = Agent(RuleBasedLLM(), tools, PermissionGate(mode="auto"), max_turns=5, name="m02")
    final = agent.run("计算 6 * 7，同时搜索 agent loop，并写入笔记")

    first = agent.messages[1]
    assert [b["name"] for b in first.tool_uses()] == ["calculator", "search_docs"]  # 一个 turn, 两个 tool_use
    assert len(agent.messages[2].tool_results()) == 2  # 两个结果在同一条 user 消息里
    assert validate_transcript(agent.messages) == []
    assert notes == ["agent_loop: " + DOCS["agent_loop"]], notes  # 笔记 = 纯工具数据, 没有混进别的文字
    assert "6 * 7 = 42" in final

    kv("notes", notes)
    print("\n  OK: schema 约束输入, 校验守住边界, 并行调用省掉来回轮次。")


if __name__ == "__main__":
    main()
