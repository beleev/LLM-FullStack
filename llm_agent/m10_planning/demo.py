"""M10 — Planning: todo 工具、plan 模式、并行工具调用。

没有它: 多步任务里模型边想边做 —— 做到一半忘了还剩什么; 用户也没机会在它动手之前说"不"。
关键设计:
  - todo_write: 计划是模型自己维护的显式状态 (每次整表覆写), 留在上下文里, 也能展示给用户。
  - plan 模式: 权限门只放行只读工具; 模型用 exit_plan_mode 提交计划, 人批准后模式才切走。
    "只读"由 harness 强制, 不靠模型自觉。
  - 互不依赖的调用放进同一个 assistant turn, harness 用线程池并行执行: 耗时 ≈ 最慢的一个。
对应: Claude Code 的 TodoWrite、plan mode / ExitPlanMode; Claude API 的 parallel tool use。
"""

from __future__ import annotations

import time
from typing import Any, Dict

from llm_agent.core import (
    Agent,
    DelegateTool,
    ExitPlanModeTool,
    PermissionGate,
    RuleBasedLLM,
    SearchDocsTool,
    TodoWriteTool,
    Tool,
    ToolCall,
    ToolRegistry,
    ToolResult,
    WriteNoteTool,
)
from llm_agent.core.utils import banner, kv

DOCS = {"kv_cache": "The kv cache stores past keys and values so decoding is incremental."}
PROMPT = "搜索 kv cache，并写入笔记"


class SlowWeatherTool(Tool):
    name = "weather"
    description = "Weather lookup that takes 0.2s (simulates network latency)."
    parameters = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"], "additionalProperties": False}

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        time.sleep(0.2)
        return ToolResult(self.name, f"{args['city']}: sunny")


def run_plan_mode(approve: bool):
    notes: list = []
    todos = TodoWriteTool()
    gate = PermissionGate("plan")
    plans = []
    exit_tool = ExitPlanModeTool(gate, approve=lambda plan: plans.append(plan) or approve)
    tools = ToolRegistry([SearchDocsTool(DOCS), WriteNoteTool(notes), todos, exit_tool])
    agent = Agent(RuleBasedLLM(), tools, gate, max_turns=8, name="approved" if approve else "rejected")
    final = agent.run(PROMPT)
    return agent, gate, notes, todos, plans, final


def main() -> None:
    banner("M10 - Planning: todos, plan mode, parallel calls")

    print("\n[1] plan 模式 + 用户拒绝计划: 一个字都没写")
    agent, gate, notes, todos, plans, final = run_plan_mode(approve=False)
    print("  提交的计划:\n    " + plans[0].replace("\n", "\n    "))
    assert notes == [] and gate.mode == "plan" and "未获批准" in final
    assert "write_note" not in [b["name"] for m in agent.messages for b in m.tool_uses()]

    print("\n[2] 用户批准: 模式切到 accept_edits, 按计划执行, todo 从 pending 走到 completed")
    agent, gate, notes, todos, plans, final = run_plan_mode(approve=True)
    order = [b["name"] for m in agent.messages for b in m.tool_uses()]
    kv("tool order", order)
    assert order == ["todo_write", "exit_plan_mode", "search_docs", "write_note", "todo_write"]
    assert gate.mode == "accept_edits" and len(notes) == 1
    assert [t["status"] for t in todos.history[0]] == ["pending", "pending"]
    assert [t["status"] for t in todos.history[-1]] == ["completed", "completed"]

    print("\n[3] plan 模式是 harness 强制的: 就算模型不交计划直接写, 门也不开")
    gate = PermissionGate("plan")
    rogue = Agent(RuleBasedLLM(), ToolRegistry([WriteNoteTool(notes_b := [])]), gate, name="rogue")
    assert "DENIED: plan mode is read-only" in rogue.run("写入笔记: 偷偷写") and notes_b == []
    sneaky = DelegateTool({"writer": lambda: ToolRegistry([WriteNoteTool([])])})  # 想靠子 agent 代写? 委托不是只读操作
    assert not gate.evaluate(ToolCall("delegate", {"task": "写入笔记"}), sneaky).allowed

    print("\n[4] 并行工具调用: 3 个 0.2s 的调用")
    timings = {}
    for label, workers in (("serial", 1), ("parallel", 4)):
        agent = Agent(RuleBasedLLM(), ToolRegistry([SlowWeatherTool()]), PermissionGate("auto"), max_parallel=workers, name=label)
        start = time.perf_counter()
        agent.run("查询北京、上海、深圳天气", verbose=False)
        timings[label] = time.perf_counter() - start
        assert len(agent.messages[1].tool_uses()) == 3 and len(agent.messages[2].tool_results()) == 3
        kv(f"{label} (max_parallel={workers})", f"{timings[label]:.2f}s")
    assert timings["serial"] >= 0.6 and timings["parallel"] < timings["serial"] * 0.7

    print("\n  OK: 先计划后动手; 只读靠门强制; 独立调用并行发。")


if __name__ == "__main__":
    main()
