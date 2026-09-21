"""M11 — Orchestrator–Workers: lead 扇出多个并行子 agent, 再综合。

没有它: 一个 agent 串行读完所有资料, 所有原文都堆在同一个上下文里 —— 慢, 而且越读越糊。
关键设计:
  - 扇出不需要新机制: lead 在一个 turn 里发多个 delegate tool_use, loop 的线程池并行执行它们。
  - 每类 worker 有不同的工具集 (researcher 只能检索, calculator 只能算), 上下文互相隔离。
  - 记两本账: 总 token (lead + 所有 worker) 往往更高; lead 的峰值上下文却小得多 —— 多智能体买的是
    并行度和干净的主上下文, 不是省钱。
  - 每个 worker 的 transcript 落盘, 出问题能逐个复盘。
对应: Anthropic multi-agent research system (lead agent + subagents); Claude Code 并行 Task。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from llm_agent.core import (
    Agent,
    CalculatorTool,
    DelegateTool,
    HookManager,
    PermissionGate,
    RuleBasedLLM,
    TfidfIndex,
    ToolRegistry,
    VectorSearchTool,
    validate_transcript,
)
from llm_agent.core.utils import banner, kv

DOCS = {
    "kv_cache": "The kv cache stores past keys and values so decoding is incremental. " + "Background detail. " * 25,
    "lora": "LoRA trains low rank adapters, cutting trainable weights below one percent. " + "Background detail. " * 25,
    "dpo": "DPO aligns a model from preference pairs without a reward model. " + "Background detail. " * 25,
}
PROMPT = "并行调研: 调研 kv cache 如何加速解码; 调研 lora 为什么省显存; 计算 4096 * 32"


def main() -> None:
    banner("M11 - Orchestrator-Workers")
    index = TfidfIndex(DOCS)
    stops = []
    hooks = HookManager()
    hooks.register("subagent_stop", lambda agent_type, summary: stops.append(agent_type))

    with tempfile.TemporaryDirectory(prefix="llm_agent_orch_") as tmp:
        delegate = DelegateTool(
            {
                "researcher": lambda: ToolRegistry([VectorSearchTool(index, k=2)]),
                "calculator": lambda: ToolRegistry([CalculatorTool()]),
            },
            hooks=hooks,
            transcript_dir=Path(tmp),
        )
        lead = Agent(RuleBasedLLM(), ToolRegistry([delegate]), PermissionGate("auto"), context_budget_chars=4000, name="lead")
        final = lead.run(PROMPT)
        print("\n" + final)

        fan_out = lead.messages[1].tool_uses()
        assert [b["input"]["agent_type"] for b in fan_out] == ["researcher", "researcher", "calculator"]
        assert len(lead.messages[2].tool_results()) == 3 and validate_transcript(lead.messages) == []
        assert sorted(stops) == ["calculator", "researcher", "researcher"]
        assert final.startswith("综合 3 个子任务结果") and "4096 * 32 = 131072" in final

        print("\n[workers]")
        for child in delegate.children:
            used = [b["name"] for m in child["messages"] for b in m.tool_uses()]
            print(f"  {child['path'].name:<28} tools={used} input_tokens={child['usage']['input_tokens']}")
            assert child["path"].exists()
            assert used == (["calculator"] if child["type"] == "calculator" else ["search_docs"])

        print("\n[对照] 同一任务交给单个 agent (自己检索 + 自己算)")
        solo = Agent(
            RuleBasedLLM(),
            ToolRegistry([VectorSearchTool(index, k=2), CalculatorTool()]),
            PermissionGate("auto"),
            context_budget_chars=4000,
            name="solo",
        )
        solo.run(PROMPT, verbose=False)

        workers_total = sum(c["usage"]["input_tokens"] for c in delegate.children)
        total = lead.usage["input_tokens"] + workers_total
        print("\n[token 记账]")
        kv("lead 峰值上下文", lead.usage["peak_context_tokens"])
        kv("solo 峰值上下文", solo.usage["peak_context_tokens"])
        kv("orchestrated 总输入", f"{total} (lead {lead.usage['input_tokens']} + workers {workers_total})")
        kv("solo 总输入", solo.usage["input_tokens"])
        assert lead.usage["peak_context_tokens"] < solo.usage["peak_context_tokens"]  # 原文留在 worker 里
        assert total > solo.usage["input_tokens"]  # 代价: 总花费更高

    print("\n  OK: 多智能体 = 用更多总 token, 换并行度和不被原文淹没的 lead 上下文。")


if __name__ == "__main__":
    main()
