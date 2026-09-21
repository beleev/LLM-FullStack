"""按学习路径顺序跑完所有默认 demo。每个 demo 都用 assert 自检, 任何断言失败都会让本脚本非零退出。

m15 (真实模型) 需要可选依赖和 API key, 不在此列: python -m llm_agent.m15_claude_api.demo
"""

from __future__ import annotations

import importlib

DEMOS = [
    "llm_agent.m01_agent_loop.demo",
    "llm_agent.m02_tool_use.demo",
    "llm_agent.m03_permissions.demo",
    "llm_agent.m04_context_memory.demo",
    "llm_agent.m05_extensibility.demo",
    "llm_agent.m06_persistence_resume.demo",
    "llm_agent.m07_subagents.demo",
    "llm_agent.m08_retrieval.demo",
    "llm_agent.m09_mcp.demo",
    "llm_agent.m10_planning.demo",
    "llm_agent.m11_orchestrator.demo",
    "llm_agent.m12_guardrails.demo",
    "llm_agent.m13_evals.demo",
    "llm_agent.m14_context_engineering.demo",
    "llm_agent.full_loop.demo",
]


def main() -> None:
    for module_name in DEMOS:
        importlib.import_module(module_name).main()
    print(f"\nALL {len(DEMOS)} DEMOS PASSED")


if __name__ == "__main__":
    main()
