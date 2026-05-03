"""Run all llm_agent demos in learning-path order."""

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
    "llm_agent.full_loop.demo",
]


def main() -> None:
    for module_name in DEMOS:
        module = importlib.import_module(module_name)
        module.main()


if __name__ == "__main__":
    main()

