"""子智能体: 隔离上下文里干活, 只把摘要交回父级。

没有它: 一次调研读 20 篇文档, 全部堆进主上下文 —— 主 agent 很快被无关细节淹没, 每一轮还要为它们重复付 token。
关键设计:
  - delegate 只是一个普通工具: 父模型在同一 turn 发多个 delegate tool_use, agent loop 的并行执行
    自然就成了 orchestrator-workers 扇出, 不需要另一套调度器。
  - 每种 agent_type 有自己的工具集与权限门; 子级拿不到父级的工具, 也看不到父级的对话。
  - 子 transcript 落盘 (可审计), 但不回流父上下文; usage 分开记账, 才能看清"总花费 vs 父上下文占用"。
代价: 总 token 通常更多 (每个子级都要重建上下文) —— 换来的是并行和干净的主上下文。
对应: Claude Code 的 Task/Agent 工具与 SubagentStop hook; Anthropic 多智能体 research 系统的 lead + subagents。
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from llm_agent.core.agent import Agent
from llm_agent.core.hooks import HookManager
from llm_agent.core.llm import LLM
from llm_agent.core.permissions import PermissionGate
from llm_agent.core.persistence import JsonlSessionStore
from llm_agent.core.schema import ToolResult
from llm_agent.core.tools import Tool, ToolRegistry
from llm_agent.core.toy_llm import RuleBasedLLM
from llm_agent.core.utils import shorten


class DelegateTool(Tool):
    name = "delegate"
    description = "Run an isolated child agent on a subtask and return only its summary."
    read_only = False  # 子级有自己的权限门、可能会写: 不标成只读, 否则 plan 模式能靠委托绕过

    def __init__(
        self,
        agent_types: Dict[str, Callable[[], ToolRegistry]],
        hooks: Optional[HookManager] = None,
        transcript_dir: Optional[Path] = None,
        llm_factory: Callable[[], LLM] = RuleBasedLLM,
        max_summary_chars: int = 200,
        guardrails: Optional[Any] = None,
    ) -> None:
        self.guardrails = guardrails  # 子级同样会读不可信数据、同样会落盘: 护栏要跟着下去
        self.max_summary_chars = max_summary_chars  # 硬上限: 子级再啰嗦, 也淹不了父上下文
        self.agent_types, self.hooks, self.transcript_dir, self.llm_factory = agent_types, hooks, transcript_dir, llm_factory
        self.parameters = {
            "type": "object",
            "properties": {"task": {"type": "string"}, "agent_type": {"type": "string", "enum": sorted(agent_types)}},
            "required": ["task"],
            "additionalProperties": False,
        }
        self.children: List[Dict[str, Any]] = []  # 每个子级的 {type, task, messages, usage, path}
        self._lock = threading.Lock()  # execute 会被线程池并发调用

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        agent_type = args.get("agent_type") or sorted(self.agent_types)[0]
        with self._lock:
            index = len(self.children)
            record: Dict[str, Any] = {"type": agent_type, "task": args["task"]}
            self.children.append(record)
        path = self.transcript_dir / f"child_{index:02d}_{agent_type}.jsonl" if self.transcript_dir else None

        child = Agent(
            llm=self.llm_factory(),
            tools=self.agent_types[agent_type](),  # 全新的工具集实例: 子级之间也不共享状态
            permissions=PermissionGate(mode="auto"),  # 子级没有人可问: 拿不准的一律拒绝
            store=JsonlSessionStore(path) if path else None,
            guardrails=self.guardrails,
            system_prompt=f"You are an isolated {agent_type} subagent. Return a short summary.",
            max_turns=4,
            name=f"child-{index}",
        )
        summary = child.run(args["task"], verbose=False)
        record.update(messages=child.messages, usage=child.usage, path=path)
        if self.hooks:
            self.hooks.on_subagent_stop(agent_type, summary)
        return ToolResult(self.name, f"[{agent_type}] {shorten(summary, self.max_summary_chars)}")
