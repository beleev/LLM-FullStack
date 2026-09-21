"""agent loop: 组装上下文 → 问模型 → (hook → 权限 → 校验 → 执行) → 结果回填 → 再问。

没有它: 模型只是一次性问答; loop 才让它能"行动-观察-再行动"。
关键设计:
  - 顺序固定为 PreToolUse hook → 权限门 → 执行。门评估的是 hook 改写后的最终调用,
    所以"hook 把 calculator 改写成 rm -rf"不可能绕过 deny 规则 (与 Claude Code 同序)。
  - 一个 assistant turn 里的多个 tool_use: 授权串行 (审批不能并发弹窗), 执行并行,
    全部 tool_result 放进同一条 user 消息 (真实 API 的硬性要求)。
  - hook 注释、skill 指令、记忆都是独立的 system 消息, 永远不拼进用户 prompt 或工具数据。
  - 上下文超预算: 先清旧工具结果, 再让模型写摘要并真的替换掉历史 (见 memory.py / persistence.py)。
对应: Claude Code / Claude Agent SDK 的主循环。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

from llm_agent.core.guardrails import Guardrails
from llm_agent.core.hooks import HookManager
from llm_agent.core.llm import LLM
from llm_agent.core.memory import (
    FileMemory,
    clear_tool_results,
    memory_messages,
    summarize_with_llm,
    total_chars,
    truncate_messages,
)
from llm_agent.core.permissions import PermissionGate
from llm_agent.core.persistence import JsonlSessionStore
from llm_agent.core.schema import Message, ModelAction, ToolCall, ToolResult
from llm_agent.core.tools import ToolRegistry
from llm_agent.core.toy_llm import RuleBasedLLM
from llm_agent.core.utils import estimate_tokens, shorten


class Agent:
    def __init__(
        self,
        llm: Optional[LLM],
        tools: ToolRegistry,
        permissions: Optional[PermissionGate] = None,
        hooks: Optional[HookManager] = None,
        memory: Optional[FileMemory] = None,
        store: Optional[JsonlSessionStore] = None,
        guardrails: Optional[Guardrails] = None,
        system_prompt: str = "You are a small teaching agent.",
        context_budget_chars: int = 1200,
        compaction: str = "summary",  # summary | truncate | none
        keep_tool_results: int = 1,
        max_turns: int = 6,
        max_parallel: int = 4,
        name: str = "agent",
        load_history: bool = False,
    ) -> None:
        self.name = name
        self.llm: LLM = llm or RuleBasedLLM()
        self.tools = tools
        self.permissions = permissions or PermissionGate(mode="default")
        self.hooks = hooks or HookManager()
        self.memory = memory
        self.store = store
        self.guardrails = guardrails
        self.system_prompt = system_prompt
        self.context_budget_chars = context_budget_chars
        self.compaction = compaction
        self.keep_tool_results = keep_tool_results
        self.max_turns = max_turns
        self.max_parallel = max_parallel
        self.messages: List[Message] = store.load() if (store and load_history) else []
        if self.messages and self.messages[-1].tool_uses():
            self.messages.pop()  # 上次崩在 tool_use 与 tool_result 之间: 丢掉悬空的调用, 否则 transcript 非法
        # id 从全量历史续号, 压缩后也不会和旧 tool_use 撞号
        self._next_id = sum(len(m.tool_uses()) for m in (store.load_all() if store and load_history else []))
        self._started = False
        self._tainted = False
        self.compactions = 0
        self.usage = {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "peak_context_tokens": 0}

    # ------------------------------------------------------------------ loop
    def run(self, prompt: str, verbose: bool = True) -> str:
        self._start_session()

        submitted = self.hooks.on_user_prompt_submit(prompt)
        if submitted.block:
            # 被拦的 prompt 本身不入上下文 (可能含密钥), 只留一条审计记录
            self._append(Message("system", f"user prompt blocked: {submitted.reason}", name="hook_block"))
            return f"blocked by UserPromptSubmit hook: {submitted.reason}"

        self._append(Message("user", prompt))
        if submitted.additional_context:
            self._append(Message("system", submitted.additional_context, name="hook_context"))
        self._tainted = False  # 污点按用户轮次计: 新的用户指令 = 新的信任起点

        for turn in range(1, self.max_turns + 1):
            action = self._ask_model(self._assemble_context(prompt))

            if action.kind == "final" or not action.tool_calls:
                final = action.content or "model returned an empty action"
                self._append(Message("assistant", action.raw_content or final))
                if verbose:
                    print(f"  [{self.name}] final: {shorten(final)}")
                self.hooks.on_stop(final)
                return final

            calls = action.tool_calls
            for call in calls:
                if not call.id:
                    self._next_id += 1
                    call.id = f"toolu_{self._next_id:04d}"
            text = [{"type": "text", "text": action.content}] if action.content else []
            # 先记下"模型要求了什么", 再去执行 —— 即使执行中崩溃, 审计日志里也有这一步
            self._append(Message("assistant", action.raw_content or text + [c.to_block() for c in calls]))
            if verbose:
                for call in calls:
                    print(f"  [{self.name}] turn {turn}: model -> tool_use {call.id} {call.name} {call.args}")

            results, notes = self._run_tools(calls, verbose)
            self._append(Message("user", [r.to_block() for r in results]))
            if notes:
                self._append(Message("system", "\n".join(notes), name="post_tool_hook"))

        final = "stopped: max_turns reached"
        self._append(Message("assistant", final))
        self.hooks.on_stop(final)
        return final

    # ----------------------------------------------------------------- tools
    def _authorize(self, call: ToolCall, verbose: bool, batch_taint: bool = False) -> Tuple[Optional[ToolCall], str]:
        """返回 (最终可执行的调用, "") 或 (None, 拒绝原因)。"""
        pre = self.hooks.on_pre_tool_use(call)
        if pre.block:
            return None, f"BLOCKED BY HOOK: {pre.reason}"
        final = pre.updated_call or call
        if final is not call and verbose:
            print(f"  [{self.name}] pre_tool_use hook rewrote -> {final.name} {final.args}")

        tool = self.tools.get(final.name)
        if self.guardrails and (self._tainted or batch_taint) and getattr(tool, "risk", "high") == "high":
            return None, "DENIED: context is tainted by untrusted data; high-risk tools are locked this turn"

        outcome = self.permissions.evaluate(final, tool)  # 评估 final 而不是 call: 改写不能绕过权限
        if verbose:
            print(f"  [{self.name}] permission {final.name} -> {outcome.decision} ({outcome.source}: {outcome.reason})")
        if not outcome.allowed:
            return None, f"DENIED: {outcome.reason}"
        return final, ""

    def _run_tools(self, calls: List[ToolCall], verbose: bool) -> Tuple[List[ToolResult], List[str]]:
        results: List[Optional[ToolResult]] = [None] * len(calls)
        approved: List[Tuple[int, ToolCall]] = []
        notes = []
        untrusted = [c for c in calls if getattr(self.tools.get(c.name), "untrusted_output", False)]
        for i, call in enumerate(calls):
            # 同批里别的调用会读不可信数据: 并行执行无法保证先后 → 对这个调用按已污染处理
            batch_taint = any(c is not call for c in untrusted)
            final, denied = self._authorize(call, verbose, batch_taint)
            if final is None:
                results[i] = ToolResult(call.name, denied, ok=False)
            else:
                approved.append((i, final))
                if final is not call:  # 审计日志必须能看出"实际执行的不是模型要求的那个"
                    notes.append(f"pre_tool_use rewrote {call.id}: {call.name} {call.args} -> {final.name} {final.args}")

        if len(approved) > 1:  # 并行: 总耗时 ≈ 最慢的那个, 而不是求和
            with ThreadPoolExecutor(max_workers=self.max_parallel) as pool:
                outs = list(pool.map(self.tools.execute, [c for _, c in approved]))
        else:
            outs = [self.tools.execute(c) for _, c in approved]
        for (i, _), out in zip(approved, outs):
            results[i] = out

        executed = {i for i, _ in approved}
        for i, (call, result) in enumerate(zip(calls, results)):
            result.tool_use_id = call.id  # 结果永远挂在模型发出的那个 id 上, 即使 hook 改写了调用
            tool = self.tools.get(result.name)
            if self.guardrails:
                result.output = self.guardrails.redact(result.output)
                if result.ok and getattr(tool, "untrusted_output", False):
                    result.output = self.guardrails.wrap_untrusted(result.output)
                    self._tainted = True
            extra = self.hooks.on_post_tool_use(result) if i in executed else ""  # 被拦下的调用没有"执行后"
            if extra:
                notes.append(extra)  # 注释走旁路 system 消息, 不拼进 result.output (否则会被写进笔记/检索)
            if verbose:
                print(f"  [{self.name}] tool_result {call.id} -> {shorten(result.output)}")
        return results, notes

    # --------------------------------------------------------------- context
    def _ask_model(self, context: List[Message]) -> ModelAction:
        action = self.llm.next(context, self.tools.schemas())
        tokens = sum(estimate_tokens(m.text) for m in context)
        self.usage["llm_calls"] += 1
        self.usage["input_tokens"] += tokens  # 每次调用都要重发整个上下文: 这就是长会话贵的原因
        self.usage["output_tokens"] += estimate_tokens(action.content) + sum(
            estimate_tokens(str(c.args)) for c in action.tool_calls
        )
        self.usage["peak_context_tokens"] = max(self.usage["peak_context_tokens"], tokens)
        return action

    def _assemble_context(self, prompt: str) -> List[Message]:
        base = [Message("system", self.system_prompt)]
        if self.memory:
            base.extend(memory_messages(self.memory, prompt))
        budget = self.context_budget_chars - total_chars(base)

        view = self.messages
        if total_chars(view) > budget:  # 第 1 档: 清旧工具结果 (只改视图, 不动 transcript)
            view = clear_tool_results(view, self.keep_tool_results)
        if total_chars(view) > budget:
            if self.compaction == "summary" and self._compact():  # 第 2 档: 摘要并替换历史
                view = clear_tool_results(self.messages, self.keep_tool_results)
            elif self.compaction == "truncate":
                view = truncate_messages(view, budget)
        return base + view

    def _compact(self) -> bool:
        """把"当前用户轮之前"的历史换成一条模型写的摘要。当前轮原样保留, 配对不会被切断。"""
        start = max((i for i, m in enumerate(self.messages) if m.is_user_prompt), default=0)
        old, tail = self.messages[:start], self.messages[start:]
        if len(old) < 2:  # 没什么可压 (或只剩上次的摘要)
            return False
        # session_start 注入的策略不能被压没: 和 pre_compact hook 的要求一起交给摘要, 标为必留
        keep = "\n".join(filter(None, [self.hooks.on_pre_compact(old)] + [m.text for m in old if m.name == "session_start"]))
        self.usage["llm_calls"] += 1
        summary = summarize_with_llm(self.llm, old, keep)
        self.messages = [summary] + tail
        if self.store:
            self.store.append_compact(summary, kept=len(tail))
        self.compactions += 1
        return True

    # ----------------------------------------------------------------- state
    def _start_session(self) -> None:
        if self._started:
            return
        self._started = True
        source = "resume" if self.messages else "startup"
        injected = any(m.name == "session_start" for m in self.messages)
        for text in self.hooks.on_session_start(source):  # hook 照常触发 (可做副作用), 但只在历史里没有时才注入
            if not injected:  # 被压缩掉了才会重新注入 —— 与 Claude Code compact 后重跑 SessionStart 一致
                self._append(Message("system", text, name="session_start"))

    def _append(self, message: Message) -> None:
        if self.guardrails:
            message = _redacted(message, self.guardrails)
        self.messages.append(message)
        if self.store:
            self.store.append(message)


def _redacted(message: Message, guard: Guardrails) -> Message:
    if isinstance(message.content, str):
        return Message(message.role, guard.redact(message.content), message.name)
    blocks = []
    for b in message.content:
        if b["type"] == "text":
            b = {**b, "text": guard.redact(b["text"])}
        elif b["type"] == "tool_result":
            b = {**b, "content": guard.redact(str(b["content"]))}
        elif b["type"] == "tool_use":
            b = {**b, "input": {k: guard.redact(v) if isinstance(v, str) else v for k, v in b["input"].items()}}
        blocks.append(b)
    return Message(message.role, blocks, message.name)
