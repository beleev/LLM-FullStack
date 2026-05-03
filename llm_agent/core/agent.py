"""Minimal runnable agent loop."""

from __future__ import annotations

from typing import List, Optional

from llm_agent.core.hooks import HookManager
from llm_agent.core.memory import FileMemory, compact_messages, memory_messages
from llm_agent.core.permissions import PermissionGate
from llm_agent.core.persistence import JsonlSessionStore
from llm_agent.core.schema import Message, ToolResult
from llm_agent.core.tools import ToolRegistry
from llm_agent.core.toy_llm import RuleBasedLLM
from llm_agent.core.utils import shorten


class Agent:
    def __init__(
        self,
        llm: Optional[RuleBasedLLM],
        tools: ToolRegistry,
        permissions: Optional[PermissionGate] = None,
        hooks: Optional[HookManager] = None,
        memory: Optional[FileMemory] = None,
        store: Optional[JsonlSessionStore] = None,
        system_prompt: str = "You are a small teaching agent.",
        context_budget_chars: int = 1200,
        max_turns: int = 6,
        name: str = "agent",
        load_history: bool = False,
    ) -> None:
        self.name = name
        self.llm = llm or RuleBasedLLM()
        self.tools = tools
        self.permissions = permissions or PermissionGate(mode="default")
        self.hooks = hooks or HookManager()
        self.memory = memory
        self.store = store
        self.system_prompt = system_prompt
        self.context_budget_chars = context_budget_chars
        self.max_turns = max_turns
        self.messages: List[Message] = store.load() if (store and load_history) else []
        self._started = False

    def run(self, prompt: str, verbose: bool = True) -> str:
        if not self._started:
            for text in self.hooks.on_session_start():
                self._append(Message("system", text, name="session_start"))
            self._started = True

        submitted = self.hooks.on_user_prompt_submit(prompt)
        if submitted.block:
            final = f"blocked by UserPromptSubmit hook: {submitted.reason}"
            self._append(Message("assistant", final))
            return final

        user_text = submitted.additional_context or prompt
        self._append(Message("user", user_text))

        for turn in range(1, self.max_turns + 1):
            context = self._assemble_context(user_text)
            action = self.llm.next(context, self.tools.names())

            if action.kind == "final":
                self._append(Message("assistant", action.content))
                if verbose:
                    print(f"  [{self.name}] final: {shorten(action.content)}")
                return action.content

            call = action.tool_call
            if call is None:
                final = "model returned an empty action"
                self._append(Message("assistant", final))
                return final

            if verbose:
                print(f"  [{self.name}] turn {turn}: model -> tool {call.name} {call.args}")

            outcome = self.permissions.evaluate(call)
            if verbose:
                print(f"  [{self.name}] permission -> {outcome.decision} ({outcome.source}: {outcome.reason})")
            if not outcome.allowed:
                result = ToolResult(call.name, f"DENIED: {outcome.reason}", ok=False)
                self._append(Message("tool", result.output, name=result.name))
                continue

            pre = self.hooks.on_pre_tool_use(call)
            if pre.block:
                result = ToolResult(call.name, f"BLOCKED BY HOOK: {pre.reason}", ok=False)
            else:
                result = self.tools.execute(pre.updated_call or call)

            extra = self.hooks.on_post_tool_use(result)
            output = result.output + (f"\n{extra}" if extra else "")
            self._append(Message("tool", output, name=result.name))
            if verbose:
                print(f"  [{self.name}] tool result -> {shorten(output)}")

        final = "stopped: max_turns reached"
        self._append(Message("assistant", final))
        return final

    def _assemble_context(self, current_prompt: str) -> List[Message]:
        base = [Message("system", self.system_prompt)]
        if self.memory:
            base.extend(memory_messages(self.memory, current_prompt))
        all_messages = base + self.messages
        return compact_messages(all_messages, max_chars=self.context_budget_chars)

    def _append(self, message: Message) -> None:
        self.messages.append(message)
        if self.store:
            self.store.append(message)

