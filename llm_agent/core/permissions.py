"""Deny-first permission gate.

This is a compact model of production agent permissions:
    1. broad deny rules win first
    2. explicit allow rules come next
    3. mode/classifier/human approval handles unknown actions
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import Callable, List, Optional

from llm_agent.core.schema import ToolCall


class Decision:
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass
class PermissionRule:
    tool: str
    pattern: str
    decision: str
    reason: str = ""


@dataclass
class PermissionOutcome:
    allowed: bool
    decision: str
    source: str
    reason: str


class PermissionGate:
    """Evaluate tool calls against policy.

    Modes:
      plan                ask for every tool call
      default             ask for unknown calls
      accept_edits         allow low/medium risk calls, ask shell
      auto                tiny classifier allows low risk, denies obvious danger
      dont_ask            allow unknown calls after deny rules
      bypass_permissions  allow everything after deny rules
    """

    def __init__(
        self,
        mode: str = "default",
        rules: Optional[List[PermissionRule]] = None,
        ask_policy: Optional[Callable[[ToolCall], bool]] = None,
    ) -> None:
        self.mode = mode
        self.rules = rules or []
        self.ask_policy = ask_policy

    def evaluate(self, call: ToolCall) -> PermissionOutcome:
        # 1) Deny-first. A broad deny beats a narrow allow.
        for rule in self.rules:
            if rule.decision == Decision.DENY and self._matches(rule, call):
                return PermissionOutcome(False, Decision.DENY, "rule", rule.reason or "deny rule")

        # 2) Explicit allow.
        for rule in self.rules:
            if rule.decision == Decision.ALLOW and self._matches(rule, call):
                return PermissionOutcome(True, Decision.ALLOW, "rule", rule.reason or "allow rule")

        # 3) Mode-specific fallback.
        if self.mode == "plan":
            return self._ask(call, "plan mode asks before actions")
        if self.mode in ("dont_ask", "bypass_permissions"):
            return PermissionOutcome(True, Decision.ALLOW, self.mode, "mode allows unknown action")
        if self.mode == "accept_edits":
            if call.name == "shell":
                return self._ask(call, "shell remains high risk")
            return PermissionOutcome(True, Decision.ALLOW, "accept_edits", "non-shell action")
        if self.mode == "auto":
            return self._auto_classify(call)
        return self._ask(call, "default mode asks for unknown action")

    def _ask(self, call: ToolCall, reason: str) -> PermissionOutcome:
        approved = bool(self.ask_policy(call)) if self.ask_policy else False
        return PermissionOutcome(
            approved,
            Decision.ALLOW if approved else Decision.DENY,
            "human",
            reason + ("; approved" if approved else "; denied"),
        )

    def _auto_classify(self, call: ToolCall) -> PermissionOutcome:
        text = self._call_text(call)
        danger = ["rm ", "sudo", "curl ", "ssh ", "chmod ", ">", "token", "secret"]
        if any(x in text for x in danger):
            return PermissionOutcome(False, Decision.DENY, "auto", "classifier saw risky pattern")
        if call.name in {"calculator", "search_docs", "read_notes", "weather", "delegate"}:
            return PermissionOutcome(True, Decision.ALLOW, "auto", "read-only or reversible tool")
        if call.name == "write_note":
            return PermissionOutcome(True, Decision.ALLOW, "auto", "bounded local write")
        return self._ask(call, "classifier unsure")

    def _matches(self, rule: PermissionRule, call: ToolCall) -> bool:
        if rule.tool not in ("*", call.name):
            return False
        if not rule.pattern:
            return True
        return fnmatch.fnmatch(self._call_text(call), rule.pattern)

    @staticmethod
    def _call_text(call: ToolCall) -> str:
        return " ".join([call.name] + [str(v) for v in call.args.values()]).lower()
