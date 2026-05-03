"""
M03 — Permissions

演示三个点:
    1) deny-first: 拒绝规则先于允许规则
    2) default: 未知动作交给人
    3) auto: 小分类器自动放行低风险, 拒绝明显危险
"""

from __future__ import annotations

from llm_agent.core import PermissionGate, PermissionRule, ToolCall
from llm_agent.core.permissions import Decision
from llm_agent.core.utils import banner, kv


def main() -> None:
    banner("M03 - Deny-first Permission Gate")

    rules = [
        PermissionRule("shell", "*rm -rf*", Decision.DENY, "destructive shell pattern"),
        PermissionRule("shell", "shell echo *", Decision.ALLOW, "echo is safe enough"),
        PermissionRule("calculator", "*", Decision.ALLOW, "calculator is reversible"),
    ]

    def human_policy(call: ToolCall) -> bool:
        # 教学 demo 不做 input(), 用固定策略模拟人工审批。
        return call.name == "shell" and str(call.args.get("command", "")).startswith("echo")

    default_gate = PermissionGate(mode="default", rules=rules, ask_policy=human_policy)
    calls = [
        ToolCall("calculator", {"expr": "1 + 2"}),
        ToolCall("shell", {"command": "echo hello"}),
        ToolCall("shell", {"command": "rm -rf /tmp/demo"}),
        ToolCall("write_note", {"text": "unknown write"}),
    ]

    print("\n[1] default mode + rules")
    for call in calls:
        out = default_gate.evaluate(call)
        print(f"  {call.name:<10} {call.args!s:<35} -> {out.decision:<5} {out.source}: {out.reason}")

    print("\n[2] auto mode classifier")
    auto_gate = PermissionGate(mode="auto")
    for call in calls:
        out = auto_gate.evaluate(call)
        print(f"  {call.name:<10} {call.args!s:<35} -> {out.decision:<5} {out.source}: {out.reason}")

    kv("principle", "deny > ask > allow")
    print("\n  OK: 权限系统是 agent 能动性的刹车和方向盘。")


if __name__ == "__main__":
    main()

