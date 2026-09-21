"""M03 — Permissions: deny > ask > allow > 模式兜底。

没有它: 模型 (或注入它的文档) 想跑什么就跑什么; 而"每步都问人"又会让 agent 没法用。
关键设计:
  - 规则三态且有序: deny 永远赢; ask 强制问人 (即使有更宽的 allow); 都没命中才轮到模式。
  - auto 模式按工具风险分级, 危险词只检查 shell 的 command 与文件工具的 path —— 不会误杀 `search "tokenizer"`。
  - 复合命令逐段评估; 命令先归一化。但字符串黑名单本质上很弱, [4] 会当场演示绕过。
对应: Claude Code settings.json 的 permissions.allow / ask / deny 与 permission modes。
"""

from __future__ import annotations

from llm_agent.core import (
    CalculatorTool,
    Decision,
    PermissionGate,
    PermissionRule,
    SearchDocsTool,
    ShellTool,
    ToolCall,
    ToolRegistry,
    WriteNoteTool,
    normalize_command,
)
from llm_agent.core.utils import banner


def main() -> None:
    banner("M03 - Permission Gate")
    tools = ToolRegistry([CalculatorTool(), SearchDocsTool({}), ShellTool(), WriteNoteTool([])])

    def show(gate: PermissionGate, call: ToolCall, expect: str, source: str) -> None:
        out = gate.evaluate(call, tools.get(call.name))
        print(f"  {call.name:<11} {str(list(call.args.values())[0]):<34} -> {out.decision:<5} {out.source}: {out.reason}")
        assert (out.decision, out.source) == (expect, source), (call, out)

    rules = [
        PermissionRule("shell", "*rm -rf*", Decision.DENY, "destructive"),
        PermissionRule("shell", "git push*", Decision.ASK, "pushing is visible to others"),
        PermissionRule("shell", "git *", Decision.ALLOW, "git is fine"),
        PermissionRule("shell", "echo *", Decision.ALLOW, "echo is harmless"),
        PermissionRule("calculator", "", Decision.ALLOW, "pure function"),
    ]
    asked = []

    def human(call: ToolCall) -> bool:  # demo 不能 input(): 用固定策略模拟"人", 并记下被问过什么
        asked.append(call.args)
        return False

    print("\n[1] default 模式 + 规则: deny > ask > allow")
    gate = PermissionGate("default", rules, ask_policy=human)
    show(gate, ToolCall("calculator", {"expr": "1 + 2"}), "allow", "rule")
    show(gate, ToolCall("shell", {"command": "git status"}), "allow", "rule")
    show(gate, ToolCall("shell", {"command": "git push origin main"}), "deny", "human")  # ask 规则压过更宽的 allow
    show(gate, ToolCall("shell", {"command": "rm -rf /tmp/demo"}), "deny", "rule")
    show(gate, ToolCall("write_note", {"text": "no rule matches"}), "deny", "human")  # 没规则 → 问人 → 人说不
    assert asked == [{"command": "git push origin main"}, {"text": "no rule matches"}]

    print("\n[2] 归一化堵住最廉价的绕过; 复合命令逐段评估")
    print(f"  normalize('RM  -r -f /') = {normalize_command('RM  -r -f /')!r}")
    for cmd in ("rm -fr /tmp/demo", "RM  -rf /tmp/demo", "rm -r -f /tmp/demo"):
        show(gate, ToolCall("shell", {"command": cmd}), "deny", "rule")
    show(gate, ToolCall("shell", {"command": "echo hi && rm -rf /"}), "deny", "rule")  # 整串其实匹配 allow "echo *"
    show(gate, ToolCall("shell", {"command": "echo hi & find / -delete"}), "deny", "human")  # 后台符 & 也是分隔符
    show(gate, ToolCall("shell", {"command": "echo $(find / -delete)"}), "deny", "human")  # 命令替换: allow 规则不生效

    print("\n[3] auto 模式: 按风险分级, 危险词只看 shell command / 文件 path")
    auto = PermissionGate("auto")
    show(auto, ToolCall("search_docs", {"query": "tokenizer bpe secret sauce"}), "allow", "auto")  # 只按子串匹配会误杀: "tokenizer" 里有 "token"
    show(auto, ToolCall("calculator", {"expr": "5 > 3"}), "allow", "auto")  # 同理: "5 > 3" 里有 ">"
    show(auto, ToolCall("write_note", {"text": "bounded write"}), "allow", "auto")
    show(auto, ToolCall("shell", {"command": "cat token.txt > /tmp/x"}), "deny", "auto")
    show(auto, ToolCall("shell", {"command": "ls"}), "deny", "human")  # 高风险且拿不准 → 问人; 没人 → fail closed

    print("\n[4] 诚实的部分: 字符串黑名单挡不住的写法 (以下全部漏过 deny 规则)")
    loose = PermissionGate("dont_ask", rules)
    for cmd in ("/bin/rm --recursive --force /", "find / -delete", "python -c 'import shutil; shutil.rmtree(\"/\")'"):
        show(loose, ToolCall("shell", {"command": cmd}), "allow", "dont_ask")
    print("  → 所以真实系统: 解析命令 AST + 默认拒绝的 allowlist + OS 沙箱; deny 列表只是最外层的便宜网。")

    print("\n[5] plan 模式: 只读; allow 规则也放不了写操作")
    plan = PermissionGate("plan", [PermissionRule("write_note", "", Decision.ALLOW)])
    show(plan, ToolCall("search_docs", {"query": "anything"}), "allow", "plan")
    show(plan, ToolCall("write_note", {"text": "blocked until approved"}), "deny", "plan")

    print("\n  OK: 权限是 agent 能动性的刹车和方向盘 —— 以及一份对黑名单局限性的清醒认识。")


if __name__ == "__main__":
    main()
