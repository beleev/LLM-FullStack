"""权限门: deny > ask > allow > 模式兜底。

没有它: 模型 (或被注入的文档) 说跑什么就跑什么。
关键设计:
  - 门只评估"最终要执行的那个调用" —— hook 改写之后才过门 (见 agent.py), 改写绕不过权限。
  - 规则按 工具名 glob + 参数 glob 匹配; shell 命令先归一化、复合命令逐段评估。
  - plan 模式 = 只读; auto 模式 = 按工具风险分级, 危险词只看 shell 的 command 和文件工具的 path。
⚠ 字符串黑名单本质上很弱 (见 normalize_command 注释): 它在枚举"坏", 而坏是无穷的。
  真实系统靠 命令解析 + allowlist + OS 沙箱, deny 规则只是最后一道便宜的网。
对应: Claude Code 的 permission modes (default/plan/acceptEdits/bypassPermissions) 与 allow/ask/deny 规则。
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from llm_agent.core.schema import ToolCall


class Decision:
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass
class PermissionRule:
    tool: str  # 工具名 glob, 如 "shell" / "mcp__weather__*" / "*"
    pattern: str  # 参数文本 glob, 空 = 匹配该工具的任何调用
    decision: str
    reason: str = ""


@dataclass
class PermissionOutcome:
    allowed: bool
    decision: str
    source: str  # rule / human / auto / 模式名 —— 审计时要知道"是谁放行的"
    reason: str


_FLAG = re.compile(r"(\*?)-([a-z]+)(\*?)$")


def normalize_command(text: str) -> str:
    """大小写、空白、短 flag 的顺序与拆分归一: `RM  -r -f /` → `rm -fr /`。

    这只堵住最廉价的绕过。堵不住的还有无穷多:
      /bin/rm、rm --recursive --force、$(echo rm) -rf、find / -delete、python -c 'shutil.rmtree'...
    所以真实系统不靠黑名单兜底: 把命令解析成 AST 逐段检查 + 默认拒绝的 allowlist
    + OS 级沙箱 (seatbelt / bubblewrap / 容器) 限制文件系统与网络。
    """
    out: List[str] = []
    for tok in text.lower().split():
        m = _FLAG.match(tok)
        prev = _FLAG.match(out[-1]) if out else None
        if m and prev and not prev.group(3) and not m.group(1):  # 相邻短 flag 合并: -r -f → -fr
            letters = "".join(sorted(set(prev.group(2) + m.group(2))))
            out[-1] = f"{prev.group(1)}-{letters}{m.group(3)}"
        elif m:
            out.append(f"{m.group(1)}-{''.join(sorted(set(m.group(2))))}{m.group(3)}")
        else:
            out.append(tok)
    return " ".join(out)


_SHELL_DANGER = ["rm ", "sudo", "curl ", "wget ", "ssh ", "chmod ", ">", "token", "secret"]
_SENSITIVE_PATH = [".env", "secret", "id_rsa", ".ssh", "credentials"]


class PermissionGate:
    """模式 (与 Claude Code 同名):
    plan                只读; 写操作一律拒绝, 直到计划获批后切换模式
    default             没有规则命中就问人
    accept_edits        低/中风险直接放行, 高风险仍问人
    auto                规则分类器: 低风险放行, 明显危险拒绝, 拿不准问人
    dont_ask            未命中规则的一律放行 (deny 规则仍生效)
    bypass_permissions  同上, 且跳过 ask 规则
    """

    def __init__(
        self,
        mode: str = "default",
        rules: Optional[List[PermissionRule]] = None,
        ask_policy: Optional[Callable[[ToolCall], bool]] = None,
    ) -> None:
        self.mode = mode
        self.rules = rules or []
        self.ask_policy = ask_policy  # 模拟"人": demo 不能 input(), 用函数代替

    def evaluate(self, call: ToolCall, tool: Optional[Any] = None) -> PermissionOutcome:
        if call.name == "shell":
            # `echo hi && rm -rf /` 整串能匹配 allow "echo *" —— 必须拆开, 任一段不过则整体不过
            parts = [p.strip() for p in re.split(r"&&|\|\||;|\||&|\n", str(call.args.get("command", ""))) if p.strip()]
            if len(parts) > 1:
                outcome = None
                for part in parts:  # 惰性: 第一段被拒就停, 不为后面的段白白打扰人
                    outcome = self._evaluate_one(ToolCall("shell", {"command": part}), tool)
                    if not outcome.allowed:
                        break
                return outcome
        return self._evaluate_one(call, tool)

    def _evaluate_one(self, call: ToolCall, tool: Optional[Any]) -> PermissionOutcome:
        for decision in (Decision.DENY, Decision.ASK, Decision.ALLOW):  # 顺序就是优先级
            if decision != Decision.DENY and self.mode == "plan":
                break  # plan 模式下 allow 规则也不能放行写操作
            if decision == Decision.ASK and self.mode == "bypass_permissions":
                continue
            if decision == Decision.ALLOW and call.name == "shell" and re.search(r"\$\(|`", str(call.args.get("command", ""))):
                continue  # 命令替换 $(...) / `...` 里能藏任何东西: `echo $(rm -rf /)` 不配享受 allow "echo *"
            for rule in self.rules:
                if rule.decision == decision and self._matches(rule, call):
                    if decision == Decision.ASK:
                        return self._ask(call, rule.reason or "ask rule")
                    return PermissionOutcome(decision == Decision.ALLOW, decision, "rule", rule.reason or f"{decision} rule")

        risk = getattr(tool, "risk", "high")  # 不认识的工具按最高风险处理
        if self.mode == "plan":
            if getattr(tool, "read_only", False):
                return PermissionOutcome(True, Decision.ALLOW, "plan", "read-only tool")
            return PermissionOutcome(False, Decision.DENY, "plan", "plan mode is read-only until the plan is approved")
        if self.mode in ("dont_ask", "bypass_permissions"):
            return PermissionOutcome(True, Decision.ALLOW, self.mode, "mode allows unknown action")
        if self.mode == "accept_edits":
            if risk == "high":
                return self._ask(call, "high-risk tool still needs approval")
            return PermissionOutcome(True, Decision.ALLOW, "accept_edits", "low/medium risk")
        if self.mode == "auto":
            return self._auto_classify(call, risk)
        return self._ask(call, "default mode asks for unknown action")

    def _ask(self, call: ToolCall, reason: str) -> PermissionOutcome:
        approved = bool(self.ask_policy(call)) if self.ask_policy else False  # 没人可问 = 拒绝 (fail closed)
        return PermissionOutcome(
            approved,
            Decision.ALLOW if approved else Decision.DENY,
            "human",
            reason + ("; approved" if approved else "; denied"),
        )

    def _auto_classify(self, call: ToolCall, risk: str) -> PermissionOutcome:
        # 危险词只对"有这种语义"的参数生效: shell 的 command、文件工具的 path。
        # 旧版对所有工具的所有参数做子串匹配, `search_docs "tokenizer"` / `calculator "5 > 3"` 全被误杀。
        if call.name == "shell":
            command = normalize_command(str(call.args.get("command", ""))) + " "
            if any(x in command for x in _SHELL_DANGER):
                return PermissionOutcome(False, Decision.DENY, "auto", "classifier saw risky shell pattern")
        path = str(call.args.get("path", "")).lower()
        if path and any(x in path for x in _SENSITIVE_PATH):
            return PermissionOutcome(False, Decision.DENY, "auto", "sensitive path")
        if risk == "low":
            return PermissionOutcome(True, Decision.ALLOW, "auto", "low-risk tool")
        if risk == "medium":
            return PermissionOutcome(True, Decision.ALLOW, "auto", "bounded local write")
        return self._ask(call, "classifier unsure about high-risk tool")

    def _matches(self, rule: PermissionRule, call: ToolCall) -> bool:
        if not fnmatch.fnmatch(call.name, rule.tool):
            return False
        if not rule.pattern:
            return True
        text = " ".join(str(v) for v in call.args.values())
        # 规则和命令走同一个归一化, 写规则的人不必关心 -rf / -fr
        return fnmatch.fnmatch(normalize_command(text), normalize_command(rule.pattern))
