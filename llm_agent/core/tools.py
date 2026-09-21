"""工具抽象 + 几个安全的教学工具 (全部模拟 / 纯计算, 不碰真实 shell 与网络)。

没有它: 模型只能"说", 不能"做"; 没有 JSON Schema, 模型只能猜参数名,
harness 也无法在执行前拦下畸形参数。
关键设计: schema() 输出 Claude API 的 tool 定义 {name, description, input_schema};
risk / read_only / untrusted_output 是给 harness (权限、护栏) 看的元数据, 不发给模型。
对应: Anthropic tool use 的 `tools=[...]`; Claude Code 内置工具的只读 / 需审批划分。
"""

from __future__ import annotations

import ast
import operator
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional

from llm_agent.core.schema import ToolCall, ToolResult
from llm_agent.core.utils import tokenize

_JSON_TYPES = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def validate_args(schema: Dict[str, Any], args: Any) -> List[str]:
    """JSON Schema 的最小子集校验: required / type / enum / additionalProperties。

    模型输出是不可信输入, 执行前校验 = 信任边界上的输入验证。
    # ponytail: 只校验一层, 嵌套 schema 请换 jsonschema 库 (或 API 的 strict: true)
    """
    if not isinstance(args, dict):
        return ["arguments must be an object"]
    props = schema.get("properties", {})
    errors = [f"missing required '{k}'" for k in schema.get("required", []) if k not in args]
    for key, value in args.items():
        if key not in props:
            if schema.get("additionalProperties") is False:
                errors.append(f"unexpected '{key}'")
            continue
        spec = props[key]
        expected = _JSON_TYPES.get(spec.get("type", ""))
        # bool 是 int 的子类, 不排除的话 True 会被当成合法 number
        if expected and (not isinstance(value, expected) or (isinstance(value, bool) and spec["type"] != "boolean")):
            errors.append(f"'{key}' should be {spec['type']}, got {type(value).__name__}")
        elif "enum" in spec and value not in spec["enum"]:
            errors.append(f"'{key}' must be one of {spec['enum']}")
    return errors


def _obj(required: List[str], **props: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


class Tool:
    name = "tool"
    description = ""
    parameters: Dict[str, Any] = _obj([])
    risk = "low"  # low / medium / high, 给 auto 模式分类器用
    read_only = True  # plan 模式只放行只读工具
    untrusted_output = False  # 输出来自外部世界 (网页 / 文档), 可能夹带 prompt injection

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        raise NotImplementedError

    def schema(self) -> Dict[str, Any]:
        return {"name": self.name, "description": self.description, "input_schema": self.parameters}


class ToolRegistry:
    def __init__(self, tools: Optional[Iterable[Tool]] = None) -> None:
        self._tools: Dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return sorted(self._tools)

    def schemas(self) -> List[Dict[str, Any]]:
        # 排序 = 确定性的工具列表; 真实 API 里工具顺序一变, prompt cache 就全失效
        return [self._tools[name].schema() for name in self.names()]

    def execute(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(call.name, f"ERROR: unknown tool {call.name!r}", ok=False)
        errors = validate_args(tool.parameters, call.args)
        if errors:  # 把错误原样告诉模型, 它下一轮可以自己改参数
            return ToolResult(call.name, "INVALID_ARGS: " + "; ".join(errors), ok=False)
        try:
            return tool.execute(call.args)
        except Exception as exc:  # 工具异常不能炸掉 loop, 变成 is_error 结果
            return ToolResult(call.name, f"ERROR: {exc}", ok=False)


_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval_arithmetic(expr: str) -> float:
    """白名单 AST 求值, 绝不用 eval()。"""

    def walk(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        # ast.Num 在 3.12 起被移除; Constant 还可能是 str/bool, 必须显式限定数值
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.operand))
        raise ValueError(f"unsupported expression: {expr!r}")

    return walk(ast.parse(expr, mode="eval"))


class CalculatorTool(Tool):
    name = "calculator"
    description = "Compute a small arithmetic expression."
    parameters = _obj(["expr"], expr={"type": "string", "description": "e.g. 2 + 3 * 4"})

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        value = _safe_eval_arithmetic(args["expr"])
        text = str(int(value)) if value.is_integer() else f"{value:.6g}"
        return ToolResult(self.name, f"{args['expr']} = {text}")


class SearchDocsTool(Tool):
    """关键词计数检索 (基线)。m08 的 VectorSearchTool 同名同 schema, 可热替换。"""

    name = "search_docs"
    description = "Search a tiny in-memory documentation corpus."
    parameters = _obj(["query"], query={"type": "string"})

    def __init__(self, docs: Dict[str, str]) -> None:
        self.docs = docs

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        words = set(tokenize(args["query"]))
        scored = []
        for title, body in self.docs.items():
            score = len(words & set(tokenize(f"{title} {body}")))  # 每个命中词等权 1 分
            if score:
                scored.append((score, title, body))
        scored.sort(reverse=True)
        if not scored:
            return ToolResult(self.name, "no matches")
        return ToolResult(self.name, "\n".join(f"{t}: {b}" for _, t, b in scored[:3]))


class WriteNoteTool(Tool):
    name = "write_note"
    description = "Append a short note to the current session notebook."
    parameters = _obj(["text"], text={"type": "string"})
    risk = "medium"
    read_only = False

    def __init__(self, notes: List[str]) -> None:
        self.notes = notes

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        text = args["text"].strip()
        if not text:
            return ToolResult(self.name, "empty note ignored", ok=False)
        self.notes.append(text)
        return ToolResult(self.name, f"note[{len(self.notes)}] saved")


class ReadNotesTool(Tool):
    name = "read_notes"
    description = "Read notes from the current session notebook."

    def __init__(self, notes: List[str]) -> None:
        self.notes = notes

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        if not self.notes:
            return ToolResult(self.name, "no notes")
        return ToolResult(self.name, "\n".join(f"{i + 1}. {n}" for i, n in enumerate(self.notes)))


class ShellTool(Tool):
    """模拟 shell: 永远不执行真实命令, 只记录"假如执行了什么"。"""

    name = "shell"
    description = "Run a (simulated) shell command."
    parameters = _obj(["command"], command={"type": "string"})
    risk = "high"
    read_only = False

    def __init__(self) -> None:
        self.executed: List[str] = []  # demo 用它断言"危险命令从未到达执行层"

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        command = args["command"].strip()
        self.executed.append(command)
        if command.startswith("echo "):
            return ToolResult(self.name, command[5:])
        if command == "date":
            return ToolResult(self.name, datetime(2026, 5, 3, 12, 0, 0).isoformat())
        return ToolResult(self.name, f"simulated shell: {command}")


class FetchDocTool(Tool):
    """模拟"抓取外部文档": 内容来自不可信来源, 是 prompt injection 的入口 (m12)。"""

    name = "fetch_doc"
    description = "Fetch an external document by name (simulated, no network)."
    parameters = _obj(["name"], name={"type": "string"})
    untrusted_output = True

    def __init__(self, pages: Dict[str, str]) -> None:
        self.pages = pages

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        if args["name"] not in self.pages:
            return ToolResult(self.name, f"not found: {args['name']}", ok=False)
        return ToolResult(self.name, self.pages[args["name"]])


class TodoWriteTool(Tool):
    """计划即状态: 模型每次整表覆写 todo, harness 只负责保存和展示。"""

    name = "todo_write"
    description = "Replace the whole todo list. Use it to plan multi-step work and track progress."
    parameters = _obj(
        ["todos"],
        todos={"type": "array", "description": "items: {content, status: pending|in_progress|completed}"},
    )
    # 只改 agent 自己的计划状态, 不碰外部世界 → plan 模式下也允许
    read_only = True

    def __init__(self) -> None:
        self.todos: List[Dict[str, str]] = []
        self.history: List[List[Dict[str, str]]] = []

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        for item in args["todos"]:
            if item.get("status") not in ("pending", "in_progress", "completed"):
                return ToolResult(self.name, f"bad status in {item}", ok=False)
        self.todos = list(args["todos"])
        self.history.append(self.todos)
        done = sum(t["status"] == "completed" for t in self.todos)
        return ToolResult(self.name, f"todos updated: {done}/{len(self.todos)} completed")


class ExitPlanModeTool(Tool):
    """模型提交计划 → 人审批 → 通过才把权限模式从 plan 切走。"""

    name = "exit_plan_mode"
    description = "Present the plan for approval. Write actions stay blocked until it is approved."
    parameters = _obj(["plan"], plan={"type": "string"})

    def __init__(self, gate: Any, approve: Callable[[str], bool], next_mode: str = "accept_edits") -> None:
        self.gate, self.approve, self.next_mode = gate, approve, next_mode

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        if not self.approve(args["plan"]):
            return ToolResult(self.name, "plan rejected by user; stay in plan mode", ok=False)
        self.gate.mode = self.next_mode
        return ToolResult(self.name, f"plan approved; mode -> {self.next_mode}")
