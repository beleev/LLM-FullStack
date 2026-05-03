"""Tool abstractions and a few safe teaching tools."""

from __future__ import annotations

import ast
import fnmatch
import operator
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from llm_agent.core.schema import ToolCall, ToolResult


class Tool:
    """Minimal callable tool interface.

    A tool has a natural-language description for the model and deterministic
    Python execution for the harness.
    """

    name = "tool"
    description = ""
    reversible = True
    risk = "low"

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        raise NotImplementedError

    def schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "reversible": self.reversible,
            "risk": self.risk,
        }


class ToolRegistry:
    def __init__(self, tools: Optional[Iterable[Tool]] = None) -> None:
        self._tools: Dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def has(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> List[str]:
        return sorted(self._tools)

    def schemas(self) -> List[Dict[str, Any]]:
        return [self._tools[name].schema() for name in self.names()]

    def execute(self, call: ToolCall) -> ToolResult:
        if call.name not in self._tools:
            return ToolResult(call.name, f"ERROR: unknown tool {call.name!r}", ok=False)
        try:
            return self._tools[call.name].execute(call.args)
        except Exception as exc:
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
    """Evaluate a tiny arithmetic expression without Python eval()."""

    def walk(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Num):
            return float(node.n)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.operand))
        raise ValueError(f"unsupported expression: {expr!r}")

    tree = ast.parse(expr, mode="eval")
    return walk(tree)


class CalculatorTool(Tool):
    name = "calculator"
    description = "Compute a small arithmetic expression."
    reversible = True
    risk = "low"

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        expr = str(args.get("expr", "0"))
        value = _safe_eval_arithmetic(expr)
        if value.is_integer():
            value_text = str(int(value))
        else:
            value_text = f"{value:.6g}"
        return ToolResult(self.name, f"{expr} = {value_text}")


class SearchDocsTool(Tool):
    name = "search_docs"
    description = "Search a tiny in-memory documentation corpus."
    reversible = True
    risk = "low"

    def __init__(self, docs: Dict[str, str]) -> None:
        self.docs = docs

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        query = str(args.get("query", "")).lower()
        words = {w for w in query.replace("_", " ").split() if w}
        scored = []
        for title, body in self.docs.items():
            text = f"{title} {body}".lower()
            score = sum(1 for word in words if word in text)
            if query and query in text:
                score += 2
            if score:
                scored.append((score, title, body))
        scored.sort(reverse=True)
        if not scored:
            return ToolResult(self.name, "no matches")
        lines = [f"{title}: {body}" for _, title, body in scored[:3]]
        return ToolResult(self.name, "\n".join(lines))


class WriteNoteTool(Tool):
    name = "write_note"
    description = "Append a short note to the current session notebook."
    reversible = False
    risk = "medium"

    def __init__(self, notes: List[str]) -> None:
        self.notes = notes

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        text = str(args.get("text", "")).strip()
        if not text:
            return ToolResult(self.name, "empty note ignored", ok=False)
        self.notes.append(text)
        return ToolResult(self.name, f"note[{len(self.notes)}] saved: {text}")


class ReadNotesTool(Tool):
    name = "read_notes"
    description = "Read notes from the current session notebook."
    reversible = True
    risk = "low"

    def __init__(self, notes: List[str]) -> None:
        self.notes = notes

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        if not self.notes:
            return ToolResult(self.name, "no notes")
        return ToolResult(self.name, "\n".join(f"{i + 1}. {n}" for i, n in enumerate(self.notes)))


class ShellTool(Tool):
    name = "shell"
    description = "Run a very small simulated shell command."
    reversible = False
    risk = "high"

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        command = str(args.get("command", "")).strip()
        if fnmatch.fnmatch(command, "echo *"):
            return ToolResult(self.name, command[5:])
        if command == "date":
            return ToolResult(self.name, datetime(2026, 5, 3, 12, 0, 0).isoformat())
        # The permission layer should block risky commands before this point.
        # We still simulate rather than executing arbitrary local shell code.
        return ToolResult(self.name, f"simulated shell: {command}")


class WeatherTool(Tool):
    name = "weather"
    description = "Return a fake weather report exposed by a fake MCP server."
    reversible = True
    risk = "low"

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        city = str(args.get("city", "Shanghai"))
        return ToolResult(self.name, f"{city}: sunny, 24C, light wind")

