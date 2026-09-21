"""路径围栏 + 文件工具 + 文件记忆工具。

没有它: `read_file ../../.ssh/id_rsa` —— 模型 (或注入它的文档) 能读写 agent 进程能碰到的任何文件。
关键设计: 所有路径先 resolve() (展开 .. 和符号链接) 再判断是否仍在 root 之内;
先拼接后检查字符串前缀是经典漏洞 (`/root/../etc`、指向外部的 symlink 都能骗过)。
"/" 开头的路径按"沙箱内的虚拟根"解释, 与 Claude memory tool 的 /memories 约定一致。
对应: Claude Code 的工作目录限制; Claude API memory tool (memory_20250818) 的客户端实现。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from llm_agent.core.schema import ToolResult
from llm_agent.core.tools import Tool, _obj


def confine(root: Path, user_path: str) -> Path:
    root = Path(root).resolve()
    target = (root / user_path.lstrip("/")).resolve()
    if target != root and not target.is_relative_to(root):
        raise PermissionError(f"path escapes sandbox: {user_path}")
    return target


class ReadFileTool(Tool):
    name = "read_file"
    description = "Read a text file inside the workspace."
    parameters = _obj(["path"], path={"type": "string"})

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        return ToolResult(self.name, confine(self.root, args["path"]).read_text(encoding="utf-8"))


class WriteFileTool(Tool):
    name = "write_file"
    description = "Write a text file inside the workspace."
    parameters = _obj(["path", "text"], path={"type": "string"}, text={"type": "string"})
    risk = "medium"
    read_only = False

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        target = confine(self.root, args["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(args["text"], encoding="utf-8")
        return ToolResult(self.name, f"wrote {len(args['text'])} chars to {args['path']}")


class MemoryTool(Tool):
    """模型自己管理的跨会话记忆: 它决定记什么、何时查; harness 只提供一个带围栏的目录。"""

    name = "memory"
    description = "Persistent memory directory /memories. Check it before starting a task; save what you learn."
    parameters = _obj(
        ["command", "path"],
        command={"type": "string", "enum": ["view", "create", "str_replace", "delete"]},
        path={"type": "string", "description": "must start with /memories"},
        text={"type": "string"},
        old={"type": "string"},
    )
    risk = "medium"
    read_only = False

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        (self.root / "memories").mkdir(parents=True, exist_ok=True)

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        path = args["path"]
        if path != "/memories" and not path.startswith("/memories/"):
            return ToolResult(self.name, "path must start with /memories", ok=False)
        # 围栏立在 memories/ 而不是 root: 否则 /memories/../x 能写到记忆目录之外
        target, cmd = confine(self.root / "memories", path[len("/memories"):]), args["command"]
        if cmd == "view":
            if target.is_dir():
                # 目录视图带每个文件的首行: 模型不必逐个打开就能判断哪条记忆相关
                files = [f"{p.name}: {p.read_text(encoding='utf-8')[:80]}" for p in sorted(target.iterdir()) if p.is_file()]
                return ToolResult(self.name, "\n".join(files) or "(empty)")
            return ToolResult(self.name, target.read_text(encoding="utf-8"))
        if cmd == "create":
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(args.get("text", ""), encoding="utf-8")
            return ToolResult(self.name, f"saved {args['path']}")
        if cmd == "str_replace":
            body = target.read_text(encoding="utf-8")
            if args.get("old", "") not in body:
                return ToolResult(self.name, "old text not found", ok=False)
            target.write_text(body.replace(args["old"], args.get("text", ""), 1), encoding="utf-8")
            return ToolResult(self.name, f"updated {args['path']}")
        target.unlink()
        return ToolResult(self.name, f"deleted {args['path']}")
