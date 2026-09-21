"""Skills: 渐进式披露 (progressive disclosure) 的任务手册。

没有它: 要么把所有操作手册常驻 system prompt (每次请求都为用不到的内容付 token),
要么不给手册 (模型每次现编流程)。
关键设计 —— 三层加载:
  1. 启动时只读 SKILL.md 的 frontmatter (name + description), 拼成一小段目录常驻上下文
  2. 模型判断相关时调用 `skill` 工具, 正文这时才进上下文
  3. 正文里再引用的脚本 / 附件, 用到才读 (本教学版未实现)
skill 正文由用户自己安装, 属于可信指令 —— 和 fetch 回来的网页 (不可信数据) 是两回事。
对应: Claude Code / Agent Skills 的 SKILL.md 与 Skill 工具。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from llm_agent.core.schema import ToolResult
from llm_agent.core.tools import Tool


def _split(text: str):
    """'---\\nname: x\\n---\\nbody' → ({'name': 'x'}, 'body')。# ponytail: 只支持单行 key: value, 够用; 复杂 YAML 换 pyyaml"""
    _, front, body = text.split("---", 2)
    meta = dict(line.split(":", 1) for line in front.strip().splitlines() if ":" in line)
    return {k.strip(): v.strip() for k, v in meta.items()}, body.strip()


class SkillRegistry:
    def __init__(self, root: Path) -> None:
        self._paths: Dict[str, Path] = {}
        self.descriptions: Dict[str, str] = {}
        for path in sorted(Path(root).glob("*/SKILL.md")):
            meta, _ = _split(path.read_text(encoding="utf-8"))  # 正文此刻被丢弃: 不进内存, 更不进上下文
            self._paths[meta["name"]] = path
            self.descriptions[meta["name"]] = meta["description"]

    def catalog(self) -> str:
        """常驻上下文的全部内容: 每个 skill 一行。"""
        return "## Skills\n" + "\n".join(f"- {n}: {d}" for n, d in self.descriptions.items())

    def load(self, name: str) -> str:
        return _split(self._paths[name].read_text(encoding="utf-8"))[1]

    def all_bodies(self) -> str:
        """反面对照: 不做渐进披露时, 常驻上下文要塞进去的全部正文。"""
        return "\n\n".join(self.load(n) for n in self._paths)


class SkillTool(Tool):
    name = "skill"
    description = "Load the full instructions of a skill listed under '## Skills'."

    def __init__(self, registry: SkillRegistry) -> None:
        self.registry = registry
        self.parameters = {
            "type": "object",
            "properties": {"name": {"type": "string", "enum": sorted(registry.descriptions)}},
            "required": ["name"],
            "additionalProperties": False,
        }

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        return ToolResult(self.name, self.registry.load(args["name"]))
