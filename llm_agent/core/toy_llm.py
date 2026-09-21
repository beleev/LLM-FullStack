"""RuleBasedLLM: 用关键词规则冒充模型, 让 harness 的行为完全确定、可断言。

没有它: 每个 demo 都要 API key, 输出还每次不同 —— 没法写 assert, 也看不清哪部分是 harness 的功劳。
关键设计:
  - 无状态: 每次从 messages 重新推导"本轮用户 prompt 之后已经调过什么", 所以第二个问题不会被上一轮的结果糊弄。
  - 先算出本轮完整计划 [pre → gather → act → post], 再发出"还没做的下一批";
    gather 阶段互不依赖的只读调用一次全发 (并行工具调用), act 阶段一次一个。
  - 只服从可信来源的指令 (hook_context、skill 工具的返回); 其它工具结果一律当数据。
    gullible=True 会关掉这条纪律, 专门给 m12 演示 prompt injection。
它不是模型: 换成 core/claude_llm.py 的 ClaudeLLM, loop 一行不用改。
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Tuple

from llm_agent.core.schema import Message, ModelAction, ToolCall

_META_TOOLS = {"todo_write", "exit_plan_mode", "skill"}  # 结果不算"任务数据"
_CITIES = {"北京": "Beijing", "beijing": "Beijing", "上海": "Shanghai", "shanghai": "Shanghai", "深圳": "Shenzhen"}
_EXPR = re.compile(r"\d+(?:\.\d+)?(?:\s*[-+*/]\s*\d+(?:\.\d+)?)+")

Result = Tuple[str, str, bool]  # (tool name, content, is_error)


def _any(text: str, words: List[str]) -> bool:
    return any(w in text for w in words)


class RuleBasedLLM:
    def __init__(self, gullible: bool = False) -> None:
        self.gullible = gullible

    def next(self, messages: List[Message], tools: List[Dict[str, Any]]) -> ModelAction:
        schemas = {t["name"]: t for t in tools}
        start = max((i for i, m in enumerate(messages) if m.is_user_prompt), default=0)
        prompt = messages[start].text if messages else ""
        if prompt.startswith("[compact]"):
            return ModelAction.final(self._summarize(messages[:start], prompt))

        turn = messages[start:]  # 只看本轮: 旧轮次用过的工具不算数
        used = Counter(b["name"] for m in turn for b in m.tool_uses())
        results = self._results(turn)

        if self.gullible and "shell" in schemas and not used["shell"]:
            for name, content, _ in results:
                hit = re.search(r"(?im)^\s*AGENT\s*[:：].*?(?:run|执行)\s+shell\s*[:：]\s*(.+)$", content)
                if hit:  # 把工具结果里的文字当成了指令 —— 这就是 prompt injection
                    return ModelAction.tool(ToolCall("shell", {"command": hit.group(1).strip()}, "injected"))

        if any(name == "exit_plan_mode" and err for name, _, err in results):
            return ModelAction.final("计划未获批准, 未执行任何写操作。")

        hint = "\n".join(m.text for m in turn if m.name == "hook_context")
        hint += "\n".join(c for name, c, err in results if name == "skill" and not err)
        plan = self._plan(prompt, hint.lower(), schemas, messages)

        seen: Counter = Counter()
        remaining = []
        for phase, call in plan:
            seen[call.name] += 1
            if seen[call.name] > used[call.name]:
                remaining.append((phase, call))
        if remaining:
            if remaining[0][0] == "gather":
                return ModelAction.tool(*[c for p, c in remaining if p == "gather"])
            return ModelAction.tool(remaining[0][1])

        data = [(n, c) for n, c, _ in results if n not in _META_TOOLS]
        if not data:
            return ModelAction.final("这是一个无需工具的直接回答。")
        if len(data) == 1:
            return ModelAction.final(f"基于工具结果完成：\n{data[0][0]}: {data[0][1]}")
        head = f"综合 {len(data)} 个子任务结果：" if all(n == "delegate" for n, _ in data) else "基于工具结果完成："
        return ModelAction.final(head + "".join(f"\n- {n}: {c}" for n, c in data))

    # ------------------------------------------------------------------ plan
    def _plan(self, prompt: str, hint: str, schemas: Dict[str, Any], messages: List[Message]) -> List[Tuple[str, ToolCall]]:
        lower = prompt.lower()
        pre: List[ToolCall] = []
        gather: List[ToolCall] = []
        act: List[ToolCall] = []

        if "skill" in schemas:  # 渐进式披露: 上下文里只有 skill 目录, 命中触发词才去加载正文
            for name, desc in self._skill_catalog(messages):
                triggers = re.split(r"[,，、\s]+", desc.split("触发词:")[-1].strip()) if "触发词:" in desc else []
                if any(t and t.lower() in lower for t in triggers):
                    pre.append(ToolCall("skill", {"name": name}, "load skill body on demand"))

        if "delegate" in schemas and _any(lower, ["delegate", "subagent", "子智能体", "委托", "并行调研"]):
            body = re.split(r"[:：]", prompt, maxsplit=1)[-1]
            parts = [p.strip() for p in re.split(r"[;；]", body) if p.strip()]
            types = schemas["delegate"]["input_schema"]["properties"].get("agent_type", {}).get("enum", [])
            for part in parts if len(parts) > 1 else [prompt]:
                args = {"task": part}
                if types:
                    want = "calculator" if _EXPR.search(part) else "researcher"
                    args["agent_type"] = want if want in types else types[0]
                gather.append(ToolCall("delegate", args, "isolate a subtask in a child agent"))
            return [("gather", c) for c in gather]  # 委托出去的活, 自己不再重复做

        if "calculator" in schemas:
            gather += [ToolCall("calculator", {"expr": e}, "need arithmetic") for e in _EXPR.findall(prompt)]
        if "search_docs" in schemas and (
            _any(lower, ["search", "搜索", "检索", "查找", "找文档", "调研"]) or "search_docs" in hint
        ):
            gather.append(ToolCall("search_docs", {"query": prompt}, "need external knowledge"))
        weather = next((n for n in sorted(schemas) if n == "weather" or n.endswith("get_weather")), None)
        if weather and _any(lower, ["weather", "天气"]):
            cities = list(dict.fromkeys(c for k, c in _CITIES.items() if k in lower)) or ["Shanghai"]
            gather += [ToolCall(weather, {"city": c}, "external (MCP) tool") for c in cities]
        for tool, pattern in (("fetch_doc", r"(?:fetch|抓取)\s*[:：]?\s*(\S+)"), ("read_file", r"读取文件\s*[:：]?\s*(\S+)")):
            hit = re.search(pattern, prompt, re.IGNORECASE)
            if tool in schemas and hit:
                gather.append(ToolCall(tool, {"name" if tool == "fetch_doc" else "path": hit.group(1)}, "read on demand"))
        if "read_notes" in schemas and _any(lower, ["read note", "读取笔记", "查看笔记"]):
            gather.append(ToolCall("read_notes", {}, "read notebook"))
        if "memory" in schemas and _any(lower, ["查看长期记忆", "recall memory"]):
            gather.append(ToolCall("memory", {"command": "view", "path": "/memories"}, "check memory first"))

        if "shell" in schemas and _any(lower, ["shell", "运行", "执行"]):
            hit = re.search(r"(?:shell|运行|执行)[:：]?\s*(.+)", prompt, re.IGNORECASE)
            act.append(ToolCall("shell", {"command": hit.group(1).strip() if hit else "echo hello"}, "user asked"))
        hit = re.search(r"写入文件\s*(\S+?)\s*[:：]\s*(.+)", prompt)
        if "write_file" in schemas and hit:
            act.append(ToolCall("write_file", {"path": hit.group(1), "text": hit.group(2).strip()}, "user asked"))
        hit = re.search(r"存入长期记忆\s*[:：]\s*(.+)", prompt)
        if "memory" in schemas and hit:
            args = {"command": "create", "path": "/memories/notes.md", "text": hit.group(1).strip()}
            act.append(ToolCall("memory", args, "persist across sessions"))
        elif "write_note" in schemas and _any(lower, ["note", "笔记", "记住", "写入", "保存"]):
            act.append(ToolCall("write_note", {"text": self._last_data(messages) or prompt}, "persist a note"))

        if "exit_plan_mode" in schemas and act:  # plan 模式: 写操作之前先交计划
            steps = "\n".join(f"{i}. {c.name} — {c.reason}" for i, c in enumerate(gather + act, 1))
            pre.append(ToolCall("exit_plan_mode", {"plan": steps}, "ask approval before acting"))

        plan = [("pre", c) for c in pre] + [("gather", c) for c in gather] + [("act", c) for c in act]
        if "todo_write" in schemas and len(gather) + len(act) >= 2:  # 多步任务才值得列 todo
            todo = lambda status: ToolCall(  # noqa: E731
                "todo_write",
                {"todos": [{"content": f"{c.name} — {c.reason}", "status": status} for c in gather + act]},
                "track multi-step work",
            )
            plan = [("pre", todo("pending"))] + plan + [("post", todo("completed"))]
        return plan

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _results(messages: List[Message]) -> List[Result]:
        names = {b["id"]: b["name"] for m in messages for b in m.tool_uses()}
        return [
            (names.get(b["tool_use_id"], "?"), str(b["content"]), bool(b.get("is_error")))
            for m in messages
            for b in m.tool_results()
        ]

    def _last_data(self, messages: List[Message]) -> str:
        skip = _META_TOOLS | {"write_note"}
        data = [c for n, c, err in self._results(messages) if n not in skip and not err]
        return data[-1] if data else ""

    @staticmethod
    def _skill_catalog(messages: List[Message]) -> List[Tuple[str, str]]:
        text = "\n".join(m.text for m in messages if m.role == "system" and "## Skills" in m.text)
        return re.findall(r"(?m)^- ([\w-]+): (.+)$", text.split("## Skills")[-1]) if text else []

    def _summarize(self, messages: List[Message], prompt: str) -> str:
        """"LLM 写摘要"的替身: 真实模型会读全文再写, 这里按结构抽取, 但产物形状相同。"""
        goals = [m.text for m in messages if m.is_user_prompt]
        calls = [f"{b['name']}({', '.join(map(str, b['input'].values()))[:30]})" for m in messages for b in m.tool_uses()]
        # 优先取助手自己下过的结论 (已是提炼过的信息), 没有才退回工具结果的开头
        facts = [m.text.split("\n", 1)[-1][:60] for m in messages if m.role == "assistant" and not m.tool_uses()] or [
            f"{n}: {c[:60]}" for n, c, err in self._results(messages) if not err and n not in _META_TOOLS
        ]
        fields = {"目标": goals, "已完成": calls, "关键结果": facts}
        for prior in (m.text for m in messages if m.name == "compact_summary"):  # 滚动摘要: 合并上一份, 而不是套娃
            for line in prior.splitlines():
                key, _, value = line.partition(": ")
                if key in fields and value != "无":
                    fields[key].insert(0, value)
        keep = prompt.split("必须保留:", 1)[1].strip() if "必须保留:" in prompt else ""
        lines = [f"{key}: {'; '.join(values) or '无'}" for key, values in fields.items()]
        return "\n".join(lines + ([f"保留: {keep}"] if keep else []))
