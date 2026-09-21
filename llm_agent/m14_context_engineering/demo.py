"""M14 — Context Engineering: 把"放什么进上下文"当成工程问题来做。

没有它: 上下文只增不减。每一轮都在为几十轮前的工具输出付费, 模型的注意力被稀释, 最后撑爆窗口。
关键设计 (都已接进 core/agent.py 的 loop):
  1. 工具结果清理  超预算先把旧 tool_result 正文换成占位符 —— 只改发给模型的视图, transcript 不动。
  2. 摘要压缩      还超 → 先触发 pre_compact hook (人指定必留信息) → 模型写摘要 → 真的替换掉历史;
                   JSONL 追加 compact_boundary, 所以 resume 读回来的上下文也变小了 (文件本身只增不减, 供审计)。
  3. 即时检索      上下文里只放"有哪些文件"的索引, 内容用到才读 (just-in-time), 而不是预先全塞进去。
  4. 记忆工具      模型自己往 /memories 写, 新会话再读回来 —— 跨会话的状态不占用任何一轮的上下文。
对应: Anthropic "effective context engineering for AI agents"; Claude API 的 context editing / compaction / memory tool。
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List

from llm_agent.core import (
    Agent,
    HookManager,
    JsonlSessionStore,
    MemoryTool,
    Message,
    ModelAction,
    PermissionGate,
    ReadFileTool,
    RuleBasedLLM,
    SearchDocsTool,
    ToolCall,
    ToolRegistry,
    total_chars,
    validate_transcript,
)
from llm_agent.core.utils import banner, estimate_tokens, kv

TOPICS = ["kv cache", "lora", "dpo", "sampling", "rope"]
DOCS = {t.replace(" ", "_"): f"{t} key fact: FACT-{i}. " + f"long background about {t}. " * 8 for i, t in enumerate(TOPICS, 1)}


class SpyLLM:
    """记录模型每次实际看到的上下文 —— 上下文工程的第一步是能看见上下文。"""

    def __init__(self) -> None:
        self.inner, self.contexts = RuleBasedLLM(), []

    def next(self, messages: List[Message], tools: List[Dict[str, Any]]) -> ModelAction:
        if tools:  # 摘要请求不带 tools, 不计入
            self.contexts.append(list(messages))
        return self.inner.next(messages, tools)


def long_session(path: Path, compaction: str, hooks: HookManager = None):
    spy = SpyLLM()
    agent = Agent(
        spy,
        ToolRegistry([SearchDocsTool(DOCS)]),
        PermissionGate("auto"),
        hooks=hooks,
        store=JsonlSessionStore(path),
        context_budget_chars=900,
        compaction=compaction,
        name=compaction,
    )
    for topic in TOPICS:
        agent.run(f"检索 {topic}", verbose=False)
    return agent, spy


def main() -> None:
    banner("M14 - Context Engineering")

    with tempfile.TemporaryDirectory(prefix="llm_agent_ctx_") as tmp:
        tmp_path = Path(tmp)

        print("\n[1] 5 轮检索, 预算 900 字符: 清理 → 压缩 逐级触发")
        compact_calls = []
        hooks = HookManager()
        hooks.register("pre_compact", lambda messages: compact_calls.append(len(messages)) or "用户只关心 FACT 编号")
        agent, spy = long_session(tmp_path / "summary.jsonl", "summary", hooks)

        assert any("[cleared:" in m.text for ctx in spy.contexts for m in ctx)  # 第 1 档生效过
        assert agent.compactions >= 1 and len(compact_calls) == agent.compactions  # 第 2 档生效, 且 hook 每次都被触发
        summary = agent.messages[0]
        print("  " + summary.text.replace("\n", "\n  ")[:600])
        assert summary.name == "compact_summary" and "检索 kv cache" in summary.text and "用户只关心 FACT 编号" in summary.text
        kv("compactions", agent.compactions)
        kv("peak context tokens", agent.usage["peak_context_tokens"])

        print("\n[2] 压缩真的缩小了'恢复出来的会话', 审计日志一条没少")
        store = JsonlSessionStore(tmp_path / "summary.jsonl")
        resumed, audit = store.load(), store.load_all()
        kv("resume 视图", f"{len(resumed)} 条 / {total_chars(resumed)} 字符")
        kv("审计全量", f"{len(audit)} 条 / {total_chars(audit)} 字符")
        assert total_chars(resumed) < total_chars(audit) * 0.6 and len(audit) == 4 * len(TOPICS)
        assert [m.text for m in resumed] == [m.text for m in agent.messages]  # 磁盘上恢复出的 == 内存里压缩后的
        assert validate_transcript(resumed) == []  # 压缩没有切断任何 tool_use/tool_result 配对
        agent_b = Agent(RuleBasedLLM(), ToolRegistry([SearchDocsTool(DOCS)]), PermissionGate("auto"), store=store, load_history=True, name="resumed")
        assert "FACT-2" in agent_b.run("检索 lora", verbose=False)

        print("\n[3] 对照: truncate 策略只裁剪视图 —— 每轮重新裁, 历史和 resume 都不会变小")
        agent_t, _ = long_session(tmp_path / "truncate.jsonl", "truncate")
        store_t = JsonlSessionStore(tmp_path / "truncate.jsonl")
        assert agent_t.compactions == 0 and len(store_t.load()) == len(store_t.load_all()) == len(agent_t.messages)
        kv("truncate resume 视图", f"{len(store_t.load())} 条 (没有变小)")
        # 更糟的是: 截断把 tool_result 拍平成普通 user 文本, 模型把它当成了新的用户输入, 后几轮直接答非所问
        derailed = [m.text for m in agent_t.messages if m.role == "assistant" and "无需工具" in m.text]
        kv("truncate 下答非所问的轮数", f"{len(derailed)}/{len(TOPICS)}")
        good = [m.text for m in agent.messages if m.role == "assistant" and "无需工具" in m.text]
        assert derailed and not good

        print("\n[4] 即时检索 vs 预加载")
        docs_dir = tmp_path / "workspace" / "docs"
        docs_dir.mkdir(parents=True)
        for name, body in DOCS.items():
            (docs_dir / f"{name}.md").write_text(body, encoding="utf-8")
        index = "可读文件:\n" + "\n".join(f"- docs/{name}.md" for name in DOCS)
        preload = "知识库全文:\n" + "\n".join(DOCS.values())
        jit = Agent(RuleBasedLLM(), ToolRegistry([ReadFileTool(tmp_path / "workspace")]), PermissionGate("auto"), system_prompt=index, name="jit")
        final = jit.run("读取文件 docs/lora.md", verbose=False)
        kv("预加载: 每次调用都要带的 tokens", estimate_tokens(preload))
        kv("即时检索: 峰值上下文 tokens", jit.usage["peak_context_tokens"])
        assert "FACT-2" in final and jit.usage["peak_context_tokens"] < estimate_tokens(preload) * 0.5

        print("\n[5] 记忆工具: 会话 1 写, 全新的会话 2 读")
        memory_root = tmp_path / "agent_home"
        Agent(RuleBasedLLM(), ToolRegistry([MemoryTool(memory_root)]), PermissionGate("auto"), name="s1").run(
            "存入长期记忆: 这个项目用 pytest, 不用 unittest"
        )
        assert (memory_root / "memories" / "notes.md").exists()
        for escape in ("/memories/../escape.md", "/memories_evil/x.md"):  # 围栏立在 memories/ 上, 不是 root 上
            call = ToolCall("memory", {"command": "create", "path": escape, "text": "x"})
            assert not ToolRegistry([MemoryTool(memory_root)]).execute(call).ok
        assert not (memory_root / "escape.md").exists()
        session2 = Agent(RuleBasedLLM(), ToolRegistry([MemoryTool(memory_root)]), PermissionGate("auto"), name="s2")
        recalled = session2.run("查看长期记忆")
        assert "pytest" in recalled and len(session2.messages) == 4  # 没有任何历史, 知识来自文件

    print("\n  OK: 上下文是预算 —— 便宜的先清, 贵的再压, 能现取的不预存, 要跨会话的写文件。")


if __name__ == "__main__":
    main()
