"""M15 — 接真实模型 (opt-in): 同一个 loop, 把 RuleBasedLLM 换成 ClaudeLLM。

不在 run_all 里。没有 SDK 或没有 ANTHROPIC_API_KEY 时, 只跑离线的格式转换检查, 然后礼貌退出 —— 不发任何网络请求。
关键设计: harness 对模型的全部依赖就是 LLM 协议的 next(); 工具 / 权限 / hook / 持久化一行不用改。
[1] 离线: 用 toy LLM 跑出一段 transcript, 转成 Messages API 请求体, 检查它满足 API 的结构约束。
[2] 在线: 真实模型 + 同一个 calculator 工具 + 同一个权限门。

    pip install anthropic && export ANTHROPIC_API_KEY=... && python -m llm_agent.m15_claude_api.demo
"""

from __future__ import annotations

import json

from llm_agent.core import Agent, CalculatorTool, HookManager, HookResult, PermissionGate, RuleBasedLLM, ToolRegistry, validate_transcript
from llm_agent.core.claude_llm import ClaudeLLM, to_api_messages
from llm_agent.core.utils import banner, kv


def main() -> None:
    banner("M15 - Real model adapter (opt-in)")

    print("\n[1] 离线: transcript → Messages API 请求体")
    hooks = HookManager()
    hooks.register("session_start", lambda source: "Answer in Chinese.")
    hooks.register("user_prompt_submit", lambda prompt: HookResult(additional_context="Show your work."))
    hooks.register("post_tool_use", lambda result: f"[audit] {result.name}")
    agent = Agent(RuleBasedLLM(), ToolRegistry([CalculatorTool()]), PermissionGate("auto"), hooks=hooks, name="offline")
    agent.run("计算 17 * 23", verbose=False)
    system, api_messages = to_api_messages(agent._assemble_context(""))
    for item in api_messages:
        print(f"    {item['role']:<9} {json.dumps(item['content'], ensure_ascii=False)[:120]}")
    kv("system", " | ".join(system.split("\n\n")))
    roles = [m["role"] for m in api_messages]
    assert roles == ["user", "assistant", "user", "assistant"], roles  # 严格交替, 以 user 开头
    assert "Answer in Chinese." in system  # 开头的 system 消息进了顶层 system
    assert api_messages[2]["content"][0]["type"] == "tool_result"  # tool_result 在最前, hook 注释排在它后面
    assert "<system-reminder>" in api_messages[2]["content"][1]["text"]
    assert api_messages[1]["content"][0]["id"] == api_messages[2]["content"][0]["tool_use_id"]

    print("\n[2] 在线: 真实模型")
    try:
        llm = ClaudeLLM()
    except RuntimeError as exc:
        print(f"  跳过: {exc}。设置好之后重跑本 demo 即可, 其余模块不受影响。")
        return
    live = Agent(llm, ToolRegistry([CalculatorTool()]), PermissionGate("auto"), max_turns=4, name="claude")
    final = live.run("请用 calculator 工具计算 (17 * 23) + 5, 然后只回答数字。")
    assert validate_transcript(live.messages) == []
    assert "396" in final, final
    print("\n  OK: 换模型 = 换一个实现了 next() 的对象。")


if __name__ == "__main__":
    main()
