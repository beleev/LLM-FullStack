"""ClaudeLLM: 用真实的 Claude Messages API 实现同一个 LLM 协议 (opt-in, 默认 demo 从不导入本文件)。

没有它: 学员会怀疑 "toy LLM 能跑, 换真模型是不是要重写 loop" —— 不用, 只换这一个对象。
关键设计:
  - transcript 本来就是 Messages API 的 block 格式, 转换只需处理三件事 (to_api_messages, 纯函数, 可离线测试):
    开头的 system 消息 → 顶层 system 参数; 中途 harness 注入的 system 消息 → user 侧 <system-reminder> 文本;
    相邻同角色消息合并, 且 tool_result 必须排在该 user 消息的最前面。
  - 响应的 content blocks 原样存回 transcript (raw_content): thinking block 必须原封不动回传。
  - tool_use.id 用 API 给的; 参数是已解析的 dict, 不要对序列化后的字符串做匹配。
依赖: `pip install anthropic` + 环境变量 ANTHROPIC_API_KEY。模型默认 claude-opus-5
(另有 claude-sonnet-5 / claude-haiku-4-5, 用 LLM_AGENT_MODEL 覆盖)。
未启用服务端 refusal fallbacks (需 beta 端点: betas=["server-side-fallback-2026-07-01"], fallbacks="default"),
这里只识别 stop_reason == "refusal" 并如实返回。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

from llm_agent.core.schema import Message, ModelAction, ToolCall


def to_api_messages(messages: List[Message]) -> Tuple[str, List[Dict[str, Any]]]:
    """内部 transcript → (system 字符串, Messages API 的 messages 列表)。"""
    system: List[str] = []
    out: List[Dict[str, Any]] = []
    for msg in messages:
        if msg.role == "system" and not out:
            system.append(msg.text)
            continue
        if msg.role == "system":  # 对话中途的 harness 注入: 降级成 user 侧提示, 任何模型都支持
            role, blocks = "user", [{"type": "text", "text": f"<system-reminder>\n{msg.text}\n</system-reminder>"}]
        else:
            role, blocks = msg.role, list(msg.blocks)
        if out and out[-1]["role"] == role:
            out[-1]["content"] += blocks
        else:
            out.append({"role": role, "content": blocks})
    for item in out:  # API 要求 tool_result 在 user 消息最前面; sorted 稳定, 其余顺序不变
        item["content"] = sorted(item["content"], key=lambda b: b["type"] != "tool_result")
    return "\n\n".join(system), out


class ClaudeLLM:
    def __init__(self, model: str = "", max_tokens: int = 16000) -> None:
        try:
            import anthropic  # 延迟导入: 没装 SDK 时, 包的其余部分照常工作
        except ImportError as exc:
            raise RuntimeError("需要可选依赖: pip install anthropic") from exc
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("未设置 ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic()  # key 由 SDK 从环境变量读取, 代码里不经手
        self.model = model or os.environ.get("LLM_AGENT_MODEL", "claude-opus-5")
        self.max_tokens = max_tokens
        self._tools: List[Dict[str, Any]] = []

    def next(self, messages: List[Message], tools: List[Dict[str, Any]]) -> ModelAction:
        system, api_messages = to_api_messages(messages)
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": api_messages,
        }
        if "haiku" not in self.model:  # haiku-4-5 不支持 adaptive thinking (它用 budget_tokens), 这里直接不开
            kwargs["thinking"] = {"type": "adaptive"}
        if system:
            kwargs["system"] = system
        # 历史里有 tool_use/tool_result 时 API 要求必须带 tools; 摘要请求传的是 [] → 沿用上一次的定义
        self._tools = tools or self._tools
        if self._tools:
            kwargs["tools"] = self._tools  # Tool.schema() 已经是 {name, description, input_schema}
        response = self.client.messages.create(**kwargs)

        if response.stop_reason == "refusal":  # 先看 stop_reason, 再读 content
            return ModelAction.final("[模型拒绝了该请求]")
        raw = [block.model_dump(exclude_none=True) for block in response.content]
        text = "\n".join(b["text"] for b in raw if b["type"] == "text")
        calls = [ToolCall(b["name"], b["input"], id=b["id"]) for b in raw if b["type"] == "tool_use"]
        if response.stop_reason == "max_tokens":  # 可能截在 tool_use 中间: 参数不完整, 绝不执行
            return ModelAction.final(text + "\n[输出在 max_tokens 处被截断]")
        if calls and tools:
            return ModelAction(kind="tool", content=text, tool_calls=calls, raw_content=raw)
        return ModelAction(kind="final", content=text, raw_content=None if calls else raw)
