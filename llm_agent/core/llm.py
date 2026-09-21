"""LLM 协议: agent loop 对"模型"的全部要求就是这一个方法。

没有它: loop 写死某个具体模型类, 换真实 API / 换评测用的替身都得改 loop。
关键设计: next(完整消息列表, 工具 schema 列表) -> ModelAction。无状态 —— 状态全在 messages 里,
所以同一个 loop 可以接 RuleBasedLLM (默认, 确定性)、ClaudeLLM (core/claude_llm.py, 需 opt-in)、
或 m13 里故意出错的 FlakyLLM。
对应: 任何 chat-completions / Messages API 客户端的最小公共接口。
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol

from llm_agent.core.schema import Message, ModelAction


class LLM(Protocol):
    def next(self, messages: List[Message], tools: List[Dict[str, Any]]) -> ModelAction:
        """tools 是 Tool.schema() 列表: 模型需要参数的 JSON Schema, 只给名字它只能猜。"""
        ...
