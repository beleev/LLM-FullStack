"""Small reusable pieces shared by the llm_agent teaching demos."""

from llm_agent.core.agent import Agent
from llm_agent.core.hooks import HookManager, HookResult
from llm_agent.core.memory import FileMemory, compact_messages
from llm_agent.core.permissions import PermissionGate, PermissionRule
from llm_agent.core.persistence import JsonlSessionStore
from llm_agent.core.schema import Message, ModelAction, ToolCall, ToolResult
from llm_agent.core.subagents import DelegateTool
from llm_agent.core.tools import (
    CalculatorTool,
    ReadNotesTool,
    SearchDocsTool,
    ShellTool,
    Tool,
    ToolRegistry,
    WeatherTool,
    WriteNoteTool,
)
from llm_agent.core.toy_llm import RuleBasedLLM

__all__ = [
    "Agent",
    "CalculatorTool",
    "DelegateTool",
    "FileMemory",
    "HookManager",
    "HookResult",
    "JsonlSessionStore",
    "Message",
    "ModelAction",
    "PermissionGate",
    "PermissionRule",
    "ReadNotesTool",
    "RuleBasedLLM",
    "SearchDocsTool",
    "ShellTool",
    "Tool",
    "ToolCall",
    "ToolRegistry",
    "ToolResult",
    "WeatherTool",
    "WriteNoteTool",
    "compact_messages",
]

