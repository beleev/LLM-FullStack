"""llm_agent 各 demo 共用的机制。默认全部 stdlib; claude_llm.py 不在这里导入 (需要可选依赖, opt-in)。"""

from llm_agent.core.agent import Agent
from llm_agent.core.guardrails import Guardrails
from llm_agent.core.hooks import HookManager, HookResult
from llm_agent.core.llm import LLM
from llm_agent.core.mcp import MCPClient, MCPError, MCPTool, mcp_tools
from llm_agent.core.memory import (
    FileMemory,
    clear_tool_results,
    summarize_with_llm,
    total_chars,
    truncate_messages,
)
from llm_agent.core.permissions import Decision, PermissionGate, PermissionRule, normalize_command
from llm_agent.core.persistence import JsonlSessionStore
from llm_agent.core.retrieval import TfidfIndex, VectorSearchTool
from llm_agent.core.sandbox import MemoryTool, ReadFileTool, WriteFileTool, confine
from llm_agent.core.schema import Message, ModelAction, ToolCall, ToolResult, validate_transcript
from llm_agent.core.skills import SkillRegistry, SkillTool
from llm_agent.core.subagents import DelegateTool
from llm_agent.core.tools import (
    CalculatorTool,
    ExitPlanModeTool,
    FetchDocTool,
    ReadNotesTool,
    SearchDocsTool,
    ShellTool,
    TodoWriteTool,
    Tool,
    ToolRegistry,
    WriteNoteTool,
    validate_args,
)
from llm_agent.core.toy_llm import RuleBasedLLM
