# llm_agent — LLM 应用层 / Agent 教学章节

> 用纯 Python stdlib 做一个可运行的 agent harness。目标不是接真实模型 API，而是讲清楚生产级 Agent 的 80% 关键机制：循环、工具、权限、上下文、记忆、扩展、持久化和子智能体。

## 设计目标

- **原理优先**：`RuleBasedLLM` 是可预测的假模型，让重点落在 harness，而不是模型能力。
- **模块自治**：每个 `mXX_*/demo.py` 都能 `python -m ...` 独立运行。
- **可组合**：`full_loop/` 把工具、权限、Hook、记忆、持久化和子智能体串起来。
- **安全可跑**：`ShellTool` 只模拟执行，不运行任意本地命令。

## 学习路径

```
m01 Agent Loop          最小 while-loop 闭环
        ↓
m02 Tool Use            tool schema / execute / result
        ↓
m03 Permissions         deny-first / ask / auto
        ↓
m04 Context & Memory    文件记忆 / 检索 / 压缩
        ↓
m05 Extensibility       hooks / skills / MCP-like tools
        ↓
m06 Persistence         append-only JSONL / resume
        ↓
m07 Subagents           隔离上下文 / summary-only return
        ↓
full_loop               组合成一个 mini agent harness
```

## 模块清单

| # | 模块 | 覆盖的技术 |
|---|------|------------|
| 01 | [Agent Loop](m01_agent_loop/) | ReAct 风格循环、流转 transcript |
| 02 | [Tool Use](m02_tool_use/) | 工具注册、schema、工具结果回填 |
| 03 | [Permissions](m03_permissions/) | deny-first、人工审批、auto 风险分类 |
| 04 | [Context & Memory](m04_context_memory/) | 文件记忆、相关检索、上下文压缩 |
| 05 | [Extensibility](m05_extensibility/) | Hook、Skill 注入、MCP-like 外部工具 |
| 06 | [Persistence](m06_persistence_resume/) | JSONL 仅追加日志、恢复会话 |
| 07 | [Subagents](m07_subagents/) | 子智能体隔离、只回传摘要 |
| ★ | [Full Loop](full_loop/) | 多机制组合的最小 Agent 系统 |

## 运行

```bash
# 单模块
python -m llm_agent.m01_agent_loop.demo
python -m llm_agent.m03_permissions.demo
python -m llm_agent.m07_subagents.demo

# 整体跑一遍
python -m llm_agent.run_all

# 组合闭环
python -m llm_agent.full_loop.demo
```

## 业界覆盖度自评

| Agent 技术 | 本目录 | 说明 |
|---|:---:|---|
| Agent loop / ReAct | yes | `m01`, `core/agent.py` |
| Tool calling | yes | `m02`, `core/tools.py` |
| Permission gate | yes | `m03`, `core/permissions.py` |
| Auto approval / risk classifier | partial | 用规则分类器模拟 |
| Context compaction | yes | `m04`, `core/memory.py` |
| File-based memory | yes | `m04`, `FileMemory` |
| Hooks | yes | `m05`, `core/hooks.py` |
| Skills | partial | 用轻量 SkillRegistry 演示指令注入 |
| MCP | partial | 用 FakeMCPServer 暴露 `WeatherTool` |
| Session persistence | yes | `m06`, JSONL transcript |
| Resume | yes | `m06`, `load_history=True` |
| Subagents | yes | `m07`, summary-only return |
| Sandbox / real shell isolation | no | 为教学安全只模拟 shell |
| Real LLM API / streaming | no | 重点是 harness 原理 |

读完本目录后，再看 Claude Code / Codex / OpenHands / LangGraph 时，可以把复杂系统拆成几条主线：
**模型怎么选动作、工具怎么执行、权限怎么拦、上下文怎么控、状态怎么留、子任务怎么隔离。**

