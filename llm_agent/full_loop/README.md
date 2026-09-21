# Full Loop — 把全部机制拼成一个 mini harness

用 `core/` 的零件组装一个 Claude-Code 式的 agent, 连续跑 5 个场景并逐一断言, 最后检查整份落盘 transcript 仍是合法的 Messages API 序列。

## 直觉

前面每个模块单独看都很小, 也都各自通过了断言; 但真实 harness 的 bug 大多出在零件之间。
典型例子: skill 正文漏进检索词导致排序变了、hook 注释被写进笔记、脱敏只做了上下文没做落盘、resume 时 session_start 被重复注入。
所以这里不引入任何新机制, 只验证"它们互不踩脚": 同一个 `Agent` 实例、同一份 JSONL、同一个权限门, 连续处理五种性质完全不同的请求。
整体形状是 一个薄 loop + 一圈确定性的 harness: 模型只负责提议下一步, 能不能做、做完怎么记, 都由确定性代码决定。

## 核心数据结构与控制流

组装全部在 `full_loop/demo.py`:

| 零件 | 来源 | 在本 demo 里的配置 |
|------|------|--------------------|
| Skills | `core/skills.py` `SkillRegistry` / `SkillTool` | 读 `m05_extensibility/skills/*/SKILL.md`; 只有 `catalog()` 目录常驻 system prompt |
| 检索 | `core/retrieval.py` `TfidfIndex` / `VectorSearchTool` | 工具名仍是 `search_docs` |
| 子智能体 | `core/subagents.py` `DelegateTool` | `researcher` 类型只有检索工具; `transcript_dir` 让子 transcript 落盘 |
| MCP | `core/mcp.py` `MCPClient` / `mcp_tools` | 用 `sys.executable` 拉起 `m09_mcp/server.py` 真实子进程 |
| 权限 | `core/permissions.py` `PermissionGate("auto")` | deny `shell` `*rm -rf*`; allow `mcp__weather__*` |
| 护栏 | `core/guardrails.py` `Guardrails` | 不可信输出包 `<untrusted_data>`、污点、密钥脱敏 |
| 记忆 / hooks / 持久化 | `core/memory.py` `core/hooks.py` `core/persistence.py` | `FileMemory`; session_start / post_tool_use / stop / subagent_stop; `JsonlSessionStore` |

一次工具调用经过的关卡 (`core/agent.py`):

```
model → tool_use
  → 先把 assistant 消息写进 transcript (执行中崩溃也有审计记录)
  → pre_tool_use hook (可拦截 / 改写)
  → 污点检查: 本轮已混入不可信数据 且 工具 risk == high → 拒绝
  → PermissionGate.evaluate(改写后的最终调用): deny > ask > allow > 模式兜底
  → 执行 (多个已批准的调用并行)
  → redact 密钥 → 不可信工具的输出 wrap_untrusted 并置污点
  → post_tool_use hook 的注释 → 独立 system 消息 (不拼进 tool_result)
  → _append: 再脱敏一次 → 内存 messages + JSONL
```

设计取舍:

- deny 规则与命令走同一个 `normalize_command`: 写规则的人只写 `*rm -rf*`, `rm -fr`、`RM -r -f` 同样命中。
- MCP 工具一律 `risk="high"`、`untrusted_output=True`: 第三方 server 自报的注解不可信, 所以必须有一条显式 allow 规则才放行 (否则 `auto` 模式会去问人, 没人可问即拒绝)。
- hook 注释、skill 正文、记忆都走各自的消息, 不拼进用户 prompt 或工具数据: 这样检索词是干净的用户原话, 写进笔记的只有工具数据。
- 子智能体只回一段有长度上限的摘要, 完整过程写进自己的 JSONL: 父上下文干净, 审计仍可追。
- 脱敏发生在进入 transcript 之前, 而不是打印时: 上下文、日志、下一次模型请求三处同时受益。

## 运行后应该看到什么

```bash
cd <仓库根目录> && python3 -m llm_agent.full_loop.demo
```

真实输出节选 (长行已截断):

```
[1] skill → 检索 → 写笔记
  [full] turn 1: model -> tool_use toolu_0001 skill {'name': 'debug'}
  [full] turn 2: model -> tool_use toolu_0002 search_docs {'query': '排查 agent loop，并写入笔记'}
  [full] tool_result toolu_0002 -> [0.55] agent_loop: Agent loop = assemble context, call model, dispatch tool, check perm...
[3] MCP 工具 (真实子进程)
  [full] permission mcp__weather__get_weather -> allow (rule: trusted local weather server)
  [full] tool_result toolu_0005 -> <untrusted_data> Shanghai: sunny, 24C, light wind </untrusted_data>
[4] 危险命令: 换了 flag 顺序也一样被拒
  [full] turn 1: model -> tool_use toolu_0006 shell {'command': 'rm -fr /tmp/demo'}
  [full] permission shell -> deny (rule: never allow destructive shell)
[5] 抓回来的文档夹带指令和密钥
  [full] tool_result toolu_0007 -> <untrusted_data injection_suspected="ignore previous instructions; AGENT:"> Restart wit...
  jsonl messages          : 32
  llm calls / input tokens: 12 / 7604
  child transcripts       : ['child_00_researcher.jsonl']
  hook events             : ['stop', 'subagent_stop:researcher', 'stop', 'stop', 'stop', 'stop']
```

`assert` 验证的内容:

- [1] 工具序列恰为 `skill → search_docs → write_note`; 笔记第一条命中 `agent_loop` (检索词没有被 skill 正文污染); 笔记里没有 `[audit]` 也没有 skill 正文 "排查流程"。
- [2] 只调用了 `delegate`; 最终回答含子级检索到的 "isolated transcripts"; 子 transcript 文件存在; `subagent_stop:researcher` 事件被触发。
- [3] 通过真实子进程拿到 "Shanghai: sunny"。
- [4] `rm -fr` 被 `*rm -rf*` 规则拒绝, 回答含 `DENIED: never allow destructive shell`, `shell.executed == []`。
- [5] 回答里带 `injection_suspected` 标记, 正常内容 `systemctl restart billing` 仍可用, shell 未被调用。
- 全局: JSONL 原文不含 `sk-live` 且含 `[REDACTED]`; `validate_transcript(store.load_all()) == []`; `stop` 事件恰好 5 次 (子 agent 不带父级的 stop hook); `session_start` 注入在 5 次 `run()` 里只出现 1 次。

## 与真实系统的差距

- 模型是 `RuleBasedLLM`: 场景 [5] 里 shell 没被调用首先是因为 toy LLM 默认不服从工具结果里的指令; 污点锁这条确定性兜底在本 demo 没有被触发 (m12 用 `gullible=True` 专门演示)。
- `ShellTool` 只模拟、从不执行; deny 规则是字符串黑名单, `/bin/rm`、`find -delete`、`python -c` 都绕得过。Claude Code 靠命令解析、allowlist 与 OS 级沙箱, 黑名单只是最后一道便宜的网。
- 脱敏是 4 条正则; 子 agent 通过 `DelegateTool(guardrails=Guardrails())` 拿到同样的护栏, 但"密钥未落盘"的断言只检查了父会话的 JSONL。
- `FileMemory` 接进了 agent (每轮按 prompt 关键词现查现拼), 但没有任何断言验证它对回答的影响。
- 上下文预算设为 6000 字符, 全程没有触发清理或压缩; 压缩与其它机制 (如 session_start 注入在压缩后的去向) 的交互在这里没有被覆盖。
- 没有人在环: `auto` 模式拿不准就拒绝 (fail closed), 没有交互式审批、没有"本次会话始终允许"。
- MCP 只有一个本地 stdio server、只用 tools; 没有 HTTP transport、认证、resources / prompts、断线重连。
- 单进程单会话: 没有并发会话、取消与中断、流式输出、成本上限。

## 常见误区

- "每个模块都单测通过了, 拼起来自然是对的。" 组合 bug 出在接缝: demo 的注释里就记着一个真实回归 —— skill 文本漏进检索词, 第一名从 `agent_loop` 变成了 `subagents`。所以要断言工具序列和笔记内容, 而不只是"跑完了"。
- "MCP 工具来自我自己启动的 server, 输出是可信的。" harness 把所有 MCP 输出按不可信处理 (包 `<untrusted_data>` 并置污点)。信任 server 能被调用 (allow 规则) 与信任它返回的文本是两件事。
- "deny 规则挡住了 `rm -fr`, 说明黑名单够用了。" 归一化只堵最廉价的绕过 (大小写、空白、短 flag 顺序与拆分)。它证明的是规则与命令必须走同一个归一化, 不是黑名单可以当安全边界。

## 自测题

1. 场景 [3] 中, 如果删掉 `mcp__weather__*` 的 allow 规则, 会发生什么? 为什么不把 MCP 工具的 risk 调成 low 来解决?

<details><summary>答案</summary>
`MCPTool.risk` 固定为 `high`; `auto` 模式对高风险工具走 `_ask`, 而 demo 没有 `ask_policy` (没人可问), 于是 fail closed 返回拒绝, tool_result 为 `DENIED: ...`, 拿不到天气。把 risk 调成 low 等于让第三方 server 自己决定自己有多安全; 正确的做法是保持默认不信任, 由使用者用一条显式规则为这个 server 授权 —— 命名空间前缀 `mcp__<server>__` 正是为了让一条 glob 规则管住整个 server。
</details>

2. 最终断言 `validate_transcript(store.load_all()) == []` 覆盖了 5 个场景的全部消息。指出至少两个如果实现得不对就会让它失败的地方。

<details><summary>答案</summary>
(a) 被权限门拒绝的调用也必须回填一条 `is_error` 的 tool_result (场景 [4]); 如果拒绝时直接跳过, 就会出现没有配对的 tool_use。(b) hook 改写调用时, 结果必须仍挂在模型发出的原始 id 上, 否则成孤儿 tool_result。(c) `post_tool_use` 注释必须写在 tool_result 那条 user 消息之后, 若插在 assistant 的 tool_use 与 tool_result 之间, "下一条消息"就不是 tool_result 了。(d) 同一 turn 的多个 tool_use, 结果必须放进同一条 user 消息。
</details>

3. 密钥 `sk-live-...` 出现在 `fetch_doc` 的输出里。它在到达磁盘之前经过了哪几处处理? 为什么不能只在打印日志时打码?

<details><summary>答案</summary>
`_run_tools` 里先对 `result.output` 做 `redact`, 再 `wrap_untrusted`; 之后 `_append` 对整条消息的 text / tool_result / tool_use.input 再统一脱敏一次, 然后才进内存 `messages` 和 JSONL。只在打印时打码的话, 明文仍在上下文里 —— 会随下一次请求发给模型提供方、可能被模型复述进回答或写进笔记, 也会原样躺在会话文件里。脱敏必须发生在"进入 transcript 之前"这个唯一入口。
</details>
