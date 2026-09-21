# M07 — Subagents: 把脏活关进隔离的上下文, 只拿回摘要

`delegate` 工具会新建一个有独立 messages、工具集和权限门的子 agent 去完成子任务, 父级只收到一段有长度上限的摘要。

## 直觉

调研类任务的中间产物 (读过的每篇文档、每次试错) 对最终结论几乎没用, 却会永久占住主上下文, 之后每一轮请求都要为它们重复付 token, 还会稀释模型对主线的注意力。
子智能体的做法是: 把这件事交给一个全新的 agent, 它在自己的上下文里折腾, 完事只交回结论。
隔离是双向的: 子级看不到父级的对话, 也拿不到父级的工具 —— 所以它同时是一种最小权限机制。
代价是总 token 通常更多 (子级要从零重建上下文), 而且父级只能看到摘要, 摘要丢了什么它不会知道。

## 核心数据结构与控制流

- `core/subagents.py: DelegateTool(agent_types, hooks, transcript_dir, llm_factory, max_summary_chars=200)` — 一个普通 `Tool`。`agent_types` 是 `{类型名: 返回 ToolRegistry 的工厂}`; 参数 schema 为 `{task (必填), agent_type (enum)}`。
- `DelegateTool.children` — 每个子级一条记录 `{type, task, messages, usage, path}`, 供 demo / 审计读取; 追加时加锁, 因为 `execute` 可能被父 loop 的线程池并发调用。
- `core/agent.py: Agent` — 子级就是同一个 `Agent` 类的另一个实例, 没有第二套运行时。
- `core/hooks.py: HookManager.on_subagent_stop(agent_type, summary)` — 子级结束时通知。

```
parent.run(prompt)
  └ model -> tool_use delegate {task, agent_type}
       └ 父级权限门 (delegate 是 low risk → auto 放行)
            └ DelegateTool.execute:
                 child = Agent(llm_factory(),
                               tools       = agent_types[agent_type](),   # 每次新建, 子级之间也不共享
                               permissions = PermissionGate("auto"),      # 无 ask_policy → 拿不准即拒绝
                               store       = child_NN_<type>.jsonl,
                               max_turns   = 4)
                 summary = child.run(task, verbose=False)                 # 子级跑完自己的完整 loop
                 hooks.on_subagent_stop(agent_type, summary)
                 return "[<type>] " + shorten(summary, max_summary_chars)
  └ tool_result(摘要) 进父 transcript → 父模型据此作答
```

关键设计决策:

- **delegate 只是工具**。不需要新的调度器: 权限、校验、tool_result 回填、并行执行全部复用现有 loop。父模型在同一 turn 发多个 `delegate`, 线程池自然把它们并行化 —— 这就是 m11 的 orchestrator-workers 扇出。
- **子级的权限门是 `auto` 且没有 `ask_policy`**。子级运行时没有人可问, `_ask` 会 fail closed; 所以低 / 中风险工具放行, 高风险工具一律拒绝。
- **摘要有硬上限**。`shorten(summary, max_summary_chars)` 在 harness 侧截断, 不依赖子级"自觉写短"; 子级再啰嗦也淹不了父上下文。
- **子 transcript 落盘但不回流**。细节可审计 (`child_00_researcher.jsonl`), 但不占父上下文; `usage` 也分开记, 才能区分"总花费"和"父上下文占用"。
- **工具集用工厂而非实例**。每次委托都新建, 避免子级之间通过有状态的工具互相影响。

## 运行后应该看到什么

```bash
cd <仓库根目录> && python3 -m llm_agent.m07_subagents.demo
```

```
  [parent] turn 1: model -> tool_use toolu_0001 delegate {'task': '请委托子智能体调研 agent loop', 'agent_type': 'researcher'}
  [parent] permission delegate -> allow (auto: low-risk tool)
  [parent] tool_result toolu_0001 -> [researcher] 基于工具结果完成： search_docs: agent_loop: The loop is small; harness systems arou...

[parent transcript]
  user      请委托子智能体调研 agent loop
  assistant [tool_use delegate {"task": "请委托子智能体调研 agent loop", "agent_type": "researcher"}]
  user      [researcher] 基于工具结果完成： search_docs: agent_loop: The loop is small; harness systems around it carry m
  ...
[child transcript — 只在磁盘上, 不在父上下文里]
  user      请委托子智能体调研 agent loop
  assistant [tool_use search_docs {"query": "请委托子智能体调研 agent loop"}]
  user      agent_loop: The loop is small; harness systems around it carry most complexity. detail detail detail
  ...
  parent peak context tokens: 107
  child  peak context tokens: 279
```

断言验证的内容:

- 父级的工具调用只有 `["delegate"]`, 子级只有 `["search_docs"]` —— 父级注册了 `shell`, 子级的工具集里没有它。
- 父 transcript 恰好 4 条 (user / tool_use / tool_result 摘要 / final), 子级的 4 条消息一条都没进来。
- `subagent_stop` hook 被触发一次, 收到 `"researcher"`。
- 子 transcript 已写入 JSONL, 行数与子级 messages 数一致。
- 父级峰值上下文 token 小于子级: 带 30 个 `detail` 的长文档只撑大了子级上下文。

## 与真实系统的差距

- 任务简报是 toy LLM 把原 prompt 原样转发 (子级收到的还是"请委托子智能体..."); 真实系统里委托 prompt 的质量 (目标、输出格式、边界、可用工具提示) 是多智能体效果的决定因素。
- 摘要是对子级最终回答的字符级硬截断, 不是语义压缩; 真实系统让子模型自己写面向父级的报告。截断可能正好切掉结论。
- `DelegateTool` 的 `hooks` 只用于 `subagent_stop`; 子 `Agent` 没有拿到父级的 HookManager 和 memory (guardrails 可通过 `DelegateTool(guardrails=...)` 传下去, 本 demo 未传)。也就是说父级的 `pre_tool_use` 安全 hook 管不到子级内部的工具调用, 而 Claude Code 的 hook 对子智能体的工具调用同样生效。
- 父级权限门只看得到 `delegate` 这一层 (`risk="low"`, `read_only=False` —— 所以 plan 模式下委托会被拒, 不能靠子级绕过只读); 子级实际能做什么由 `agent_types` 的工具集和子级自己的 `auto` 门决定, 且子级没有人可问。
- Claude Code 的子智能体由带 frontmatter 的 markdown 文件定义 (description、工具白名单、模型、独立 system prompt), 可按类型选不同模型; 这里只有一个工具集工厂和一句固定 system prompt。
- 没有子级超时 / 取消 / 总预算控制, 只有 `max_turns=4`; 没有后台运行, 也不能向已结束的子级追问。
- 本 demo 只有一个子级、串行执行; 并行扇出与 token 两本账见 m11。

## 常见误区

- **"用子智能体是为了省 token"** — 通常更费: 每个子级都要重建上下文、多几次模型调用。省下的是父上下文的占用 (以及并行带来的墙钟时间), 不是总花费。
- **"子级继承父级的上下文和工具, 只是换个线程跑"** — 子级的 messages 从空开始, 只收到 `task` 这一个字符串; 工具集来自 `agent_types`, 与父级的注册表无关。父级想让子级知道的一切都必须写进 task。
- **"子级在沙箱里, 所以给它什么工具都安全"** — 隔离的是上下文, 不是副作用。子级的工具照样作用于真实世界, 而且它的调用不经过父级的权限门和 hook; 安全性取决于你给该 agent_type 配了哪些工具。

## 自测题

1. 父级开着 `default` 模式、有人审批; 为什么子级不沿用父级的门, 而是新建一个 `PermissionGate("auto")`?
<details><summary>答案</summary>

子级在一次工具调用内部同步跑完整个 loop, 期间没有交互通道可以弹审批; 并行扇出时更不可能同时问人。`auto` 且不带 `ask_policy` 的门让低 / 中风险工具直接放行, 高风险走 `_ask` 时因无人可问而 fail closed 拒绝。沿用父级的门则可能把"问人"语义带进无人值守的环境, 或把父级已有的宽松授权泄漏给子级。

</details>

2. 把 `max_summary_chars` 调到 5000, demo 的哪条断言最可能先失败? 为什么?
<details><summary>答案</summary>

`parent.usage["peak_context_tokens"] < child.usage["peak_context_tokens"]`。摘要不再被截断, 子级最终回答 (内含整篇带 120 个 detail 的文档) 会完整进入父级的 tool_result, 父级最后一次模型调用的上下文就包含了这篇长文档再加上自己的消息, 峰值不再小于子级。这正说明上下文隔离靠的是 harness 侧的硬上限。

</details>

3. 子 transcript 已经不回流父上下文了, 为什么还要落盘?
<details><summary>答案</summary>

父级只看到摘要, 摘要可能遗漏、截断或出错。落盘的子 transcript 是唯一能回答"子级到底搜了什么、依据是什么、哪些调用被拒"的证据, 用于审计和调试; 它与父 JSONL 分文件存放, 既可追溯, 又不增加任何一次模型调用的输入。

</details>
