# M01 — Agent Loop

让模型能"行动 - 观察 - 再行动"的那个循环: 问模型, 有 `tool_use` 就执行并回填 `tool_result`, 没有就结束。

## 直觉

单次问答的模型只能"说": 稍复杂的算式它心算不准, 也看不到任何外部信息。
Agent loop 把"模型的决定"和"确定性代码的执行"接成闭环: 模型提出调用, harness 执行, 结果作为新的观察回到上下文, 模型再决定下一步。
没有这个循环, 工具、权限、记忆都无处挂载。
循环本身非常薄 (`Agent.run` 约 40 行), 真正的复杂度在它周围: 工具校验、权限、上下文管理、持久化。
本模块只用一个计算器, 目的是把 transcript 的形状看清楚。

## 核心数据结构与控制流

- `core/schema.py: Message` — `content` 要么是 `str`, 要么是 content block 列表 (Claude Messages API 同款):
  - assistant: `{"type":"tool_use","id","name","input"}`
  - user: `{"type":"tool_result","tool_use_id","content","is_error"}`
- `core/schema.py: ModelAction` — 模型一步的输出: `kind` 为 `tool` (带 `tool_calls`) 或 `final`。
- `core/schema.py: validate_transcript` — 检查每个 `tool_use` 在下一条 user 消息里有同 id 的 `tool_result`, 且没有孤儿 `tool_result`; 返回空列表才算合法。
- `core/llm.py: LLM` — 协议只有一个方法 `next(messages, tools) -> ModelAction`, 无状态。
- `core/toy_llm.py: RuleBasedLLM` — 关键词规则冒充模型, 让每一步都可断言。
- `core/agent.py: Agent.run` — 主循环。

```
run(prompt)
  append user(prompt)
  for turn in 1..max_turns:
      action = llm.next(system + messages, tool_schemas)
      if final / 没有 tool_calls:  append assistant(text); return
      给每个 call 分配 id (toolu_0001 ...)
      append assistant([tool_use ...])          # 先记"模型要求了什么"
      results = hook -> 权限门 -> 校验 -> 执行   # 见 m02 / m03
      append user([tool_result ...])            # 同 id 回填
  return "stopped: max_turns reached"
```

关键设计决定:

- 模型的"决定"也进 transcript, 而且在执行之前写入: 执行中途崩溃, 审计记录里仍有这一步; 否则日志里只剩孤零零的工具结果, 既无法审计, 也无法原样发给真实 API。
- `tool_result` 用 user 角色: 在 Messages API 里只有 user / assistant 两种轮次, 工具结果是"外部世界给模型的输入", 靠 `tool_use_id` 指回对应的请求。
- LLM 无状态, 状态全在 `messages` 里。`RuleBasedLLM.next` 每次从"最后一条真正的用户 prompt"(`Message.is_user_prompt`, 携带 `tool_result` 的 user 消息不算) 往后统计已用工具, 所以旧轮次的调用不算数。
- `max_turns` 是硬上限: 模型永远可能一直要求调用工具, loop 必须自己能停。
- system prompt 不进 `agent.messages`, 每次在 `_assemble_context` 里现拼, 所以 demo 的 transcript 只有 4 条消息。

## 运行后应该看到什么

```bash
cd "/Volumes/My Shared Files/LLMFullStack"
python3 -m llm_agent.m01_agent_loop.demo
```

```
[1] 一次完整闭环
  [m01] turn 1: model -> tool_use toolu_0001 calculator {'expr': '2 + 3 * 4'}
  [m01] permission calculator -> allow (dont_ask: mode allows unknown action)
  [m01] tool_result toolu_0001 -> 2 + 3 * 4 = 14
  [m01] final: 基于工具结果完成： calculator: 2 + 3 * 4 = 14

[transcript]
  user      请计算 2 + 3 * 4
  assistant [{'type': 'tool_use', 'id': 'toolu_0001', 'name': 'calculator', 'input': {'expr': '2 + 3 * 4'}}]
  user      [{'type': 'tool_result', 'tool_use_id': 'toolu_0001', 'content': '2 + 3 * 4 = 14', 'is_error': False}]
  assistant 基于工具结果完成：
calculator: 2 + 3 * 4 = 14

[2] 同一会话的第二个问题: 必须重新调用工具, 不能拿上一轮结果充数
  [m01] turn 1: model -> tool_use toolu_0002 calculator {'expr': '10 / 4'}
  ...
  [m01] tool_result toolu_0002 -> 10 / 4 = 2.5
  [m01] final: 基于工具结果完成： calculator: 10 / 4 = 2.5
  llm calls               : 4
```

`assert` 验证的内容:

- 最终回答包含 `2 + 3 * 4 = 14` (计算器真的算对了, 而不只是"没崩")。
- transcript 形状恰好是 `user[text] -> assistant[tool_use] -> user[tool_result] -> assistant[text]`。
- `validate_transcript(agent.messages) == []`: id 配对完整。
- 第二个问题的回答含 `10 / 4 = 2.5` 且不含 `14`; 全会话 `tool_use` 总数为 2 (第二问重新调用了工具)。
- `llm_calls == 4`: 每个问题两次模型调用, 一次决定调工具, 一次读结果作答。

关于旧 bug: 计算器曾用 `ast.Num` 判断数字节点, 该类在 Python 3.12 被移除, 在 3.12+ 上工具每次都返回 `ERROR` (工具异常会被 `ToolRegistry.execute` 转成 `is_error` 结果, loop 不会崩); 旧 demo 没有断言答案内容, 于是照样打印 OK。现在用 `ast.Constant` 并显式限定 `int / float`, 且第一条 `assert` 就是为了让这种"静默失败"无处可藏。

## 与真实系统的差距

- `RuleBasedLLM` 是关键词规则 + 正则, 不是模型: 它不会推理、不会出错, 也不会在工具报错后换思路。换真模型见 `core/claude_llm.py` (opt-in)。
- 真实 API 用 `stop_reason` (`tool_use` / `end_turn` / `max_tokens` / `refusal` ...) 表示一步的结束方式; 这里只有 `kind` 为 `tool` 或 `final` 两种, 没有处理截断、重试、限流、流式输出和用户中断。
- `tool_use` 的 id 在真实 API 里由模型侧生成; toy LLM 留空, 由 `Agent` 顺序编号。
- `usage` 里的 token 数是 `core/utils.py: estimate_tokens` 的粗估, 不是计费口径。
- 真实 API 的 system prompt 是顶层参数而不是一条消息; 这里统一用 `Message("system", ...)`, 转换由 `claude_llm.to_api_messages` 负责。

## 常见误区

- "Agent 是一种特殊的模型。" 不是。模型只输出"我想调用什么", 循环、执行、停止条件全在普通代码里; 同一个 loop 可以换任意实现了 `next()` 的对象。
- "工具结果是 assistant 说的话。" 不是。它是 harness 执行后的观察, 以 user 角色回填; 写成 assistant 文本会让模型把它当成自己的结论, 也不符合 API 的消息格式。
- "demo 打印了 OK 就说明跑对了。" 工具异常会被转成 `is_error` 结果, loop 照常结束; 只有断言最终内容才能发现工具其实一直在报错。

## 自测题

1. 为什么 `tool_result` 用 user 角色, 并且必须和 `tool_use` 同 id、紧跟在下一条消息里?

<details><summary>答案</summary>
对模型而言, 工具结果是来自外部世界的输入, 而 Messages API 的对话只有 user / assistant 交替, 所以它属于 user 侧。一个 assistant turn 可以带多个 `tool_use`, 只有靠 `tool_use_id` 才能知道哪个结果对应哪个请求; 缺失或错位的配对会被真实 API 直接拒绝, `validate_transcript` 检查的就是这一点。
</details>

2. 同一会话里问第二个算式, toy LLM 为什么不会直接拿第一问的结果作答? 如果去掉这个机制会怎样?

<details><summary>答案</summary>
`RuleBasedLLM.next` 先找到最后一条 `is_user_prompt` 的消息, 只在这之后的切片里统计 `used` 和 `results`。上一轮的 `calculator` 调用不在切片内, 所以本轮计划里的调用仍是"还没做"。如果对全量历史统计, 第二问会被判定为"calculator 已经用过", 直接用旧结果 `14` 作答 —— demo 里 `"14" not in final2` 就是防这个回归。
</details>

3. `Agent.run` 为什么在执行工具之前就把 assistant 的 `tool_use` 消息写进 transcript?

<details><summary>答案</summary>
transcript 同时是审计日志和下一次模型调用的输入。先写入, 即使执行阶段崩溃或被拒绝, 记录里也保留了"模型要求过什么"; 而且无论执行成功、校验失败还是权限拒绝, 后面都会补上一条同 id 的 `tool_result` (失败时 `is_error: true`), 配对始终完整。
</details>
