# M15 — 接真实模型 (opt-in)

同一个 agent loop, 把 `RuleBasedLLM` 换成调用 Claude Messages API 的 `ClaudeLLM`; 本模块不在 `run_all` 里, 默认 demo 从不导入它。

## 直觉

前面 14 个模块的"模型"都是关键词规则, 学员自然会怀疑: 换成真模型是不是要重写 loop?
答案是不用: harness 对模型的全部依赖就是 `LLM` 协议的一个方法 `next(messages, tools) -> ModelAction`, 工具、权限、hook、持久化一行不改。
能这么干净, 是因为 transcript 从 m01 起就用 Messages API 同款的 content block (`tool_use` / `tool_result` 按 id 配对)。
剩下的适配工作只有"格式上的最后一公里": system 消息放哪、相邻同角色消息怎么合并、模型返回的原始 block 怎么存回去。
这一步做错的后果很具体: API 直接 400 (角色不交替、tool_result 不在最前、thinking block 被改动), 或者密钥被写进代码。

**重要声明**: 本仓库的开发与测试环境没有安装 `anthropic` SDK, 也没有 API key; 在线路径 (`ClaudeLLM.next` 和 demo 的第 [2] 段) 从未对真实 API 运行过, 只有离线的 `to_api_messages` 被断言覆盖。

## 核心数据结构与控制流

- `core/llm.py` — `LLM` 协议: `next(messages, tools) -> ModelAction`, 无状态, 状态全在 messages 里。
- `core/claude_llm.py` — `to_api_messages(messages) -> (system, api_messages)`: 纯函数, 不需要 SDK, 可离线测试。
- `core/claude_llm.py` — `ClaudeLLM`: 构造时延迟 `import anthropic`, 缺 SDK 或缺 `ANTHROPIC_API_KEY` 抛 `RuntimeError`; 模型默认 `claude-opus-5`, 环境变量 `LLM_AGENT_MODEL` 可覆盖 (如 `claude-sonnet-5`、`claude-haiku-4-5`)。
- `core/schema.py` — `ModelAction.raw_content`: API 返回的原始 content blocks; `ToolCall.id`: 真实模型自带 id, 为空时才由 agent 分配 `toolu_NNNN`。

```
to_api_messages(内部 transcript)
  1. 开头连续的 system 消息            → 拼成顶层 system 参数
  2. 对话中途的 system 消息 (hook 注释、
     hook_context 等 harness 注入)      → role=user 的 <system-reminder> 文本块
  3. 相邻同角色消息                     → 合并成一条 (保证 user/assistant 严格交替)
  4. 每条消息内按"是否 tool_result"稳定排序 → tool_result 排最前, 其余顺序不变

ClaudeLLM.next(messages, tools)
  kwargs = {model, max_tokens=16000, messages} (+thinking=adaptive 若非 haiku, +system, +tools; tools 为空时沿用上次的)
  response = client.messages.create(**kwargs)
  stop_reason == "refusal"      → final("[模型拒绝了该请求]")        先看 stop_reason 再读 content
  raw = [block.model_dump(exclude_none=True) ...]
  stop_reason == "max_tokens"   → final + 截断提示 (即使含 tool_use 也不执行: 参数可能不完整)
  有 tool_use block 且带 tools  → ModelAction(kind="tool", tool_calls=用 API 给的 id, raw_content=raw); 否则 → final
Agent.run 把 action.raw_content 原样 append 进 transcript (含 thinking block)
```

设计取舍:

- 转换写成纯函数: 适配器里最容易错的是格式, 而格式可以不联网就断言。
- 中途的 system 消息降级成 user 侧 `<system-reminder>`: 这种写法任何模型都接受, 且 harness 注入与用户原话、工具数据在文本上仍可区分。
- 原始 blocks 原样存回: 带 thinking 的工具循环要求把 thinking block 原封不动回传, 只存拍平后的文本会丢掉它们。
- 用 API 给的 `tool_use.id`, 参数直接用已解析的 dict: 不对序列化后的 JSON 字符串做匹配。
- key 只由 SDK 从环境变量读, 代码不经手; SDK 延迟导入, 所以没装 SDK 时包的其余部分照常工作。

## 运行后应该看到什么

```bash
cd <仓库根目录> && python3 -m llm_agent.m15_claude_api.demo          # 无 SDK / 无 key: 只跑离线部分
pip install anthropic && export ANTHROPIC_API_KEY=... && python3 -m llm_agent.m15_claude_api.demo   # 在线 (会产生真实调用与费用)
```

无 SDK 环境下的真实输出 (长行已截断):

```
[1] 离线: transcript → Messages API 请求体
    user      [{"type": "text", "text": "计算 17 * 23"}, {"type": "text", "text": "<system-reminder>\nShow your work.\n</system-reminder...
    assistant [{"type": "tool_use", "id": "toolu_0001", "name": "calculator", "input": {"expr": "17 * 23"}}]
    user      [{"type": "tool_result", "tool_use_id": "toolu_0001", "content": "17 * 23 = 391", "is_error": false}, {"type": "text", "...
    assistant [{"type": "text", "text": "基于工具结果完成：\ncalculator: 17 * 23 = 391"}]
  system                  : You are a small teaching agent. | Answer in Chinese.

[2] 在线: 真实模型
  跳过: 需要可选依赖: pip install anthropic。设置好之后重跑本 demo 即可, 其余模块不受影响。
```

离线段的 `assert` 验证:

- 角色序列恰为 `user, assistant, user, assistant`: 以 user 开头、严格交替 (`hook_context` 被并进第一条 user, `post_tool_hook` 注释被并进携带 tool_result 的那条 user)。
- `session_start` hook 注入的 "Answer in Chinese." 位于用户 prompt 之前, 因而进了顶层 `system`。
- 第三条消息的第 0 个 block 是 `tool_result`, 第 1 个才是 `<system-reminder>` 文本 (hook 注释排在工具结果之后)。
- assistant 的 `tool_use.id` 与下一条 user 的 `tool_result.tool_use_id` 相等。

在线段 (仅在有 SDK 和 key 时执行, 本仓库未跑过): 断言 transcript 通过 `validate_transcript`, 且最终回答包含 `396`。

## 与真实系统的差距

- 在线路径未经真实 API 验证 (见上方声明)。`model_dump` 出来的 block 能否原样被 API 接受需要实测; 压缩请求 (`summarize_with_llm` 传空工具列表) 会沿用上一次的 tools, 因为历史含 tool_use 时 API 要求必须声明 tools。
- 未启用服务端 refusal fallbacks (需要 beta 端点与 `fallbacks` 参数): 这里只识别 `stop_reason == "refusal"` 并如实返回一句话。
- 没有 streaming: 非流式 + `max_tokens=16000`, 长输出要等整段返回; 更大的 `max_tokens` 需要改用流式。
- 没有 prompt caching: 每轮全价重发整个前缀。也没有用 `response.usage` 记账, `Agent.usage` 仍是字符估算。
- 没有自定义重试、限流退避与按错误类型分类处理 (只有 SDK 自带的默认重试); 任何 API 异常都会直接抛出中断 loop。
- `stop_reason` 只处理 refusal 与 max_tokens (截断时不执行任何工具调用); `pause_turn` 等未处理。
- thinking 只有"adaptive 或不开"两档: 模型名含 haiku 时不发送 thinking 参数 (`claude-haiku-4-5` 用的是 `budget_tokens` 形式), 没有暴露 effort 等调节项。
- 中途 system 消息一律降级为 user 文本; 部分新模型原生支持 messages 数组里的 system 角色, 这里没有利用。
- 认证只认 `ANTHROPIC_API_KEY`, 比 SDK 自身支持的凭据来源更窄。

## 常见误区

- "接真实模型要改 agent loop。" loop 只调用 `llm.next()`。需要改的是适配器里的格式转换, 而这部分是纯函数, 已被离线断言覆盖。
- "thinking block 只是调试信息, 存 transcript 时可以丢掉或只留文本。" 工具循环里上一轮 assistant 的 thinking block 必须原样回传, 所以 `Agent` 存的是 `raw_content`, 不是拍平后的文本。
- "system 消息可以出现在对话任意位置。" 顶层 `system` 只有一个; 本适配器把开头的 system 消息拼进去, 中途的 harness 注入转成 user 侧 `<system-reminder>` 并与相邻 user 消息合并, 否则会破坏角色交替。

## 自测题

1. `post_tool_use` hook 的注释在内部 transcript 里是一条独立的 system 消息, 位于 tool_result 那条 user 消息之后。转换后它去了哪里? 为什么必须排在 tool_result 后面?

<details><summary>答案</summary>
它被转成 role=user 的 `<system-reminder>` 文本块; 因为前一条也是 user, 两者被合并成同一条 user 消息。API 要求 tool_result 出现在紧跟 tool_use 的那条 user 消息的最前面, 所以 `to_api_messages` 最后对每条消息的 blocks 做稳定排序, 把 tool_result 提到前面, 其余相对顺序不变。demo 断言了 `content[0]` 是 tool_result、`content[1]` 含 `<system-reminder>`。
</details>

2. `Agent.run` 里有 `if not call.id: 分配 toolu_NNNN`。为什么接真实模型时绝不能覆盖 API 返回的 id?

<details><summary>答案</summary>
`raw_content` 是原样存回的, 其中 tool_use block 带着 API 的 id; 下一条 user 消息里的 tool_result 必须引用同一个 id。如果 harness 另起一个 id, transcript 里的 tool_use.id 与 tool_result.tool_use_id 就对不上, 下一次请求会因孤儿 tool_result 被拒。所以只有 toy LLM (id 为空) 才由 agent 分配; `_run_tools` 也始终把结果挂在模型发出的那个 id 上, 即使 hook 改写了调用。
</details>

3. 为什么 `ClaudeLLM` 在 `__init__` 里才 `import anthropic`, 而不是在文件顶部? 这与"m15 不在 run_all 里"是什么关系?

<details><summary>答案</summary>
包的卖点是零依赖、确定性、可断言。顶部导入会让任何导入 `claude_llm` 的代码在没装 SDK 时失败; 延迟导入后, 纯函数 `to_api_messages` 无需 SDK 即可导入和测试, 缺 SDK 或缺 key 时构造函数抛 `RuntimeError`, demo 捕获后打印跳过信息并正常退出, 不发任何网络请求。不进 `run_all` 是同一原则的另一半: 默认路径的结果不应依赖网络、费用和非确定的模型输出。
</details>
