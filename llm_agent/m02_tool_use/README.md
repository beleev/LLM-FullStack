# M02 — Tool Use

工具 = 给模型看的 JSON Schema + 由确定性代码执行的 `execute`; 本模块演示 schema、执行前校验和并行工具调用。

## 直觉

模型只能产出文本; "调用工具"其实是模型按约定格式写出工具名和参数, 由 harness 去执行。
如果只告诉模型工具名, 参数名和类型它只能猜 (`expression` 还是 `expr`?)。
反过来, 模型的输出对 harness 是不可信输入: 不校验就执行, 坏参数会让工具在深处崩溃, 错误信息对模型毫无帮助。
所以 schema 同时服务两头: 对模型是说明书, 对 harness 是信任边界上的输入验证。
校验失败不抛异常, 而是回一个 `is_error` 的 `tool_result`, 模型下一轮可以自己改。

## 核心数据结构与控制流

- `core/tools.py: Tool` — `name / description / parameters` 加三个只给 harness 看的元数据 `risk / read_only / untrusted_output`。
- `core/tools.py: Tool.schema` — 输出 `{name, description, input_schema}`, 即 Claude API `tools=[...]` 的元素; 元数据不发给模型。
- `core/tools.py: validate_args` — JSON Schema 最小子集: `required / type / enum / additionalProperties`, 只校验一层。
- `core/tools.py: ToolRegistry.execute` — 未知工具 -> `ERROR: unknown tool`; 校验失败 -> `INVALID_ARGS: ...`; 工具抛异常 -> `ERROR: ...`; 三者都是 `ok=False` 的 `ToolResult`, 不会炸掉 loop。
- `core/agent.py: Agent._run_tools / _authorize` — 一个 assistant turn 里多个 `tool_use` 的处理。

```
assistant: [tool_use A, tool_use B]            # 一个 turn, 多个调用
   |
   |  for call in calls (串行):  PreToolUse hook -> 权限门
   |      拒绝 -> results[i] = is_error 结果 ("DENIED: ...")
   |      通过 -> approved
   v
approved > 1 ?  ThreadPoolExecutor.map(tools.execute)  :  直接执行
   |                 每个 execute 内部: validate_args -> tool.execute
   v
user: [tool_result A, tool_result B]           # 全部放进同一条 user 消息, 顺序与 tool_use 一致
```

关键设计决定:

- 授权串行、执行并行: 审批可能要问人, 不能并发弹窗; 而互不依赖的只读调用并行后总耗时约等于最慢的那个。
- 被拒绝的调用也占一个 `tool_result` 位置 (`is_error: true`), 否则同一 turn 里其它调用的配对会被破坏。
- `result.tool_use_id` 永远取模型发出的那个 id, 即使 hook 改写了调用。
- `ToolRegistry.schemas()` 按名字排序: 工具列表确定, 真实 API 下才不会因为顺序变化让 prompt cache 失效。
- `bool` 是 `int` 的子类, `validate_args` 显式排除, 否则 `True` 会被当成合法 `number`。
- toy LLM 的计划分 `gather` (互不依赖, 一次全发) 和 `act` (写操作, 一次一个): 先并行收集, 再串行动手。

## 运行后应该看到什么

```bash
cd "/Volumes/My Shared Files/LLMFullStack"
python3 -m llm_agent.m02_tool_use.demo
```

```
[1] 发给模型的工具定义 (Claude API 格式)
  {"name": "calculator", "description": "Compute a small arithmetic expression.", "input_schema": {"type": "object", "properties": {"expr": {"type": "string", ...}}, "required": ["expr"], "additionalProperties": false}}
  ...
[2] 执行前校验: 模型给错参数时, 得到的是可自我纠正的错误, 不是崩溃
  {'expression': '1+1'}                                -> INVALID_ARGS: missing required 'expr'; unexpected 'expression'
  {'expr': 42}                                         -> INVALID_ARGS: 'expr' should be string, got int
  {'expr': "__import__('os').system('id')"}            -> ERROR: unsupported expression: "__import__('os').system('id')"

[3] 并行 gather → 串行 act
  [m02] turn 1: model -> tool_use toolu_0001 calculator {'expr': '6 * 7'}
  [m02] turn 1: model -> tool_use toolu_0002 search_docs {'query': '计算 6 * 7，同时搜索 agent loop，并写入笔记'}
  [m02] permission calculator -> allow (auto: low-risk tool)
  [m02] permission search_docs -> allow (auto: low-risk tool)
  [m02] tool_result toolu_0001 -> 6 * 7 = 42
  [m02] tool_result toolu_0002 -> agent_loop: Agent loop = assemble context, call model, run tools, repeat.
  [m02] turn 2: model -> tool_use toolu_0003 write_note {'text': 'agent_loop: Agent loop = assemble context, ...'}
  [m02] permission write_note -> allow (auto: bounded local write)
  [m02] tool_result toolu_0003 -> note[1] saved
```

`assert` 验证的内容:

- 每个 schema 的键恰好是 `{name, description, input_schema}`, 且 `input_schema.type == "object"`。
- 三个坏调用都得到 `ok=False` 的结果 (没有异常逃出 `execute`); 正确调用 `1+1` 返回 `1+1 = 2`。
- 第一个 assistant 消息带两个 `tool_use` (`calculator`, `search_docs`), 下一条 user 消息带两个 `tool_result`。
- `validate_transcript` 返回空: 并行调用的 id 配对完整。
- 笔记内容恰好等于 `search_docs` 的原始输出 (纯工具数据, 没有混进别的文字); 最终回答含 `6 * 7 = 42`。

注意第三个坏调用: 它通过了 schema (确实是字符串), 是被 `_safe_eval_arithmetic` 的白名单 AST 拒绝的。schema 只管形状, 语义安全要靠工具自己。

## 与真实系统的差距

- `validate_args` 只是一层子集: 不校验嵌套对象、`items`、`minimum / pattern / oneOf` 等。例如 `todo_write` 的数组元素只能在工具内部手工检查。真实系统用完整的 jsonschema 库, 或用 API 的 `strict: true` 让采样阶段就满足 schema。
- 这里只要通过授权就全部并行, 不区分工具是否并发安全; 生产 harness 通常只并行只读工具, 写操作串行。也没有单个工具的超时和输出长度上限。
- "并行调用"由 `RuleBasedLLM` 的 gather / act 规则决定; 真实模型是否在一个 turn 里发多个 `tool_use` 取决于模型和提示, harness 只能支持, 不能强制。
- 所有工具都是模拟或纯计算 (`ShellTool` 从不执行真实命令), 没有真实 I/O 的失败模式: 网络超时、部分写入、巨大输出。
- 工具 `description` 在真实系统里是提示工程的重点 (何时用、何时不用、返回什么); 这里只有一句话。

## 常见误区

- "schema 只是文档。" 它同时是模型的参数说明书和 harness 的校验依据; 少了前者模型猜参数, 少了后者坏参数直达工具内部。
- "校验失败应该抛异常终止。" 模型给错参数是常态。把错误原文作为 `is_error` 的 `tool_result` 回填, 模型通常下一轮就能改对; 抛异常只会丢掉整个会话。
- "并行工具调用是模型在并行执行。" 模型只是在一个 turn 里写了多个 `tool_use`; 并行是 harness 用线程池做的, 而且全部结果必须合在同一条 user 消息里回去。

## 自测题

1. demo [2] 的三个坏调用里, 为什么前两个是 `INVALID_ARGS`, 第三个却是 `ERROR`? 这说明了什么?

<details><summary>答案</summary>
前两个违反 schema (缺必填参数 / 多余参数 / 类型不对), 在 `validate_args` 就被拦下, 工具代码没运行。第三个是合法字符串, 通过了 schema, 进入 `CalculatorTool.execute` 后被白名单 AST 求值器以 `ValueError` 拒绝, 再由 `ToolRegistry.execute` 的 `except` 转成 `is_error` 结果。结论: schema 只约束形状, 语义层面的安全 (这里是"绝不 eval") 必须由工具自己保证, 两层缺一不可。
</details>

2. 一个 turn 里有两个 `tool_use`, 其中一个被权限门拒绝。transcript 里应该怎么记? 为什么不能只回填成功的那个?

<details><summary>答案</summary>
两个都要有 `tool_result`, 放在同一条 user 消息里; 被拒绝的那个内容是 `DENIED: ...` 且 `is_error: true` (`Agent._run_tools` 里先占位 `results[i]`)。真实 API 要求每个 `tool_use` 都有同 id 的结果, 缺一个整条请求就非法; 而且模型需要知道"被拒绝了", 才能换方案而不是重复请求。
</details>

3. 为什么授权是串行的, 执行却可以并行?

<details><summary>答案</summary>
授权可能需要人来决定, 同时弹出多个审批既无法操作也难以审计, 而且 hook / 权限门可能读写共享状态, 串行最简单可靠。执行阶段的调用都已获批且 (在 gather 阶段) 互不依赖, 用 `ThreadPoolExecutor.map` 并行后耗时约为最慢的一个, `map` 还保持了结果顺序, 回填时与 `tool_use` 一一对应。
</details>
