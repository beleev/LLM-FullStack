# M09 — MCP: 工具活在另一个进程里

用一个真实的 stdio JSON-RPC 2.0 server 演示: 工具由外部进程通过标准协议提供, 但进了 agent 之后和内置工具走完全相同的校验与权限路径。

## 直觉

到 m08 为止, 每个工具都是 agent 进程里的一个 `Tool` 子类。想接天气、数据库、工单系统, 就得在 agent 里各写一个类, 而且别的 agent、别的语言用不了。
MCP (Model Context Protocol) 把"工具提供方"拆成独立进程: 它用 JSON-RPC 自报工具清单 (`tools/list`) 并接受调用 (`tools/call`), agent 这边只需要一个通用客户端。
对模型来说什么都没变: 它看到的还是 `{name, description, input_schema}`。MCP 是 harness 与 server 之间的协议, 不是模型的新能力。
代价是信任问题: server 是第三方代码, 它的工具描述、自报的"只读"注解、返回内容都不可信。没有统一的权限门, 装一个 server 就等于给 agent 开了个后门。

## 核心数据结构与控制流

| 位置 | 作用 |
| --- | --- |
| `m09_mcp/server.py` | 独立脚本, 不 import 本包。`for line in sys.stdin` 逐行读 JSON-RPC, 只实现 `initialize` / `tools/list` / `tools/call`; 其它方法回 `-32601` |
| `core/mcp.py` `MCPClient` | `subprocess.Popen(argv, stdin=PIPE, stdout=PIPE)` 拉起 server, 握手, `request()` 收发 |
| `core/mcp.py` `MCPTool` / `mcp_tools()` | 把 `tools/list` 的每一项包成 `Tool`: 名字 `mcp__<server>__<tool>`, `parameters = inputSchema` |
| `core/tools.py` `ToolRegistry.execute` | 先用 `inputSchema` 本地校验, 再调 `MCPTool.execute` |
| `core/permissions.py` `PermissionGate` | 规则的工具名是 glob, `mcp__weather__*` 一条管住整个 server |

```
Agent / demo              MCPClient                         server.py (子进程)
MCPClient(name, argv) --> Popen(argv), 起读线程
                          initialize (id=1) --------------> {protocolVersion, capabilities:{tools}, serverInfo}
                          notifications/initialized ------> 没有 id = notification, 不回复
mcp_tools(client) ------> tools/list --------------------->  {tools:[{name, description, inputSchema}]}
ToolRegistry.execute
  validate_args(inputSchema)      # 畸形参数到不了 server
  MCPTool.execute ------> tools/call {name, arguments} --->  {content:[{type:"text",...}], isError}
close(): 关 stdin --------------------------------------->  读到 EOF, 循环结束, 退出码 0
```

关键设计 (为什么这样做):

- **换行分隔 + stdout 专用**: stdio 传输里 stdout 就是协议通道, 所以 server 的日志只能写 stderr, 每条响应后必须 `flush()`, 否则客户端一直等到超时。
- **读线程 + 队列**: `readline()` 没有超时参数。后台线程把 stdout 的每一行放进 `queue.Queue`, `request()` 用 `get(timeout=5.0)` 取, server 卡死时抛 `MCPError` 而不是把 agent 挂住。
- **一把锁**: loop 会用线程池并行执行同一 turn 的多个调用 (demo [4]), 它们共用一条管道。客户端的假设是"发完请求后读到的下一行就是它的响应", 所以"发送 + 取响应"必须在锁内成对完成。
- **命名空间前缀**: 防止两个 server 的工具重名, 也让权限规则可以按 server 粒度写。
- **`MCPTool.risk = "high"`, `untrusted_output = True`**: 一律按第三方代码处理, 不读 server 自报的注解。auto 模式下高风险工具会走 `_ask`; demo 没有配 `ask_policy` (没人可问), 于是 fail closed —— 这就是输出里 `deny (human: ...)` 的来历。`untrusted_output` 只有在 Agent 配了 `Guardrails` 时才生效 (见 m12), 本 demo 没配。
- **两种失败分开**: 方法不存在是协议错误 (JSON-RPC `error`, 客户端抛 `MCPError`); 工具自己执行失败是正常的 `result` 加 `isError: true`, 变成 `ToolResult(ok=False)` 回给模型, 让它有机会改参数重试。
- **唯一的子进程**: 只 spawn 调用方给的 argv 列表 (`sys.executable` + 同目录 `server.py`), 不经过 shell, 没有字符串拼接命令。

## 运行后应该看到什么

在仓库根目录 (`llm_agent/` 的上一级) 运行:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m llm_agent.m09_mcp.demo
```

真实输出节选 ([3] [4] 各省略了最后一行 final):

```
[1] 握手 + 工具发现
  serverInfo              : {'name': 'toy-weather', 'version': '0.1'}
    mcp__weather__add  {'a': {'type': 'number'}, 'b': {'type': 'number'}}
    mcp__weather__get_weather  {'city': {'type': 'string'}}

[2] 两种失败: 协议错误 vs 工具错误
  protocol error          : weather: -32601 Method not found
  tool error              : isError=True tool error: KeyError('city')

[3] 同一个权限门: 没有 allow 规则, auto 模式不放行第三方工具
  [no-rule] turn 1: model -> tool_use toolu_0001 mcp__weather__get_weather {'city': 'Beijing'}
  [no-rule] permission mcp__weather__get_weather -> deny (human: classifier unsure about high-risk tool; denied)
  [no-rule] tool_result toolu_0001 -> DENIED: classifier unsure about high-risk tool; denied

[4] 加一条 mcp__weather__* 规则: 两个城市并行调用
  [with-rule] turn 1: model -> tool_use toolu_0001 mcp__weather__get_weather {'city': 'Beijing'}
  [with-rule] turn 1: model -> tool_use toolu_0002 mcp__weather__get_weather {'city': 'Shanghai'}
  [with-rule] permission mcp__weather__get_weather -> allow (rule: trusted weather server)
  [with-rule] permission mcp__weather__get_weather -> allow (rule: trusted weather server)
  [with-rule] tool_result toolu_0001 -> Beijing: sunny, 24C, light wind
  [with-rule] tool_result toolu_0002 -> Shanghai: sunny, 24C, light wind
```

`assert` 验证的事:

- 发现到的工具名恰好是 `mcp__weather__add` 和 `mcp__weather__get_weather`。
- `resources/list` 抛 `MCPError` 且含 `-32601`; 绕过本地校验直接 `call_tool("get_weather", {})` 得到 `isError=True` 而不是异常。
- `add(2, 3)` 返回 `("5", False)`。
- 没有 allow 规则时最终回答含 `DENIED`; 加规则后两个城市的结果都在, 同一 assistant turn 有 2 个 `tool_use`, 且 `validate_transcript` 无问题。
- `with` 退出后 `proc.poll() == 0`: 关 stdin 让 server 正常退出, 不留孤儿进程。

## 与真实系统的差距

- 只实现了 tools。MCP 规范还有 resources、prompts、sampling、roots、elicitation, 以及 ping、进度、取消、`tools/list` 分页, 这里都没有。
- 只有 stdio 传输。没有 Streamable HTTP, 因而也没有 OAuth 授权、会话管理、断线重连。
- 没有真正的能力协商: 客户端发空的 `capabilities`, 不检查 server 返回的 `protocolVersion` 和 `capabilities`; 不处理 `notifications/tools/list_changed`, 工具清单在启动时读一次就固定了。
- 严格的一问一答 (一把锁串行化所有请求): 请求 id 在锁内取号, 迟到的旧响应按 id 丢弃, stdout 上的非 JSON 行报 `MCPError`; 但 server 主动发来的请求 / 通知会被直接丢掉。真实客户端用 `id -> 等待者` 的映射做多路复用, 并处理 server→client 的消息。
- `call_tool` 只拼接 `type == "text"` 的内容块, 忽略图片、音频、嵌入资源和 `structuredContent`。
- Claude Code 从 `.mcp.json` 等配置加载 server、逐个 server 让用户确认是否信任; 这里 argv 写死在 demo 里, server 崩溃后也不会重启。

## 常见误区

- **"模型需要专门支持 MCP。"** 不需要。模型只看到普通的工具定义并发出普通的 `tool_use`; MCP 是 harness 和工具进程之间的事, 换掉它模型无感。
- **"server 声明自己只读, 就可以免审批。"** 注解是第三方自己写的, 不是事实。`MCPTool` 无视它们, 统一 `risk = "high"`, 放行只能来自用户写的规则。
- **"工具执行失败就该返回 JSON-RPC error。"** `error` 留给协议层问题 (方法不存在、请求非法)。工具失败要走 `isError: true` 的正常结果, 这样错误文本能作为 `tool_result` 进上下文, 模型才能自我纠正。

## 自测题

1. 有人在 `server.py` 的 `call_tool` 里加了一句 `print("calling", name)` 调试, 会发生什么? 应该怎么打日志?

<details><summary>答案</summary>
stdout 是协议通道, 这行文本会先于真正的响应进入客户端队列, `request()` 解析失败并抛 `MCPError` (走 agent 路径时被 `ToolRegistry.execute` 兜成 `ERROR` 结果); 真正的响应要等下一次请求时才按 id 被丢弃。日志必须写 stderr (`print(..., file=sys.stderr)`)。同理, 响应后不 `flush()` 会让客户端等满 5 秒超时。
</details>

2. 每个请求已经有自增 id 了, `MCPClient` 为什么还需要 `threading.Lock`?

<details><summary>答案</summary>
因为这个客户端不按 id 分发响应, 而是假设"我发完之后队列里的下一行就是我的"。两个线程并发时, A 可能取走 B 的响应。锁把"发送 + 取一行"变成原子操作, id 只用来事后校验。想要真正的并发, 需要一个读线程按 id 把响应投递给各自的等待者 —— 那是真实 MCP 客户端的做法。
</details>

3. demo [3] 里 gate 是 auto 模式, 为什么日志显示的来源是 `human` 而不是 `auto`? [4] 里一条规则为什么能覆盖 server 的所有工具?

<details><summary>答案</summary>
`_auto_classify` 对 low/medium 风险直接放行, 对 high 风险"拿不准", 转交 `_ask`。demo 没有提供 `ask_policy`, 没人可问就按拒绝处理 (fail closed), 所以来源是 `human`、结果是 deny。[4] 能一条规则覆盖, 是因为规则的工具名用 `fnmatch` 做 glob 匹配, 而所有该 server 的工具都带 `mcp__weather__` 前缀; 规则的优先级 (deny > ask > allow) 高于模式兜底, 所以 allow 规则先于 auto 分类器生效。
</details>
