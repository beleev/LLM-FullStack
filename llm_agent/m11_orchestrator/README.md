# M11 — Orchestrator–Workers: lead 扇出并行子 agent 再综合

一个 lead agent 在同一个 turn 里委托三个子 agent 并行干活, 只收回摘要; 再用 token 账本说明多智能体到底买到了什么。

## 直觉

让一个 agent 串行读完所有资料, 所有原文都会堆在同一个上下文里: 慢, 每一轮还要为这些原文重复付 token, 而且越读越糊。
orchestrator-workers 的做法是: lead 只负责拆任务和综合, 每个 worker 在自己的隔离上下文里读原文, 只交回一段很短的摘要。
它不需要新的调度机制 —— m07 的 `delegate` 工具加上 m10 的并行工具调用, 合起来就是扇出。
要看清的是代价: 每个 worker 都要重建自己的上下文, 总 token 通常更高。多智能体买的是并行度和干净的主上下文, 不是省钱。

## 核心数据结构与控制流

| 位置 | 作用 |
| --- | --- |
| `core/subagents.py` `DelegateTool` | 普通 `Tool`。`agent_types: Dict[str, Callable[[], ToolRegistry]]` 决定每类 worker 的工具集; schema 里 `agent_type` 是 enum |
| `DelegateTool.execute` | 登记 `children[index]` (加锁) -> 新建子 `Agent` -> `child.run(task)` -> `subagent_stop` hook -> 返回截断后的摘要 |
| `DelegateTool.children` | 每个子级的 `{type, task, messages, usage, path}`, 供审计和记账 |
| `core/agent.py` `Agent._run_tools` | 同一 turn 的多个 `delegate` 被线程池并行执行 (`max_parallel` 默认 4) |
| `core/agent.py` `Agent.usage` | `input_tokens` 每次调用累加整个上下文; `peak_context_tokens` 记单次最大值 |
| `core/persistence.py` `JsonlSessionStore` | 子 transcript 写到 `child_<index>_<type>.jsonl` |

```
lead turn 1   assistant = [delegate(researcher, A), delegate(researcher, B), delegate(calculator, C)]
  _run_tools  授权 x3 (auto: low-risk tool) -> ThreadPoolExecutor
     |- DelegateTool.execute -> Agent(child-0, tools=[search_docs]) .run(A) --.
     |- DelegateTool.execute -> Agent(child-1, tools=[search_docs]) .run(B) --+-> 各自: JSONL 落盘
     '- DelegateTool.execute -> Agent(child-2, tools=[calculator])  .run(C) --'   -> subagent_stop hook
                                                                                  -> "[type] " + 摘要 (<= 200 字符)
lead turn 2   一条 user 消息里 3 个 tool_result -> 综合成最终回答
```

关键设计 (为什么这样做):

- **没有调度器**: 扇出就是"一个 assistant turn 里的多个 `tool_use`", 复用 loop 已有的授权、并行、结果回填和 transcript 配对。少一套机制, 就少一处会和主循环不一致的地方。
- **工具集是工厂函数**: 每次委托都 `self.agent_types[agent_type]()` 新建一套工具实例, 子级之间不共享状态; researcher 只拿到 `search_docs`, calculator 只拿到 `calculator`。工具集就是能力边界, 比在提示词里说"你只负责检索"可靠。
- **子级只收到 task 字符串**: 看不到父对话, 也拿不到父级的工具 (包括 `delegate` 本身, 所以不会递归扇出)。
- **子级权限门是 auto 且没有 `ask_policy`**: 子 agent 没有人可问, 拿不准的高风险调用按拒绝处理 (fail closed)。
- **`max_summary_chars = 200` 是硬上限**: 子级再啰嗦也淹不了父上下文。这是 lead 峰值上下文小的直接原因。
- **锁只包住"分配序号 + 登记 children"**: `execute` 会被多个线程同时调用; 但 `child.run` 在锁外, 否则并行就退化成串行了。
- **usage 分开记**: lead 的账和每个 worker 的账分开, 才能同时回答"总共花了多少"和"主上下文被占了多少"这两个不同的问题。

## 运行后应该看到什么

在仓库根目录 (`llm_agent/` 的上一级) 运行:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m llm_agent.m11_orchestrator.demo
```

真实输出节选 (长行用 `...` 截断)。worker 在并行线程里领取序号, `child_00 / child_01 / child_02` 与任务的对应关系在不同运行间可能不同:

```
  [lead] turn 1: model -> tool_use toolu_0001 delegate {'task': '调研 kv cache 如何加速解码', 'agent_type': 'researcher'}
  [lead] turn 1: model -> tool_use toolu_0002 delegate {'task': '调研 lora 为什么省显存', 'agent_type': 'researcher'}
  [lead] turn 1: model -> tool_use toolu_0003 delegate {'task': '计算 4096 * 32', 'agent_type': 'calculator'}
  [lead] permission delegate -> allow (auto: low-risk tool)
  [lead] permission delegate -> allow (auto: low-risk tool)
  [lead] permission delegate -> allow (auto: low-risk tool)
  [lead] tool_result toolu_0001 -> [researcher] 基于工具结果完成： search_docs: [0.59] kv_cache: The kv cache stores past keys and ...
  [lead] tool_result toolu_0002 -> [researcher] 基于工具结果完成： search_docs: [0.46] lora: LoRA trains low rank adapters, cutting...
  [lead] tool_result toolu_0003 -> [calculator] 基于工具结果完成： calculator: 4096 * 32 = 131072
...
[workers]
  child_00_researcher.jsonl    tools=['search_docs'] input_tokens=215
  child_01_researcher.jsonl    tools=['search_docs'] input_tokens=212
  child_02_calculator.jsonl    tools=['calculator'] input_tokens=58
...
[token 记账]
  lead 峰值上下文              : 247
  solo 峰值上下文              : 375
  orchestrated 总输入        : 770 (lead 285 + workers 485)
  solo 总输入                : 413
```

`assert` 验证的事:

- lead 的第一个 assistant turn 里有 3 个 `delegate`, `agent_type` 依次是 `researcher, researcher, calculator`。
- 下一条 user 消息含 3 个 `tool_result`, `validate_transcript(lead.messages)` 无问题 —— 扇出没有破坏 tool_use / tool_result 配对。
- `subagent_stop` hook 触发 3 次, 类型排序后为 `calculator, researcher, researcher` (排序是因为完成顺序不确定)。
- 最终回答以"综合 3 个子任务结果"开头, 且包含 `4096 * 32 = 131072`。
- 每个 worker 的 transcript 文件存在; researcher 只用过 `search_docs`, calculator 只用过 `calculator`。
- 两本账: lead 峰值上下文 < solo 峰值上下文 (247 < 375), 同时 orchestrated 总输入 > solo 总输入 (770 > 413)。

## 与真实系统的差距

- 玩具 lead 不会迭代: 按分号把 prompt 切成子任务, 扇出一轮就综合。Anthropic 的 multi-agent research 系统里, lead 会先制定计划, 看完结果再决定要不要追加一轮 subagent。
- 没有引用环节 (真实系统有单独的 citation 步骤把结论对回来源), 综合也只是把三段摘要拼成列表。
- 摘要是 `shorten()` 截断, 不是模型写的。真实 subagent 自己提炼要点, 截断可能正好切掉关键句。
- 委托的 brief 只有一句任务。真实系统里 lead 要写清目标、输出格式、可用工具和任务边界, 否则 worker 会重复劳动或跑偏。
- token 数是 `estimate_tokens` 的字符粗估, 对照组也不严格: 玩具 solo 只用整句 prompt 做了一次检索。结论的方向 (总量更高、主上下文更小) 是真实的, 具体倍数不是 —— Anthropic 公开的数据是多智能体系统的 token 用量约为普通对话的 15 倍。
- 并行用的是本地线程。真实系统要面对 API 速率限制、单个 worker 超时或失败后的重试; 这里一个 worker 卡住, lead 的整个 turn 就跟着等。
- demo 的 transcript 写在临时目录里, 运行结束即删除; demo 也没有对并行耗时做断言 (那是 m10 [4] 的内容)。

## 常见误区

- **"多智能体更省 token。"** 相反。每个 worker 都要重新付 system prompt、任务描述和原文的钱, 本 demo 是 770 对 413。省下的是 lead 的上下文空间和墙钟时间。
- **"需要一个专门的调度器或消息总线。"** 这里的全部"编排"就是一个工具加上 loop 已有的并行执行。只有当 worker 之间需要互相通信、或 lead 不想同步等待时, 才需要更重的机制。
- **"子 agent 能看到父级的对话。"** 它只看到 `task` 这一个字符串。父级知道而没写进 task 的背景, 对子级就不存在 —— 所以委托描述的质量直接决定结果质量。

## 自测题

1. 为什么 lead 的峰值上下文比 solo 小, 而 orchestrated 的总输入 token 反而更大? 两件事矛盾吗?

<details><summary>答案</summary>
不矛盾, 它们量的是不同的东西。检索回来的长原文只进了 worker 的上下文, lead 只看到每个 worker 不超过 200 字符的摘要, 所以 lead 单次调用的上下文小。但总账要把三个 worker 加进来: 每个 worker 都有自己的 system prompt、任务和原文, 而且每次模型调用都重发整个上下文, 这些加起来 (485) 再加 lead 的 (285) 超过了 solo 的 413。
</details>

2. `DelegateTool.execute` 里的锁保护了什么? 为什么 `child.run(...)` 不放在锁里? 序号可能变化, lead 看到的结果顺序会不会也乱?

<details><summary>答案</summary>
锁保护的是"读 `len(self.children)` 得到序号 + append 登记"这个读改写, 否则两个线程可能拿到同一个序号、写同一个 transcript 文件。`child.run` 如果放进锁里, 三个 worker 就只能一个接一个跑, 并行失效。序号取决于哪个线程先抢到锁, 所以文件编号和任务的对应关系可能变; 但 lead 看到的结果顺序不会乱, 因为 `pool.map` 按提交顺序返回结果, 每条结果还挂在各自的 `tool_use_id` 上。
</details>

3. 子 agent 的权限门是 `PermissionGate(mode="auto")` 且没有 `ask_policy`。如果某类 worker 的工具集里有一个 `risk="high"` 的工具, 子 agent 调用它会怎样? 为什么这样设计?

<details><summary>答案</summary>
auto 模式对高风险工具"拿不准", 转交 `_ask`; 没有 `ask_policy` 就视为无人应答, 结果是拒绝, 子 agent 收到一条 `DENIED` 的 tool_result。这样设计是因为子 agent 在后台线程里运行, 没有人可以实时审批; 在无人监督的地方默认拒绝 (fail closed), 比默认放行安全。注意当前实现把子级的门写死了, 没有传入 allow 规则的入口, 所以高风险工具在子级里实际上不可用; 真实系统会让子级继承或显式配置一份权限规则。
</details>
