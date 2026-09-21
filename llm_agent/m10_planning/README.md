# M10 — Planning: todo、plan 模式、并行工具调用

三件和"先想后做"有关的机制: 把计划变成显式状态, 让用户在动手前有否决权, 让互不依赖的调用同时跑。

## 直觉

多步任务里, 模型如果边想边做, 做到第三步常常忘了还剩什么; 用户也只能在事后发现它改了不该改的东西。
`todo_write` 让模型把计划写成一份显式的列表, 留在 transcript 里, 也能展示给人看。
plan 模式把"只许看、不许动"做成权限门的一个模式: 模型必须用 `exit_plan_mode` 提交计划, 人批准后才解锁写操作。
关键在于这是 harness 强制的 —— 模型不交计划直接写, 门照样不开 (demo [3])。
并行工具调用解决的是另一个问题: 三个互不依赖、各 0.2s 的查询, 串行要 0.6s, 同一个 turn 发出去只要约 0.2s。

## 核心数据结构与控制流

| 位置 | 作用 |
| --- | --- |
| `core/tools.py` `TodoWriteTool` | 每次整表覆写 `self.todos`, 校验 status 只能是 `pending / in_progress / completed`; `read_only = True` |
| `core/tools.py` `ExitPlanModeTool(gate, approve, next_mode)` | 调 `approve(plan)`; 通过则 `gate.mode = next_mode` (默认 `accept_edits`), 否则返回 `ok=False` |
| `core/permissions.py` `PermissionGate._evaluate_one` | plan 模式下只评估 deny 规则, 然后只放行 `tool.read_only` 的工具 |
| `core/agent.py` `Agent._run_tools` | 授权串行, 执行用 `ThreadPoolExecutor(max_workers=max_parallel)` |
| `core/toy_llm.py` `RuleBasedLLM._plan` | 玩具模型: 算出 `pre -> gather -> act -> post`, gather 阶段的调用一次全发 |

plan 模式的一次完整运行 (demo [2]):

```
gate.mode = "plan"
turn 1  todo_write([pending, pending])   read_only -> allow (source=plan)
turn 2  exit_plan_mode(plan)             read_only -> allow -> approve(plan)?
          拒绝: ToolResult(ok=False), mode 仍是 plan, 模型收尾, 什么都没写   (demo [1])
          批准: gate.mode = "accept_edits"
turn 3  search_docs                      allow (accept_edits: low/medium risk)
turn 4  write_note                       allow (medium 风险, 现在才放行)
turn 5  todo_write([completed, completed])
```

一个 turn 里多个 `tool_use` 的处理 (`_run_tools`):

```
for call in calls:  _authorize(call)        # 串行: PreToolUse hook -> 权限门
pool.map(tools.execute, approved)           # 并行; 只有 1 个获批时直接调用
按原顺序回填 results, 被拒的也有一条 is_error 的 tool_result
所有 tool_result 放进同一条 user 消息
```

关键设计 (为什么这样做):

- **整表覆写而不是增删改单项**: 没有 id、没有局部更新的歧义, 调用是幂等的; 模型每次都要重述整个计划, 最新状态总在上下文的近处。harness 只负责保存, 不调度 —— 推进计划的是模型。
- **`todo_write` 标成只读**: 它只改 agent 自己的计划状态, 不碰外部世界, 所以 plan 模式下也能用, 否则"先列计划"这一步本身就会被拒。
- **plan 模式下 allow 规则也失效**: `_evaluate_one` 在 plan 模式只看 deny 规则就跳出规则循环。否则一条早先配好的 `allow write_note` 就能让 plan 模式形同虚设。
- **模式是门的状态, 由工具翻转**: `ExitPlanModeTool` 持有 `gate` 的引用; 审批由注入的 `approve` 回调代表"人" (demo 不能 `input()`)。批准发生在工具执行里, 所以模型无法绕过审批自己改模式。
- **授权串行、执行并行**: 审批弹窗不能并发, 顺序也要确定; 真正耗时的是执行。`pool.map` 保序, 结果始终挂回模型发出的那个 `tool_use_id`。

## 运行后应该看到什么

在仓库根目录 (`llm_agent/` 的上一级) 运行:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m llm_agent.m10_planning.demo
```

真实输出节选 (长行用 `...` 截断; 最后两行的耗时每次运行略有不同):

```
[1] plan 模式 + 用户拒绝计划: 一个字都没写
  [rejected] turn 1: model -> tool_use toolu_0001 todo_write {'todos': [{'content': 'search_docs — need external knowledge', 'status': 'pending'}, ...]}
  [rejected] permission todo_write -> allow (plan: read-only tool)
  [rejected] tool_result toolu_0001 -> todos updated: 0/2 completed
  [rejected] turn 2: model -> tool_use toolu_0002 exit_plan_mode {'plan': '1. search_docs — need external knowledge\n2. write_note — persist a note'}
  [rejected] permission exit_plan_mode -> allow (plan: read-only tool)
  [rejected] tool_result toolu_0002 -> plan rejected by user; stay in plan mode
  [rejected] final: 计划未获批准, 未执行任何写操作。
...
[2] 用户批准: 模式切到 accept_edits, 按计划执行, todo 从 pending 走到 completed
  [approved] tool_result toolu_0002 -> plan approved; mode -> accept_edits
  [approved] permission write_note -> allow (accept_edits: low/medium risk)
  tool order              : ['todo_write', 'exit_plan_mode', 'search_docs', 'write_note', 'todo_write']

[3] plan 模式是 harness 强制的: 就算模型不交计划直接写, 门也不开
  [rogue] turn 1: model -> tool_use toolu_0001 write_note {'text': '写入笔记: 偷偷写'}
  [rogue] permission write_note -> deny (plan: plan mode is read-only until the plan is approved)

[4] 并行工具调用: 3 个 0.2s 的调用
  serial (max_parallel=1) : 0.61s
  parallel (max_parallel=4): 0.21s
```

`assert` 验证的事:

- [1] 拒绝后: 笔记为空, `gate.mode` 仍是 `plan`, 最终回答含"未获批准", transcript 里根本没有 `write_note` 的 `tool_use`。
- [2] 批准后: 工具顺序恰为 `todo_write, exit_plan_mode, search_docs, write_note, todo_write`; 模式变成 `accept_edits`; 笔记正好 1 条; todo 第一版全是 `pending`, 最后一版全是 `completed`。
- [3] 工具集里没有 `exit_plan_mode` 的 agent 直接写: 结果含 `DENIED: plan mode is read-only`, 笔记仍为空。
- [4] 两种配置下都是 1 个 assistant turn 含 3 个 `tool_use`、下一条 user 消息含 3 个 `tool_result`; 串行耗时 >= 0.6s, 并行耗时 < 串行的 70%。

## 与真实系统的差距

- 玩具模型只写两版 todo (全 pending, 全 completed), 从不使用 `in_progress`, 也不会在某步失败后改计划。真实模型逐步更新, 并根据结果重排。
- harness 只保存 todo: 不在界面展示, 也不会在后续上下文里主动提醒模型当前的 todo 状态; 旧轮次被压缩后, 计划只活在摘要里。
- 计划被拒后玩具模型直接收尾。Claude Code 里用户可以给出修改意见, 模型留在 plan 模式改完再提交。
- `read_only` 是工具作者自己声明的类属性, harness 无法验证, 标错就是漏洞。例如 `DelegateTool` 必须标 `read_only = False`: 子 agent 用的是自己的 auto 模式权限门, 若委托被当成只读, plan 模式就能靠子级绕过。
- 同一批获批的调用一律并行, 不区分读写。真实系统通常只并行只读调用, 写操作串行; 这里因为玩具模型只在 gather 阶段 (只读) 成批发调用, 才没暴露问题。也没有单个工具的超时与取消, 一个调用卡住整批都等。
- Claude API 里是否并行由模型决定 (一个 assistant 消息里多个 `tool_use` 块), 可用 `tool_choice` 的 `disable_parallel_tool_use` 关闭; 这里的 `max_parallel` 只是 harness 侧的线程数。

## 常见误区

- **"plan 模式就是在 system prompt 里叮嘱模型先别动手。"** 提示词只是建议。这里的只读由权限门强制: demo [3] 的模型直接发 `write_note`, 得到的是 `DENIED`。
- **"todo 列表是 harness 用来调度任务的。"** harness 不读 todo 的内容, 也不据此决定下一步。它是模型给自己 (和用户) 看的外部记忆, 执行顺序仍然由模型每一轮的输出决定。
- **"并行是 harness 把调用拆开同时跑。"** harness 只能并行模型放在同一个 turn 里的调用; 有依赖的调用 (先搜索再写笔记) 必须跨 turn。并行省的是墙钟时间, 不省 token。

## 自测题

1. 用户早先配置了规则 `PermissionRule("write_note", "", Decision.ALLOW)`, 然后进入 plan 模式。`write_note` 会被放行吗? 如果配的是 deny 规则呢?

<details><summary>答案</summary>
不会放行。`_evaluate_one` 按 deny、ask、allow 的顺序查规则, 但在 plan 模式下处理完 deny 就 `break`, ask 和 allow 规则都不看, 随后因为 `write_note.read_only` 为 False 而拒绝。deny 规则在 plan 模式下照常生效, 连只读工具也能被它拒掉。这样设计是因为 plan 模式的承诺是"批准之前零写入", 不能被一条历史规则打穿。
</details>

2. 模型在一个 turn 里发了 3 个调用, 其中第 2 个被权限门拒绝。执行层和 transcript 里分别发生什么?

<details><summary>答案</summary>
三个调用依次过 `_authorize`; 第 2 个立即得到一条 `ok=False` 的 `DENIED` 结果, 不进线程池。第 1、3 个并行执行。最后三条 `tool_result` 按原顺序放进同一条 user 消息, 各自挂在对应的 `tool_use_id` 上。每个 `tool_use` 都必须有配对的 `tool_result`, 否则真实 API 会直接拒绝这段历史。
</details>

3. `todo_write` 为什么设计成"整表覆写", 而不是 `add_todo / update_todo / delete_todo` 三个操作?

<details><summary>答案</summary>
增量操作需要稳定的 id, 模型一旦记错 id 或漏掉一次更新, harness 里的状态就和模型以为的状态分叉, 而且很难发现。整表覆写没有分叉的可能: 模型最后一次写的就是全部真相, 调用幂等, 实现只有一次赋值。代价是每次多花一点输出 token, 对十来项的列表可以忽略。
</details>
