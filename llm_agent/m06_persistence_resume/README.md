# M06 — Persistence & Resume: append-only JSONL 会话日志

把 transcript 的每条消息立刻追加成 JSONL 的一行, 于是会话可以在新进程里恢复, 也可以在事后逐步复盘。

## 直觉

agent 的全部"状态"其实就是 messages 列表; 它只活在内存里, 进程一退会话就没了。
更糟的是出了事故无从复盘: 模型当时要求调用什么、harness 放行了什么、工具返回了什么, 都查不到。
解法朴素: 每产生一条消息就往文件尾追加一行 JSON, resume 就是把这些行读回来当 messages。
要点在"记什么"和"不恢复什么": assistant 的 `tool_use` 必须和 `tool_result` 一样落盘 (否则只剩孤零零的结果, 既不能审计也不能喂回真实 API); 而权限绝不能随 transcript 恢复。

## 核心数据结构与控制流

- `core/persistence.py: JsonlSessionStore` — `append(message)` 追加一行 `{"role", "content", "name"}`; `load()` 给 resume 用, `load_all()` 给审计用。
- `core/schema.py: Message.to_dict / from_dict` — content 要么是 str, 要么是 block 列表 (`tool_use` / `tool_result`), 序列化后与 Messages API 的消息形状一致。
- `core/schema.py: validate_transcript` — 检查每个 `tool_use` 都在下一条 user 消息里配齐同 id 的 `tool_result`。
- `core/agent.py: Agent._append` — 所有进入 `self.messages` 的消息唯一入口, 同时写 store。
- `core/agent.py: Agent.__init__(store=..., load_history=True)` — resume 入口。

```
写:  run() ─ _append(user prompt) ─ _append(assistant tool_use) ─ 执行工具 ─ _append(user tool_result) ─ _append(final)
                                          └ 先落盘"模型要什么", 再去执行
读:  Agent(store, load_history=True)
       messages  = store.load()                          # 上下文
       _next_id  = store.load_all() 里 tool_use 的总数     # id 续号
     首次 run() → _start_session():
       source = "resume" if messages else "startup"
       hook 返回的文字若已在历史的 session_start 消息里 → 不再注入
```

关键设计决策:

- **只追加、不改写**。写入逻辑只有 open("a") + 一行 JSON, 已写的行永不变化, 审计链天然完整; 代价是文件只增不减。
- **先记录意图, 再执行**。assistant 的 `tool_use` 在工具运行之前就落盘, 即使工具执行中进程崩溃, 日志里也有"模型要求了这一步"。
- **恢复上下文 ≠ 恢复授权**。`PermissionGate` 根本不在 JSONL 里; 新会话必须自己传一个新的门。demo 里会话 A 是 `auto`, 会话 B 退回 `default`, 写笔记要重新问"人"。上一次的"同意"是对当时情境的同意, 不是永久授权。
- **tool_use id 跨会话续号**。toy LLM 不带 id, 由 agent 分配 `toolu_NNNN`; 从 `load_all()` (全量历史, 而非压缩后的视图) 计数, 保证压缩后也不会与旧 id 撞号。
- **压缩不删历史**。`append_compact(summary, kept)` 追加一条 `compact_boundary` 记录; `load()` 遇到它就把视图换成"摘要 + 保留的尾部", `load_all()` 则跳过 boundary 返回全部原始消息。本模块的 demo 不触发压缩, 演示在 m14。

## 运行后应该看到什么

```bash
cd <仓库根目录> && python3 -m llm_agent.m06_persistence_resume.demo
```

```
[raw jsonl]
  {"role": "system", "content": "Policy: answer in Chinese.", "name": "session_start"}
  {"role": "user", "content": "搜索 agent loop", "name": null}
  {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_0001", "name": "search_docs", "input": {"query": "搜索 agent loop"}}], "name": null}
  {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_0001", "content": "agent_loop: Agent loop uses ...
  {"role": "assistant", "content": "基于工具结果完成：\nsearch_docs: agent_loop: Agent loop uses messages, tools, ...

[2] session B: 新进程读回 transcript 继续; 引用'刚才的结果'
  [session-B] turn 1: model -> tool_use toolu_0002 write_note {'text': 'agent_loop: Agent loop uses messages, tools, permissions and persistence.'}
  [session-B] permission write_note -> allow (human: default mode asks for unknown action; approved)
  [session-B] tool_result toolu_0002 -> note[1] saved
  jsonl lines             : 9
```

断言验证的内容:

- 会话 A 的 JSONL 里 block 类型恰为 `["tool_use", "tool_result"]` —— 模型发起的调用也被记录, 不只是结果。
- 会话 B 读回的消息数等于文件行数 (5 条全部恢复)。
- "把刚才结果写入笔记"写进去的正是会话 A 的工具数据, 原样无杂质 —— 证明 B 真的看到了 A 的上下文。
- `validate_transcript(agent_b.messages) == []`: 跨会话拼接后的 transcript 仍是合法的 tool_use / tool_result 配对序列。
- `session_start` hook 收到的 source 依次是 `["startup", "resume"]`, 但 transcript 里的 `session_start` 消息只有 1 条。
- tool_use id 为 `["toolu_0001", "toolu_0002"]`, 跨会话续号。

输出里 `permission write_note -> allow (human: ...)` 说明会话 B 的写操作是重新问出来的, 不是从 A 继承的。

## 与真实系统的差距

- 没有 fsync、没有文件锁: 断电可能丢不止一行; 两个进程同时 resume 同一文件会交错写。容错只有两条: `load()` 跳过写了一半的坏行; resume 时若最后一条是没有结果的 `tool_use` (崩在调用与结果之间), `Agent` 会丢掉它。
- 崩溃在 `tool_use` 落盘之后、`tool_result` 落盘之前, 会留下悬空的 `tool_use`; resume 时代码不会自动修复 (`validate_transcript` 只在 demo 里调用, `Agent` 加载时不检查), 这样的 transcript 直接发给真实 Messages API 会被拒绝。
- "block 格式与 Messages API 一致"不等于"整个文件可原样发送": JSONL 里有 role 为 `system` 的 harness 注入消息和 `name` 字段, API 的 messages 数组不接受; 发送前要经 `core/claude_llm.py: to_api_messages` 转换 (开头的 system → 顶层 `system` 参数, 中途的 system → user 侧 `<system-reminder>` 文本, 相邻同角色合并)。
- 每行只有 role / content / name: 没有时间戳、session id、消息 uuid / parent 链、模型名、usage、cwd; Claude Code 的会话文件带这些元数据, 才能支持会话列表、分叉与回退。
- 没有会话发现机制 (`claude --resume` 的选择列表、`--continue` 取最近一次); 这里要手动给路径。
- 只持久化了对话。工具侧的外部状态 (demo 里的 `notes` 列表、真实系统里的文件改动) 不在 JSONL 里, resume 不会回滚或重放它们。
- resume 时只要历史里已有 `session_start` 消息就不再注入 (hook 仍会以 source="resume" 被调用); 它若被压缩掉, 会在下次 resume 重新注入, 同时其文本会作为"必须保留"交给摘要 (见 m14)。
- 日志是明文。这里只有传了 `guardrails` 才在 `_append` 里脱敏 (m12), 本 demo 没传。

## 常见误区

- **"resume 就是把上次的状态全恢复"** — 只恢复上下文。权限、审批记录、工具的外部状态都不恢复; 前者是刻意的安全决策, 后者是做不到。
- **"只存用户消息和最终回答就够了"** — 缺了 `tool_use` / `tool_result`, 模型 resume 后不知道自己做过什么、依据是什么, 审计也无从谈起; 而且 tool_result 必须紧跟同 id 的 tool_use, 缺任何一半都不是合法 API 输入。
- **"append-only 意味着 resume 出来的上下文只会越来越大"** — 文件只增不减, 但 `load()` 遇到 `compact_boundary` 会用摘要替换之前的历史, 恢复出的上下文真的变小 (m14)。审计用 `load_all()` 仍能看到全部。

## 自测题

1. `_next_id` 为什么从 `store.load_all()` 而不是 `store.load()` 计数?
<details><summary>答案</summary>

`load()` 在压缩后只返回"摘要 + 尾部", 旧的 `tool_use` 已不在视图里, 按它计数会从较小的数字重新发号, 与磁盘上旧记录的 id 撞号, 审计时无法区分。`load_all()` 是全量历史, 计数单调递增。

</details>

2. 会话 B 用 `PermissionGate("default")` 且去掉 `ask_policy`, 会发生什么? 这说明了什么?
<details><summary>答案</summary>

`default` 模式对未命中规则的调用要问人, 没有 `ask_policy` 就是无人可问, `_ask` fail closed 返回拒绝, `write_note` 得到 `DENIED`, `notes` 为空。说明授权完全来自新会话自己的门, transcript 里"上次允许过"的痕迹不起任何作用。

</details>

3. 为什么 assistant 的 `tool_use` 消息要在 `_run_tools` 之前就 `_append`, 而不是等结果出来再一起写?
<details><summary>答案</summary>

append-only 日志的价值在于崩溃时也完整。工具执行可能卡死、抛错或让进程退出; 先写意图, 日志里至少留下"模型要求了什么", 事后能定位到出事的那一步。代价是可能留下没有配对 `tool_result` 的悬空 `tool_use`, 真实系统要在 resume 时补一条错误结果或裁掉它。

</details>
