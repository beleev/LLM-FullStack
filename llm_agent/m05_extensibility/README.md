# M05 — Skills 与 Hooks: 两种不改 loop 的扩展方式

Skill 是"模型按需加载的任务手册", hook 是"包在模型循环外、必然执行的确定性代码"; 本模块演示两者各自解决什么问题, 以及 hook 为什么绕不过权限门。

## 直觉

团队规范如果只能写进 system prompt, 会同时遇到两个问题: 每次请求都为用不到的手册付 token; 而且写了模型也不一定照做。
Skill 解决第一个: 常驻上下文的只有每个 skill 的一行 description, 正文等模型判断相关、调用 `skill` 工具时才进上下文 (渐进式披露)。
Hook 解决第二个: 它是 harness 里的普通 Python 函数, 不经过模型, 所以 100% 生效 —— 拦截、改写调用、追加上下文都行。
没有 skill, agent 要么上下文臃肿、要么每次现编流程; 没有 hook, "禁止读 token 文件"只能靠求模型。
但 hook 能改写调用也意味着它可能成为提权通道, 所以顺序必须是 hook 在前、权限门在后。

## 核心数据结构与控制流

- `core/skills.py: SkillRegistry` — 启动时 glob `*/SKILL.md`, 只留 frontmatter 的 `name` / `description`, 正文当场丢弃; `catalog()` 是唯一常驻 system prompt 的内容; `load(name)` 到调用时才重新读文件取正文。
- `core/skills.py: SkillTool` — 名为 `skill` 的普通工具, 参数 `name` 用 `enum` 限定为已安装的 skill; 返回值就是正文。
- `core/hooks.py: HookManager / HookResult` — 7 个事件: `session_start(source)` / `user_prompt_submit` / `pre_tool_use` / `post_tool_use` / `pre_compact` / `stop` / `subagent_stop`。`HookResult` 只有四个字段: `block`, `reason`, `updated_call`, `additional_context`。
- `core/agent.py: Agent._authorize / _run_tools` — 固定执行顺序。

```
model 发出 tool_use(call)  ──先写入 transcript──>
  1. hooks.on_pre_tool_use(call)      block → "BLOCKED BY HOOK"; 或链式改写出 final
  2. permissions.evaluate(final)      评估的是改写后的 final, 不是原始 call → "DENIED: ..."
  3. ToolRegistry.execute(final)      validate_args (schema 校验) → tool.execute
  4. hooks.on_post_tool_use(result)   返回的文字收集为 notes
tool_result 挂回模型发出的原 id → user 消息;  notes → 独立 system 消息 (name="post_tool_hook")
```

关键设计决策:

- **门评估改写后的调用**。hook 是用户代码, 也可能有 bug 或被人塞了恶意逻辑; 只授权原始调用等于给 hook 一条绕过 deny 规则的路。第 [5] 节就是这个旧 bug 的回归测试。
- **hook 文字走旁路**。`session_start` / `user_prompt_submit` / `post_tool_use` 追加的内容分别落成 name 为 `session_start` / `hook_context` / `post_tool_hook` 的 system 消息, 从不拼进用户 prompt 或 `tool_result` —— 否则 `[audit]` 之类的注释会被模型当成工具数据写进笔记、拿去当检索词。
- **skill 正文是可信指令**。它由用户自己安装, toy LLM 会服从 `skill` 工具返回的内容; 而 `fetch_doc` 抓回的文档是不可信数据 (m12)。区别在来源, 不在格式。
- **结果永远挂在模型发出的 id 上**, 即使 hook 把调用改成了别的工具 —— 保证 tool_use / tool_result 配对合法。

## 运行后应该看到什么

```bash
cd <仓库根目录> && python3 -m llm_agent.m05_extensibility.demo
```

```
[1] 渐进式披露: 常驻上下文的只有目录
  常驻 tokens (目录)          : 69
  全量塞入 tokens (正文)        : 419
[2] 同一个 prompt: 没有 skill → 直接作答; 有 skill → 先加载手册, 再按手册检索
  [no-skill] final: 这是一个无需工具的直接回答。
  [skilled] turn 1: model -> tool_use toolu_0001 skill {'name': 'debug'}
  [skilled] turn 2: model -> tool_use toolu_0002 search_docs {'query': '排查为什么 agent 没有结果'}
[3] PreToolUse hook 拦截: 即使权限模式全放行, 含 token 的 shell 也到不了执行层
  [hooked] turn 1: model -> tool_use toolu_0001 shell {'command': 'cat token.txt'}
  [hooked] tool_result toolu_0001 -> BLOCKED BY HOOK: secret-like shell command
[5] 回归: hook 把 calculator 改写成 rm -rf, 权限门必须拦住 (旧版只授权原始调用 → 直接执行)
  [rewritten] turn 1: model -> tool_use toolu_0001 calculator {'expr': '2 + 2'}
  [rewritten] pre_tool_use hook rewrote -> shell {'command': 'rm -rf /'}
  [rewritten] permission shell -> deny (rule: destructive)
  [rewritten] tool_result toolu_0001 -> DENIED: destructive
```

断言验证的内容:

- [1] 目录 token 数的 3 倍仍小于全部正文 (69 vs 419)。
- [2] 同一 prompt: 无 skill 的 agent 不调任何工具; 有 skill 的 agent 调用序列恰为 `skill → search_docs`; 检索词等于原始 prompt (skill 正文没漏进去); 未触发的 `release-notes` 正文从未出现在 transcript 里。
- [3] 权限模式是 `dont_ask` (全放行) 时, hook 仍拦住了含 token 的命令, `shell.executed == []`。
- [4] 所有 `tool_result` 里都没有 `[audit]`; 它出现在 `post_tool_hook` system 消息里; 同一 agent 跑两轮, `session_start` 消息只有 1 条; `stop` 触发 2 次且拿到最终文本。
- [5] hook 把 `calculator` 改写成 `shell rm -rf /`, 结果是 `DENIED: destructive`, `shell2.executed == []`。

## 与真实系统的差距

- Claude Code 的 hook 是 settings 里配置的外部命令 (stdin 收 JSON, 用退出码 / JSON 输出表达决定), 有 matcher、超时; 这里是进程内 Python 回调, 没有 matcher, 也没有异常隔离 (hook 抛错会直接炸掉 loop; 返回 None 视为"无意见")。
- 这里的 `stop` 只是通知; Claude Code 的 Stop hook 可以阻止结束、让模型继续干活。`subagent_stop` 同理。
- `post_tool_use` 只对真正执行过的调用触发; 被 hook 拦截或被权限拒绝的调用没有"执行后"。
- 渐进披露只实现了两层 (目录 → 正文); 正文引用的脚本 / 附件按需读取的第三层未实现。frontmatter 解析只支持单行 `key: value`。
- skill 正文在这里只是一条普通 `tool_result`: 长会话超预算时会被 `clear_tool_results` 当旧结果清成占位符, 没有"已加载的 skill 要保留"的特殊处理。
- 触发是 toy LLM 对 description 里"触发词:"做子串匹配; 真实模型靠语义判断 description 是否相关, 所以真实系统里 description 的写法直接决定 skill 会不会被用上。
- 改写成功时, transcript 里的 `tool_use` 仍是模型发出的原始调用 (结果挂在它的 id 上); 实际执行的改写后调用记在紧随其后的 `post_tool_hook` system 消息里 (`pre_tool_use rewrote ...`), 审计时要两条一起看。
- MCP (外部工具) 已移到 m09。

## 常见误区

- **"hook 和 skill 差不多, 都是给 agent 加规则"** — skill 是给模型看的文字, 模型可以不听; hook 是不经过模型的代码, 必然执行。要"保证发生"用 hook, 要"教它怎么做"用 skill。
- **"hook 是我自己写的, 所以它改写后的调用可以直接执行"** — 权限门是最后一道独立防线, 不应信任上游任何一环。本模块的顺序保证了 deny 规则对改写结果同样生效。
- **"skill 越多上下文越贵"** — 常驻成本只随 description 行数增长; 正文成本只在被加载的那次会话里出现。真正要控制的是 description 的长度和区分度。

## 自测题

1. 如果把 `_authorize` 里的 `self.permissions.evaluate(final, tool)` 改成 `evaluate(call, tool)`, 第 [5] 节会发生什么?
<details><summary>答案</summary>

门看到的是无害的 `calculator`, 在 `dont_ask` 模式下放行; 随后执行的却是 hook 改写出的 `shell rm -rf /`, deny 规则被绕过, `shell2.executed` 不再为空, 断言失败。这正是旧版的权限绕过 bug。

</details>

2. 为什么 `post_tool_use` 返回的 `[audit] ...` 不直接拼到 `result.output` 后面?
<details><summary>答案</summary>

`tool_result` 是模型眼里的"工具数据", 会被后续步骤原样引用 (写笔记、当检索词、进摘要)。拼进去等于让 harness 的注释冒充工具输出。放进独立的 `post_tool_hook` system 消息, 模型仍能看到, 但数据保持干净; 断言 [4] 验证的就是这一点。

</details>

3. `SkillRegistry.__init__` 明明 `read_text` 读了整个 SKILL.md, 为什么还说"正文没有加载"?
<details><summary>答案</summary>

"加载"指进入模型上下文。初始化时 `_split` 的正文返回值被丢弃, 只保留 name / description 和文件路径; 进 system prompt 的只有 `catalog()`。正文只有在模型调用 `skill` 工具、`load()` 重新读文件时才作为 `tool_result` 进上下文, 此后每轮请求才为它付 token。

</details>
