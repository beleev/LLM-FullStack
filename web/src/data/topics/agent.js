// 阶段 6 · llm_agent: 12 章的完整页面定义 (不再依赖 models.js 的 baseTopicPages)。
// 每页的 points 里恰好有一条 key: true —— 这一章真正的脊梁。
const A = 'llm_agent/'
const C = `${A}core/`

export default {
  stage: 'agent',
  chapters: [
    { route: 'agent-mcp', label: 'MCP · stdio JSON-RPC', hint: 'initialize → tools/list → tools/call, mcp__server__tool' },
    { route: 'agent-planning', label: '计划 · todo 与 plan 模式', hint: 'todo_write、只读 plan 模式、并行工具调用' },
    { route: 'agent-orchestrator', label: 'Orchestrator–workers', hint: '并行扇出、只回摘要、两本 token 账' },
    { route: 'agent-guardrails', label: '护栏 · 注入 / 围栏 / 脱敏', hint: 'untrusted_data、污点规则、confine、redact' },
    { route: 'agent-evals', label: 'Agent 评测', hint: '终态判分、轨迹检查、pass@k vs pass^k' },
    { route: 'agent-context-engineering', label: '上下文工程', hint: '清工具结果 → 摘要、即时检索、memory 工具' },
  ],
  pages: {
    'agent-loop': {
      title: 'Agent loop · 从"会说"到"会做"的最小闭环',
      subtitle: '读完你能说清: 一次工具调用在 loop 里要经过哪几站, 以及工具报错为什么不该把 loop 炸掉。',
      tldr: '拼上下文 → 问模型 → 过 hook → 过权限门 → 执行 → 结果回灌, 一直转到模型说"完事了"或者撞上 max_turns。',
      question: '工具报错了, 为什么不抛异常终止 loop, 而要包成一条 is_error 的 tool_result 再喂回去?',
      code: 'llm_agent/core/{agent.py,schema.py,llm.py} · m01_agent_loop · m15_claude_api',
      points: [
        {
          title: '模型只是一个 next()',
          body: 'loop 对模型的全部要求只有一个方法: next(messages, tool_schemas) → ModelAction。模型自己不存状态, 状态全在 messages 里。所以教学用的 RuleBasedLLM、真实的 ClaudeLLM、评测用的会随机偷懒的 FlakyLLM 可以互换, Agent.run 一行都不用改。',
        },
        {
          title: 'tool_use 和 tool_result 靠 id 配对',
          body: 'assistant 发出 {type: tool_use, id, name, input}; harness 执行完, 把 {type: tool_result, tool_use_id, content} 放进下一条 user 消息。结果挂在 user 角色下是 Messages API 的约定 —— 它是外界的观察, 回填给模型看。一个 turn 里发了 3 个 tool_use, 3 条结果必须放进同一条 user 消息, 少一条真实 API 直接 400。',
        },
        {
          key: true,
          title: '错误和拒绝都是观察, 不是异常',
          body: '未知工具、参数不合 schema、工具内部抛异常、权限门拒绝 —— 全都变成 is_error=true 的结果回填, loop 继续转。模型下一轮自己改参数、换做法。代价是每次问模型都要重发整个上下文, 累计 input token 随轮数近似平方增长。',
        },
      ],
      links: [
        { from: 'llm_infer.generate', to: 'Agent.run', body: '推理只负责生成 token; Agent 把生成出来的东西解释成对外界的动作。' },
        { from: 'ModelAction.tool_calls', to: 'Agent._run_tools', body: '授权串行走 (审批弹窗不能并发), 执行并行跑, 总耗时约等于最慢的那一个。' },
        { from: 'ToolResult.to_block', to: 'Message("user", [tool_result…])', body: '结果永远挂回模型发出的那个 id 上, 就算 hook 中途改写过调用也一样。' },
        { from: 'core/llm.py:LLM', to: 'core/claude_llm.py:ClaudeLLM', body: 'transcript 本来就是 Messages API 格式, 换真模型只换这一个对象 (m15, 需要自己开)。' },
      ],
      sourceRows: [
        { concept: '主循环', code: 'core/agent.py:Agent.run', takeaway: 'for turn in range(1, max_turns+1): 拿到 final 就返回, 否则执行工具、回填结果。循环走完 = "stopped: max_turns reached"。' },
        { concept: 'content block', code: 'core/schema.py:ToolCall.to_block / ToolResult.to_block', takeaway: '{type:tool_use,id,name,input} 与 {type:tool_result,tool_use_id,content,is_error}, 和真实 API 同构。' },
        { concept: '配对校验', code: 'core/schema.py:validate_transcript', takeaway: '每个 tool_use 都要在下一条 user 消息里找到同 id 的 tool_result, 否则这段历史发不出去。' },
        { concept: 'LLM 协议', code: 'core/llm.py:LLM', takeaway: 'tools 传的是带 JSON Schema 的完整定义, 不是一串工具名 —— 只给名字模型只能猜参数。' },
        { concept: 'token 记账', code: 'core/agent.py:Agent._ask_model', takeaway: 'input_tokens 每次都把整个上下文再加一遍: 长会话贵就贵在这里。' },
      ],
      snippetTitle: 'Agent loop 骨架 (content block 版)',
      snippet: `messages.append(Message("user", prompt))
for turn in range(max_turns):
    action = llm.next(assemble(system, memory, messages), tools.schemas())
    if action.kind == "final":
        messages.append(Message("assistant", action.content))
        return action.content

    calls = action.tool_calls              # 一轮可以有多个 tool_use
    messages.append(Message("assistant", [c.to_block() for c in calls]))
    results = run_tools(calls)             # 授权串行, 执行并行; 出错也是结果
    for call, r in zip(calls, results):
        r.tool_use_id = call.id            # 靠 id 配对
    messages.append(Message("user", [r.to_block() for r in results]))
return "stopped: max_turns reached"`,
      run: 'python -m llm_agent.m01_agent_loop.demo',
      source: [`${C}agent.py:Agent.run`, `${C}schema.py:validate_transcript`],
    },

    'agent-tools-permissions': {
      title: '工具与权限 · 想让它动手, 先想清楚怎么拦',
      subtitle: '读完你能照着 deny → ask → allow → 模式的顺序手推一次裁决, 也能说出字符串黑名单为什么注定漏。',
      tldr: '工具 = name + description + JSON Schema; 模型填的参数是不可信输入, 执行前先 validate_args。权限门按 deny → ask → allow → 模式兜底裁决, 评的是 hook 改写之后那个真正要执行的调用。',
      question: '`rm -fr /`、`RM  -r -f /` 能绕过 deny "*rm -rf*" 吗? 归一化之后还剩哪些绕法, 这说明黑名单的什么本质?',
      code: 'llm_agent/core/{tools.py,permissions.py} · m02_tool_use · m03_permissions',
      points: [
        {
          title: 'schema 两头都要用',
          body: 'schema() 输出 {name, description, input_schema} 给模型看; 同一份 schema 还要给 harness 在执行前校验 required / type / enum / additionalProperties。校验失败原样回填, 模型下一轮自己改。risk / read_only / untrusted_output 是给 harness 看的元数据, 不发给模型。',
        },
        {
          key: true,
          title: 'deny > ask > allow > 模式兜底',
          body: '顺序就是优先级: 广义的拒绝必须压过狭义的允许。deny 规则在任何模式下都生效, 连 bypass_permissions 也拦得住。六种模式只在规则都没命中时才兜底: plan 只读、default 问人、accept_edits 放行低中风险、auto 按风险分类、dont_ask 全放、bypass_permissions 连 ask 规则都跳过。没人可问时 ask 等于拒绝 (fail closed)。',
        },
        {
          title: '先归一化, 再逐段判',
          body: 'normalize_command 把大小写、空白和相邻短 flag 的顺序统一: `RM  -r -f /` 变成 `rm -fr /`。复合命令按 && || ; | & 拆开逐段评估, 否则 `echo hi && rm -rf /` 能蹭到 allow "echo *"。含 `$()` 或反引号的命令一律不享受 allow 规则 —— 命令替换里能藏任何东西。8 种写法 × 3 种模式打过一遍, 24 次全部拦住。',
        },
        {
          title: '但字符串黑名单天生很弱',
          body: '源码注释里那句话值得抄下来: 黑名单是在枚举"坏", 而坏是无穷的。归一化只堵住最廉价的绕过, 剩下的还有 /bin/rm、rm --recursive --force、find / -delete、python -c shutil.rmtree……真正的边界是默认拒绝的 allowlist + 把命令解析成 AST 逐段检查 + OS 级沙箱 (seatbelt / bubblewrap / 容器)。deny 规则只是最后一道很便宜的网。',
        },
      ],
      links: [
        { from: 'Tool.schema()', to: 'llm.next(messages, tools)', body: '模型看到的是参数的 JSON Schema; 只给它一串名字, 它只能猜参数。' },
        { from: 'validate_args', to: 'ToolResult(ok=False)', body: '信任边界上的输入验证: 坏参数变成 INVALID_ARGS 结果, 根本进不到工具里。' },
        { from: 'PreToolUse hook', to: 'PermissionGate.evaluate(final)', body: '门评估的是改写后的调用, 所以 hook 不是提权通道 (见 Hooks 一章)。' },
        { from: 'PermissionRule("mcp__weather__*")', to: 'MCPTool', body: '工具名 glob 让一条规则管住整个 MCP server。' },
      ],
      sourceRows: [
        { concept: '参数校验', code: 'core/tools.py:validate_args', takeaway: 'bool 是 int 的子类, 不单独排掉的话 True 会被当成合法 number 放过去。' },
        { concept: '执行不抛异常', code: 'core/tools.py:ToolRegistry.execute', takeaway: '未知工具、坏参数、工具内部异常 —— 一律变成 ok=False 的结果。' },
        { concept: '裁决顺序', code: 'core/permissions.py:PermissionGate._evaluate_one', takeaway: 'for decision in (DENY, ASK, ALLOW): 循环顺序就是优先级; plan 模式下处理完 deny 直接 break。' },
        { concept: '归一化', code: 'core/permissions.py:normalize_command', takeaway: '规则和命令走同一个函数, 写规则的人不必关心 -rf 还是 -fr。' },
        { concept: '复合命令', code: 'core/permissions.py:PermissionGate.evaluate', takeaway: '拆成段逐段评估, 第一段被拒就停 —— 不为后面的段白白打扰人。' },
      ],
      snippetTitle: '权限评估顺序',
      snippet: `def evaluate_one(call, tool):
    for decision in (DENY, ASK, ALLOW):          # 顺序就是优先级
        if decision != DENY and mode == "plan":
            break                                # plan: allow 规则也不能放行写操作
        if decision == ASK and mode == "bypass_permissions":
            continue
        for rule in rules:
            if rule.decision == decision and matches(rule, call):
                return ask_human(call) if decision == ASK else outcome(decision)
    return mode_fallback(call, tool.risk)        # plan / default / accept_edits / auto / …

def matches(rule, call):
    text = " ".join(map(str, call.args.values()))
    return fnmatch(call.name, rule.tool) and \\
           fnmatch(normalize_command(text), normalize_command(rule.pattern))`,
      run: 'python -m llm_agent.m03_permissions.demo',
      source: [`${C}permissions.py:normalize_command`, `${C}permissions.py:_evaluate_one`, `${C}tools.py:validate_args`],
    },

    'agent-context-memory': {
      title: '上下文与记忆 · 模型这一轮到底看见了什么',
      subtitle: '读完你能说清一条记忆从 .md 文件走到模型眼前的完整路径, 也能解释中文查询为什么会得 0 分。',
      tldr: 'FileMemory 就是一堆 Markdown 文件; 每轮按当前 prompt 现查现拼, 作为独立的 system 消息进上下文, 不写进 transcript。',
      question: '分词器只认 [a-z0-9]+ 时, 一句中文查询的检索得分是多少? 为什么关键词计数会让一篇凑满常见词的 FAQ 挤掉正确答案?',
      code: 'llm_agent/core/{memory.py,retrieval.py,utils.py} · m04_context_memory · m08_retrieval',
      points: [
        {
          title: '记忆是文件, 所以可读可改',
          body: '记忆落成普通 .md 文件 (CLAUDE.md 的迷你版): 你能读它、改它、把它提交进版本库。每轮现查现拼、不写进 transcript, 所以文件一改, 下一轮立刻生效 —— 不需要重启会话, 也不用求模型"记住"。',
        },
        {
          key: true,
          title: '检索质量先死在分词上',
          body: '中文句子里没有空格, 正则 [a-z0-9]+ 会把整句丢光, 查询向量是空的, 所有文档得分恒为 0 —— 模型拿到 no matches 只能瞎编。字符 bigram (Lucene CJKAnalyzer 同款) 不需要词典: 「显存碎片」切成 显存 / 存碎 / 碎片, 「存碎」是噪声, 但噪声几乎不会在别的文档出现, 影响很小。',
        },
        {
          title: 'idf 让罕见词说了算',
          body: '关键词计数会让一篇凑满 how / the / model 的 FAQ 排第一。idf = ln(1 + (N−df+0.5)/(df+0.5)) 修正它: 几乎每篇都有的词权重趋近 0 (the 只有 0.33), 罕见词主导排序 (fragmentation 是 1.79)。工具名和 schema 都不变, 还是 search_docs, 所以这次升级对 loop 和模型完全透明。',
        },
      ],
      links: [
        { from: 'FileMemory.search', to: 'memory_messages', body: '相关文件变成 name="memory" 的 system 消息, 不进 transcript。' },
        { from: 'utils.tokenize', to: 'TfidfIndex', body: '记忆检索和文档检索共用同一个分词器, 中文都靠 bigram。' },
        { from: 'SearchDocsTool', to: 'VectorSearchTool', body: '同名同 schema 热替换; 把 embed() 换成神经向量就是稠密检索。' },
        { from: 'clear_tool_results / summarize_with_llm', to: 'agent-context-engineering', body: '三档瘦身怎么接进 loop、怎么让 resume 出来的会话也变小, 见 m14。' },
      ],
      sourceRows: [
        { concept: '文件记忆', code: 'core/memory.py:FileMemory', takeaway: 'add 落成 Markdown, search 用 token 交集打分取前 3 条。' },
        { concept: '中文分词', code: 'core/utils.py:tokenize', takeaway: '英文按词切, 中文连续段切相邻两字的 bigram。' },
        { concept: 'TF-IDF 索引', code: 'core/retrieval.py:TfidfIndex', takeaway: '文档和查询都归一化, 稀疏点积就是余弦; 语料里没有的词直接丢掉。' },
        { concept: '热替换', code: 'core/retrieval.py:VectorSearchTool', takeaway: 'name 仍然是 search_docs: 升级检索不用动 agent loop。' },
        { concept: '三档瘦身', code: 'core/memory.py:clear_tool_results', takeaway: '最便宜的一档: 旧 tool_result 正文换成占位符, 配对结构原样保留。' },
      ],
      snippetTitle: '上下文组装 + 检索',
      snippet: `base = [Message("system", system_prompt)]
base += memory_messages(memory, prompt)        # 每轮现查, 不写进 transcript
budget = context_budget - total_chars(base)

view = messages
if total_chars(view) > budget:                 # 第 1 档: 清旧工具结果 (只改视图)
    view = clear_tool_results(view, keep_last)
if total_chars(view) > budget:                 # 第 2 档: 模型写摘要, 真的替换历史
    compact()
context = base + view

def tokenize(text):                            # 英文按词, 中文 bigram
    for run in re.findall(r"[a-z0-9]+|[一-鿿]+", text.lower()):
        yield from ([run] if run[0].isascii() else bigrams(run))`,
      run: 'python -m llm_agent.m08_retrieval.demo',
      source: [`${C}utils.py:tokenize`, `${C}retrieval.py:TfidfIndex`],
    },

    'agent-extensibility': {
      title: 'Hooks / Skills · 按上下文成本给扩展点分层',
      subtitle: '读完你能判断一条团队规范该写成 hook 还是 skill, 也能说清为什么 skill 正文能当指令、网页不能。',
      tldr: '能用确定性代码办的事写成 hook (零 token、100% 执行); 需要模型自己判断要不要用的写成 skill (常驻只有一行描述, 正文用到才加载)。',
      question: '如果顺序写成"权限门 → PreToolUse hook → 执行", 一个把 calculator 改写成 shell rm -rf / 的 hook 会让 deny 规则发生什么?',
      code: 'llm_agent/core/{hooks.py,skills.py} · llm_agent/m05_extensibility',
      points: [
        {
          title: 'Hook 必然执行, 但不能提权',
          body: 'hook 只能做三件事: 拦截、改写调用、追加上下文。七个事件 (SessionStart / UserPromptSubmit / PreToolUse / PostToolUse / PreCompact / Stop / SubagentStop) 跟 Claude Code 同名。顺序固定成 hook → 权限门 → 执行, 门评估的是改写之后那个调用。旧顺序下改写发生在检查之后, deny 规则形同虚设, 这是典型的 TOCTOU。',
        },
        {
          title: 'Hook 的输出走旁路',
          body: 'hook 追加的文字变成独立的 system 消息, 不拼进用户 prompt, 也不拼进 tool_result。否则审计标记会被模型当成工具数据写进笔记、拿去当检索词。被 UserPromptSubmit 拦下的 prompt 本身不进上下文 —— 它可能含密钥。',
        },
        {
          title: 'Skill 分三层加载',
          body: '① 启动时只读 SKILL.md 的 frontmatter, 拼成一行一条的目录; ② 模型觉得相关时调 skill 工具, 正文这时才进上下文; ③ 正文引用的附件用到才读。m05 demo: 目录 69 token, 全部正文 419 token, 而这个差价每一轮都在付。代价是模型只凭一行 description 决定要不要加载 —— 写得含糊, 这个 skill 就永远不会被触发。',
        },
        {
          key: true,
          title: 'skill 正文是指令, 抓回来的网页是数据',
          body: '两者都以 tool_result 的形式进上下文, 看起来一模一样, 但来源完全不同。SKILL.md 是你自己装进来的, 它的正文当指令执行天经地义。fetch 回来的网页谁都能写, 它只能当数据读。这条线一模糊, prompt injection 就不是意外而是必然。整个阶段最该带走的就是这一句。',
        },
      ],
      links: [
        { from: 'HookManager.on_pre_tool_use', to: 'PermissionGate.evaluate(final)', body: '链式改写后的最终调用才过门。' },
        { from: 'SkillRegistry.catalog()', to: 'system 消息', body: '常驻的只有目录; description 写得含糊, skill 就永远不会被触发。' },
        { from: 'SkillTool.execute', to: 'tool_result', body: '正文作为工具结果进上下文: 用户自己装的可信指令, 不同于 fetch 回来的网页。' },
        { from: 'on_pre_compact', to: 'summarize_with_llm(keep=…)', body: '人来点名"摘要里必须留下什么"。' },
      ],
      sourceRows: [
        { concept: '事件表', code: 'core/hooks.py:EVENTS', takeaway: '七个生命周期事件, 与 Claude Code hooks 同名。' },
        { concept: '链式改写', code: 'core/hooks.py:HookManager.on_pre_tool_use', takeaway: '后一个 hook 看到前一个的改写结果; 任何一个 block 就立刻返回。' },
        { concept: '改写后过门', code: 'core/agent.py:Agent._authorize', takeaway: 'permissions.evaluate(final, tool) —— 评估 final, 不是模型发出的那个 call。' },
        { concept: '只读 frontmatter', code: 'core/skills.py:SkillRegistry', takeaway: '启动时正文就被丢掉: 不进内存, 更不进上下文。' },
        { concept: '按需加载', code: 'core/skills.py:SkillTool', takeaway: 'name 参数的 enum 就是已装 skill 列表, 模型不可能加载一个不存在的 skill。' },
      ],
      snippetTitle: 'hook → 权限门 → 执行',
      snippet: `def authorize(call):
    pre = hooks.on_pre_tool_use(call)          # 可能拦截, 可能改写
    if pre.block:
        return None, f"BLOCKED BY HOOK: {pre.reason}"
    final = pre.updated_call or call

    outcome = permissions.evaluate(final)      # ★ 评估 final, 不是 call
    if not outcome.allowed:
        return None, f"DENIED: {outcome.reason}"
    return final, ""

# skills: 常驻的只有目录
system += "## Skills\\n" + "\\n".join(f"- {n}: {d}" for n, d in descriptions.items())
# 模型调用 skill(name) 时, 正文才作为 tool_result 进入上下文`,
      run: 'python -m llm_agent.m05_extensibility.demo',
      source: [`${C}agent.py:_authorize`, `${C}skills.py:SkillRegistry`],
    },

    'agent-state-subagents': {
      title: '持久化与子智能体 · 状态能恢复, 上下文要隔离',
      subtitle: '读完你能说出 resume 带回了什么、没带回什么, 也能解释子 agent 读的 20 篇文档为什么挤不进父级上下文。',
      tldr: '会话的全部状态就是 messages, 所以每产生一条就往 JSONL 尾部追加一行。resume 就是把这些行读回来, 但只读回消息。权限得由新会话自己重新建立。',
      question: '为什么恢复 transcript 不应该等于恢复上次的 bypass 权限?',
      code: 'llm_agent/core/{persistence.py,subagents.py} · m06_persistence_resume · m07_subagents',
      points: [
        {
          title: '只追加, 先记意图再执行',
          body: '写入逻辑只有 open("a") + 一行 JSON, 写下去的行永不改变, 审计链天然完整。assistant 的 tool_use 在工具跑之前就落盘: 就算执行中进程崩了, 日志里也留着"模型要求了这一步"。容错只有两条: load() 跳过写了一半的坏行; resume 时若最后一条是没有结果的悬空 tool_use, Agent 主动丢掉它 —— 留着这段 transcript 就是非法的。',
        },
        {
          key: true,
          title: 'resume 恢复的是上下文, 不是信任',
          body: 'PermissionGate 根本不在 JSONL 里。会话 A 跑在 auto 模式, 会话 B 退回 default, 写笔记要重新问一次人。上次的"同意"是对当时那个情境的同意, 不是永久授权。把权限写进可被恢复的状态, 等于让一个磁盘上的文件给自己提权。hooks、skills、工具在外面留下的改动, 同样不会跟着回来。',
        },
        {
          title: '压缩不删历史, 只加一条边界',
          body: '压缩时往文件尾追加一条 compact_boundary。load() (给 resume 用) 遇到它就把视图换成"摘要 + 保留的尾部", load_all() (给审计用) 跳过它返回全量。于是文件只增不减, 恢复出来的上下文却真的变小了。tool_use id 从 load_all() 计数续号, 压缩之后也不会和旧 id 撞号。',
        },
        {
          title: '子 agent 的细节留在子 agent 那里',
          body: '一次调研读 20 篇文档, 中间产物对最终结论几乎没用, 却会永久占住主上下文, 之后每一轮都要为它们重复付 token。delegate 给子任务开一个全新的 Agent: 自己的工具集、自己的 auto 权限门 (没人可问, 拿不准就拒)、自己的 transcript。父级只收到一条摘要, 而且长度上限在 harness 侧硬截断 (max_summary_chars=200), 不指望子级"自觉写短"。子 transcript 落盘可审计, 但从不回流。',
        },
      ],
      links: [
        { from: 'JsonlSessionStore.load', to: 'Agent(load_history=True)', body: '旧 transcript (压缩后的视图) 成为新会话的上下文。' },
        { from: 'PermissionGate(mode="default")', to: 'resume', body: '恢复状态和恢复权限是两回事。' },
        { from: 'DelegateTool.execute', to: 'agent-orchestrator', body: '同一轮发多个 delegate 就是并行扇出, 见 m11。' },
        { from: 'store.load_all()', to: 'Agent._next_id', body: 'id 从全量历史续号, 而不是从压缩后的视图 —— 否则会和磁盘上的旧记录撞号。' },
      ],
      sourceRows: [
        { concept: 'JSONL append', code: 'core/persistence.py:JsonlSessionStore.append', takeaway: '一行一条消息, 从不就地改写 —— 审计日志只增不减。' },
        { concept: '坏行容错', code: 'core/persistence.py:JsonlSessionStore._records', takeaway: 'json.loads 抛 JSONDecodeError 就跳过这一行: 坏一行不该让整个会话无法恢复。' },
        { concept: '压缩边界', code: 'core/persistence.py:JsonlSessionStore.load', takeaway: '遇到 compact_boundary 就把此前的视图换成 [摘要] + 保留的尾部。' },
        { concept: '悬空调用', code: 'core/agent.py:Agent.__init__', takeaway: '最后一条若是没配上结果的 tool_use, 直接 pop 掉, 否则 transcript 非法。' },
        { concept: '隔离委托', code: 'core/subagents.py:DelegateTool', takeaway: '每种 agent_type 有自己的工具集工厂与 auto 权限门; 子 transcript 落盘但不回流。' },
      ],
      snippetTitle: 'resume 与委托',
      snippet: `store = JsonlSessionStore(path)
agent1 = Agent(llm, tools, store=store)
agent1.run("搜索 agent loop")

# resume: 只恢复消息; 权限门是新会话自己的
agent2 = Agent(llm, tools, store=store, load_history=True,
               permissions=PermissionGate(mode="default"))

# 子智能体: agent_type → 全新的工具集; 父级只拿摘要
delegate = DelegateTool({"researcher": lambda: ToolRegistry([search])},
                        transcript_dir=tmp)
parent = Agent(llm, ToolRegistry([delegate]))`,
      run: 'python -m llm_agent.m07_subagents.demo',
      source: [`${C}persistence.py:JsonlSessionStore`, `${C}subagents.py:execute`],
    },

    'agent-full-loop': {
      title: 'mini Agent harness · 把所有零件接到同一个 loop 上',
      subtitle: '读完你能指着一个失败场景说出"这一层原本该挡住它", 也能看出哪些复杂度根本不在模型里。',
      tldr: 'loop 本身只有十几行, 而且从 m01 到 m14 基本没改过; 变的是它周围挂了几个确定性零件 —— 可靠性全在那些零件上。',
      question: '一个 Agent 产品的工程复杂度, 到底有多少在模型之外?',
      code: 'llm_agent/full_loop/demo.py · llm_agent/core/',
      points: [
        {
          key: true,
          title: '所有动作走同一条执行面',
          body: '内置工具、MCP 工具、子智能体都以同一个 Tool 接口接进 ToolRegistry。于是它们共用同一套参数校验、同一个权限门、同一批 hook、同一份审计日志。安全策略只需要在一处写对。任何绕开它的"特殊通道"迟早就是漏洞 —— 这也是为什么 MCP 工具没有后门, 它和 calculator 走的是同一条路。',
        },
        {
          title: '纵深防御: 每层都假设别的层会失守',
          body: 'demo 里的防线互不依赖。deny 规则挡住 rm -fr, 但它是字符串规则, 会被绕过。auto 分类器按风险和敏感路径再兜一次底。护栏把不可信输出标记出来、给本轮上下文打污点、把密钥抹掉。ShellTool 干脆只模拟不执行, 这是教学版的最后一道物理边界。各层检查的维度不同, 任何单层都有已知的绕过方式, 叠起来才可靠。',
        },
        {
          title: '五个场景, 逐个 assert',
          body: 'skill → 检索 → 写笔记; 委托子智能体; 调用真实 MCP 子进程; rm -fr 换了 flag 顺序照样被拒; 抓回来的文档夹带注入指令和密钥。跑完还要断言两件事: 落盘的 JSONL 里不含 sk-live 只有 [REDACTED], 整份日志的 tool_use / tool_result 配对完好。12 次模型调用, 7586 个累计 input token。',
        },
      ],
      links: [
        { from: 'llm_infer/full_engine', to: 'llm_agent/full_loop', body: '前者负责把 token 服务出去, 后者负责编排行动。' },
        { from: 'Agent.run', to: 'full_loop/demo.py', body: '同一个 loop, 在多种工具和扩展下一行不改。' },
        { from: 'full_loop', to: 'm09–m14', body: '外部工具、计划、多智能体、护栏、评测、上下文工程, 全都是 loop 周围的确定性系统。' },
        { from: 'store.load_all()', to: 'validate_transcript', body: '整份审计日志仍然是一段合法的 Messages API 序列。' },
      ],
      sourceRows: [
        { concept: '组合入口', code: 'full_loop/demo.py:main', takeaway: '所有 core 机制在一处组装: 工具、规则、hooks、记忆、存储、护栏。' },
        { concept: '安全规则', code: 'PermissionRule("shell", "*rm -rf*", DENY)', takeaway: '规则和命令走同一个归一化, 所以 rm -fr 也算命中。' },
        { concept: '护栏下传', code: 'DelegateTool(guardrails=Guardrails())', takeaway: '子级同样会读不可信数据、同样会落盘, 护栏要跟着下去。' },
        { concept: '密钥不落盘', code: 'assert "sk-live" not in raw', takeaway: '脱敏发生在进 transcript 之前, 所以 JSONL 里只有 [REDACTED]。' },
        { concept: '配对完好', code: 'validate_transcript(store.load_all())', takeaway: '跑了五个场景之后, 整份日志仍可原样喂给真实 API。' },
      ],
      snippetTitle: 'full_loop 组装',
      snippet: `tools = ToolRegistry([
    VectorSearchTool(index), WriteNoteTool(notes), SkillTool(skills),
    FetchDocTool(PAGES), ShellTool(), delegate, *mcp_tools(mcp),
])
permissions = PermissionGate(mode="auto", rules=[
    PermissionRule("shell", "*rm -rf*", Decision.DENY, "never allow destructive shell"),
    PermissionRule("mcp__weather__*", "", Decision.ALLOW, "trusted local weather server"),
])
agent = Agent(
    RuleBasedLLM(), tools, permissions,
    hooks=hooks, memory=memory, store=store, guardrails=Guardrails(),
    system_prompt="You are a small teaching agent.\\n" + skills.catalog(),
    context_budget_chars=6000, max_turns=6,
)`,
      run: 'python -m llm_agent.full_loop.demo',
      source: [`${A}full_loop/demo.py:main`],
    },

    'agent-mcp': {
      title: 'MCP · 用一个协议把外部工具接进来',
      subtitle: '读完你能读懂一次 MCP 调用的全部报文, 也能说出协议管什么、不管什么。',
      tldr: 'MCP 的 stdio transport 就是子进程 stdin/stdout 上逐行的 JSON-RPC 2.0: initialize → tools/list → tools/call, 工具名加前缀 mcp__<server>__<tool>。',
      question: '协议解决了"怎么接进来", 那"能不能信"由谁负责? 一个 MCP server 自报 readOnlyHint=true, harness 应该信吗?',
      code: 'llm_agent/core/mcp.py · llm_agent/m09_mcp/{server.py,demo.py}',
      points: [
        {
          title: '协议就是三个方法',
          body: 'initialize 对齐版本和能力; tools/list 返回 name + description + inputSchema; tools/call 执行。请求带递增的 id, 响应带同一个 id; 没有 id 的是通知, 不用回复。stdout 是协议通道, 日志只能写 stderr —— 往 stdout 打一行调试信息就会把报文流搞坏。',
        },
        {
          title: '两种错误不要混',
          body: '方法不存在 → JSON-RPC 的 error 字段 (-32601), 这是协议层的问题; 工具跑失败 → 正常的 result, 里面 isError=true。后者会变成 is_error 的 tool_result 回填给模型, 让它换个做法; 前者是 client 该报警的 bug。',
        },
        {
          key: true,
          title: '协议不解决信任, 所以没有后门',
          body: 'inputSchema 就是 JSON Schema, 本地先 validate_args, 坏参数根本发不到 server。MCPTool.risk 固定为 high、untrusted_output=True: 没有 allow mcp__weather__* 规则就得问人, 输出开护栏后会被包进 untrusted_data。server 自报的 readOnlyHint 一律不信 —— 恶意 server 当然会说自己只读。',
        },
      ],
      links: [
        { from: 'MCPClient.list_tools', to: 'ToolRegistry.register', body: '外部能力最终仍以 Tool 进入统一执行面。' },
        { from: 'mcp__weather__get_weather', to: 'tools/call name="get_weather"', body: '前缀用来防重名、方便按 server 写规则; 发给 server 时剥掉。' },
        { from: 'PermissionRule("mcp__weather__*", ALLOW)', to: 'MCPTool', body: '一条 glob 规则管住整个 server。' },
        { from: 'MCPTool.untrusted_output', to: 'agent-guardrails', body: '第三方输出可能夹带 prompt injection。' },
      ],
      sourceRows: [
        { concept: '握手', code: 'core/mcp.py:MCPClient.__init__', takeaway: 'Popen(argv) 不经过 shell; initialize 之后发 notifications/initialized。' },
        { concept: '请求-响应', code: 'core/mcp.py:MCPClient.request', takeaway: '加锁保证并行工具调用时请求和响应成对; 读线程 + 队列给 readline 加超时。' },
        { concept: '命名与风险', code: 'core/mcp.py:MCPTool', takeaway: 'name = mcp__{server}__{tool}; risk 固定 high, server 自报的注解不可信。' },
        { concept: 'server 主循环', code: 'm09_mcp/server.py:main', takeaway: '逐行读 JSON; 没有 id 的消息不回复; 每次写完必须 flush。' },
        { concept: '工具失败 ≠ 协议错误', code: 'm09_mcp/server.py:handle', takeaway: '工具异常 → isError=true 的 result; 未知方法 → error -32601。' },
      ],
      snippetTitle: 'MCP client 骨架',
      snippet: `proc = Popen(argv, stdin=PIPE, stdout=PIPE, text=True)   # 不经过 shell

def request(method, params):
    send({"jsonrpc": "2.0", "id": next_id(), "method": method, "params": params})
    resp = json.loads(readline(timeout=5))
    if "error" in resp:
        raise MCPError(resp["error"])                     # 协议层错误
    return resp["result"]

request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}})
send({"jsonrpc": "2.0", "method": "notifications/initialized"})   # 通知: 无 id, 无回复

for spec in request("tools/list", {})["tools"]:
    registry.register(MCPTool(name=f"mcp__{server}__{spec['name']}",
                              parameters=spec["inputSchema"], risk="high"))

result = request("tools/call", {"name": "get_weather", "arguments": {"city": "Beijing"}})
text, is_error = result["content"][0]["text"], result.get("isError")`,
      run: 'python -m llm_agent.m09_mcp.demo',
      source: [`${C}mcp.py:request`, `${C}mcp.py:MCPTool`, `${A}m09_mcp/server.py:handle`],
    },

    'agent-planning': {
      title: '计划 · todo、plan 模式与并行工具调用',
      subtitle: '读完你能说清"批准之前零写入"是怎么被强制的, 以及哪些调用值得放进同一个 turn。',
      tldr: '模型边想边做, 做到第三步常常忘了还剩什么。用户也只能事后发现它改了不该改的东西。todo_write 把计划变成显式状态, plan 模式把否决权还给人。',
      question: '如果 plan 模式只是 system prompt 里的一句"请先不要修改文件", 它和现在的实现差在哪?',
      code: 'llm_agent/core/tools.py (TodoWriteTool · ExitPlanModeTool) · llm_agent/m10_planning',
      points: [
        {
          title: '计划就是一份留在上下文里的状态',
          body: 'todo 列表一直在上下文里, 模型做到一半不会忘了还剩什么, 用户也能看到进度。每次整表覆写而不是增量 patch: 没有 id 要对齐、调用是幂等的, 模型每一轮都得重新面对完整计划。todo_write 只改 agent 自己的状态、不碰外部世界, 所以标成只读, plan 模式下也能用 —— 否则"先列个计划"这一步本身就会被拒。',
        },
        {
          key: true,
          title: 'plan 模式是一扇门, 不是一句提示',
          body: 'PermissionGate("plan") 对非只读工具一律 DENY, 连 allow 规则都不看 (处理完 deny 规则就 break), 否则一条早先配好的 allow 就能让 plan 模式形同虚设。demo [3]: 模型不交计划直接写, 门照样不开。翻转模式的动作发生在 ExitPlanModeTool.execute 里 —— 人点了同意它才改 gate.mode, 模型没有别的路径能自己改。',
        },
        {
          title: 'read_only 是自己声明的, 标错就是漏洞',
          body: 'harness 无法验证一个工具是不是真的只读, 全靠工具作者写对类属性。DelegateTool 显式写了 read_only = False 并在注释里说明原因: 子 agent 跑的是自己那扇 auto 模式的门, 里面的 write_note 是 medium 风险直接放行。委托要是被当成只读, plan 模式就能靠一层委托绕过去。',
        },
        {
          title: '并行只发生在"执行"这一步',
          body: '模型在一个 turn 里发 3 个 tool_use, harness 用线程池跑。3 个各 0.2s 的调用, 串行 0.63s, 并行 0.21s。并行还省掉两次模型往返, 以及那两次重发的整个上下文。授权仍然串行 —— 审批弹窗不能并发, 顺序也要确定。有依赖的调用 (先搜索再写笔记) 只能跨 turn。',
        },
      ],
      links: [
        { from: 'TodoWriteTool', to: 'tool_result "todos updated: 1/2 completed"', body: '进度回填给模型, 也能直接渲染给用户看。' },
        { from: 'ExitPlanModeTool.execute', to: 'gate.mode = "accept_edits"', body: '人批准之后才切模式; 拒绝就留在 plan。' },
        { from: 'ModelAction.tool_calls', to: 'ThreadPoolExecutor', body: '并行只在执行阶段; 结果按原顺序放回同一条 user 消息。' },
        { from: 'DelegateTool.read_only = False', to: 'agent-state-subagents', body: '委托不是只读: 子级有自己的门, 可能会写。' },
      ],
      sourceRows: [
        { concept: 'todo 状态', code: 'core/tools.py:TodoWriteTool', takeaway: 'status 只能是 pending / in_progress / completed; read_only=True 所以 plan 模式可用。' },
        { concept: '提交计划', code: 'core/tools.py:ExitPlanModeTool', takeaway: 'approve(plan) 为真才改 gate.mode, 否则返回 is_error 结果。' },
        { concept: 'plan 兜底', code: 'core/permissions.py:PermissionGate._evaluate_one', takeaway: 'mode == "plan": 只读放行, 其余 DENY "plan mode is read-only until the plan is approved"。' },
        { concept: '并行执行', code: 'core/agent.py:Agent._run_tools', takeaway: 'pool.map 保序; 总耗时约等于最慢的那个, 而不是求和。' },
      ],
      snippetTitle: 'plan 模式的一次完整往返',
      snippet: `gate = PermissionGate("plan")                       # 只读
tools = [search_docs, write_note, TodoWriteTool(),
         ExitPlanModeTool(gate, approve=ask_user)]

# turn 1  todo_write([{搜索, pending}, {写笔记, pending}])   -> 允许: 只改自身状态
# turn 2  exit_plan_mode(plan="1. 搜索 2. 写笔记")
#           用户拒绝 -> is_error 结果, 仍在 plan 模式, 一个字都没写
#           用户批准 -> gate.mode = "accept_edits"
# turn 3  search_docs(...)                                  -> 只读, 一直允许
# turn 4  write_note(...)                                   -> 现在才放行
# turn 5  todo_write([... completed, ... completed])`,
      run: 'python -m llm_agent.m10_planning.demo',
      source: [`${C}tools.py:ExitPlanModeTool`, `${C}tools.py:TodoWriteTool`],
    },

    'agent-orchestrator': {
      title: 'Orchestrator–workers · 并行扇出与两本 token 账',
      subtitle: '读完你能判断一个任务值不值得拆给多个 worker, 也能说清多智能体到底买到了什么。',
      tldr: '多智能体不省钱: m11 demo 里 lead 峰值上下文 247 (单 agent 375), 但总输入 770 (单 agent 413)。花更多 token 买到的是并行度和一个没被原文淹没的主上下文。',
      question: '什么样的任务值得用多智能体? 如果子任务之间强依赖、需要共享同一份上下文, 会发生什么?',
      code: 'llm_agent/core/subagents.py · llm_agent/m11_orchestrator',
      points: [
        {
          title: '扇出不需要新机制',
          body: 'lead 在一个 turn 里发多个 delegate tool_use, loop 原有的线程池自然就把它们并行跑了 —— 不用写任务队列, 也不用消息总线。好的抽象会复用: 并行工具调用 + 一个会新建子 agent 的普通工具 = orchestrator–workers。',
        },
        {
          key: true,
          title: '两本账要分开记',
          body: 'lead 峰值上下文 247 vs 单 agent 375 —— worker 读了多少原文都留在自己那里, lead 只收一条摘要。但总 token 是 770 vs 413: 每个 worker 都要从零重建上下文 (system + 任务简报), 文档短时这份固定开销占主导。文档长、轮数多时反过来: 单 agent 每轮重发全部已读文档是平方增长, 而每个 worker 只重发自己那一份。真实系统里 worker 还会各自多探索 (Anthropic 报告多智能体约为聊天的 15× token)。',
        },
        {
          title: '最小权限的 worker',
          body: 'researcher 只有检索工具, calculator 只会算。子级用 auto 权限门, 而且没有人可问 —— 高风险工具走到"问人"那一步就 fail closed 拒绝。SubagentStop hook 让父级能审计每个子级的收尾。隔离的是上下文, 不是副作用: 子级的工具照样作用于真实世界, 安全性取决于你给这个 agent_type 配了什么。',
        },
      ],
      links: [
        { from: 'Agent._run_tools', to: 'DelegateTool.execute × N', body: '同一轮的多个 delegate 由线程池并行执行 (max_parallel=4)。' },
        { from: 'child.run(task)', to: 'ToolResult("[researcher] …summary")', body: '父级上下文里只有这一行。' },
        { from: 'child.usage', to: 'token 记账', body: 'usage 分开记, 才能看清"总花费"和"父上下文占用"是两件事。' },
        { from: 'agent-state-subagents', to: 'agent-orchestrator', body: 'm07 讲隔离, m11 讲扇出和记账。' },
      ],
      sourceRows: [
        { concept: '子级工厂', code: 'core/subagents.py:DelegateTool', takeaway: 'agent_types 是 名字 → 返回全新 ToolRegistry 的工厂; 子级之间也不共享状态。' },
        { concept: '并发安全', code: 'core/subagents.py:DelegateTool.execute', takeaway: 'execute 会被线程池并发调用, children 列表用锁保护。' },
        { concept: '峰值上下文', code: 'core/agent.py:Agent._ask_model', takeaway: 'peak_context_tokens 与 input_tokens 是两本不同的账。' },
        { concept: '对照实验', code: 'm11_orchestrator/demo.py:main', takeaway: 'assert lead 峰值 < solo 峰值, 且 orchestrated 总输入 > solo 总输入。' },
      ],
      snippetTitle: '扇出就是"一轮里的多个 tool_use"',
      snippet: `delegate = DelegateTool({
    "researcher": lambda: ToolRegistry([VectorSearchTool(index)]),
    "calculator": lambda: ToolRegistry([CalculatorTool()]),
})
lead = Agent(llm, ToolRegistry([delegate]))

# lead 的 turn 1: 同时发出 3 个 tool_use
#   delegate(task="检索 kv cache", agent_type="researcher")
#   delegate(task="检索 lora",     agent_type="researcher")
#   delegate(task="计算 4096*32",  agent_type="calculator")
# loop: 授权串行 → 线程池并行执行 → 3 个 tool_result 放进同一条 user 消息
# lead 的 turn 2: 只看到 3 条摘要, 综合作答

total = lead.usage["input_tokens"] + sum(c["usage"]["input_tokens"] for c in delegate.children)`,
      run: 'python -m llm_agent.m11_orchestrator.demo',
      source: [`${C}subagents.py:execute`],
    },

    'agent-guardrails': {
      title: '护栏 · prompt injection、路径围栏与密钥脱敏',
      subtitle: '读完你能分清哪几层只是在"劝"模型、哪几层跟模型信不信没关系, 并保证系统里至少有一层是后者。',
      tldr: '工具取回来的东西是数据, 不是指令。标记和特征检测只降低模型上当的概率; 真正兜底的是三条不看模型脸色的规则: 污点、路径围栏、脱敏。',
      question: '注入检测的正则挡得住换个说法的攻击吗? 如果假设模型一定会上当, 你的系统还剩哪几道防线?',
      code: 'llm_agent/core/{guardrails.py,sandbox.py,agent.py} · llm_agent/m12_guardrails',
      points: [
        {
          title: '标记只降低概率',
          body: 'wrap_untrusted 把不可信输出包进 untrusted_data, 命中注入特征时再加一个 injection_suspected 标记。这是给模型一个"这是数据"的强提示。但正则只认它见过的说法, 攻击者换个措辞就绕过去了。模型也可能就是不听。这一层降低的是概率, 不是可能性。',
        },
        {
          key: true,
          title: '污点规则切断 lethal trifecta',
          body: '私有数据 + 不可信内容 + 对外通道, 三者同时成立才出事。本轮上下文一旦混入不可信数据, _tainted 置位, 高风险工具一律 DENIED。即使用户早先配过 allow 规则也不例外。_tainted 要等下一条真正的用户指令才重置。m12 demo 的结果很说明问题: 模型还是上当了, 但 shell.executed 是空的。',
        },
        {
          title: '先 resolve, 再检查',
          body: 'confine() 先把 (root / path) 展开 .. 和符号链接, 再判断是否还在 root 内。先拼接再比字符串前缀是经典漏洞: /work/../etc 也以 /work 开头, 却早就逃出去了。脱敏则发生在内容进 transcript 之前。带捕获组的规则保留 password= 这样的 key 名, 只抹掉值。于是日志仍然可读, 密钥也不会进下一次模型请求。',
        },
      ],
      links: [
        { from: 'Tool.untrusted_output', to: 'Guardrails.wrap_untrusted', body: 'fetch_doc、MCP 工具的输出都来自外部世界。' },
        { from: 'Agent._tainted', to: 'Agent._authorize', body: '污点检查排在权限门之前: allow 规则也救不了。' },
        { from: 'confine(root, path)', to: 'ReadFileTool / WriteFileTool / MemoryTool', body: '所有文件类工具共用同一个围栏。' },
        { from: 'Guardrails.redact', to: 'Agent._append', body: '进 transcript / JSONL 之前脱敏, 密钥不会进下一次模型请求。' },
      ],
      sourceRows: [
        { concept: '包裹与标记', code: 'core/guardrails.py:Guardrails.wrap_untrusted', takeaway: '文档自带闭合标签会被转义 —— 那是想提前"越狱"出数据区。' },
        { concept: '污点锁', code: 'core/agent.py:Agent._authorize', takeaway: 'guardrails and _tainted and risk == "high" → DENIED, 确定性, 与模型无关。' },
        { concept: '同批污染', code: 'core/agent.py:Agent._run_tools', takeaway: '同一轮里只要有别的调用会读不可信数据, 这个调用也按已污染处理 —— 并行执行保证不了先后。' },
        { concept: '路径围栏', code: 'core/sandbox.py:confine', takeaway: '"/" 开头按沙箱内的虚拟根解释; resolve 之后才判断。' },
        { concept: '脱敏', code: 'core/guardrails.py:Guardrails.redact', takeaway: '带捕获组的规则保留 password= 这类 key 名, 只抹掉值。' },
      ],
      snippetTitle: '三层防线',
      snippet: `# 1. 标记 (概率性): 结果进上下文之前
if tool.untrusted_output:
    result.output = f"<untrusted_data{flag}>\\n{redact(result.output)}\\n</untrusted_data>"
    tainted = True

# 2. 污点 (确定性): 下一次工具授权时
if tainted and tool.risk == "high":
    return DENIED("context is tainted by untrusted data")

# 3. 围栏 (确定性): 任何文件路径
def confine(root, user_path):
    target = (root / user_path.lstrip("/")).resolve()   # 先展开 .. 和 symlink
    if not target.is_relative_to(root.resolve()):
        raise PermissionError("path escapes sandbox")
    return target`,
      run: 'python -m llm_agent.m12_guardrails.demo',
      source: [`${C}sandbox.py:confine`, `${C}guardrails.py:Guardrails`],
    },

    'agent-evals': {
      title: 'Agent 评测 · 终态判分、轨迹检查与 pass^k',
      subtitle: '读完你能给自己的 agent 设计一个任务集, 并知道该盯 pass@k 还是 pass^k。',
      tldr: '"跑一下看着还行"不是评测。agent 有随机性, 单次成功什么也说明不了: 同一个 0.65 的单次通过率, pass@3 是 0.97, pass^3 只有 0.25。',
      question: '单次成功率 90% 的 agent, 连续 8 次都做对的概率是多少? 哪类产品应该盯 pass@k, 哪类必须盯 pass^k?',
      code: 'llm_agent/m13_evals/demo.py',
      points: [
        {
          title: '以环境终态判分',
          body: '任务 = prompt + 一个全新的环境 + 一个看终态的 grader: 笔记到底写了没、危险命令到底跑了没 (env.shell.executed == [])。模型说"我已经写好了"不算数, 反过来措辞不同也可能真做对了。每次试验都要一个全新环境, 试验之间不能互相污染。',
        },
        {
          title: '结果对, 过程也要对',
          body: '轨迹检查另外约束过程: 工具调用次数超预算、成功执行了禁用工具 (forbidden:shell) 都算失败。demo 里的回归就是这么被抓到的。有人为了少弹确认框把模式改成 dont_ask, 还删了 deny 规则。calc 和 safety 两个任务悄悄坏掉, 而最终回答看起来完全正常。',
        },
        {
          key: true,
          title: 'pass@k 和 pass^k 随 k 走向两端',
          body: '同一个单次通过率 0.65: pass@3 = 0.97 (至少成一次), pass^3 = 0.25 (三次全成)。pass@k 衡量能力上限, 适合有验证器、可以重试挑最优的场景 (写代码跑测试)。pass^k 衡量可靠性, 动作不可撤销又没人复核的 agent (退款、发邮件、改库) 必须盯它。代码里用的是无偏估计 1−C(n−c,k)/C(n,k) 与 C(c,k)/C(n,k), 比直接拿 p̂ 求幂更准。',
        },
      ],
      links: [
        { from: 'LLM 协议', to: 'FlakyLLM', body: '评测用的替身只要实现 next(): 以概率 p "懒得用工具"。' },
        { from: 'Agent.messages', to: '轨迹检查', body: 'tool_use / tool_result block 让"执行了什么"可以程序化断言。' },
        { from: 'baseline vs candidate', to: '回归列表', body: '逐任务列出变差的项: 改 prompt、换模式之前先跑一遍。' },
      ],
      sourceRows: [
        { concept: '任务定义', code: 'm13_evals/demo.py:Task', takeaway: 'grade(final, env) + max_tool_calls + forbidden 三样一起构成一个任务。' },
        { concept: '单次试验', code: 'm13_evals/demo.py:run_task', takeaway: '用 tool_use id → name 的映射找出"真正成功执行"了哪些工具。' },
        { concept: 'pass@k', code: 'm13_evals/demo.py:pass_at_k', takeaway: '1 − C(n−c, k) / C(n, k): HumanEval 同款无偏估计。' },
        { concept: 'pass^k', code: 'm13_evals/demo.py:pass_hat_k', takeaway: 'C(c, k) / C(n, k): τ-bench 的可靠性指标。' },
        { concept: '随机失误替身', code: 'm13_evals/demo.py:FlakyLLM', takeaway: '同一个 loop、同一套工具, 只换模型对象。' },
      ],
      snippetTitle: '评测骨架',
      snippet: `def run_task(task, make_env, llm):
    env = make_env(llm)                          # 每次试验一个全新环境
    final = env.agent.run(task.prompt)
    executed = successful_tool_names(env.agent.messages)
    violations = [n for n in executed if n in task.forbidden]
    if len(executed) > task.max_tool_calls:
        violations.append("too many tool calls")
    return task.grade(final, env) and not violations   # 终态 + 轨迹

c = sum(run_task(task, baseline, FlakyLLM(p=0.3, seed=s)) for s in range(n))
pass_at_k  = 1 - comb(n - c, k) / comb(n, k)     # k 次里至少成一次
pass_hat_k = comb(c, k) / comb(n, k)             # k 次全部成功`,
      run: 'python -m llm_agent.m13_evals.demo',
      source: [`${A}m13_evals/demo.py:run_task`, `${A}m13_evals/demo.py:pass_at_k`, `${A}m13_evals/demo.py:pass_hat_k`],
    },

    'agent-context-engineering': {
      title: '上下文工程 · 把"放什么进窗口"当成工程问题',
      subtitle: '读完你能按成本从低到高排出四种省上下文的办法, 并知道哪种丢掉的东西还能找回来。',
      tldr: '上下文是预算: 便宜的先清, 贵的再压, 能现取的不预存, 要跨会话的写文件。',
      question: '被"清掉"的工具结果和被"截断"丢掉的对话, 哪个还能找回来? 为什么?',
      code: 'llm_agent/core/{memory.py,agent.py,persistence.py,sandbox.py} · llm_agent/m14_context_engineering',
      points: [
        {
          key: true,
          title: '第 1 档: 工具结果最胖, 也最快过时',
          body: 'clear_tool_results 只保留最近 keep_last 个结果的正文, 其余换成 [cleared: N chars]。零模型开销, 配对结构一点不动, 而且 tool_use 块还在 —— 模型知道当时读的是哪个文件, 需要时再调一次工具就取回来了。这正是 just-in-time 检索的做法: 上下文里只留指针, 不留内容。',
        },
        {
          title: '第 2 档: 摘要是有损的, 要点名保留',
          body: '还超预算才让模型写摘要并真的替换历史, 代价是一次模型调用。_compact 只压"当前用户轮之前"的部分, 当前轮原样保留 —— 否则会切断 tool_use / tool_result 配对。行号、数值这类细节摘要最容易丢, PreCompact hook 让人指定"必须保留什么"。m14 demo: 5 轮检索触发 3 次压缩, 峰值上下文 246 token。',
        },
        {
          title: '压缩之后文件不会变小, 变小的是视图',
          body: 'JSONL 只是追加了一条 compact_boundary。审计用 load_all() 仍能读到全量 20 条 / 3049 字符, 而 resume 用 load() 读回来的只有 5 条 / 1026 字符。对照的反例是硬截断 truncate: 它只裁当轮视图, 每轮重新裁一遍, 历史和 resume 都不会变小, demo 里 5 轮有 3 轮答非所问。',
        },
        {
          title: '不进窗口的才是最省的',
          body: '即时检索: 预加载每次调用都要带 331 token, 改成上下文里只放索引、用到才读, 峰值降到 107。memory 工具更彻底: 模型自己往 /memories 写文件, 全新会话再读回来。跨会话的知识不占任何一轮的上下文, 任何压缩都碰不到它。',
        },
      ],
      links: [
        { from: 'Agent._assemble_context', to: 'clear_tool_results → _compact', body: '由便宜到贵的级联; truncate 只裁视图, 历史和 resume 都不会变小。' },
        { from: 'hooks.on_pre_compact', to: 'summarize_with_llm(keep)', body: '人来指定摘要里必须留下什么。' },
        { from: 'store.append_compact', to: 'JsonlSessionStore.load', body: '文件只增不减供审计, load() 读回的是压缩后的视图。' },
        { from: 'MemoryTool', to: 'confine(root, "/memories/…")', body: '记忆目录也在路径围栏里。' },
      ],
      sourceRows: [
        { concept: '级联', code: 'core/agent.py:Agent._assemble_context', takeaway: '第 1 档只改视图; 第 2 档才动 self.messages。' },
        { concept: '清工具结果', code: 'core/memory.py:clear_tool_results', takeaway: '返回新列表, 不改原消息; 占位符里留下原来的长度。' },
        { concept: '摘要压缩', code: 'core/agent.py:Agent._compact', takeaway: '从最后一条真正的用户输入处切开: old 换成摘要, tail 原样保留。' },
        { concept: '反例', code: 'core/memory.py:truncate_messages', takeaway: '头 2 + 尾 2 + 中间每条 32 字符: 语义和配对结构都会坏。' },
        { concept: 'memory 工具', code: 'core/sandbox.py:MemoryTool', takeaway: 'view / create / str_replace / delete, 路径必须以 /memories 开头。' },
      ],
      snippetTitle: '逐级降级',
      snippet: `def assemble_context(prompt):
    base = [system] + memory_messages(memory, prompt)
    budget = context_budget - total_chars(base)

    view = messages
    if total_chars(view) > budget:                       # 第 1 档: 零成本
        view = clear_tool_results(view, keep_last=1)
    if total_chars(view) > budget and compact():         # 第 2 档: 一次模型调用
        view = clear_tool_results(messages, keep_last=1)
    return base + view

def compact():
    start = last_user_prompt_index(messages)             # 当前轮原样保留
    old, tail = messages[:start], messages[start:]
    keep = hooks.on_pre_compact(old)                     # 人点名必须保留的要点
    summary = summarize_with_llm(llm, old, keep)
    messages[:] = [summary] + tail
    store.append_compact(summary, kept=len(tail))        # resume 也变小`,
      run: 'python -m llm_agent.m14_context_engineering.demo',
      source: [`${C}agent.py:_assemble_context`, `${C}agent.py:_compact`, `${C}memory.py:clear_tool_results`],
    },
  },
}
