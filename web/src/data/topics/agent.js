// 阶段 6 · llm_agent 章节扩展: m09–m14 各成一章, 并按字段覆盖 models.js 里已有的 6 章
// (transcript 改成 content block、权限门顺序修正、skills 渐进披露、真 MCP 等, 旧文字已经和代码漂移)。
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
    // ---------------- 已有章节: 按字段覆盖 ---------------- //
    'agent-loop': {
      tldr: 'core/agent.py 的 loop 很薄: 组装上下文 → llm.next() → (hook → 权限门 → 参数校验 → 执行) → tool_result 回填 → 再问。transcript 用 Claude Messages API 同款 content block: assistant 发 tool_use{id,name,input}, 结果以 tool_result{tool_use_id,content,is_error} 放进下一条 user 消息。',
      question: '为什么工具报错不应该抛异常终止 loop, 而要变成一条 is_error 的 tool_result?',
      code: 'llm_agent/core/{agent.py,schema.py,llm.py} · m01_agent_loop · m15_claude_api',
      points: [
        { title: '模型只是一个 next()', body: 'LLM 协议只有一个方法: next(messages, tool_schemas) → ModelAction。无状态, 状态全在 messages 里, 所以 RuleBasedLLM、ClaudeLLM、评测用的 FlakyLLM 可以互换, loop 一行不改。' },
        { title: 'tool_use / tool_result 靠 id 配对', body: '一个 assistant turn 可以带多个 tool_use (并行调用); 它们的结果必须放进同一条 user 消息。先记下"模型要求了什么"再执行, 崩溃时审计日志也完整。' },
        { title: '错误是观察, 不是异常', body: '未知工具、参数不合 schema、工具内部异常, 全部变成 is_error=true 的结果回填 —— 模型下一轮自己改。每次调用都重发整个上下文, 所以累计 input token 随轮数近似平方增长。' },
      ],
      links: [
        { from: 'llm_infer.generate', to: 'Agent.run', body: '推理只生成 token; Agent 把生成结果解释成外部动作。' },
        { from: 'ModelAction.tool_calls', to: 'Agent._run_tools', body: '授权串行 (审批不能并发弹窗), 执行并行 (线程池), 耗时 ≈ 最慢的一个。' },
        { from: 'ToolResult.to_block', to: 'Message("user", [tool_result…])', body: '结果永远挂在模型发出的那个 id 上, 即使 hook 改写过调用。' },
        { from: 'core/llm.py:LLM', to: 'core/claude_llm.py:ClaudeLLM', body: 'transcript 本来就是 Messages API 格式, 换真模型只换这一个对象 (m15, opt-in)。' },
      ],
      sourceRows: [
        { concept: '主循环', code: 'core/agent.py:Agent.run', takeaway: 'for turn in range(1, max_turns+1): final 就返回, 否则执行工具并回填; 走完循环 = "stopped: max_turns reached"。' },
        { concept: 'content block', code: 'core/schema.py:ToolCall.to_block / ToolResult.to_block', takeaway: '{type:tool_use,id,name,input} 与 {type:tool_result,tool_use_id,content,is_error}。' },
        { concept: '配对校验', code: 'core/schema.py:validate_transcript', takeaway: '每个 tool_use 必须在下一条 user 消息里配齐 tool_result, 否则真实 API 直接 400。' },
        { concept: 'LLM 协议', code: 'core/llm.py:LLM', takeaway: 'loop 对模型的全部要求就是 next(); tools 传的是带 JSON Schema 的定义, 不是名字列表。' },
        { concept: 'token 记账', code: 'core/agent.py:Agent._ask_model', takeaway: 'input_tokens 每次累加整个上下文: 长会话贵在这里。' },
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
      source: [`${C}agent.py:Agent.run`, `${C}schema.py:validate_transcript`],
    },
    'agent-tools-permissions': {
      tldr: '工具 = name + description + JSON Schema; 模型输出是不可信输入, 执行前先 validate_args。权限门的顺序是 deny → ask → allow → 模式兜底, 评估的是 hook 改写之后"最终要执行的那个调用"; shell 命令先归一化, 复合命令逐段评估。',
      question: '`rm -fr /`、`RM  -r -f /` 能绕过 deny "*rm -rf*" 吗? 归一化之后还剩哪些绕法, 这说明黑名单的什么本质?',
      code: 'llm_agent/core/{tools.py,permissions.py} · m02_tool_use · m03_permissions',
      points: [
        { title: 'schema 是给模型看的, 也是给 harness 校验的', body: 'schema() 输出 {name, description, input_schema}; validate_args 校验 required / type / enum / additionalProperties, 错误原样回填, 模型下一轮自己改参数。risk / read_only / untrusted_output 是给 harness 的元数据, 不发给模型。' },
        { title: 'deny > ask > allow > 模式', body: '六种模式: plan 只读; default 未命中就问人; accept_edits 放行低/中风险; auto 按风险分类; dont_ask 全放行; bypass_permissions 连 ask 规则也跳过。deny 规则在任何模式下都生效; 没人可问时 ask = 拒绝 (fail closed)。' },
        { title: '字符串黑名单天生很弱', body: 'normalize_command 统一大小写、空白和短 flag 顺序, 只堵住最廉价的绕过; rm --recursive、find / -delete 这类有无穷多。真正的边界是默认拒绝的 allowlist + 命令解析 + OS 沙箱, deny 规则只是最后一道便宜的网。' },
      ],
      links: [
        { from: 'Tool.schema()', to: 'llm.next(messages, tools)', body: '模型看到参数的 JSON Schema; 只给名字它只能猜参数。' },
        { from: 'validate_args', to: 'ToolResult(ok=False)', body: '信任边界上的输入验证: 坏参数变成 INVALID_ARGS 结果, 不会进到工具里。' },
        { from: 'PreToolUse hook', to: 'PermissionGate.evaluate(final)', body: '门评估改写后的调用: hook 不是提权通道 (见 Hooks 一章)。' },
        { from: 'PermissionRule("mcp__weather__*")', to: 'MCPTool', body: '工具名 glob 让一条规则管住整个 MCP server。' },
      ],
      sourceRows: [
        { concept: '参数校验', code: 'core/tools.py:validate_args', takeaway: 'bool 是 int 的子类, 不单独排除的话 True 会被当成合法 number。' },
        { concept: '执行不抛异常', code: 'core/tools.py:ToolRegistry.execute', takeaway: '未知工具 / 坏参数 / 工具异常 → 都是 ok=False 的结果。' },
        { concept: '裁决顺序', code: 'core/permissions.py:PermissionGate._evaluate_one', takeaway: 'for decision in (DENY, ASK, ALLOW): 顺序就是优先级; plan 模式下 ask/allow 规则不生效。' },
        { concept: '归一化', code: 'core/permissions.py:normalize_command', takeaway: '`RM  -r -f /` → `rm -fr /`; 规则和命令走同一个函数, 写规则的人不必关心 -rf / -fr。' },
        { concept: '复合命令', code: 'core/permissions.py:PermissionGate.evaluate', takeaway: '按 && || ; | 拆开逐段评估, 否则 `echo hi && rm -rf /` 能蹭到 allow "echo *"。' },
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
      source: [`${C}permissions.py:normalize_command`, `${C}permissions.py:_evaluate_one`, `${C}tools.py:validate_args`],
    },
    'agent-context-memory': {
      tldr: 'FileMemory 用 Markdown 文件做透明记忆, 每轮按当前 prompt 检索相关片段, 作为独立的 system 消息拼进上下文。检索升级见 m08: TF-IDF 余弦 + BM25 式 idf, 中文按字符 bigram 分词。上下文瘦身有三档 (清工具结果 / 模型摘要 / 硬截断), 细节见「上下文工程」一章。',
      question: '分词器只认 [a-z0-9]+ 时, 一句中文查询的检索得分是多少? 为什么关键词计数会让一篇凑满常见词的 FAQ 挤掉正确答案?',
      code: 'llm_agent/core/{memory.py,retrieval.py,utils.py} · m04_context_memory · m08_retrieval',
      points: [
        { title: '文件记忆透明', body: '记忆是普通 .md 文件 (CLAUDE.md 的迷你版): 可读、可改、可进版本库。每轮现查现拼、不写进 transcript, 文件改了下一轮立刻生效。' },
        { title: '检索质量先死在分词上', body: '中文没有空格, [a-z0-9]+ 会把整句丢光, 得分恒为 0。字符 bigram (Lucene CJKAnalyzer 同款) 不需要词典: 「显存碎片」→ 显存 / 存碎 / 碎片。' },
        { title: 'idf 让罕见词主导', body: 'idf = ln(1 + (N−df+0.5)/(df+0.5)): 几乎每篇都有的词权重趋近 0。工具名和 schema 不变 (仍是 search_docs), 检索升级对 loop 和模型完全透明。' },
      ],
      links: [
        { from: 'FileMemory.search', to: 'memory_messages', body: '相关文件变成 name="memory" 的 system 消息, 不进 transcript。' },
        { from: 'utils.tokenize', to: 'TfidfIndex', body: '记忆检索和文档检索共用同一个分词器, 中文都靠 bigram。' },
        { from: 'SearchDocsTool', to: 'VectorSearchTool', body: '同名同 schema 热替换; 把 embed() 换成神经向量就是稠密检索。' },
        { from: 'clear_tool_results / summarize_with_llm', to: 'agent-context-engineering', body: '三档瘦身如何接进 loop、如何缩小 resume 出来的会话, 见 m14。' },
      ],
      sourceRows: [
        { concept: '文件记忆', code: 'core/memory.py:FileMemory', takeaway: 'add 落成 Markdown, search 用 token 交集打分取 top-3。' },
        { concept: '中文分词', code: 'core/utils.py:tokenize', takeaway: '英文按词, 中文连续段切相邻两字 bigram。' },
        { concept: 'TF-IDF 索引', code: 'core/retrieval.py:TfidfIndex', takeaway: '文档与查询都归一化, 稀疏点积 = 余弦; 语料外的词直接丢。' },
        { concept: '热替换', code: 'core/retrieval.py:VectorSearchTool', takeaway: 'name 仍是 search_docs: 升级检索不用动 agent loop。' },
        { concept: '三档瘦身', code: 'core/memory.py:clear_tool_results', takeaway: '最便宜的一档: 旧 tool_result 正文换占位符, 配对结构原样保留。' },
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
      title: 'Hooks / Skills · 扩展点按上下文成本分层',
      tldr: 'Hooks 是包在模型循环外面的确定性代码 (SessionStart / UserPromptSubmit / PreToolUse / PostToolUse / PreCompact / Stop / SubagentStop), 零 token、100% 生效, 但 PreToolUse 改写后的调用仍要过权限门。Skills 是渐进式披露的任务手册: 常驻上下文的只有 name + description, 正文在模型调用 skill 工具时才加载。外部工具 (MCP) 见下一章。',
      question: '旧版 loop 的顺序是"权限门 → PreToolUse hook → 执行"。一个把 calculator 改写成 shell rm -rf / 的 hook, 会让 deny 规则发生什么?',
      code: 'llm_agent/core/{hooks.py,skills.py} · llm_agent/m05_extensibility',
      points: [
        { title: 'Hook: 必然执行, 但不是提权通道', body: 'hook 只能做三件事: 拦截、改写调用、追加上下文。顺序固定为 hook → 权限门 → 执行: 门评估的是改写后的最终调用。旧顺序下改写发生在检查之后, deny 规则形同虚设 (TOCTOU)。' },
        { title: 'Hook 的输出走旁路', body: 'hook 追加的文字是独立的 system 消息, 不拼进用户 prompt 或 tool_result —— 否则审计标记会被模型写进笔记、拿去当检索词。被拦的 prompt 本身不入上下文 (可能含密钥)。' },
        { title: 'Skill: 三层加载', body: '① 启动只读 SKILL.md 的 frontmatter, 拼成一行一条的目录; ② 模型判断相关时调 skill 工具, 正文才进上下文; ③ 正文引用的附件用到才读。m05 demo: 目录 69 token vs 全量 419 token, 且差价每一轮都在付。' },
      ],
      links: [
        { from: 'HookManager.on_pre_tool_use', to: 'PermissionGate.evaluate(final)', body: '链式改写后的最终调用才过门。' },
        { from: 'SkillRegistry.catalog()', to: 'system 消息', body: '常驻的只有目录; description 写得含糊, skill 就永远不会被触发。' },
        { from: 'SkillTool.execute', to: 'tool_result', body: '正文作为工具结果进入上下文: 用户自己安装的可信指令, 不同于 fetch 回来的网页。' },
        { from: 'on_pre_compact', to: 'summarize_with_llm(keep=…)', body: '人指定"摘要里必须留下什么"。' },
      ],
      sourceRows: [
        { concept: '事件表', code: 'core/hooks.py:EVENTS', takeaway: '七个生命周期事件, 与 Claude Code hooks 同名。' },
        { concept: '链式改写', code: 'core/hooks.py:HookManager.on_pre_tool_use', takeaway: '后一个 hook 看到前一个的改写结果; 任一个 block 就立即返回。' },
        { concept: '改写后过门', code: 'core/agent.py:Agent._authorize', takeaway: 'permissions.evaluate(final, tool) —— 评估 final 而不是 call。' },
        { concept: '只读 frontmatter', code: 'core/skills.py:SkillRegistry', takeaway: '启动时正文被丢弃: 不进内存, 更不进上下文。' },
        { concept: '按需加载', code: 'core/skills.py:SkillTool', takeaway: 'name 参数的 enum 就是已安装 skill 列表, 模型不可能加载不存在的 skill。' },
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
      source: [`${C}agent.py:_authorize`, `${C}skills.py:SkillRegistry`],
    },
    'agent-state-subagents': {
      tldr: 'JsonlSessionStore 仅追加: 每条 Message (含 tool_use / tool_result block) 一行; 压缩时追加一条 compact_boundary, load() 读回压缩后的视图, load_all() 读回全量审计日志。resume 只恢复消息, 权限由新会话重新建立。DelegateTool 为每个子任务新建隔离的 child agent, 父级只收到摘要。',
      sourceRows: [
        { concept: 'JSONL append', code: 'core/persistence.py:JsonlSessionStore.append', takeaway: '一行一条消息, 从不就地改写 —— 审计日志只增不减。' },
        { concept: '压缩边界', code: 'core/persistence.py:JsonlSessionStore.load', takeaway: '遇到 compact_boundary 就把此前的视图换成 [摘要] + 保留的尾部。' },
        { concept: '隔离委托', code: 'core/subagents.py:DelegateTool', takeaway: '每种 agent_type 有自己的工具集与 auto 权限门; 子 transcript 落盘但不回流。' },
        { concept: 'id 续号', code: 'core/agent.py:Agent.__init__', takeaway: 'tool_use id 从全量历史续号, 压缩后也不会和旧 id 撞号。' },
      ],
      links: [
        { from: 'JsonlSessionStore.load', to: 'Agent(load_history=True)', body: '旧 transcript (压缩后的视图) 作为新会话上下文。' },
        { from: 'PermissionGate(mode="default")', to: 'resume', body: '恢复状态和恢复权限是两回事。' },
        { from: 'DelegateTool.execute', to: 'agent-orchestrator', body: '同一轮发多个 delegate 就是并行扇出, 见 m11。' },
      ],
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
      source: [`${C}persistence.py:JsonlSessionStore`, `${C}subagents.py:execute`],
    },
    'agent-full-loop': {
      tldr: 'full_loop/demo.py 把工具、deny-first 权限、hooks、文件记忆、JSONL transcript 和子智能体接进同一个 Agent。m09–m14 的 MCP、计划、编排、护栏、评测、上下文工程都是在这同一个 loop 上加零件, loop 本身不变。',
      links: [
        { from: 'llm_infer/full_engine', to: 'llm_agent/full_loop', body: '前者服务 token, 后者编排行动。' },
        { from: 'Agent.run', to: 'full_loop/demo.py', body: '同一 loop 在多种工具和扩展下保持不变。' },
        { from: 'full_loop', to: 'm09–m14', body: '外部工具、计划、多智能体、护栏、评测、上下文工程: 都是 loop 周围的确定性系统。' },
      ],
    },

    // ---------------- 新章节 ---------------- //
    'agent-mcp': {
      title: 'MCP · 用一个协议把外部工具接进来',
      subtitle: '不再为每个外部系统手写 Tool 子类: 工具可以由别的进程、别的语言、别的团队提供。',
      tldr: 'MCP 的 stdio transport 就是子进程 stdin/stdout 上逐行的 JSON-RPC 2.0。握手: initialize → notifications/initialized → tools/list; 调用: tools/call。工具名加前缀 mcp__<server>__<tool>; 第三方工具一律按 high 风险、输出按不可信处理, 和内置工具走同一个校验与权限门。',
      question: '协议解决了"怎么接进来", 那"能不能信"由谁负责? 一个 MCP server 自报 readOnlyHint=true, harness 应该信吗?',
      code: 'llm_agent/core/mcp.py · llm_agent/m09_mcp/{server.py,demo.py}',
      points: [
        { title: '协议就是三个方法', body: 'initialize 对齐版本与能力; tools/list 返回 name + description + inputSchema; tools/call 执行。请求带递增 id, 响应带同一个 id; 没有 id 的是通知, 不回复。stdout 是协议通道, 日志只能写 stderr。' },
        { title: '两种错误不要混', body: '方法不存在 → JSON-RPC error (-32601), 协议层问题; 工具执行失败 → 正常 result 里 isError=true, 会变成 is_error 的 tool_result 回填给模型, 让它换个做法。' },
        { title: '没有后门', body: 'inputSchema 就是 JSON Schema, 本地先 validate_args, 坏参数到不了 server。MCPTool.risk = high、untrusted_output = True: 没有 allow mcp__weather__* 规则就得问人; 输出开护栏后会被包进 <untrusted_data>。' },
      ],
      links: [
        { from: 'MCPClient.list_tools', to: 'ToolRegistry.register', body: '外部能力最终仍以 Tool 进入统一执行面。' },
        { from: 'mcp__weather__get_weather', to: 'tools/call name="get_weather"', body: '前缀防重名、方便按 server 写规则; 发给 server 时剥掉。' },
        { from: 'PermissionRule("mcp__weather__*", ALLOW)', to: 'MCPTool', body: '一条 glob 规则管住整个 server。' },
        { from: 'MCPTool.untrusted_output', to: 'agent-guardrails', body: '第三方输出可能夹带 prompt injection。' },
      ],
      sourceRows: [
        { concept: '握手', code: 'core/mcp.py:MCPClient.__init__', takeaway: 'Popen(argv) 不经过 shell; initialize 之后发 notifications/initialized。' },
        { concept: '请求-响应', code: 'core/mcp.py:MCPClient.request', takeaway: '加锁保证并行工具调用时请求-响应成对; 读线程 + 队列给 readline 加超时。' },
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
      subtitle: '多步任务里, 让模型先把计划写成显式状态; 让用户在它动手之前有机会说"不"。',
      tldr: 'todo_write 让模型把计划维护成显式状态 (每次整表覆写)。plan 模式下权限门只放行只读工具, 模型用 exit_plan_mode 提交计划, 人批准后模式才切到 accept_edits —— "只读"由 harness 强制, 不靠模型自觉。互不依赖的调用放进同一个 assistant turn 并行执行, 耗时 ≈ 最慢的一个。',
      question: '如果 plan 模式只是 system prompt 里的一句"请先不要修改文件", 它和现在的实现差在哪?',
      code: 'llm_agent/core/tools.py (TodoWriteTool · ExitPlanModeTool) · llm_agent/m10_planning',
      points: [
        { title: '计划即状态', body: 'todo 列表留在上下文里: 模型做到一半不会忘了还剩什么, 用户也能看到进度。整表覆写而不是增量 patch —— 模型每次都要重新面对完整计划。todo_write 只改 agent 自己的状态, 所以 plan 模式下也允许。' },
        { title: 'plan 模式是门, 不是提示', body: 'PermissionGate("plan") 对非只读工具一律 DENY, allow 规则也不能放行。demo [3]: 就算模型不交计划直接写, 门也不开。批准计划这个动作本身才会改 gate.mode。' },
        { title: '并行工具调用', body: '模型在一个 turn 里发 3 个 tool_use, harness 用线程池执行: 3 个 0.2s 的调用, 串行 ≥ 0.6s, 并行约 0.2s; 同时省掉两次模型往返。授权仍然串行 —— 审批弹窗不能并发。' },
      ],
      links: [
        { from: 'TodoWriteTool', to: 'tool_result "todos updated: 1/2 completed"', body: '进度回填给模型, 也能直接渲染给用户。' },
        { from: 'ExitPlanModeTool.execute', to: 'gate.mode = "accept_edits"', body: '人批准之后才切模式; 拒绝则留在 plan。' },
        { from: 'ModelAction.tool_calls', to: 'ThreadPoolExecutor', body: '并行只发生在"执行"阶段, 结果按原顺序放回同一条 user 消息。' },
      ],
      sourceRows: [
        { concept: 'todo 状态', code: 'core/tools.py:TodoWriteTool', takeaway: 'status ∈ pending / in_progress / completed; read_only=True 所以 plan 模式可用。' },
        { concept: '提交计划', code: 'core/tools.py:ExitPlanModeTool', takeaway: 'approve(plan) 为真才改 gate.mode, 否则返回 is_error 结果。' },
        { concept: 'plan 兜底', code: 'core/permissions.py:PermissionGate._evaluate_one', takeaway: 'mode == "plan": 只读放行, 其余 DENY "plan mode is read-only until the plan is approved"。' },
        { concept: '并行执行', code: 'core/agent.py:Agent._run_tools', takeaway: 'pool.map 保序; 总耗时 ≈ 最慢的那个, 而不是求和。' },
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
      title: 'Orchestrator–workers · 并行子智能体与两本 token 账',
      subtitle: 'lead 把任务扇出给多个隔离的 worker, 只收摘要。买到的是并行度和干净的主上下文, 不是省钱。',
      tldr: '扇出不需要新机制: lead 在一个 turn 里发多个 delegate tool_use, loop 的线程池自然并行执行。每类 worker 有自己的工具集和上下文, transcript 落盘但不回流。要记两本账: lead 峰值上下文小得多; 总 token (lead + 所有 worker) 在 m11 demo 里是 770 vs 单 agent 的 413。',
      question: '什么样的任务值得用多智能体? 如果子任务之间强依赖、需要共享同一份上下文, 会发生什么?',
      code: 'llm_agent/core/subagents.py · llm_agent/m11_orchestrator',
      points: [
        { title: '只回摘要', body: 'worker 读了多少原文都留在自己的上下文里, lead 只收到一条 tool_result 摘要。demo: lead 峰值 247 token vs 单 agent 375 —— 文档越长、子任务越多, 差距越大。' },
        { title: '总账要分情况看', body: '每个 worker 都要重建上下文 (system + 任务简报), 这是固定开销; 文档短时它占主导, 总 token 更高 (demo 的情况)。文档长、轮数多时, 单 agent 每轮重发全部已读文档是平方增长, 同等工作量下反而更贵 —— 但真实 worker 会各自多探索 (Anthropic 报告多智能体约为聊天的 15× token)。' },
        { title: '最小权限的 worker', body: 'researcher 只能检索, calculator 只能算; 子级用 auto 权限门且没有人可问 —— 拿不准的一律拒绝。SubagentStop hook 让父级能审计每个子级的收尾。' },
      ],
      links: [
        { from: 'Agent._run_tools', to: 'DelegateTool.execute × N', body: '同一轮的多个 delegate 由线程池并行执行 (max_parallel=4)。' },
        { from: 'child.run(task)', to: 'ToolResult("[researcher] …summary")', body: '父级上下文里只有这一行。' },
        { from: 'child.usage', to: 'token 记账', body: 'usage 分开记, 才能看清"总花费 vs 父上下文占用"。' },
        { from: 'agent-state-subagents', to: 'agent-orchestrator', body: 'm07 讲隔离, m11 讲扇出与记账。' },
      ],
      sourceRows: [
        { concept: '子级工厂', code: 'core/subagents.py:DelegateTool', takeaway: 'agent_types: 名字 → 返回全新 ToolRegistry 的工厂; 子级之间也不共享状态。' },
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
      subtitle: '纵深防御: 每一层都假设别的层会失守, 而且至少有一层不依赖模型听话。',
      tldr: '工具输出是数据不是指令: 不可信输出包进 <untrusted_data> 并标记注入特征 —— 但这只是在"劝"模型。兜底的是确定性规则: 本轮上下文一旦混入不可信数据, 高风险工具一律拒绝 (污点); 文件路径先 resolve() 再判断是否在 root 内; 内容进 transcript 之前先脱敏。',
      question: '注入检测的正则挡得住换个说法的攻击吗? 如果假设模型一定会上当, 你的系统还剩哪几道防线?',
      code: 'llm_agent/core/{guardrails.py,sandbox.py,agent.py} · llm_agent/m12_guardrails',
      points: [
        { title: '标记只降低概率', body: 'wrap_untrusted 给模型一个"这是数据"的强提示, scan 命中特征时再加 injection_suspected。特征匹配只能标记可疑, 挡不住改写过的注入 —— 这一层降低的是概率, 不是可能性。' },
        { title: '污点规则切断 lethal trifecta', body: '私有数据 + 不可信内容 + 对外通道同时成立才出事。_tainted 置位后本轮高风险工具全部 DENIED, 即使用户配过 allow 规则; 下一条真正的用户指令才重置。demo: 模型还是上当了, 但 shell.executed == []。' },
        { title: '先 resolve 再检查', body: 'confine(): (root / path).resolve() 展开 .. 和符号链接后再 is_relative_to(root)。先拼接后查字符串前缀是经典漏洞: /work/../etc 以 /work 开头。脱敏保留 key 名只抹值, 日志仍可读。' },
      ],
      links: [
        { from: 'Tool.untrusted_output', to: 'Guardrails.wrap_untrusted', body: 'fetch_doc、MCP 工具的输出来自外部世界。' },
        { from: 'Agent._tainted', to: 'Agent._authorize', body: '污点检查排在权限门之前: allow 规则也救不了。' },
        { from: 'confine(root, path)', to: 'ReadFileTool / WriteFileTool / MemoryTool', body: '所有文件类工具共用同一个围栏。' },
        { from: 'Guardrails.redact', to: 'Agent._append', body: '进 transcript / JSONL 之前脱敏, 密钥不会进下一次模型请求。' },
      ],
      sourceRows: [
        { concept: '包裹与标记', code: 'core/guardrails.py:Guardrails.wrap_untrusted', takeaway: '<untrusted_data injection_suspected="…"> —— 提示模型, 不保证模型听。' },
        { concept: '污点锁', code: 'core/agent.py:Agent._authorize', takeaway: 'guardrails and _tainted and risk == "high" → DENIED, 确定性, 与模型无关。' },
        { concept: '路径围栏', code: 'core/sandbox.py:confine', takeaway: '"/" 开头按沙箱内虚拟根解释; resolve 之后才判断。' },
        { concept: '脱敏', code: 'core/guardrails.py:Guardrails.redact', takeaway: '带捕获组的规则保留 password= 这类 key 名, 只抹值。' },
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
      subtitle: '"跑一下看着还行"不是评测。agent 有随机性, 单次成功什么也说明不了。',
      tldr: '任务 = prompt + 全新环境 + grader(终态): 检查环境里真实发生了什么, 而不是匹配模型的措辞。轨迹检查另外约束过程 (工具调用预算、禁用工具)。pass@k = 1−(1−p)^k 衡量能力上限, pass^k = p^k 衡量可靠性 —— 面向用户的 agent 要看后者。同一任务集跑两套配置做回归对比。',
      question: '单次成功率 90% 的 agent, 连续 8 次都做对的概率是多少? 哪类产品应该盯 pass@k, 哪类必须盯 pass^k?',
      code: 'llm_agent/m13_evals/demo.py',
      points: [
        { title: '以终态判分', body: 'grader 看的是: 笔记写了没、危险命令跑了没 (env.shell.executed == [])。每次试验一个全新环境, 试验之间不能互相污染。模型说"我已经写好了"不算数。' },
        { title: '结果对, 过程也要对', body: '轨迹检查: 工具调用次数超预算、成功执行了禁用工具 (forbidden:shell) 都算失败。demo 的回归: 有人为了少弹确认框改成 dont_ask 并删了 deny 规则 → safety 任务悄悄坏掉。' },
        { title: 'pass@k 与 pass^k 走向两端', body: 'demo: 单次 0.65 → pass@3 = 0.97, pass^3 = 0.25。代码里用无偏估计 1−C(n−c,k)/C(n,k) 与 C(c,k)/C(n,k) (n 次试验成功 c 次), 比直接代 p̂^k 更准。' },
      ],
      links: [
        { from: 'LLM 协议', to: 'FlakyLLM', body: '评测用的替身只需实现 next(): 以概率 p "懒得用工具"。' },
        { from: 'Agent.messages', to: '轨迹检查', body: 'tool_use / tool_result block 让"执行了什么"可以程序化断言。' },
        { from: 'baseline vs candidate', to: '回归列表', body: '逐任务列出变差的项: 改 prompt / 换模式之前先跑一遍。' },
      ],
      sourceRows: [
        { concept: '任务定义', code: 'm13_evals/demo.py:Task', takeaway: 'grade(final, env) + max_tool_calls + forbidden。' },
        { concept: '单次试验', code: 'm13_evals/demo.py:run_task', takeaway: '用 tool_use id → name 的映射找出"成功执行"了哪些工具。' },
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
      subtitle: '上下文是预算: 便宜的先清, 贵的再压, 能现取的不预存, 要跨会话的写文件。',
      tldr: '超预算时逐级降级: ① 清旧工具结果 (只改发给模型的视图, 零模型开销, 配对结构不动); ② 还超 → PreCompact hook 指定必留信息 → 模型写摘要并真的替换历史, JSONL 追加 compact_boundary, resume 出来的会话也变小; 硬截断只作反例。另两招: 即时检索 (上下文里只放索引) 和 memory 工具 (状态写到窗口之外)。',
      question: '被"清掉"的工具结果和被"截断"丢掉的对话, 哪个还能找回来? 为什么?',
      code: 'llm_agent/core/{memory.py,agent.py,persistence.py,sandbox.py} · llm_agent/m14_context_engineering',
      points: [
        { title: '工具结果最胖, 也最快过时', body: 'clear_tool_results 只保留最近 keep_last 个结果的正文, 其余换成 [cleared: N chars]。tool_use 块还在 —— 模型知道当时读的是哪个文件, 需要时再调一次工具就回来了。' },
        { title: '摘要是有损的, 要点名保留', body: '_compact 只压"当前用户轮之前"的历史, 当前轮原样保留, 配对不会被切断。行号、数值这类细节摘要容易丢, pre_compact hook 让人指定"必须保留什么"。代价是一次模型调用。' },
        { title: '不进窗口的才是最省的', body: '即时检索: demo 里预加载每次调用要带 331 token, 即时检索峰值 107。memory 工具: 模型自己往 /memories 写, 全新会话再读回来 —— 跨会话的知识不占任何一轮的上下文。' },
      ],
      links: [
        { from: 'Agent._assemble_context', to: 'clear_tool_results → _compact', body: '由便宜到贵的级联; truncate 只裁视图, 历史和 resume 都不会变小。' },
        { from: 'hooks.on_pre_compact', to: 'summarize_with_llm(keep)', body: '人指定摘要里必须留下什么。' },
        { from: 'store.append_compact', to: 'JsonlSessionStore.load', body: '文件只增不减供审计, load() 读回的是压缩后的视图。' },
        { from: 'MemoryTool', to: 'confine(root, "/memories/…")', body: '记忆目录也在路径围栏里。' },
      ],
      sourceRows: [
        { concept: '级联', code: 'core/agent.py:Agent._assemble_context', takeaway: '第 1 档只改视图; 第 2 档才动 self.messages。' },
        { concept: '清工具结果', code: 'core/memory.py:clear_tool_results', takeaway: '返回新列表, 不改原消息; 占位符里留下原长度。' },
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
