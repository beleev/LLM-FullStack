<template>
  <div>
    <h1 class="page-title">Agent 应用层 · 把模型接成可行动系统</h1>
    <p class="page-subtitle">
      <RepoLink path="llm_agent/" label="llm_agent/" tiny /> 默认不接真实 API, 用行为完全确定的
      <code class="inline">RuleBasedLLM</code> 顶上 —— 这样每个 demo 都能用 <code class="inline">assert</code>
      证明自己声称的事。这一阶段要看的不是模型多聪明, 而是它的输出怎么被结构化、检查、执行和记录。
      设计刻意对齐 Claude Code: 权限模式名、hook 事件名、<code class="inline">mcp__server__tool</code> 命名、content block 格式。
    </p>

    <ChapterIntro
      tldr="loop 本身只有十几行, 从 m01 到 m14 几乎没改过; 可靠性全在它周围: 工具注册表、权限门、上下文与记忆、hooks、JSONL transcript、子智能体隔离。"
      question="为什么一个能调用工具的模型, 还不能直接等价于一个可靠 Agent 产品?"
      :goals="[
        '读懂一个 Agent loop 的最小骨架: messages / tool_use / tool_result',
        '说清权限门、上下文、hooks、持久化各自拦住了什么',
        '看出一个真实 Agent harness 是从这些零件怎么搭起来的',
      ]"
      :codes="[
        { path: 'llm_agent/core/' },
        { path: 'llm_agent/full_loop/' },
        { path: 'llm_agent/m01_agent_loop/' },
        { path: 'llm_agent/m02_tool_use/' },
      ]"
      :prereq="{ name: 'infer-engine', label: '阶段 5.5 · mini-vLLM 引擎' }"
      :next-step="{ name: 'agent-loop', label: '阶段 6.1 · Agent loop' }"
    />

    <section class="section">
      <h2>1. 应用层依赖链</h2>
      <p class="lead">
        从"能生成 token"到"能把事做完", 中间多出来的全是控制面。先把 loop 跑通,
        再一层层加工具、权限、上下文、扩展、持久化和子智能体 —— 每加一层能力, 就得同时加一道边界。
      </p>
      <EvolutionChain
        title="从 completion 到 mini Agent harness"
        subtitle="每一层都让模型多一点能力, 也多一道确定性边界。"
        :steps="agentChain"
      />
    </section>

    <section class="section">
      <h2>2. Agent harness 的六个账本</h2>
      <p class="lead">
        读 Agent 代码时别只盯着那句模型调用。先把下面六个账本各自在哪儿维护找出来, 整个系统就清楚了。
      </p>
      <div class="grid grid-3">
        <div v-for="p in primitives" :key="p.name" class="card primitive-card">
          <h3>{{ p.name }} <span class="tag">{{ p.tag }}</span></h3>
          <p class="desc">{{ p.desc }}</p>
          <pre class="code">{{ p.code }}</pre>
          <p class="hint"><CodeRef :value="p.file" base="llm_agent/" tiny /></p>
        </div>
      </div>
    </section>

    <section class="section">
      <h2>3. 模型与 harness 的边界</h2>
      <p class="lead">
        <code class="inline">RuleBasedLLM</code> 故意写得很弱。这一分工才是重点: 模型负责"选哪个动作"(概率性),
        harness 负责"这个动作能不能做、做出来记在哪"(确定性)。安全相关的判断一条都不能依赖模型听话。
      </p>
      <div class="grid grid-2" style="gap: 16px;">
        <div class="card">
          <h3>模型负责选择动作 <span class="tag">probabilistic</span></h3>
          <pre class="code">{{ modelSide }}</pre>
          <p class="hint">
            换成真实 LLM, 这里就变成 tool calling 的 JSON 输出。后面的工具执行和权限逻辑一行都不用跟着改
            —— loop 对模型的全部要求只有一个 <code class="inline">next()</code>。
          </p>
        </div>
        <div class="card">
          <h3>harness 负责执行边界 <span class="tag">deterministic</span></h3>
          <pre class="code">{{ harnessSide }}</pre>
          <p class="hint">
            Agent 产品的可靠性几乎全在这几行里: 能不能拒掉危险动作、能不能恢复状态、能不能压住上下文、
            出事之后能不能逐行复盘。
          </p>
        </div>
      </div>
    </section>

    <section class="section">
      <h2>4. 模块索引 · 运行时看什么</h2>
      <p class="lead">
        每个 demo 都可以单独跑; <code class="inline">python -m llm_agent.run_all</code>
        会按学习路径整体跑一遍 (<RepoLink path="llm_agent/run_all.py" label="llm_agent/run_all.py" tiny />)。
      </p>
      <div class="card" style="padding: 0; overflow-x: auto;">
        <table class="agent-table">
          <thead>
            <tr>
              <th>模块</th>
              <th>要点</th>
              <th>和真实 Agent 的关系</th>
              <th>原始代码</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="m in agentModules" :key="m.id">
              <td class="axis">{{ m.name }}</td>
              <td>{{ m.concept }}</td>
              <td>{{ m.link }}</td>
              <td class="mono small"><RepoLink :path="m.file" tiny /></td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <section class="section">
      <h2>5. full_loop · mini-Claude-Code-style harness</h2>
      <p class="lead">
        <RepoLink path="llm_agent/full_loop/demo.py" label="full_loop/demo.py" tiny /> 把所有机制接到同一个 Agent 中:
        检索、笔记、skill、fetch、shell、delegate 与 MCP 工具共用一条执行面。
      </p>
      <div class="grid grid-2" style="gap: 16px;">
        <div class="card">
          <h3>组合入口 <span class="tag">main</span></h3>
          <pre class="code">{{ fullLoopCode }}</pre>
        </div>
        <div class="card">
          <h3>运行现象 <span class="tag">五个场景</span></h3>
          <div class="run-list">
            <div v-for="r in runRows" :key="r.title" class="run-row">
              <span class="pill">{{ r.step }}</span>
              <div>
                <strong>{{ r.title }}</strong>
                <p>{{ r.body }}</p>
              </div>
            </div>
          </div>
          <p class="hint">
            对应命令: <code class="inline">python -m llm_agent.full_loop.demo</code>
            (<RepoLink path="llm_agent/full_loop/demo.py" label="源码" tiny />)
          </p>
        </div>
      </div>
    </section>

    <!-- 本章挂载的实验台 (data/labmap/*.js) 与章末自测 (data/quiz/*.js), 没配置时不渲染 -->

    <LabMount />

    <QuizCard />


    <ChapterNav
      :prev="{ name: 'infer-engine', label: '阶段 5.5 · mini-vLLM 引擎', hint: '推理服务提供 token, Agent harness 编排动作' }"
      :next="{ name: 'agent-loop', label: '阶段 6.1 · Agent loop', hint: '先看最小 while-loop 闭环' }"
    />
  </div>
</template>

<script setup>
import LabMount from '@/components/LabMount.vue'
import QuizCard from '@/components/QuizCard.vue'
import ChapterIntro from '@/components/ChapterIntro.vue'
import ChapterNav from '@/components/ChapterNav.vue'
import EvolutionChain from '@/components/EvolutionChain.vue'
import CodeRef from '@/components/CodeRef.vue'
import RepoLink from '@/components/RepoLink.vue'
import { agentModules } from '@/data/models.js'

const agentChain = [
  {
    name: 'Loop',
    year: 'm01',
    pain: 'completion 只能给文本, 不能把观察结果接回下一轮。',
    fix: 'messages → ModelAction → ToolResult → messages 的最小闭环。',
    color: 'var(--left)',
  },
  {
    name: 'Tools',
    year: 'm02',
    pain: '模型不知道外部世界, 也不能自己执行 Python。',
    fix: 'Tool schema 暴露能力, execute 由确定性代码完成。',
    color: 'var(--accent)',
  },
  {
    name: 'Permission',
    year: 'm03',
    pain: '能行动后就有破坏面。',
    fix: 'deny > ask > allow + 六种模式兜底, 风险可控地放权。',
    color: 'var(--warn)',
  },
  {
    name: 'Context',
    year: 'm04',
    pain: '历史和记忆会挤爆窗口。',
    fix: '文件记忆检索 + 清旧工具结果 + 模型摘要压缩。',
    color: 'var(--eye)',
  },
  {
    name: 'Extensible',
    year: 'm05',
    pain: '所有能力都写进 prompt 会越来越贵。',
    fix: 'Hooks / 渐进披露的 Skills / MCP 工具分层接入。',
    color: 'var(--right)',
  },
  {
    name: 'State + Team',
    year: 'm06-m07',
    pain: '长任务要恢复, 子任务不能污染主上下文。',
    fix: 'append-only transcript + isolated subagent summary return。',
    color: 'var(--left)',
  },
]

const primitives = [
  {
    name: 'Transcript',
    tag: 'messages',
    desc: '用 Messages API 同款 content block: assistant 发 tool_use, 结果以 tool_result 放进下一条 user 消息, 靠 id 配对。',
    file: 'llm_agent/core/schema.py',
    code: `Message("user", prompt)
Message("assistant", [{"type": "tool_use", "id": "toolu_0001", ...}])
Message("user", [{"type": "tool_result", "tool_use_id": "toolu_0001", ...}])`,
  },
  {
    name: 'Tool Pool',
    tag: 'actions',
    desc: '工具池是模型可选动作集合, 也是执行层的唯一入口。',
    file: 'llm_agent/core/tools.py',
    code: `tools = ToolRegistry([
    CalculatorTool(),
    SearchDocsTool(DOCS),
    WriteNoteTool(notes),
])`,
  },
  {
    name: 'Permission',
    tag: 'guard',
    desc: '权限门在工具执行前裁决: deny > ask > allow > 模式兜底, 评估的是 hook 改写后的最终调用。',
    file: 'llm_agent/core/permissions.py',
    code: `final = pre_hook.updated_call or call
outcome = permissions.evaluate(final, tool)  # 评估改写后的调用
if not outcome.allowed:
    return ToolResult(call.name, "DENIED", ok=False)`,
  },
  {
    name: 'Memory',
    tag: 'context',
    desc: '透明文件记忆按需检索进入上下文; 超预算先清旧工具结果, 不够再让模型写摘要。',
    file: 'llm_agent/core/memory.py',
    code: `base += memory_messages(memory, prompt)
view = clear_tool_results(messages, keep_last=1)
if total_chars(view) > budget: compact()   # 模型写摘要`,
  },
  {
    name: 'Hooks',
    tag: 'policy',
    desc: '生命周期事件让策略、审计和改写不用塞进模型 prompt。',
    file: 'llm_agent/core/hooks.py',
    code: `hooks.register("pre_tool_use", block_secret_shell)
hooks.register("post_tool_use", audit_tool_result)`,
  },
  {
    name: 'Persistence',
    tag: 'resume',
    desc: 'JSONL 仅追加保存 transcript, 可恢复但不自动恢复权限。',
    file: 'llm_agent/core/persistence.py',
    code: `store.append(message)
messages = store.load()   # load_history=True`,
  },
]

const modelSide = `prompt = latest_user(messages)
if "计算" in prompt:
    return ModelAction.tool(
        ToolCall("calculator", {"expr": expr})
    )

if last_message_is_tool_result:
    return ModelAction.final(summary)`

const harnessSide = `action = llm.next(context, tools.schemas())
final = hooks.on_pre_tool_use(call).updated_call or call
outcome = permissions.evaluate(final, tool)
if outcome.allowed:
    result = tools.execute(final)      # 先 validate_args
else:
    result = ToolResult(name, "DENIED", ok=False)

messages.append(Message("user", [result.to_block()]))
store.append(message)`

const fullLoopCode = `tools = ToolRegistry([
    VectorSearchTool(index), WriteNoteTool(notes), SkillTool(skills),
    FetchDocTool(PAGES), ShellTool(), delegate, *mcp_tools(mcp),
])

agent = Agent(
    llm=RuleBasedLLM(),
    tools=tools,
    permissions=PermissionGate(mode="auto", rules=rules),
    hooks=hooks, memory=memory, store=store,
    guardrails=Guardrails(),
)`

const runRows = [
  { step: '1', title: 'skill → 检索 → 写笔记', body: '先按需加载 SKILL.md 正文, 再 TF-IDF 检索, 把结果写入笔记。' },
  { step: '2', title: 'delegate isolated research', body: '父 Agent 调子 Agent, 父级只收到 summary。' },
  { step: '3', title: 'MCP 工具 (真实子进程)', body: 'stdio JSON-RPC server 暴露 mcp__weather__* 工具, 由 allow 规则放行。' },
  { step: '4', title: '危险命令被拒', body: 'rm -fr 换了 flag 顺序, 归一化后仍命中 deny 规则。' },
  { step: '5', title: '文档夹带指令和密钥', body: '不可信输出被标记、密钥被脱敏, 污点规则锁住高风险工具。' },
]
</script>

<style scoped>
table.agent-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
table.agent-table th {
  text-align: left;
  padding: 12px 14px;
  background: var(--bg-elev);
  color: var(--text-muted);
  font-size: 11px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.7px;
  border-bottom: 1px solid var(--border-strong);
}
table.agent-table td {
  padding: 10px 14px;
  border-bottom: 1px solid var(--border);
  vertical-align: top;
}
table.agent-table .axis {
  color: var(--text);
  font-weight: 600;
  white-space: nowrap;
}
table.agent-table .small {
  color: var(--text-muted);
  font-size: 12px;
}
.hint {
  margin-top: 10px;
  font-size: 12px;
  color: var(--text-dim);
  line-height: 1.6;
}
.run-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.run-row {
  display: grid;
  grid-template-columns: 34px 1fr;
  gap: 12px;
  align-items: start;
  padding: 12px;
  background: var(--bg-elev);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
}
.run-row strong {
  display: block;
  font-size: 13px;
  margin-bottom: 2px;
}
.run-row p {
  color: var(--text-muted);
  font-size: 12px;
  line-height: 1.5;
}
</style>
