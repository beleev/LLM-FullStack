<template>
  <div>
    <h1 class="page-title">Agent 应用层 · 把模型接成可行动系统</h1>
    <p class="page-subtitle">
      <RepoLink path="llm_agent/" label="llm_agent/" tiny /> 不接真实 LLM API, 而是用可预测的
      <code class="inline">RuleBasedLLM</code> 把应用层 harness 摊开:
      工具、权限、上下文、记忆、扩展、持久化和子智能体如何围绕一个很薄的 loop 工作。
    </p>

    <ChapterIntro
      tldr="Agent 的核心不是复杂 while-loop, 而是 loop 周围的确定性系统: tool registry、permission gate、context/memory、hooks、JSONL transcript 和 subagent isolation。"
      question="为什么一个能调用工具的模型, 还不能直接等价于一个可靠 Agent 产品?"
      :goals="[
        '看清一个 Agent loop 的最小骨架: messages / tool / result',
        '知道 permission / context / hooks / persistence 各自负责什么',
        '理解一个真实 Agent harness 是怎么从这些零件搭起来的',
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
        从能生成 token 到能执行任务, 中间多出来的是控制面。先把 loop 跑通,
        再逐层加工具、权限、上下文、扩展、持久化和子智能体。
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
        读 Agent 代码时, 不要只盯模型调用。先找下面六个账本分别在哪里维护。
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
        教学版 <code class="inline">RuleBasedLLM</code> 故意很弱, 因为本章要看的不是模型聪明程度,
        而是模型输出如何被结构化、检查、执行和记录。
      </p>
      <div class="grid grid-2" style="gap: 16px;">
        <div class="card">
          <h3>模型负责选择动作 <span class="tag">probabilistic</span></h3>
          <pre class="code">{{ modelSide }}</pre>
          <p class="hint">
            换成真实 LLM 时, 这里会变成 function calling / tool calling 的 JSON 输出。
            后面的工具执行和权限逻辑不需要跟着模型一起重写。
          </p>
        </div>
        <div class="card">
          <h3>harness 负责执行边界 <span class="tag">deterministic</span></h3>
          <pre class="code">{{ harnessSide }}</pre>
          <p class="hint">
            Agent 产品的可靠性主要来自这里: 能否拒绝危险动作、恢复状态、压缩上下文并留下审计记录。
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
        search、note、weather、shell、delegate 五类工具共用一条执行面。
      </p>
      <div class="grid grid-2" style="gap: 16px;">
        <div class="card">
          <h3>组合入口 <span class="tag">main</span></h3>
          <pre class="code">{{ fullLoopCode }}</pre>
        </div>
        <div class="card">
          <h3>运行现象 <span class="tag">四个任务</span></h3>
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

    <ChapterNav
      :prev="{ name: 'infer-engine', label: '阶段 5.5 · mini-vLLM 引擎', hint: '推理服务提供 token, Agent harness 编排动作' }"
      :next="{ name: 'agent-loop', label: '阶段 6.1 · Agent loop', hint: '先看最小 while-loop 闭环' }"
    />
  </div>
</template>

<script setup>
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
    fix: 'deny-first + ask/default/auto 模式, 风险可控地放权。',
    color: 'var(--warn)',
  },
  {
    name: 'Context',
    year: 'm04',
    pain: '历史和记忆会挤爆窗口。',
    fix: '文件记忆检索 + 头尾保留 + 中间摘要压缩。',
    color: 'var(--eye)',
  },
  {
    name: 'Extensible',
    year: 'm05',
    pain: '所有能力都写进 prompt 会越来越贵。',
    fix: 'Hooks / Skills / MCP-like tools 分层接入。',
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
    desc: '用户、工具、助手输出都进入同一条消息流, 供下一轮模型决策。',
    file: 'llm_agent/core/schema.py',
    code: `Message(role="user", content=prompt)
Message(role="tool", name="calculator", content="2+2=4")
Message(role="assistant", content=final)`,
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
    desc: '权限门在工具执行前做 deny-first 判断。',
    file: 'llm_agent/core/permissions.py',
    code: `outcome = permissions.evaluate(call)
if not outcome.allowed:
    return ToolResult(call.name, "DENIED", ok=False)`,
  },
  {
    name: 'Memory',
    tag: 'context',
    desc: '透明文件记忆按需检索进入上下文, 历史超预算时压缩。',
    file: 'llm_agent/core/memory.py',
    code: `base.extend(memory_messages(memory, prompt))
context = compact_messages(base + messages, max_chars=budget)`,
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

const harnessSide = `action = llm.next(context, tools.names())
outcome = permissions.evaluate(action.tool_call)
if outcome.allowed:
    result = tools.execute(action.tool_call)
else:
    result = ToolResult(name, "DENIED", ok=False)

messages.append(tool_result_message(result))
store.append(message)`

const fullLoopCode = `tools = ToolRegistry([
    SearchDocsTool(DOCS), WriteNoteTool(notes),
    WeatherTool(), ShellTool(), DelegateTool(DOCS),
])

agent = Agent(
    llm=RuleBasedLLM(),
    tools=tools,
    permissions=PermissionGate(mode="auto", rules=deny_rules),
    hooks=build_hooks(),
    memory=FileMemory(...),
    store=JsonlSessionStore(...),
)`

const runRows = [
  { step: '1', title: 'gather → act → persist', body: '搜索文档后把结果写入笔记, 展示多步工具调用。' },
  { step: '2', title: 'delegate isolated research', body: '父 Agent 调子 Agent, 父级只收到 summary。' },
  { step: '3', title: 'external MCP-like tool', body: 'Fake MCP server 暴露 weather 工具。' },
  { step: '4', title: 'denied dangerous action', body: 'rm -rf 被 deny-first 规则拒绝。' },
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
