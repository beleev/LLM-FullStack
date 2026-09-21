<!--
  MCP 握手时序 (对应 llm_agent/core/mcp.py 的 MCPClient / MCPTool 与 m09_mcp/server.py)。
  只讲一件事: MCP 就是 stdio 上逐行的 JSON-RPC 2.0; 外部工具进来以后, 和内置工具走同一个校验 + 权限门。
-->
<template>
  <LabFrame
    title="MCP — 一次外部工具调用的完整报文"
    sub="左边是 agent 进程 (MCP client), 右边是被拉起的 server 子进程, 中间每个箭头是 stdio 上的一行 JSON。按播放走完 initialize → tools/list → tools/call; 点任意一步看真实报文。换场景和权限规则, 数一数有几条请求真的到了 server。"
    module="llm_agent/m09"
    run="python -m llm_agent.m09_mcp.demo"
    :challenge="{
      ask: '场景选「参数类型错」(city 传了数字 123)。这次调用会产生几条 tools/call 报文? server 会看到这个坏参数吗?',
      answer: '0 条, server 根本看不到。tools/list 返回的 inputSchema 就是 JSON Schema, client 把它当作本地工具的 parameters, 执行前先 validate_args —— 坏参数在本地变成 INVALID_ARGS 的 is_error 结果回填给模型。同理, 权限门也在发报文之前: 没有 allow mcp__weather__* 规则时, MCP 工具按 high 风险处理, 无人审批就 fail closed。协议只负责「怎么接进来」, 不负责「能不能信」。',
    }"
  >
    <template #controls>
      <div class="row">
        <button v-for="s in SCENES" :key="s.id" type="button" :class="{ active: scene === s.id }" @click="scene = s.id">{{ s.label }}</button>
      </div>
      <div class="row">
        <button type="button" :class="{ active: allowRule }" :aria-pressed="allowRule" @click="allowRule = !allowRule">{{ allowRule ? '☑' : '☐' }} allow 规则 mcp__weather__*</button>
      </div>
      <StepPlayer :stepper="stepper" />
    </template>

    <div class="seq">
      <div class="heads"><span>agent 进程 · MCPClient</span><span>server 子进程 · stdio</span></div>
      <button
        v-for="(s, i) in steps" :key="i" type="button" class="step" :class="[s.dir, s.state, { dim: i > stepper.step.value, now: i === stepper.step.value }]"
        @click="stepper.pause(); stepper.step.value = i"
      >
        <span class="mono title">{{ s.dir === 'req' ? '──▶ ' : s.dir === 'res' ? '◀── ' : '● ' }}{{ s.title }}</span>
      </button>
    </div>
    <div class="payload">
      <p class="note">{{ cur.note }}</p>
      <pre v-if="cur.json" class="mono">{{ JSON.stringify(cur.json, null, 1) }}</pre>
    </div>

    <template #stats>
      <div class="kv"><span>已发 JSON-RPC 请求</span><b>{{ sent.requests }}</b></div>
      <div class="kv"><span>到达 server 的 tools/call</span><b :class="sent.calls ? '' : 'good'">{{ sent.calls }}</b></div>
      <div class="kv"><span>模型看到的工具名</span><b class="small">mcp__weather__get_weather</b></div>
      <div class="kv"><span>结局</span><b :class="outcome.cls">{{ outcome.text }}</b></div>
      <p class="lab-note">前缀 <code class="inline">mcp__server__tool</code> 有两个用途: 防止和内置工具重名; 让一条规则就能管住整个 server。发给 server 时前缀会被剥掉, 还原成它自己的工具名。</p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import StepPlayer from '@/components/lab/StepPlayer.vue'
import { useStepper } from '@/composables/useStepper.js'

const SCENES = [
  { id: 'ok', label: '正常调用', name: 'get_weather', args: { city: 'Beijing' } },
  { id: 'badargs', label: '参数类型错', name: 'get_weather', args: { city: 123 } },
  { id: 'toolerr', label: '工具内部出错', name: 'get_weather', args: { city: 'Atlantis' } },
  { id: 'nomethod', label: 'server 不支持的方法', name: 'get_weather', args: { city: 'Beijing' } },
]
const scene = ref('ok')
const allowRule = ref(true)

const TOOLS = [
  { name: 'get_weather', description: 'Return a (fake) weather report for a city.', inputSchema: { type: 'object', properties: { city: { type: 'string' } }, required: ['city'], additionalProperties: false } },
  { name: 'add', description: 'Add two numbers.', inputSchema: { type: 'object', properties: { a: { type: 'number' }, b: { type: 'number' } }, required: ['a', 'b'], additionalProperties: false } },
]
const rpc = (id, method, params) => ({ jsonrpc: '2.0', id, method, params })

const steps = computed(() => {
  const sc = SCENES.find((s) => s.id === scene.value)
  const full = `mcp__weather__${sc.name}`
  const st = [
    { dir: 'local', title: 'spawn server 子进程', note: 'Popen(argv, stdin=PIPE, stdout=PIPE): 只拉起调用方给的 argv, 不经过 shell。之后 stdout 就是协议通道 —— server 的日志只能写 stderr。' },
    { dir: 'req', title: 'initialize', json: rpc(1, 'initialize', { protocolVersion: '2025-06-18', capabilities: {}, clientInfo: { name: 'llm_agent', version: '0.1' } }), note: '握手第一步: 双方对齐协议版本和能力。每条请求带递增的 id, 响应必须带同一个 id。' },
    { dir: 'res', title: 'result: serverInfo + capabilities', json: { jsonrpc: '2.0', id: 1, result: { protocolVersion: '2025-06-18', capabilities: { tools: {} }, serverInfo: { name: 'toy-weather', version: '0.1' } } }, note: 'server 声明自己支持 tools 能力。' },
    { dir: 'req', title: 'notifications/initialized', json: { jsonrpc: '2.0', method: 'notifications/initialized' }, note: '没有 id = 通知 (notification): 规范要求 server 不回复。' },
    { dir: 'req', title: 'tools/list', json: rpc(2, 'tools/list', {}), note: '问 server: 你有哪些工具?' },
    { dir: 'res', title: 'result: tools[2]', json: { jsonrpc: '2.0', id: 2, result: { tools: TOOLS } }, note: `每个工具自带 JSON Schema。client 把它们包成 MCPTool 注册进 ToolRegistry, 名字变成 ${full.replace(sc.name, '*')}; risk 一律按 high (server 自报的只读注解不可信)。` },
  ]
  if (scene.value === 'nomethod') {
    st.push({ dir: 'req', title: 'resources/list', json: rpc(3, 'resources/list', {}), note: 'client 试探一个这个 server 没实现的方法。' })
    st.push({ dir: 'res', state: 'bad', title: 'error -32601', json: { jsonrpc: '2.0', id: 3, error: { code: -32601, message: 'Method not found' } }, note: '协议层错误用 error 字段 (没有 result)。它和「工具执行失败」是两回事 —— 对比「工具内部出错」场景。' })
    return st
  }
  st.push({ dir: 'local', title: `模型发出 tool_use ${full}`, json: { type: 'tool_use', id: 'toolu_0001', name: full, input: sc.args }, note: '对模型来说, 它和内置工具没有任何区别: 都是工具列表里的一个 name + input_schema。' })
  const allowed = allowRule.value
  st.push({ dir: 'local', state: allowed ? 'ok' : 'bad', title: `权限门: ${allowed ? 'ALLOW (rule)' : 'ASK → 无人 = DENY'}`, note: allowed ? '命中 allow mcp__weather__* → 放行。' : '没有规则命中; MCPTool.risk = high → 问人; 无人可问 → fail closed。报文不会发出。' })
  if (!allowed) return st
  // ★ inputSchema 在本地先校验: 坏参数到不了 server
  const bad = typeof sc.args.city === 'number'
  st.push({ dir: 'local', state: bad ? 'bad' : 'ok', title: `validate_args: ${bad ? 'INVALID_ARGS' : '通过'}`, json: bad ? { type: 'tool_result', tool_use_id: 'toolu_0001', content: "INVALID_ARGS: 'city' should be string, got int", is_error: true } : null, note: bad ? '用 tools/list 拿到的 inputSchema 在本地校验, 错误直接回填给模型, 让它下一轮改参数。' : '参数符合 inputSchema。' })
  if (bad) return st
  st.push({ dir: 'req', title: 'tools/call', json: rpc(3, 'tools/call', { name: sc.name, arguments: sc.args }), note: `注意 name 是 "${sc.name}": 前缀 mcp__weather__ 已被剥掉。` })
  const err = scene.value === 'toolerr'
  const text = err ? "tool error: KeyError('Atlantis')" : 'Beijing: sunny, 24C, light wind'
  st.push({ dir: 'res', state: err ? 'bad' : 'ok', title: `result: isError=${err}`, json: { jsonrpc: '2.0', id: 3, result: { content: [{ type: 'text', text }], isError: err } }, note: err ? '工具失败是「正常结果」: 仍然走 result, 只是 isError=true。协议没坏, 模型可以据此换个做法。(教学 server 其实什么城市都答, 这里的异常是示意。)' : '结果是 content block 列表, client 取出其中的 text。' })
  st.push({ dir: 'local', title: err ? 'tool_result 回填 (is_error)' : 'tool_result 回填 (标记为不可信数据)', json: { type: 'tool_result', tool_use_id: 'toolu_0001', content: err ? text : `<untrusted_data>\n${text}\n</untrusted_data>`, is_error: err }, note: 'MCP 工具的输出来自第三方: untrusted_output = True, 开了护栏就会被包进 <untrusted_data>, 见「护栏」一章。' })
  return st
})

const stepper = useStepper(() => steps.value.length, { interval: 1100 })
watch(steps, (s) => { stepper.pause(); stepper.step.value = s.length - 1 }, { immediate: true })
const cur = computed(() => steps.value[stepper.step.value] || steps.value[0])
const sent = computed(() => {
  const seen = steps.value.slice(0, stepper.step.value + 1).filter((s) => s.dir === 'req' && s.json.id)
  return { requests: seen.length, calls: seen.filter((s) => s.json.method === 'tools/call').length }
})
const outcome = computed(() => {
  const last = steps.value[steps.value.length - 1]
  if (scene.value === 'nomethod') return { text: '协议错误 -32601', cls: 'bad' }
  if (!allowRule.value) return { text: '权限门拒绝, 未发报文', cls: 'good' }
  if (last.title.startsWith('validate')) return { text: '本地拦下坏参数', cls: 'good' }
  return scene.value === 'toolerr' ? { text: 'isError 结果回填', cls: '' } : { text: '成功', cls: 'good' }
})
</script>

<style scoped>
.seq { display: flex; flex-direction: column; gap: 4px; }
.heads { display: flex; justify-content: space-between; font-size: 11px; color: var(--text-dim); padding: 0 4px 4px; border-bottom: 1px solid var(--border); margin-bottom: 4px; }
.step { min-height: 0; padding: 5px 10px; width: 78%; text-align: left; border: 1px solid var(--border); border-radius: var(--radius-sm); background: var(--bg-elev); color: var(--text); }
.step.req { border-color: var(--accent); }
.step.res { align-self: flex-end; text-align: right; border-color: var(--left); }
.step.local { width: 60%; border-style: dashed; }
.step.bad { border-color: var(--danger); color: var(--danger); }
.step.dim { opacity: 0.3; }
.step.now { outline: 2px solid var(--warn); outline-offset: 1px; }
.title { font-size: 11px; word-break: break-all; }
.payload { margin-top: 12px; min-height: 150px; }
.note { font-size: 12px; color: var(--text-muted); line-height: 1.7; }
.payload pre { margin-top: 6px; font-size: 11px; line-height: 1.45; background: var(--code-bg); color: var(--code-text); border-radius: var(--radius-sm); padding: 8px 10px; overflow-x: auto; max-height: 260px; }
.small { font-size: 12px !important; word-break: break-all; text-align: right; }
@media (max-width: 600px) { .step, .step.local { width: 92%; } }
</style>
