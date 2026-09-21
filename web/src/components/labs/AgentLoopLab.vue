<!--
  Agent loop 步进器 (对应 llm_agent/core/agent.py 的 Agent.run)。
  只讲一件事: loop 就是"往 messages 里追加 content block", 而每次问模型都要重发整个列表。
-->
<template>
  <LabFrame
    title="Agent loop — 消息列表是怎么一块一块长出来的"
    sub="任务: 查 kv cache 文档, 再算 16×64。按播放看 transcript 逐块增长; 悬停任意 tool_use / tool_result 看它靠哪个 id 配对。打开开关观察: 工具出错不会炸掉 loop, 并行调用少问一次模型, max_turns 到了会被强制停下。"
    module="llm_agent/m01"
    run="python -m llm_agent.m01_agent_loop.demo"
    :challenge="{
      ask: '关掉并行、打开「工具出错」, 先猜: 模型一共被调用几次? 累计 input tokens 大约是最终上下文的几倍?',
      answer: '4 次 (search → 写错参数的 calculator → 改对的 calculator → final)。每次调用都要把 system + 全部历史重新发一遍, 所以累计 input (258) 已是最终上下文 (116) 的 2 倍多, 并随轮数近似平方增长 —— 这就是长会话贵、需要并行工具调用和上下文压缩的原因。错误没有抛异常, 而是变成 is_error=true 的 tool_result 回填, 模型下一轮自己改对了参数。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: parallel }" @click="parallel = !parallel">并行工具调用: {{ parallel ? '开' : '关' }}</button>
        <button type="button" :class="{ active: toolError }" @click="toolError = !toolError">工具出错 (参数名写错): {{ toolError ? '开' : '关' }}</button>
      </div>
      <LabSlider v-model="maxTurns" label="max_turns" :min="1" :max="6" />
      <StepPlayer :stepper="stepper" :label="`block ${stepper.step.value + 1}/${frames.length}`" />
    </template>

    <div class="msgs">
      <div v-for="(m, i) in visible" :key="i" class="msg" :class="m.role">
        <span class="role mono">{{ m.role }}<em v-if="m.call"> · 第 {{ m.call }} 次模型调用的输出</em></span>
        <button
          v-for="(b, j) in m.blocks" :key="j" type="button" class="block mono"
          :class="[b.type, { err: b.is_error, linked: hoverId && (b.id === hoverId || b.tool_use_id === hoverId) }]"
          @mouseenter="hoverId = b.id || b.tool_use_id || ''" @mouseleave="hoverId = ''"
          @focus="hoverId = b.id || b.tool_use_id || ''" @blur="hoverId = ''"
        >{{ show(b) }}</button>
      </div>
    </div>

    <template #stats>
      <div class="kv"><span>模型调用次数</span><b>{{ now.calls }}</b></div>
      <div class="kv"><span>当前上下文</span><b>{{ now.ctx }} tok</b></div>
      <div class="kv"><span>累计 input tokens</span><b :class="now.input > 2 * now.ctx ? 'bad' : ''">{{ now.input }}</b></div>
      <div class="kv"><span>结束方式</span><b :class="sim.forced ? 'bad' : 'good'">{{ done ? (sim.forced ? 'max_turns 强停' : 'final') : '进行中' }}</b></div>
      <p class="lab-note">
        tool_result 放在 <b>user</b> 消息里, 用 tool_use_id 指回 assistant 的 tool_use; 同一轮的多个结果必须放进同一条消息。
        累计 input = 每次调用时的上下文之和: 历史越长, 每多一轮越贵。
      </p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import StepPlayer from '@/components/lab/StepPlayer.vue'
import { useStepper } from '@/composables/useStepper.js'

const parallel = ref(false)
const toolError = ref(false)
const maxTurns = ref(6)
const hoverId = ref('')

const SYSTEM = 'You are a small teaching agent.'
const DOC = '[0.62] kv_cache: the kv cache is the model memory of past keys and values, decode becomes incremental.'
// 与 core/utils.py:estimate_tokens 同一个粗估: 中文 1 字/token, 其余 4 字符/token
const est = (s) => { const cjk = (s.match(/[一-鿿]/g) || []).length; return cjk + Math.floor((s.length - cjk + 3) / 4) }
const flat = (b) => (b.type === 'text' ? b.text : b.type === 'tool_use' ? `[tool_use ${b.name} ${JSON.stringify(b.input)}]` : String(b.content))
const show = (b) => (b.type === 'text' ? b.text
  : b.type === 'tool_use' ? `tool_use  id=${b.id}  ${b.name}(${JSON.stringify(b.input)})`
    : `tool_result  tool_use_id=${b.tool_use_id}  is_error=${b.is_error}  ${b.content}`)

// 工具执行: 参数先过 JSON Schema 校验, 失败也回填成 is_error 结果 (tools.py: ToolRegistry.execute)
const execute = (name, input) => {
  if (name === 'search_docs') return { content: DOC, is_error: false }
  if (!('expr' in input)) return { content: "INVALID_ARGS: missing required 'expr'; unexpected 'expression'", is_error: true }
  return { content: '16*64 = 1024', is_error: false }
}

const sim = computed(() => {
  // 脚本化的"模型": 每个元素是一次模型调用想发出的 tool_use 列表
  const search = { name: 'search_docs', input: { query: 'kv cache' } }
  const bad = { name: 'calculator', input: { expression: '16*64' } }
  const good = { name: 'calculator', input: { expr: '16*64' } }
  const calc = toolError.value ? [[bad], [good]] : [[good]]
  const plan = parallel.value ? [[search, ...calc[0]], ...calc.slice(1)] : [[search], ...calc]

  const msgs = [{ role: 'user', blocks: [{ type: 'text', text: '查一下 kv cache 文档, 再算 16*64' }] }]
  const ctxAt = [] // 每次模型调用时的上下文 token 数
  const ctx = () => est(SYSTEM) + msgs.reduce((s, m) => s + m.blocks.reduce((t, b) => t + est(flat(b)), 0), 0)
  let id = 0, forced = true
  // ★ 这就是 Agent.run: for turn in range(max_turns): 问模型 → final 则返回, 否则执行工具并回填
  for (let turn = 0; turn < maxTurns.value; turn++) {
    ctxAt.push(ctx())
    const uses = plan[turn]
    if (!uses) {
      msgs.push({ role: 'assistant', call: ctxAt.length, blocks: [{ type: 'text', text: 'kv cache 保存历史 K/V 让 decode 增量进行; 16×64 = 1024。' }] })
      forced = false
      break
    }
    const blocks = uses.map((u) => ({ type: 'tool_use', id: `toolu_${String(++id).padStart(4, '0')}`, ...u }))
    msgs.push({ role: 'assistant', call: ctxAt.length, blocks })
    msgs.push({ role: 'user', blocks: blocks.map((b) => ({ type: 'tool_result', tool_use_id: b.id, ...execute(b.name, b.input) })) })
  }
  if (forced) msgs.push({ role: 'assistant', blocks: [{ type: 'text', text: 'stopped: max_turns reached' }] })
  return { msgs, ctxAt, forced, final: ctx() }
})

// 帧 = 逐个 block 揭开
const frames = computed(() => sim.value.msgs.flatMap((m, i) => m.blocks.map((_, j) => [i, j])))
const stepper = useStepper(() => frames.value.length, { interval: 900 })
watch(sim, () => { stepper.pause(); stepper.step.value = frames.value.length - 1 }, { immediate: true })

const visible = computed(() => {
  const [mi, bi] = frames.value[stepper.step.value] || [0, 0]
  return sim.value.msgs.slice(0, mi + 1).map((m, i) => (i < mi ? m : { ...m, blocks: m.blocks.slice(0, bi + 1) }))
})
const done = computed(() => stepper.step.value >= frames.value.length - 1)
const now = computed(() => {
  const calls = visible.value.filter((m) => m.call).length
  const ctxs = sim.value.ctxAt.slice(0, calls)
  const ctx = est(SYSTEM) + visible.value.reduce((s, m) => s + m.blocks.reduce((t, b) => t + est(flat(b)), 0), 0)
  return { calls, ctx, input: ctxs.reduce((a, b) => a + b, 0) }
})
</script>

<style scoped>
.msgs { display: flex; flex-direction: column; gap: 8px; min-height: 320px; }
.msg { border: 1px solid var(--border); border-left-width: 3px; border-radius: var(--radius-sm); padding: 6px 8px; display: flex; flex-direction: column; gap: 4px; }
.msg.user { border-left-color: var(--eye); }
.msg.assistant { border-left-color: var(--accent); }
.role { font-size: 11px; color: var(--text-dim); }
.role em { font-style: normal; color: var(--accent); }
.block { text-align: left; font-size: 11px; line-height: 1.5; min-height: 0; padding: 4px 6px; white-space: pre-wrap; word-break: break-all; background: var(--bg-elev); border: 1px solid var(--border); border-radius: 4px; color: var(--text); cursor: default; }
.block.tool_use { border-color: var(--accent); }
.block.tool_result { border-color: var(--left); }
.block.err { border-color: var(--danger); color: var(--danger); }
.block.linked { outline: 2px solid var(--warn); outline-offset: 1px; }
</style>
