<!--
  Plan 模式 (对应 llm_agent/core/tools.py:TodoWriteTool / ExitPlanModeTool 与 permissions.py:_evaluate_one 的 plan 分支)。
  只讲一件事: "批准之前只许看不许动"是权限门强制的, 不是 prompt 里求模型自觉。
-->
<template>
  <LabFrame
    title="Plan 模式 — 只读是门强制的, 不是模型自觉的"
    sub="模型先用 todo_write 把计划写成显式状态, 再用 exit_plan_mode 提交。点左边的按钮扮演用户: 拒绝或批准。点右边任意一个工具, 看它在当前模式下被怎么裁决、裁决走的哪条分支。三个开关分别演示: 一条 allow 规则在 plan 模式下为什么不管用, 以及把 delegate 错标成只读会捅出多大的洞。"
    module="llm_agent/m10"
    run="python -m llm_agent.m10_planning.demo"
    :challenge="{
      ask: '打开「配一条 allow write_note 规则」, 停在 plan 模式 —— write_note 会被放行吗? 再打开「把 delegate 标成只读」, 计划还没批准, 磁盘上为什么已经多了一条笔记?',
      answer: 'allow 规则不放行。_evaluate_one 按 deny → ask → allow 的顺序查规则, 但 plan 模式下处理完 deny 就 break, ask 和 allow 规则根本不看; 否则一条早先配好的 allow 就能让 plan 模式形同虚设 (deny 规则在 plan 模式下照常生效, 连只读工具也能被拒)。delegate 那个洞更隐蔽: 它的 read_only 是工具作者自己声明的类属性, harness 无法验证, 标错就是漏洞。子 agent 跑的是自己那扇 auto 模式的门, 里面的 write_note 是 medium 风险, 直接放行 —— 于是「批准前零写入」的承诺被一层委托绕开了。所以 core/subagents.py 里 DelegateTool 显式写了 read_only = False, 并且在注释里说明了原因。',
    }"
  >
    <template #controls>
      <div class="row">
        <button v-for="s in STAGES" :key="s.id" type="button" :class="{ active: stage === s.id }" @click="stage = s.id">{{ s.label }}</button>
      </div>
      <div class="row">
        <button type="button" :class="{ active: allowRule }" :aria-pressed="allowRule" @click="allowRule = !allowRule">{{ allowRule ? '☑' : '☐' }} 配一条 allow write_note 规则</button>
        <button type="button" :class="{ active: fakeRO }" :aria-pressed="fakeRO" @click="fakeRO = !fakeRO">{{ fakeRO ? '☑' : '☐' }} 把 delegate 标成只读 (漏洞)</button>
        <button type="button" :class="{ active: human }" :aria-pressed="human" @click="human = !human">{{ human ? '☑' : '☐' }} 有人审批高风险工具</button>
      </div>
    </template>

    <div class="panes">
      <section class="todo">
        <h4>todo_write 的最新一版 <em>整表覆写, 不做增量 patch</em></h4>
        <div v-for="(t, i) in todos" :key="i" class="todo-row">
          <span class="pill" :class="t.status">{{ STATUS[t.status] }}</span><span>{{ t.content }}</span>
        </div>
        <p class="plan mono">exit_plan_mode(plan="1. search_docs 2. write_note") → {{ stageNote }}</p>
      </section>

      <section class="tools">
        <h4>工具 <em>当前 gate.mode = {{ mode }}</em></h4>
        <button
          v-for="t in TOOLS" :key="t.name" type="button" class="trow" :class="[judge(t).ok ? 'ok' : 'deny', { sel: sel === t.name }]"
          @click="sel = t.name"
        >
          <span class="mono tn">{{ t.name }}</span>
          <span class="meta mono">read_only={{ ro(t) }} · risk={{ t.risk }}</span>
          <span class="mono vd">{{ judge(t).ok ? 'allow' : 'deny' }}</span>
        </button>
      </section>
    </div>

    <div class="trace">
      <b class="mono">{{ sel }}</b> 的裁决路径:
      <span v-for="(s, i) in trace" :key="i" class="tr" :class="s.state">{{ s.label }}</span>
    </div>

    <template #stats>
      <div class="kv"><span>gate.mode</span><b :class="mode === 'plan' ? '' : 'good'">{{ mode }}</b></div>
      <div class="kv"><span>放行 / 拒绝</span><b>{{ counts.ok }} / {{ counts.no }}</b></div>
      <div class="kv"><span>直接写进的笔记</span><b :class="notes ? 'bad' : 'good'">{{ notes }}</b></div>
      <div class="kv"><span>经子 agent 写进的笔记</span><b :class="viaChild ? 'bad' : 'good'">{{ viaChild }}</b></div>
      <p class="lab-note">
        todo_write 标成只读, 是因为它只改 agent 自己的计划状态, 不碰外部世界 —— 否则"先列个计划"这一步本身就会被 plan 模式拒掉。
        真正翻转模式的动作发生在 ExitPlanModeTool.execute 里: 人点了同意, 它才去改 gate.mode, 模型没有别的路径能自己改。
      </p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'

const STAGES = [
  { id: 'plan', label: 'plan 模式 · 计划待批' },
  { id: 'rejected', label: '用户拒绝计划' },
  { id: 'approved', label: '用户批准 → accept_edits' },
]
const STATUS = { pending: '待办', in_progress: '进行中', completed: '已完成' }
// 元数据全部取自 core/tools.py 的类属性
const TOOLS = [
  { name: 'todo_write', ro: true, risk: 'low' },
  { name: 'exit_plan_mode', ro: true, risk: 'low' },
  { name: 'search_docs', ro: true, risk: 'low' },
  { name: 'read_notes', ro: true, risk: 'low' },
  { name: 'write_note', ro: false, risk: 'medium' },
  { name: 'delegate', ro: false, risk: 'low' },
  { name: 'shell', ro: false, risk: 'high' },
]

const stage = ref('plan')
const allowRule = ref(false)
const fakeRO = ref(false)
const human = ref(false)
const sel = ref('write_note')

const mode = computed(() => (stage.value === 'approved' ? 'accept_edits' : 'plan'))
const ro = (t) => (t.name === 'delegate' && fakeRO.value ? true : t.ro)

// ★ 照搬 permissions.py:_evaluate_one 的分支顺序
const judge = (t) => {
  if (mode.value === 'plan') {
    // deny 规则先看 (本实验没配 deny 规则), 然后直接 break —— ask / allow 规则一概不查
    return ro(t)
      ? { ok: true, why: 'plan: read-only tool' }
      : { ok: false, why: 'plan: plan mode is read-only until the plan is approved' }
  }
  if (allowRule.value && t.name === 'write_note') return { ok: true, why: 'rule: allow write_note' }
  if (t.risk === 'high') {
    return human.value
      ? { ok: true, why: 'human: high-risk tool still needs approval; approved' }
      : { ok: false, why: 'human: 无人可问, fail closed' }
  }
  return { ok: true, why: 'accept_edits: low/medium risk' }
}

const trace = computed(() => {
  const t = TOOLS.find((x) => x.name === sel.value) || TOOLS[0]
  const plan = mode.value === 'plan'
  const hitAllow = allowRule.value && t.name === 'write_note'
  return [
    { label: 'deny 规则: 未命中', state: 'pass' },
    plan
      ? { label: 'plan 模式 → break, 不查 ask / allow 规则', state: hitAllow ? 'kill' : 'pass' }
      : { label: hitAllow ? 'allow 规则: 命中 write_note' : 'ask / allow 规则: 未命中', state: hitAllow ? 'ok' : 'pass' },
    { label: plan ? `模式兜底: read_only=${ro(t)}` : `模式兜底: accept_edits, risk=${t.risk}`, state: 'pass' },
    { label: judge(t).why, state: judge(t).ok ? 'ok' : 'kill' },
  ]
})

const counts = computed(() => {
  const ok = TOOLS.filter((t) => judge(t).ok).length
  return { ok, no: TOOLS.length - ok }
})
const todos = computed(() => {
  const s = stage.value === 'approved' ? 'completed' : 'pending'
  return [
    { content: 'search_docs — 需要外部知识', status: s },
    { content: 'write_note — 把结论落成笔记', status: s },
  ]
})
const stageNote = computed(() => ({
  plan: '等待用户答复; 在此之前写操作一律被拒',
  rejected: 'plan rejected by user; stay in plan mode (ok=False)',
  approved: 'plan approved; mode -> accept_edits',
}[stage.value]))

const notes = computed(() => (stage.value === 'approved' ? 1 : 0))
// 委托被错标成只读 → plan 模式放行 delegate → 子 agent 用自己的 auto 门写下了笔记
const viaChild = computed(() => (mode.value === 'plan' && fakeRO.value ? 1 : 0))
</script>

<style scoped>
.panes { display: grid; grid-template-columns: minmax(230px, 1fr) minmax(280px, 1.2fr); gap: 14px; min-width: 520px; }
.panes h4 { font-size: 12px; color: var(--text-muted); font-weight: 500; margin-bottom: 8px; }
.panes h4 em { font-style: normal; color: var(--text-dim); font-size: 11px; }
.todo-row { display: flex; gap: 8px; align-items: center; font-size: 12px; color: var(--text-muted); padding: 5px 0; border-bottom: 1px dashed var(--border); }
.pill { font-size: 10px; padding: 1px 6px; border-radius: 10px; border: 1px solid var(--border-strong); color: var(--text-dim); white-space: nowrap; }
.pill.completed { border-color: var(--left); color: var(--left); }
.plan { margin-top: 10px; font-size: 10.5px; line-height: 1.6; color: var(--text-dim); background: var(--bg-elev); border: 1px solid var(--border); border-radius: 3px; padding: 6px; }
.trow { display: grid; grid-template-columns: 110px 1fr 48px; gap: 8px; align-items: center; width: 100%; text-align: left; font-size: 11px; padding: 6px; margin-bottom: 4px; min-height: 0; background: var(--bg-elev); border: 1px solid var(--border); border-left-width: 3px; border-radius: 3px; color: var(--text-muted); }
.trow.ok { border-left-color: var(--left); }
.trow.deny { border-left-color: var(--danger); }
.trow.sel { outline: 2px solid var(--accent); outline-offset: 1px; }
.trow .tn { color: var(--text); }
.trow .meta { font-size: 10px; color: var(--text-dim); }
.trow .vd { text-align: right; }
.trow.ok .vd { color: var(--left); }
.trow.deny .vd { color: var(--danger); }
.trace { margin-top: 12px; font-size: 12px; color: var(--text-muted); display: flex; flex-wrap: wrap; gap: 6px; align-items: center; line-height: 1.7; }
.tr { font-size: 11px; padding: 2px 7px; border-radius: 3px; border: 1px solid var(--border); }
.tr.pass { color: var(--text-dim); }
.tr.ok { border-color: var(--left); color: var(--left); }
.tr.kill { border-color: var(--danger); color: var(--danger); }
@media (max-width: 900px) { .panes { grid-template-columns: 1fr; min-width: 0; } }
</style>
