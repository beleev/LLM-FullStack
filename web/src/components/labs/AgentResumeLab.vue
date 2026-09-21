<!--
  Resume 与隔离 (对应 llm_agent/core/persistence.py:JsonlSessionStore、agent.py 的 resume 分支、subagents.py:DelegateTool)。
  只讲一件事: 磁盘上有, 不等于回到上下文里 —— resume 只带回消息, 权限和子 transcript 都不回来。
-->
<template>
  <LabFrame
    title="Resume — 哪些东西回来了, 哪些没有"
    sub="会话 A 搜了一次文档就'关掉进程'。点任意一行 JSONL 看它 resume 时的下场。打开三个故障开关制造崩溃现场。换一套新会话的权限门, 看同一句'写进笔记'的裁决怎么变。下半部分: 拖动子 agent 读几份文档, 看子 transcript 涨到多大, lead 上下文却纹丝不动。"
    module="llm_agent/m06 · m07"
    run="python -m llm_agent.m06_persistence_resume.demo"
    :challenge="{
      ask: '三个故障开关全打开, 磁盘上有 7 行。resume 出来几条消息? 再把新会话的权限门换成 default 且无人可问。上一次明明放行过 write_note, 这次为什么被拒?',
      answer: '5 条。写了一半的第 7 行 json.loads 会抛 JSONDecodeError, load() 直接跳过它: 坏一行不该让整个会话无法恢复。第 6 行是崩在 tool_use 和 tool_result 之间留下的悬空调用, Agent.__init__ 主动把它丢掉。留着的话, 这段 transcript 发给真实 Messages API 会直接 400。权限被拒是因为 PermissionGate 根本不在 JSONL 里: 授权是当前运行环境的决定, 不是可以被文件恢复的状态。否则一个磁盘上的文件就能给自己提权。同理, hooks、skills、工具的外部副作用、子 agent 的 transcript 也都不会跟着 resume 回来。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: dangling }" :aria-pressed="dangling" @click="dangling = !dangling">{{ dangling ? '☑' : '☐' }} 崩在 tool_use 和 tool_result 之间</button>
        <button type="button" :class="{ active: halfLine }" :aria-pressed="halfLine" @click="halfLine = !halfLine">{{ halfLine ? '☑' : '☐' }} 最后一行只写了一半</button>
        <button type="button" :class="{ active: replay }" :aria-pressed="replay" @click="replay = !replay">{{ replay ? '☑' : '☐' }} resume 重放 session_start</button>
      </div>
      <div class="row">
        <span class="tip">新会话的权限门:</span>
        <button v-for="g in GATES" :key="g.id" type="button" class="mono" :class="{ active: gate === g.id }" @click="gate = g.id">{{ g.label }}</button>
      </div>
      <LabSlider v-model="docs" label="子 agent 读几份文档" :min="1" :max="5" />
    </template>

    <div class="cols">
      <section>
        <h4>磁盘 · session.jsonl <em>只追加, 从不就地改写</em></h4>
        <button
          v-for="r in disk" :key="r.id" type="button" class="line mono" :class="[r.fate, { sel: sel === r.id }]"
          @click="sel = sel === r.id ? -1 : r.id"
        ><span class="no">{{ r.id }}</span>{{ r.text }}</button>
      </section>
      <section>
        <h4>新进程 · resume 出来的 agent.messages</h4>
        <div v-for="(m, i) in resumed" :key="i" class="line mono" :class="{ dup: m.dup, fresh: m.fresh }">
          <span class="no">{{ i + 1 }}</span>{{ m.text }}
        </div>
        <div class="line mono verdict" :class="verdict.ok ? 'ok' : 'deny'">
          <span class="no">门</span>write_note → {{ verdict.text }}
        </div>
      </section>
    </div>

    <p class="detail">{{ detail.tip }}</p>
    <pre v-if="detail.raw" class="raw mono">{{ detail.raw }}</pre>

    <div class="iso">
      <h4>磁盘 · child_00_researcher.jsonl <em>落盘可审计, 但不属于 lead 的上下文</em></h4>
      <div class="barrow">
        <span class="bar child" :style="{ width: childTok / SCALE * 100 + '%' }" />
        <span class="mono lbl">{{ childTok }} tok · {{ 3 + docs }} 条消息</span>
      </div>
      <h4>lead 上下文里只有这一条 <em>摘要在 harness 侧硬截断到 200 字符</em></h4>
      <div class="barrow">
        <span class="bar lead" :style="{ width: LEAD / SCALE * 100 + '%' }" />
        <span class="mono lbl">{{ LEAD }} tok · 1 条摘要</span>
      </div>
    </div>

    <template #stats>
      <div class="kv"><span>磁盘行数</span><b>{{ disk.length }}</b></div>
      <div class="kv"><span>恢复出的消息</span><b :class="restored < disk.length ? 'bad' : ''">{{ restored }}</b></div>
      <div class="kv"><span>被跳过 / 被丢弃</span><b :class="halfLine || dangling ? 'bad' : 'good'">{{ +halfLine }} / {{ +dangling }}</b></div>
      <div class="kv"><span>子 transcript / lead 收到</span><b :class="childTok > 3 * LEAD ? 'bad' : ''">{{ childTok }} / {{ LEAD }} tok</b></div>
      <p class="lab-note">
        <b>回来了:</b> 消息 (含 tool_use / tool_result 配对)、tool_use id 的续号起点。<br />
        <b>没回来:</b> 权限模式 —— JSONL 里根本没有这个字段; hooks 和 skills —— 由新进程自己装; 工具在外面留下的改动 —— 存储管不着;
        子 agent 的 transcript —— 它在磁盘上, 但从来不属于 lead 的上下文。
      </p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'

const GATES = [
  { id: 'default', label: 'default · 无人可问' },
  { id: 'human', label: 'default · 有人审批' },
  { id: 'auto', label: 'auto' },
]
const replay = ref(false)
const halfLine = ref(false)
const dangling = ref(false)
const gate = ref('default')
const docs = ref(1)
const sel = ref(-1)

// 会话 A 写下的 5 行, 与 m06 demo 的 [raw jsonl] 逐行对应 (text 为缩写, raw 为原行)
const BASE = [
  { role: 'system', text: 'system  name=session_start  "Policy: answer in Chinese."', raw: '{"role":"system","content":"Policy: answer in Chinese.","name":"session_start"}' },
  { role: 'user', text: 'user  "搜索 agent loop"', raw: '{"role":"user","content":"搜索 agent loop","name":null}' },
  { role: 'assistant', use: true, text: 'assistant  [tool_use toolu_0001 search_docs]', raw: '{"role":"assistant","content":[{"type":"tool_use","id":"toolu_0001","name":"search_docs","input":{"query":"搜索 agent loop"}}],"name":null}' },
  { role: 'user', text: 'user  [tool_result toolu_0001 "agent_loop: …"]', raw: '{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_0001","content":"agent_loop: Agent loop uses messages, tools, permissions and persistence."}],"name":null}' },
  { role: 'assistant', text: 'assistant  "基于工具结果完成：agent_loop…"', raw: '{"role":"assistant","content":"基于工具结果完成：\\nsearch_docs: agent_loop: Agent loop uses messages, tools, permissions and persistence.","name":null}' },
]

const disk = computed(() => {
  const rows = BASE.map((r, i) => ({ ...r, id: i + 1, fate: 'restored' }))
  // 崩在"写下 tool_use"之后、"写下 tool_result"之前: 日志里留下一个没有结果的调用
  if (dangling.value) rows.push({ id: rows.length + 1, role: 'assistant', use: true, fate: 'dropped', text: 'assistant  [tool_use toolu_0002 write_note]  ← 崩在这里', raw: '{"role":"assistant","content":[{"type":"tool_use","id":"toolu_0002","name":"write_note","input":{"text":"…"}}],"name":null}' })
  // 进程被 kill 时正写到一半: 这一行不是合法 JSON
  if (halfLine.value) rows.push({ id: rows.length + 1, role: '?', broken: true, fate: 'skipped', text: '(半行) {"role":"user","content":[{"type":"tool_re', raw: '{"role":"user","content":[{"type":"tool_re' })
  return rows
})

const resumed = computed(() => {
  // load(): json.loads 抛 JSONDecodeError 的行跳过, 其余照单全收
  const kept = disk.value.filter((r) => !r.broken)
  // ★ Agent.__init__: 最后一条若是悬空的 tool_use, 丢掉 —— 否则这段 transcript 不是合法 API 输入
  if (kept.length && kept[kept.length - 1].use) kept.pop()
  const out = kept.map((r) => ({ text: r.text }))
  // session_start: 历史里已经有了就不再注入, 否则每次 resume 都会重复注入同一段
  if (replay.value) out.push({ dup: true, text: 'system  name=session_start  ← 又注入了一遍' })
  out.push({ fresh: true, text: 'user  "把刚才的结果写进笔记"  ← 新会话的第一句' })
  return out
})
// 真正从磁盘恢复出来的条数 (不算重放的 session_start 和新会话的第一句)
const restored = computed(() => resumed.value.length - 1 - (replay.value ? 1 : 0))

const verdict = computed(() => {
  if (gate.value === 'auto') return { ok: true, text: 'allow (auto: bounded local write) —— medium 风险直接放行' }
  if (gate.value === 'human') return { ok: true, text: 'allow (human: 重新问了一次, 人批准了)' }
  return { ok: false, text: 'deny (human: 无人可问, fail closed) —— 上次的"同意"不跟着 resume 回来' }
})

const FATE = {
  restored: '恢复进新会话的上下文。',
  dropped: '被丢掉。这是崩在 tool_use 与 tool_result 之间留下的悬空调用: 每个 tool_use 都必须有配对的 tool_result, 留着它整段 transcript 就是非法的。',
  skipped: '被跳过。json.loads 在这一行抛 JSONDecodeError, load() 吞掉异常继续读下一行 —— 坏一行不该让整个会话无法恢复。',
}
const detail = computed(() => {
  const r = disk.value.find((x) => x.id === sel.value)
  if (!r) return { tip: '点左边任意一行, 看它的原始 JSON 和 resume 时的下场。append-only 的意思是: 这些行写下去就永不改变, 出事故时能逐行复盘"模型当时要求了什么、harness 放行了什么"。', raw: '' }
  return { tip: `第 ${r.id} 行 (${r.role}): ${FATE[r.fate]}`, raw: r.raw }
})

// 子 agent: 自己的工具集、自己的上下文; 父级只拿一条硬截断到 200 字符的摘要
// docs = 1 时与 m07 demo 一致: child 峰值 279 tok, parent 峰值 107 tok
const LEAD = 107
const SCALE = 59 + 5 * 220
const childTok = computed(() => 59 + docs.value * 220)
</script>

<style scoped>
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; min-width: 520px; }
.cols > section { min-width: 0; }
.iso { margin-top: 16px; min-width: 300px; }
h4 { font-size: 12px; color: var(--text-muted); font-weight: 500; margin: 0 0 6px; }
h4 em { font-style: normal; color: var(--text-dim); font-size: 11px; }
.iso h4 { margin-top: 10px; }
.line { display: block; width: 100%; text-align: left; font-size: 10.5px; line-height: 1.5; padding: 4px 6px; margin-bottom: 3px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; background: var(--bg-elev); border: 1px solid var(--border); border-radius: 3px; color: var(--text-muted); min-height: 0; }
.line .no { display: inline-block; width: 20px; color: var(--text-dim); }
.line.restored { border-left: 3px solid var(--left); }
.line.dropped { border-left: 3px solid var(--danger); color: var(--danger); }
.line.skipped { border-left: 3px solid var(--warn); color: var(--warn); }
.line.sel { outline: 2px solid var(--accent); outline-offset: 1px; }
.line.dup { border-color: var(--warn); color: var(--warn); }
.line.fresh { border-color: var(--eye); color: var(--text); }
.line.verdict { white-space: normal; margin-top: 6px; }
.line.verdict.ok { border-color: var(--left); color: var(--text); }
.line.verdict.deny { border-color: var(--danger); color: var(--danger); }
.tip { font-size: 12px; color: var(--text-muted); }
.detail { margin-top: 10px; font-size: 12.5px; color: var(--text-muted); line-height: 1.7; }
.raw { margin-top: 6px; font-size: 10.5px; line-height: 1.6; color: var(--text-dim); background: var(--bg-elev); border: 1px solid var(--border); border-radius: 3px; padding: 6px 8px; white-space: pre-wrap; word-break: break-all; }
.barrow { display: flex; align-items: center; gap: 8px; }
.barrow .lbl { font-size: 11px; color: var(--text-muted); white-space: nowrap; }
.bar { height: 22px; min-width: 6px; border-radius: 3px; display: block; }
.bar.child { background: color-mix(in srgb, var(--warn) 30%, transparent); border: 1px solid var(--warn); }
.bar.lead { background: color-mix(in srgb, var(--left) 30%, transparent); border: 1px solid var(--left); }
@media (max-width: 900px) { .cols { grid-template-columns: 1fr; min-width: 0; } }
</style>
