<!--
  token 级语法约束实验台 (对应 llm_infer/m14_structured_output/grammar.py:token_row / compile_token_table)。
  一件事: 真实 token 是多字符的, 一个 token 可能一口气跨过几个语法状态 —— 所以要把 "状态 × token → 落点" 离线编译成表。
-->
<template>
  <LabFrame
    title="结构化输出 — 把 FSM 预编译成 token mask 表"
    sub='迷你 JSON 语法: {"key":value, …}, value = 字符串 | 数字 | true | false | null。灰掉的 token 在当前状态下非法 (logit 会被置 −inf)。点一个合法 token 前进; 悬停一个多字符 token, 看它在状态图上一口气走过哪几个状态。'
    module="llm_infer/m14"
    run="python -m llm_infer.m14_structured_output.demo"
    :challenge="{
      ask: '重置后, 用最少的 token 拼出 {&quot;id&quot;:42}。需要几个? 再勾上「只用单字符词表」拼同一个串。为什么 true 这个 token 在「等 value」状态合法, 而 tru 也合法、}} 却永远非法?',
      answer: '多字符词表 5 个 token: {&quot; → id → &quot;: → 42 → }, 再加 EOS (其中 &quot;: 一口气跨了 2 个状态); 单字符要 9 个。合法性的定义是「整段字符都能沿 DFA 走通」。true 从 value? 走 4 步落到「, 或 }」。tru 走 3 步停在字面量中间, 之后只能接 e。}} 的第二个 } 在结束态无路可走, 对所有状态都是 −1。这张 S×V 表离线算一次, 在线每步只查一行 —— 否则每步要对全词表逐字符试走, 纯 CPU 开销卡在 GPU 前向和采样之间。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :disabled="!hist.length" @click="hist.pop()">撤销</button>
        <button type="button" :disabled="!hist.length" @click="hist.splice(0)">重置</button>
        <button type="button" :disabled="finished" @click="randomStep">随机合法一步</button>
        <button type="button" :class="{ active: singleOnly }" :aria-pressed="singleOnly" @click="singleOnly = !singleOnly; hist.splice(0)">只用单字符词表</button>
      </div>
      <p class="out mono" aria-live="polite">输出: <b>{{ output || '∅' }}</b><span v-if="finished" class="okmsg"> ⏹ EOS · JSON.parse ✓ → {{ parsed }}</span></p>
    </template>

    <svg viewBox="0 0 560 165" role="img" :aria-label="`语法状态图, 当前状态 ${GROUPS[curGroup].label}`">
      <line v-for="(e, k) in EDGES" :key="k" :x1="GROUPS[e[0]].x" :y1="GROUPS[e[0]].y" :x2="GROUPS[e[1]].x" :y2="GROUPS[e[1]].y" class="edge" />
      <g v-for="(g, k) in GROUPS" :key="k" :class="['st', { cur: k === curGroup, via: hoverPath.includes(k) }]">
        <rect :x="g.x - 30" :y="g.y - 13" width="60" height="26" rx="6" />
        <text :x="g.x" :y="g.y + 4">{{ g.label }}</text>
        <text v-if="hoverPath.includes(k)" :x="g.x + 26" :y="g.y - 8" class="ord">{{ hoverPath.indexOf(k) + 1 }}</text>
      </g>
    </svg>
    <p class="hint mono">{{ hoverMsg }}</p>

    <div class="vocab">
      <button v-for="t in shown" :key="t" type="button" class="tokbtn mono" :class="{ multi: VOCAB[t].length > 1 && t !== EOS_ID, junk: JUNK.includes(VOCAB[t]) }"
        :aria-disabled="!ok(t)" @click="ok(t) && hist.push(t)"
        @mouseenter="hover = t" @mouseleave="hover = -1" @focus="hover = t" @blur="hover = -1">{{ show(VOCAB[t]) }}</button>
    </div>

    <template #stats>
      <div class="kv"><span>当前状态合法 token</span><b>{{ finished ? 0 : legal.length }} / {{ shown.length }}</b></div>
      <div class="kv"><span>上一个 token 跨过的状态数</span><b :class="lastCross > 1 ? 'good' : ''">{{ lastCross }}</b></div>
      <div class="kv"><span>已用 token / 已出字符</span><b>{{ hist.length }} / {{ output.length }}</b></div>
      <div class="kv"><span>预编译表 S × V</span><b>{{ STATES.length }} × {{ VOCAB.length }}</b></div>
      <p class="lab-note">
        在线每步: 查表一行 O(1)。不预编译: 每步对全词表逐字符试走 = {{ totalChars }} 次转移 (真实词表 V≈10⁵ 时是几十万次, 纯 CPU)。
        EOS 只在结束态合法, 所以输出一定是完整 JSON, 不靠模型自觉。
      </p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, reactive, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import { mulberry32 } from '@/utils/labmath.js'

// ---- 字符级 DFA: step(状态名, 字符) → 状态名 | null。字面量 / 长度上限都需要自己的状态 ----
const LIT = { t: 'true', f: 'false', n: 'null' }
const az = (c) => c >= 'a' && c <= 'z', dg = (c) => c >= '0' && c <= '9'
const step = (s, c) => {
  const [kind, a, b] = s.split(':')
  if (kind === 'open') return c === '{' ? 'keyq' : null
  if (kind === 'keyq') return c === '"' ? 'key:0' : null
  if (kind === 'key' || kind === 'str') {                       // [a-z]{1,6} 再闭引号
    if (az(c) && +a < 6) return `${kind}:${+a + 1}`
    return c === '"' && +a > 0 ? (kind === 'key' ? 'colon' : 'after') : null
  }
  if (kind === 'colon') return c === ':' ? 'val' : null
  if (kind === 'val') return c === '"' ? 'str:0' : c >= '1' && c <= '9' ? 'num:1' : LIT[c] ? `lit:${LIT[c]}:1` : null
  if (kind === 'lit') return c !== a[+b] ? null : +b + 1 === a.length ? 'after' : `lit:${a}:${+b + 1}`
  if (kind === 'num' && dg(c)) return +a < 4 ? `num:${+a + 1}` : null
  if (kind === 'num' || kind === 'after') return c === ',' ? 'keyq' : c === '}' ? 'done' : null
  return null                                                   // done: 只有 EOS
}
const SINGLE = [...'{}":,adefgilmnrstu0124'], JUNK = ['[', '}}', '\n', "'", '::']
const MULTI = ['{"', '":', '",', '"}', '":"', '","', 'e":', 'd":', '1}', '2}', 'true', 'false', 'null', 'tru', 'name', 'age', 'id', '42', '10']
const VOCAB = ['<EOS>', ...SINGLE, ...MULTI, ...JUNK], EOS_ID = 0
const STATES = (() => { const seen = ['open']; for (const s of seen) for (const c of SINGLE) { const n = step(s, c); if (n && !seen.includes(n)) seen.push(n) } return seen })()
const DONE = STATES.indexOf('done')

// ★ token_row: 从状态 s 逐字符试走 token; 全走通 → 记落点, 否则 −1。对每个状态各跑一遍 = 预编译
const walk = (s, piece) => { const path = []; let cur = STATES[s]; for (const c of piece) { cur = step(cur, c); if (!cur) return null; path.push(STATES.indexOf(cur)) } return path }
const table = STATES.map((_, s) => VOCAB.map((p, t) => (t === EOS_ID ? (s === DONE ? DONE : -1) : walk(s, p)?.at(-1) ?? -1)))
const totalChars = VOCAB.slice(1).reduce((a, p) => a + p.length, 0)

// ---- 状态图: 把 DFA 状态归并成 10 个展示状态 ----
const GROUPS = [
  { id: 'open', label: '等 {', x: 40, y: 30 }, { id: 'keyq', label: '等 key "', x: 130, y: 30 }, { id: 'key', label: 'key 中', x: 220, y: 30 },
  { id: 'colon', label: '等 :', x: 310, y: 30 }, { id: 'val', label: 'value?', x: 400, y: 30 }, { id: 'str', label: '字符串中', x: 515, y: 30 },
  { id: 'num', label: '数字中', x: 515, y: 85 }, { id: 'lit', label: '字面量中', x: 515, y: 140 }, { id: 'after', label: ', 或 }', x: 310, y: 140 }, { id: 'done', label: '结束', x: 130, y: 140 },
]
const EDGES = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [4, 6], [4, 7], [5, 8], [7, 8], [6, 8], [8, 1], [8, 9], [6, 9]]
const groupOf = (s) => GROUPS.findIndex((g) => g.id === STATES[s].split(':')[0])
const dedupe = (xs) => xs.filter((x, i) => x !== xs[i - 1])

const hist = reactive([]), singleOnly = ref(false), hover = ref(-1)
let rand = mulberry32(7)
const states = computed(() => hist.reduce((acc, t) => [...acc, table[acc.at(-1)][t]], [0]))   // 每步查表前进
const state = computed(() => states.value.at(-1))
const finished = computed(() => hist.at(-1) === EOS_ID)
const curGroup = computed(() => groupOf(state.value))
const shown = computed(() => VOCAB.map((_, t) => t).filter((t) => !singleOnly.value || VOCAB[t].length === 1 || t === EOS_ID))
const legal = computed(() => shown.value.filter((t) => table[state.value][t] >= 0))
const output = computed(() => hist.filter((t) => t !== EOS_ID).map((t) => VOCAB[t]).join(''))
const parsed = computed(() => { try { return JSON.stringify(JSON.parse(output.value)) } catch { return '✗ 解析失败' } })
const crossed = (s, t) => (t === EOS_ID ? [] : dedupe((walk(s, VOCAB[t]) || []).map(groupOf)).filter((g, i) => i > 0 || g !== groupOf(s)))
const lastCross = computed(() => (hist.length ? crossed(states.value.at(-2), hist.at(-1)).length : 0))
const hoverPath = computed(() => (hover.value > 0 && !finished.value ? crossed(state.value, hover.value) : []))
const ok = (t) => !finished.value && table[state.value][t] >= 0   // aria-disabled 而非 disabled: 非法 token 也要能悬停看原因
const show = (p) => (p === '\n' ? '\\n' : p)
const hoverMsg = computed(() => {
  if (hover.value < 0) return `当前状态: ${GROUPS[curGroup.value].label} (DFA 状态 #${state.value} ${STATES[state.value]})`
  const p = show(VOCAB[hover.value])
  if (!ok(hover.value)) return `「${p}」在当前状态走不通 → next_state = −1 → logit 置 −inf`
  if (hover.value === EOS_ID) return 'EOS: 只在结束态合法'
  return hoverPath.value.length > 1 ? `「${p}」一口气跨了 ${hoverPath.value.length} 个状态: ${hoverPath.value.map((g) => GROUPS[g].label).join(' → ')}`
    : `「${p}」→ ${hoverPath.value.length ? GROUPS[hoverPath.value[0]].label : '留在 ' + GROUPS[curGroup.value].label}`
})
const randomStep = () => { const c = legal.value; if (c.length) hist.push(c[Math.floor(rand() * c.length)]) }
</script>

<style scoped>
/* 窄屏: 保住可读的最小宽度, 由 .lab-viz 横向滚动; 只有 .draggable 把手拦截触摸 */
svg { min-width: 500px; touch-action: pan-x pan-y; }
.out { font-size: 13px; color: var(--text-muted); word-break: break-all; }
.out b { color: var(--text); font-weight: 500; }
.okmsg { color: var(--left); }
.edge { stroke: var(--border-strong); stroke-width: 1; }
.st rect { fill: var(--bg-elev); stroke: var(--border-strong); }
.st text { font-size: 10px; fill: var(--text-muted); text-anchor: middle; }
.st.via rect { stroke: var(--warn); stroke-width: 2; fill: color-mix(in srgb, var(--warn) 18%, var(--bg-elev)); }
.st.cur rect { stroke: var(--accent); stroke-width: 2.5; fill: var(--accent-soft); }
.st.cur text, .st.via text { fill: var(--text); }
.st .ord { font-size: 9px; fill: var(--warn); font-weight: 700; }
.hint { font-size: 11px; color: var(--text-muted); min-height: 2.6em; line-height: 1.5; margin: 4px 0 8px; }
.vocab { display: flex; flex-wrap: wrap; gap: 5px; }
.tokbtn { min-width: 34px; min-height: 30px; padding: 2px 8px; font-size: 12px; }
.tokbtn.multi:not([aria-disabled='true']) { border-color: var(--left); }
.tokbtn[aria-disabled='true'] { opacity: 0.28; cursor: not-allowed; }
.tokbtn.junk { text-decoration: line-through; }
</style>
