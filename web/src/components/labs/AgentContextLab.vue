<!--
  上下文预算 (对应 llm_agent/core/memory.py 的三档瘦身 + agent.py:_assemble_context / _compact)。
  只讲一件事: 同样是"塞不下了", 截断 / 清工具结果 / 模型摘要 丢掉的信息完全不同。
-->
<template>
  <LabFrame
    title="上下文预算 — 塞不下的时候, 丢什么?"
    sub="一段 14 条消息的会话, 里面埋了 6 个之后还要用的事实。拖预算滑杆, 切换策略, 看堆叠条怎么缩、右边哪些事实活了下来。点任意一条消息, 看它在模型眼里变成了什么样。"
    module="llm_agent/m14"
    run="python -m llm_agent.m14_context_engineering.demo"
    :challenge="{
      ask: '预算 6000。先猜: 「硬截断」和「清旧工具结果」哪个留下的事实多? 被清掉的工具结果里的事实, 和被截断丢掉的事实, 有什么本质区别?',
      answer: '清工具结果留下的多, 而且丢的那几条是「可重取」的: tool_use 块原样保留, 模型还知道当时读的是哪个文件, 再调一次工具就回来了 —— 这就是 just-in-time 检索: 上下文里只留「指针」, 不留「内容」。硬截断丢的是对话本身 (中途的决定), 还把 tool_use/tool_result 拍平成文本, 没处可取。摘要最省空间, 但它是有损的、要多花一次模型调用, 行号、数值这类细节要靠 PreCompact hook 点名保留。真实的 loop 是级联: 先清 (零成本), 不够再摘要。',
    }"
  >
    <template #controls>
      <div class="row">
        <button v-for="s in STRATS" :key="s.id" type="button" :class="{ active: strat === s.id }" @click="strat = s.id">{{ s.label }}</button>
      </div>
      <LabSlider v-model="budget" label="上下文预算" :min="2000" :max="14000" :step="500" unit=" tok" />
      <LabSlider v-model="keep" label="保留最近几个工具结果" :min="0" :max="3" />
      <div class="row">
        <button type="button" :class="{ active: hook }" :aria-pressed="hook" @click="hook = !hook">{{ hook ? '☑' : '☐' }} PreCompact hook: 摘要必须保留文件位置与失败用例</button>
        <button type="button" :class="{ active: memory }" :aria-pressed="memory" @click="memory = !memory">{{ memory ? '☑' : '☐' }} memory 工具: 把决定写进 /memories</button>
      </div>
    </template>

    <div class="stack" :title="`预算 ${budget}`">
      <div v-for="s in segs" :key="s.k" class="seg" :class="s.k" :style="{ width: s.v / scale * 100 + '%' }" :title="`${s.label} ${s.v}`" />
      <div class="mark" :style="{ left: budget / scale * 100 + '%' }"><span class="mono">预算</span></div>
    </div>
    <p class="legend"><span v-for="s in segs" :key="s.k"><i :class="s.k" />{{ s.label }} {{ fmtNum(s.v) }}</span></p>

    <div class="msgs">
      <button v-for="m in view.msgs" :key="m.i" type="button" class="m" :class="[m.kind, m.change, { sel: sel === m.i }]" @click="sel = m.i">
        <span class="mono idx">{{ m.i }}</span>
        <span class="bar" :style="{ width: Math.max(2, m.after / 2600 * 100) + '%' }" />
        <span class="mono tk">{{ m.after }}<em v-if="m.after !== m.tok"> ← {{ m.tok }}</em></span>
      </button>
    </div>
    <p class="detail"><b class="mono">msg[{{ sel }}] {{ cur.role }}</b> {{ cur.text }}<br /><span class="now">模型现在看到: {{ cur.seen }}</span></p>

    <template #stats>
      <div class="kv"><span>压缩后 / 预算</span><b :class="view.total > budget ? 'bad' : 'good'">{{ fmtNum(view.total) }} / {{ fmtNum(budget) }}</b></div>
      <div class="kv"><span>事实: 保留 · 可重取 · 丢失</span><b :class="count.lost ? 'bad' : 'good'">{{ count.kept }} · {{ count.refetch }} · {{ count.lost }}</b></div>
      <div class="kv"><span>tool_use/result 配对</span><b :class="view.paired ? 'good' : 'bad'">{{ view.paired ? '完好' : '拍平成文本' }}</b></div>
      <div class="kv"><span>额外模型调用</span><b>{{ view.calls }}</b></div>
      <ul class="facts">
        <li v-for="f in facts" :key="f.id" :class="f.status"><button type="button" @click="sel = f.msg">{{ ICON[f.status] }} {{ f.label }} <span class="mono">msg[{{ f.msg }}]</span></button></li>
      </ul>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { fmtNum, sum } from '@/utils/labmath.js'

const STRATS = [{ id: 'none', label: '什么都不做' }, { id: 'truncate', label: '硬截断 (反例)' }, { id: 'clear', label: '清旧工具结果' }, { id: 'summary', label: '模型摘要' }, { id: 'cascade', label: '级联: 先清, 不够再摘要 (真实 loop)' }]
const ICON = { kept: '✔', refetch: '↻', lost: '✘' }
const SYSTEM = 800, SKILLS = 360, MEM = 60, SUMMARY = 220, CLEARED = 6, GIST = 8 // token; 32 字符 ≈ 8 token

// kind: u=用户输入 a=assistant(text/tool_use) r=tool_result。 fact.off = 事实距消息开头多少 token
const MSGS = [
  { role: 'user', kind: 'u', tok: 40, text: '目标: 修复登录超时 bug, 不要改 public API。' },
  { role: 'assistant', kind: 'a', tok: 30, text: 'tool_use grep("timeout")' },
  { role: 'user', kind: 'r', tok: 1800, text: 'tool_result: grep 输出 1800 tok …… 超时常量在 auth/session.py:42 ……' },
  { role: 'assistant', kind: 'a', tok: 30, text: 'tool_use read_file("auth/session.py")' },
  { role: 'user', kind: 'r', tok: 2600, text: 'tool_result: session.py 全文 2600 tok' },
  { role: 'assistant', kind: 'a', tok: 80, text: '看完 session 与 refresh 流程、比较两种做法之后 —— 决定: 用方案 B (续期 token), 不改表结构。 tool_use edit(…)' },
  { role: 'user', kind: 'r', tok: 300, text: 'tool_result: edit ok' },
  { role: 'assistant', kind: 'a', tok: 30, text: 'tool_use run_tests()' },
  { role: 'user', kind: 'r', tok: 2200, text: 'tool_result: 测试日志 2200 tok …… test_refresh 失败: 期望 3600 实际 1800 ……' },
  { role: 'assistant', kind: 'a', tok: 30, text: 'tool_use read_file("tests/test_refresh.py")' },
  { role: 'user', kind: 'r', tok: 2400, text: 'tool_result: test_refresh.py 全文 2400 tok' },
  { role: 'user', kind: 'u', tok: 30, text: '顺便把超时改成可配置的。' },
  { role: 'assistant', kind: 'a', tok: 30, text: 'tool_use read_file("settings.toml")' },
  { role: 'user', kind: 'r', tok: 1500, text: 'tool_result: settings.toml 1500 tok …… 配置项放在 [auth] 段 ……' },
]
const FACTS = [
  { id: 'A', label: '目标与约束', msg: 0, off: 0 }, { id: 'B', label: '超时常量的文件:行号', msg: 2, off: 900, detail: true },
  { id: 'C', label: '中途的决定 (方案 B)', msg: 5, off: 20, decision: true }, { id: 'D', label: '失败用例与数值', msg: 8, off: 1200, detail: true },
  { id: 'E', label: '用户最新指令', msg: 11, off: 0 }, { id: 'F', label: '配置项位置', msg: 13, off: 300 },
]
const LAST_USER = 11 // 当前用户轮的起点: 摘要只压它之前的历史, 当前轮原样保留

const strat = ref('cascade'), budget = ref(6000), keep = ref(1), hook = ref(false), memory = ref(false), sel = ref(5)
const base = computed(() => SYSTEM + SKILLS + (memory.value ? MEM : 0))
const full = computed(() => base.value + sum(MSGS.map((m) => m.tok)))

const view = computed(() => {
  const room = budget.value - base.value
  let msgs = MSGS.map((m, i) => ({ ...m, i, after: m.tok, change: '', seen: '原样' }))
  let paired = true, calls = 0, summarized = false
  const total = () => sum(msgs.map((m) => m.after))
  // memory.py:clear_tool_results —— 只保留最近 keep 个结果的正文; tool_use 块和配对结构不动
  const clear = () => {
    const rs = msgs.filter((m) => m.kind === 'r' && m.change !== 'gone')
    rs.slice(0, Math.max(0, rs.length - keep.value)).forEach((m) => Object.assign(m, { after: CLEARED, change: 'cleared', seen: `[cleared: ${m.tok * 4} chars] (tool_use 还在, 可以重新调用)` }))
  }
  // agent.py:_compact —— 当前用户轮之前的历史换成一条模型写的摘要
  const compact = () => {
    msgs.forEach((m) => { if (m.i < LAST_USER) Object.assign(m, { after: m.i === 0 ? SUMMARY : 0, change: m.i === 0 ? 'summary' : 'gone', seen: m.i === 0 ? '[compact summary of 11 messages] 目标 / 已完成 / 关键结果 / 待办' : '(已并入摘要)' }) })
    calls = 1; summarized = true
  }
  if (full.value > budget.value) {
    if (strat.value === 'truncate') {
      // memory.py:truncate_messages —— 头 2 + 尾 2 各留 edge, 中间每条只留 32 字符
      const edge = Math.max(10, Math.floor(room / 6))
      msgs.forEach((m, i) => {
        const keepTok = i < 2 || i >= msgs.length - 2 ? Math.min(m.tok, edge) : Math.min(m.tok, GIST)
        Object.assign(m, { after: keepTok, change: keepTok < m.tok ? 'cut' : '', seen: keepTok < m.tok ? `只剩开头 ${keepTok} tok 的纯文本` : '拍平成纯文本' })
      })
      paired = false
    } else if (strat.value === 'clear') clear()
    else if (strat.value === 'summary') { compact(); clear() }
    else if (strat.value === 'cascade') { clear(); if (base.value + total() > budget.value) { compact(); clear() } } // ★ 由便宜到贵
  }
  return { msgs, total: base.value + total(), paired, calls, summarized }
})

const facts = computed(() => FACTS.map((f) => {
  const m = view.value.msgs[f.msg]
  let status = 'kept'
  if (m.change === 'cut') status = f.off < m.after ? 'kept' : 'lost'
  else if (m.change === 'cleared') status = 'refetch'
  else if (m.change === 'gone') status = f.detail ? (hook.value ? 'kept' : 'lost') : 'kept' // 摘要留得住目标和决定, 留不住行号和数值
  if (status === 'lost' && f.decision && memory.value) status = 'kept' // 写进 /memories 的东西在窗口之外, 任何压缩都碰不到
  return { ...f, status }
}))
const count = computed(() => Object.fromEntries(['kept', 'refetch', 'lost'].map((s) => [s, facts.value.filter((f) => f.status === s).length])))

const segs = computed(() => {
  const by = (pred) => sum(view.value.msgs.filter(pred).map((m) => m.after))
  return [
    { k: 'sys', label: 'system', v: SYSTEM }, { k: 'skl', label: 'skills 目录', v: SKILLS }, { k: 'mem', label: 'memory', v: memory.value ? MEM : 0 },
    { k: 'sum', label: '摘要', v: by((m) => m.change === 'summary') }, { k: 'txt', label: '对话', v: by((m) => m.kind !== 'r' && m.change !== 'summary') }, { k: 'res', label: '工具结果', v: by((m) => m.kind === 'r') },
  ].filter((s) => s.v > 0)
})
const scale = computed(() => Math.max(full.value, budget.value) * 1.02)
const cur = computed(() => view.value.msgs[sel.value])
</script>

<style scoped>
.stack { position: relative; display: flex; height: 22px; border: 1px solid var(--border); border-radius: 4px; margin-top: 14px; }
.seg { height: 100%; }
.sys, i.sys { background: var(--text-dim); } .skl, i.skl { background: var(--accent); } .mem, i.mem { background: var(--right); }
.sum, i.sum { background: var(--left); } .txt, i.txt { background: var(--eye); } .res, i.res { background: var(--warn); }
.mark { position: absolute; top: -12px; bottom: -4px; border-left: 2px dashed var(--danger); }
.mark span { position: absolute; top: -4px; left: 4px; font-size: 10px; color: var(--danger); white-space: nowrap; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 12px; font-size: 11px; color: var(--text-dim); margin: 6px 0 12px; }
.legend i { display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin-right: 4px; }
.msgs { display: flex; flex-direction: column; gap: 3px; }
.m { display: grid; grid-template-columns: 22px 1fr 110px; gap: 8px; align-items: center; min-height: 0; padding: 2px 6px; border: 1px solid transparent; border-radius: 4px; background: none; color: var(--text-muted); }
.m .idx { font-size: 10px; color: var(--text-dim); }
.m .bar { height: 12px; border-radius: 2px; background: var(--eye); }
.m.r .bar { background: var(--warn); } .m.summary .bar { background: var(--left); }
.m.cleared .bar, .m.cut .bar { opacity: 0.45; } .m.gone { opacity: 0.35; }
.m .tk { font-size: 10px; text-align: right; } .m .tk em { font-style: normal; color: var(--text-dim); }
.m.sel { border-color: var(--accent); }
.detail { margin-top: 10px; font-size: 12px; color: var(--text-muted); line-height: 1.7; min-height: 64px; }
.detail b { color: var(--text); font-weight: 500; } .now { color: var(--accent); }
.facts { list-style: none; display: flex; flex-direction: column; gap: 2px; }
.facts button { min-height: 0; padding: 2px 4px; border: 0; background: none; font-size: 12px; text-align: left; color: inherit; }
.facts .kept { color: var(--left); } .facts .refetch { color: var(--warn); } .facts .lost { color: var(--danger); }
.facts .mono { font-size: 10px; color: var(--text-dim); }
</style>
