<!--
  Orchestrator–workers 记账 (对应 llm_agent/core/subagents.py 的 DelegateTool + agent.py 的并行执行)。
  只讲一件事: 子智能体把"读了什么"留在自己的上下文里, lead 只收摘要 —— 换来干净的主上下文和并行, 代价要用 token 账本看。
-->
<template>
  <LabFrame
    title="Orchestrator–workers — 三本账: lead 上下文 / 总 token / 墙钟时间"
    sub="一个调研任务拆成 N 个子任务, 每个子任务要读几份长文档。比较「单 agent 全自己读」和「lead 扇出给 worker」。拖滑杆, 再点任意一个 worker 条, 看它读了多少、lead 实际收到多少。"
    module="llm_agent/m11"
    run="python -m llm_agent.m11_orchestrator.demo"
    :challenge="{
      ask: '初始状态就是 m11 demo 的情形 (3 个子任务、各读 1 份短文档): orchestrator 总 token 更多。现在把「读几份文档」和「每份文档」都拖大。orchestrator 的总 token 什么时候比单 agent 多, 什么时候反而少? 最后把「worker 多探索」拖到 3×。',
      answer: '活少、文档短时 orchestrator 更贵: 每个 worker 都要重建上下文 (system + 任务简报), 这份固定开销占了主导 —— m11 demo 就是这种情况 (770 vs 413)。活一多, 单 agent 的账是平方增长的: 每一轮都要重发此前读过的全部文档。而每个 worker 只重发自己那一份, 所以同等工作量下总 token 反而更少。但真实系统里 worker 会各自多翻几份资料 (Anthropic 报告多智能体约为普通聊天的 15× token), 把探索倍数拖上去总账又反超了。不变的只有两件事: lead 峰值上下文始终小得多, 并行的墙钟时间始终短得多 —— 这才是多智能体买到的东西。',
    }"
  >
    <template #controls>
      <div class="row">
        <button v-for="m in MODES" :key="m.id" type="button" :class="{ active: mode === m.id }" @click="mode = m.id">{{ m.label }}</button>
        <button type="button" @click="seed++">换一组耗时</button>
      </div>
      <LabSlider v-model="n" label="子任务数 N" :min="1" :max="8" />
      <LabSlider v-model="r" label="每个子任务读几份文档" :min="1" :max="5" />
      <LabSlider v-model="t" label="每份文档 (tool_result)" :min="500" :max="8000" :step="500" unit=" tok" />
      <LabSlider v-model="explore" label="worker 多探索" :min="1" :max="3" :step="0.5" unit="×" />
    </template>

    <div class="gantt">
      <div v-for="row in plan.rows" :key="row.id" class="g-row">
        <span class="mono g-lbl">{{ row.label }}</span>
        <div class="g-track">
          <button
            v-for="(b, i) in row.bars" :key="i" type="button" class="g-bar" :class="[b.kind, { sel: sel === b.worker && b.worker != null }]"
            :style="{ left: b.start / plan.span * 100 + '%', width: b.dur / plan.span * 100 + '%' }"
            :title="`${b.text} · ${b.dur.toFixed(0)}s`" @click="sel = b.worker ?? sel"
          >{{ b.text }}</button>
        </div>
      </div>
      <p class="axis mono">0s ─── 墙钟时间 ─── {{ plan.span.toFixed(0) }}s (线程池 max_parallel = 4)</p>
    </div>
    <p v-if="mode !== 'single'" class="detail">
      <b>worker {{ selW + 1 }}</b> 在自己的上下文里读了 <b class="mono">{{ fmtNum(acct.workerRead) }}</b> tok 文档、花了 <b class="mono">{{ fmtNum(acct.workerInput) }}</b> input tok;
      lead 只收到一条 <b class="mono">{{ S }}</b> tok 的摘要 tool_result。完整 transcript 落盘在 child_{{ String(selW).padStart(2, '0') }}.jsonl, 可审计, 但不回流。
    </p>
    <p v-else class="detail">单 agent: {{ n * r }} 份文档全部堆在同一个上下文里, 每多一轮都要把前面读过的再发一遍。</p>

    <template #stats>
      <div class="kv"><span>lead / 主 agent 峰值上下文</span><b :class="mode === 'single' ? 'bad' : 'good'">{{ fmtNum(acct.peak) }}</b></div>
      <div class="kv"><span>总 input tokens</span><b :class="acct.total > acct.single.total ? 'bad' : mode === 'single' ? '' : 'good'">{{ fmtNum(acct.total) }}</b></div>
      <div class="kv"><span>相对单 agent</span><b>{{ (acct.total / acct.single.total).toFixed(2) }}×</b></div>
      <div class="kv"><span>墙钟时间</span><b :class="mode === 'parallel' ? 'good' : ''">{{ plan.wall.toFixed(1) }}s</b></div>
      <p class="lab-note">delegate 只是一个普通工具: lead 在同一轮发出 N 个 tool_use, agent loop 的并行执行自然就成了扇出, 不需要另一套调度器。</p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { fmtNum, mulberry32, range, sum } from '@/utils/labmath.js'

const MODES = [{ id: 'single', label: '单 agent 全自己读' }, { id: 'serial', label: 'orchestrator · worker 串行' }, { id: 'parallel', label: 'orchestrator · worker 并行' }]
const mode = ref('parallel'), n = ref(3), r = ref(1), t = ref(500), explore = ref(1), seed = ref(1), sel = ref(0)
const selW = computed(() => Math.min(sel.value, n.value - 1))

// 固定开销 (token): 主 system+工具定义 / 子 system+工具定义 / 用户问题 / 每次 tool_use / 任务简报 / 摘要
const B = 1500, BW = 800, Q = 50, U = 30, BRIEF = 150, S = 300
const TURN = 3, POOL = 4 // 每轮 (模型调用+工具) 约 3s; 线程池 4
// ★ 一个 agent 连续读 k 份文档: 第 j 次调用要重发此前全部 j 份 → input 之和是 k 的平方级
const session = (base, k) => ({ total: sum(range(k + 1).map((j) => base + j * (t.value + U))), peak: base + k * (t.value + U) })

const acct = computed(() => {
  const single = session(B + Q, n.value * r.value)
  const kw = Math.round(r.value * explore.value)
  const worker = session(BW + BRIEF, kw)
  const leadPeak = B + Q + n.value * (U + BRIEF + S)
  const lead = (B + Q) + leadPeak // 两次调用: 扇出 + 汇总
  const orch = { total: lead + n.value * worker.total, peak: leadPeak }
  const cur = mode.value === 'single' ? single : orch
  return { ...cur, single, workerInput: worker.total, workerRead: kw * t.value, kw }
})

const plan = computed(() => {
  const rand = mulberry32(seed.value * 97 + 3)
  const kw = acct.value.kw
  const durs = range(n.value).map(() => (kw + 1) * TURN * (0.7 + 0.6 * rand()))
  const serialSpan = 2 * TURN + sum(durs)
  if (mode.value === 'single') {
    const d = (n.value * r.value + 1) * TURN
    return { rows: [{ id: 'a', label: 'agent', bars: [{ kind: 'lead', start: 0, dur: d, text: `${n.value * r.value} 份文档 + final` }] }], wall: d, span: Math.max(d, serialSpan) }
  }
  // 线程池: 谁先空闲谁接下一个 (pool.map 的效果); 串行 = 池大小 1
  const free = Array(mode.value === 'parallel' ? POOL : 1).fill(TURN)
  const rows = durs.map((d, i) => {
    const k = free.indexOf(Math.min(...free)), start = free[k]
    free[k] = start + d
    return { id: i, label: `worker ${i + 1}`, bars: [{ kind: 'worker', worker: i, start, dur: d, text: `读 ${kw} 份 → 摘要` }] }
  })
  const end = Math.max(...free)
  rows.unshift({ id: 'lead', label: 'lead', bars: [{ kind: 'lead', start: 0, dur: TURN, text: '扇出' }, { kind: 'lead', start: end, dur: TURN, text: '汇总' }] })
  return { rows, wall: end + TURN, span: Math.max(serialSpan, end + TURN) }
})
</script>

<style scoped>
.gantt { display: flex; flex-direction: column; gap: 4px; min-width: 420px; }
.g-row { display: grid; grid-template-columns: 64px 1fr; gap: 8px; align-items: center; }
.g-lbl { font-size: 11px; color: var(--text-dim); }
.g-track { position: relative; height: 24px; border-bottom: 1px dashed var(--border); }
.g-bar { position: absolute; top: 2px; height: 20px; min-height: 0; padding: 0 4px; font-size: 10px; line-height: 18px; overflow: hidden; white-space: nowrap; border-radius: 3px; border: 1px solid var(--accent); background: var(--accent-soft); color: var(--text); }
.g-bar.lead { border-color: var(--eye); background: color-mix(in srgb, var(--eye) 22%, transparent); cursor: default; }
.g-bar.sel { outline: 2px solid var(--warn); outline-offset: 1px; }
.axis { font-size: 10px; color: var(--text-dim); margin-top: 4px; }
.detail { margin-top: 12px; font-size: 13px; color: var(--text-muted); line-height: 1.7; }
.detail b { color: var(--text); font-weight: 500; }
</style>
