<!--
  mini-vLLM 引擎主循环实验台 (llm_infer/full_engine)。
  一步永远是同四件事: 调度 → 前向 → 采样 → 后处理。难的只有调度那一步 ——
  它要同时管住 KV 显存 (block 够不够)、公平 (谁先进) 和延迟 (每步算几个 token)。
  最后一个开关演示一类真实存在的活锁: 队首进不来时如果不落到 decode, 整条时间线就冻住。
-->
<template>
  <LabFrame
    title="引擎主循环 — 三条队列、一个 block 池、一个调度分支"
    sub="上面三行是请求队列 (waiting / running / finished), 下面是物理 KV block 池, 每格一块。按播放看每一步调度器走了哪条分支、为什么。滑杆能把它调到崩: block 池调小到必然抢占, 到达速率调高到排长队。"
    module="llm_infer/full_engine"
    run="python -m llm_infer.full_engine.demo"
    :challenge="{
      ask: '打开最后那个红色开关 (队首进不来时也只走 prefill 分支), 再把 block 池拖到 10 以下。时间线会怎样? 为什么关掉就能恢复?',
      answer: '时间线冻住。队首拿不到 block → prefill 分支选不出任何序列 → 返回空 batch → running 一个 token 也不前进 → 没有序列结束 → block 永远不释放 → 队首永远进不来。这就是活锁, 本仓库的 engine.py 早期版本就死在这里。修法只有一行: prefill 进不来时落到 decode (Python 里就是 `_admit(budget) or _schedule_running(budget)` 的那个 or)。关键不在于 decode 更重要, 而在于必须保证「最老的那条序列总能前进」—— 有进度才有释放, 有释放才有名额。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: chunked }" @click="setMode('chunk')">decode 优先 + 分块 prefill</button>
        <button type="button" :class="{ active: !chunked && !livelock }" @click="setMode('prefill')">prefill 优先 (进不来就落到 decode)</button>
        <button type="button" class="danger-btn" :class="{ active: livelock }" @click="setMode('bug')">⚠ 队首进不来时也只走 prefill 分支</button>
      </div>
      <div class="row">
        <button type="button" :class="{ active: prefixOn }" @click="prefixOn = !prefixOn">
          前缀缓存 {{ prefixOn ? '开' : '关' }}
        </button>
        <span class="tip">5 条请求共享同一段 30 token 的 system prompt</span>
      </div>
      <LabSlider v-model="pool" label="KV block 池" :min="6" :max="40" unit=" 块" />
      <LabSlider v-model="budget" label="max_batch_tokens" :min="8" :max="96" :step="8" />
      <LabSlider v-model="rate" label="到达速率" :min="0.2" :max="5" :step="0.2" unit=" 条/步" />
      <StepPlayer :stepper="stepper" :label="`step ${f.t + 1}`" />
    </template>

    <div class="engine">
      <div v-for="q in queues" :key="q.name" class="qrow">
        <span class="qlab mono">{{ q.name }}</span>
        <button
          v-for="c in q.items" :key="c.id" type="button" class="chip"
          :class="{ active: sel === c.id, dimmed: sel >= 0 && sel !== c.id }"
          :style="{ borderColor: color(c.id), background: c.n ? fill(c.id) : 'transparent' }"
          @click="sel = sel === c.id ? -1 : c.id"
        >R{{ c.id }}<em v-if="c.txt">{{ c.txt }}</em></button>
        <span v-if="!q.items.length" class="tip">—</span>
      </div>

      <p class="branch">
        <b class="tag" :class="branchCls">{{ f.branch }}</b>
        <span v-if="sim.dead && stepper.step.value >= sim.frames.length - 1" class="tag dead">活锁</span>
        {{ f.why }}
      </p>

      <div class="cells" :style="{ gridTemplateColumns: `repeat(${Math.min(pool, 20)}, 20px)` }">
        <span
          v-for="(o, i) in f.cells" :key="i" class="cell"
          :class="{ dim: sel >= 0 && o !== sel, free: o === null, shared: o === -1 }"
          :style="o >= 0 ? { background: fill(o), borderColor: color(o) } : null"
          :title="o === null ? '空闲 block' : o === -1 ? '共享前缀 block (ref_count > 1)' : `R${o} 的 block`"
        >{{ o === -1 ? '∷' : o === null ? '' : o }}</span>
      </div>
      <p class="lab-note info">
        <template v-if="selSeq">
          R{{ selSeq.id }}: prompt {{ selSeq.prompt }} token, 第 {{ selSeq.arrive + 1 }} 步到达 ·
          TTFT {{ selSeq.ttft === null ? '未出首 token' : selSeq.ttft.toFixed(0) + ' ms' }} ·
          最大 TBT {{ selSeq.tbt ? selSeq.tbt.toFixed(0) + ' ms' : '—' }} ·
          已出 {{ selSeq.out }}/{{ MAX_NEW }} token
        </template>
        <template v-else>点队列里的 R 按钮, 看那条请求占了哪些 block、它的 TTFT / TBT 是多少。∷ 是被所有请求共享的前缀 block。</template>
      </p>
    </div>

    <template #stats>
      <div class="kv"><span>吞吐 (tok/s)</span><b :class="sim.dead ? 'bad' : 'good'">{{ sim.thr.toFixed(1) }}</b></div>
      <div class="kv"><span>平均 TTFT</span><b>{{ sim.ttft ? sim.ttft.toFixed(0) + ' ms' : '—' }}</b></div>
      <div class="kv">
        <span>最大 TBT</span>
        <b v-if="!sim.maxTbt">—</b>
        <b v-else :class="sim.maxTbt > 120 ? 'bad' : 'good'">{{ sim.maxTbt.toFixed(0) }} ms</b>
      </div>
      <div class="kv"><span>抢占次数</span><b :class="sim.preempts ? 'bad' : 'good'">{{ sim.preempts }}</b></div>
      <div class="kv"><span>block 利用率</span><b>{{ (sim.util * 100).toFixed(0) }}%</b></div>
      <div class="kv">
        <span>实算 + 命中 = 需要</span>
        <b :class="sim.comp + sim.hit === sim.need ? 'good' : 'bad'">{{ sim.comp }} + {{ sim.hit }} = {{ sim.need }}</b>
      </div>
      <p v-if="sim.dead" class="lab-note">
        活锁: 连续 6 步选不出任何序列, 之后也不会变 —— block 只在序列结束时释放, 而没有序列能前进。
        真实引擎里这表现为"服务没崩, 但所有请求都不动了"。
      </p>
      <p v-else-if="sim.comp + sim.hit !== sim.need" class="lab-note">
        实算比"需要 − 命中"多出来的部分, 是被抢占的序列回来重算的 KV。文本没丢, 丢的是算力。
      </p>
      <p class="lab-note">
        耗时用的是 m06 那个代价模型: 每步 20 ms 固定开销 + 0.25 ms × 本步 token 数, 不是实测。
        对照 demo: 64 块池 + 预算 48 时 305 个待算 token = 233 实算 + 72 命中; 9 块的小池 82 步、抢占 4 次, 输出仍与朴素 greedy 逐 token 相同。
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

const BS = 8                       // block_size, 与 demo 一致
const SYS = 30                     // 共享 system prompt 的 token 数
const MAX_SEQS = 4                 // max_batch_seqs
const MAX_NEW = 12
const QLEN = [15, 12, 14, 13, 12]  // 每条请求各自的问句长度
const FIXED_MS = 20, PER_TOK_MS = 0.25
const PAL = ['var(--accent)', 'var(--left)', 'var(--eye)', 'var(--right)', 'var(--warn)']
const color = (i) => (i < 0 ? 'var(--text-dim)' : PAL[i % PAL.length])
const fill = (i) => `color-mix(in srgb, ${color(i)} 40%, transparent)`

const pool = ref(22)   // 22 是这组请求的临界点: 往下拖一格就开始抢占
const budget = ref(48)
const rate = ref(2)
const prefixOn = ref(true)
const chunked = ref(true)
const livelock = ref(false)
const sel = ref(-1)

const setMode = (m) => { chunked.value = m === 'chunk'; livelock.value = m === 'bug' }

const sim = computed(() => {
  const P = prefixOn.value ? Math.floor(SYS / BS) : 0   // 可复用的完整前缀 block 数
  const NB = pool.value, B = budget.value
  const seqs = QLEN.map((q, i) => ({
    id: i, arrive: Math.floor(i / rate.value), prompt: SYS + q, len: SYS + q,
    computed: 0, out: 0, ttft: null, last: 0, tbt: 0, done: false, shared: false, donor: false,
  }))
  const waiting = [], running = [], finished = []
  let warm = false, preempts = 0, comp = 0, hit = 0, ms = 0, idle = 0, dead = false
  const frames = [], startMs = []
  // 共享的前缀 block 只占池子一份: donor 是最早算出它们的那条, shared 是后来命中的那些
  const blocksOf = (s) => Math.ceil(s.len / BS) - (s.shared || s.donor ? P : 0)
  const used = () => (warm ? P : 0) + running.reduce((a, s) => a + blocksOf(s), 0)

  for (let t = 0; t < 400; t++) {
    startMs[t] = ms
    for (const s of seqs) if (s.arrive === t) waiting.push(s)
    const batch = []
    let bud = B, just = false, why = ''

    // 给 running 里每条序列它还欠的 token; block 不够就抢占最年轻的
    const schedRunning = () => {
      const todo = [...running]
      while (todo.length && bud > 0) {
        const s = todo.shift()
        let selfKilled = false
        while (used() > NB) {
          const v = todo.length ? todo.pop() : s
          running.splice(running.indexOf(v), 1)
          v.computed = 0                       // ★ recompute 式抢占: 还 block, 已生成的文本留着, KV 之后重算
          waiting.unshift(v)
          preempts++; just = true
          why = `block 池撑不住了, 踢掉最年轻的 R${v.id}: 还 block、num_computed 归 0、回 waiting 队首`
          if (v === s) { selfKilled = true; break }
        }
        if (selfKilled) continue
        const n = Math.min(s.len - s.computed, bud)
        if (n > 0) { batch.push([s, n]); bud -= n }
      }
    }
    // 从 waiting 队首 FCFS 接新请求; 命中前缀的 token 不占预算、不用算
    const admit = () => {
      while (waiting.length && running.length < MAX_SEQS && bud > 0) {
        const s = waiting[0]
        const h = warm ? Math.min(P, Math.floor((s.len - 1) / BS)) * BS : 0
        let n = s.len - h
        if (chunked.value) n = Math.min(n, bud)
        else if (n > bud && batch.length) break
        const need = Math.ceil(s.len / BS) - (h ? P : 0)
        if (used() + need > NB) {
          why = `队首 R${s.id} 还要 ${need} 个 block, 池里只剩 ${NB - used()} 个 → 进不来`
          break
        }
        waiting.shift()
        s.computed = h; s.shared = h > 0; hit += h
        running.push(s); batch.push([s, n]); bud -= n
      }
    }

    if (chunked.value) {
      schedRunning()
      if (!just) admit()                       // 刚抢占过就别立刻把腾出的 block 又占掉
    } else {
      admit()
      if (!batch.length && !livelock.value) schedRunning()   // ★ 这个 fallback 就是活锁修复
    }

    // 前向 + 采样 + 后处理
    const nTok = batch.reduce((a, [, n]) => a + n, 0)
    ms += nTok ? FIXED_MS + PER_TOK_MS * nTok : 0
    comp += nTok
    for (const [s, n] of batch) {
      s.computed += n
      if (P && !warm && s.computed >= P * BS) { warm = true; s.donor = true }   // 头一条算完整前缀的, 把这几块捐给大家
      if (s.computed < s.len) continue                    // prefill 还没追平, 这一步没有 logits 可采
      s.out++; s.len++
      if (s.ttft === null) s.ttft = ms - startMs[s.arrive]
      else s.tbt = Math.max(s.tbt, ms - s.last)
      s.last = ms
      if (s.out >= MAX_NEW) {
        s.done = true
        running.splice(running.indexOf(s), 1)
        finished.push(s)
      }
    }

    const kinds = batch.map(([, n]) => (n === 1 ? 'D' : 'P'))
    const branch = !batch.length ? '空转' : just ? '抢占' : kinds.every((k) => k === 'D') ? 'decode'
      : kinds.every((k) => k === 'P') ? 'prefill' : '混批'
    if (!why) {
      const pre = batch.filter(([, n]) => n > 1)
      const stuck = running.length >= MAX_SEQS ? `max_batch_seqs=${MAX_SEQS} 的并发名额满了` : 'token 预算已经被 decode 吃完'
      why = branch === 'decode' ? `running 各拿 1 个 token${waiting.length ? `; waiting 里还压着 ${waiting.length} 条, ${stuck}` : ''}`
        : branch === 'prefill' ? `只有新请求有活干: R${pre.map(([s]) => s.id).join(' R')} 各算 ${pre.map(([, n]) => n).join(' / ')} 个 token`
          : `running 先各拿 1 个, 剩下的 ${pre.reduce((a, [, n]) => a + n, 0)} 个预算切给 R${pre.map(([s]) => s.id).join(' R')} 的 prefill chunk`
    }
    const cells = []
    if (warm) for (let i = 0; i < P; i++) cells.push(-1)
    for (const s of running) for (let i = 0; i < blocksOf(s); i++) cells.push(s.id)
    while (cells.length < NB) cells.push(null)

    const inBatch = Object.fromEntries(batch.map(([s, n]) => [s.id, n]))
    frames.push({
      t, branch, why, cells: cells.slice(0, NB),
      waiting: waiting.map((s) => ({ id: s.id, n: 0, txt: s.computed === 0 && s.out ? ` 重算 ${s.len}` : '' })),
      running: running.map((s) => ({ id: s.id, n: inBatch[s.id] || 0, txt: ` ${s.computed}/${s.len}` })),
      finished: finished.map((s) => ({ id: s.id, n: 0, txt: ` ${s.out} tok` })),
    })

    if (!batch.length && waiting.length) { if (++idle > 5) { dead = true; break } } else idle = 0
    if (!waiting.length && !running.length && seqs.every((s) => s.arrive <= t)) break
  }

  const done = seqs.filter((s) => s.done)
  const totalOut = seqs.reduce((a, s) => a + s.out, 0)
  return {
    frames, dead, preempts, comp, hit, seqs,
    need: seqs.reduce((a, s) => a + s.len - 1, 0),
    thr: dead || ms === 0 ? 0 : (totalOut / ms) * 1000,   // 卡死之后再也不会有 token 出来, 稳态吞吐就是 0
    ttft: done.length ? done.reduce((a, s) => a + s.ttft, 0) / done.length : 0,
    maxTbt: Math.max(0, ...seqs.map((s) => s.tbt)),
    util: frames.reduce((a, fr) => a + fr.cells.filter((c) => c !== null).length, 0) / frames.length / NB,
  }
})

const stepper = useStepper(() => sim.value.frames.length, { interval: 420 })
const f = computed(() => sim.value.frames[Math.min(stepper.step.value, sim.value.frames.length - 1)])
const queues = computed(() => [
  { name: 'waiting', items: f.value.waiting },
  { name: 'running', items: f.value.running },
  { name: 'finished', items: f.value.finished },
])
const selSeq = computed(() => sim.value.seqs.find((s) => s.id === sel.value))
const branchCls = computed(() => ({ 抢占: 'warn', 空转: 'dead', decode: 'ok', prefill: 'ok', 混批: 'ok' }[f.value.branch]))
// 换参数后回到第 0 步, 从头看这组配置怎么演化
watch(sim, () => { stepper.pause(); stepper.step.value = 0 }, { immediate: true })
</script>

<style scoped>
.engine { display: flex; flex-direction: column; gap: 8px; }
.qrow { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; min-height: 30px; }
.qlab { width: 62px; flex: none; font-size: 11px; color: var(--text-dim); }
.chip { min-height: 26px; padding: 2px 8px; border-width: 2px; font-size: 12px; }
.chip em { font-style: normal; font-size: 10px; color: var(--text-dim); margin-left: 3px; }
.chip.active { outline: 2px solid var(--accent); outline-offset: 1px; }
.chip.dimmed { opacity: 0.4; }
.tip { font-size: 11px; color: var(--text-dim); }
.branch { display: flex; flex-wrap: wrap; align-items: baseline; gap: 8px; font-size: 12px; color: var(--text-muted); line-height: 1.7; margin: 4px 0 2px; }
.branch .tag { font-size: 11px; }
.tag.ok { border-color: var(--left); color: var(--left); }
.tag.warn { border-color: var(--warn); color: var(--warn); }
.tag.dead { border-color: var(--danger); color: var(--danger); }
.cell.free { border-style: dashed; opacity: 0.5; }
.cell.shared { border-style: dashed; border-color: var(--text-dim); }
.danger-btn.active { background: var(--danger); border-color: var(--danger); color: #fff; }
.info { margin-top: 6px; min-height: 3.2em; }
</style>
