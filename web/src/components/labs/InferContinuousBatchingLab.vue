<!-- 连续批处理实验台 (llm_infer/m03): 同一条请求轨迹, 静态 batch vs iteration-level 调度。 -->
<template>
  <LabFrame
    title="连续批处理 — 静态 batch vs 逐步重组"
    sub="每一行是一个 batch 槽位, 每一格是一步 (一次前向)。同一批请求、同样的到达时刻, 只换调度策略。虚线格 = 请求已经结束, 槽位却还被占着等同批最长的那条。"
    module="llm_infer/m03"
    run="python -m llm_infer.m03_continuous_batching.demo"
    :challenge="{
      ask: '把到达率拖到最大 (请求一开始就全到齐), 静态 batch 还会比连续批差吗? 差在哪?',
      answer: '还是差。请求全到齐时静态 batch 不用再等人。但同一批里输出长度不同: 短请求结束后, 槽位要空转到最长的那条结束 (虚线格)。连续批在每一步结束后立刻把空出来的槽位补上新请求, 所以利用率接近 100%, 平均延迟也低。差距来自「输出长度参差不齐」, 而不是到达时间。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: mode === 'static' }" @click="mode = 'static'">静态 batch (整批进, 整批出)</button>
        <button type="button" :class="{ active: mode === 'cont' }" @click="mode = 'cont'">连续批 (每步重组)</button>
        <button type="button" @click="seed++">换一组请求</button>
      </div>
      <LabSlider v-model="rate" label="到达率 (请求/步)" :min="0.1" :max="3" :step="0.1" />
      <LabSlider v-model="slots" label="batch 槽位数 B" :min="2" :max="6" />
      <StepPlayer :stepper="stepper" :label="`t = ${stepper.step.value}`" />
      <div class="row reqs">
        <button
          v-for="r in reqs" :key="r.id" type="button" class="req" :class="{ active: sel === r.id }"
          :style="{ borderColor: color(r.id) }" @click="sel = sel === r.id ? -1 : r.id"
        >R{{ r.id }}</button>
      </div>
    </template>

    <div class="cells" :style="{ gridTemplateColumns: `44px repeat(${cur.T}, 20px)` }">
      <template v-for="(row, s) in cur.grid" :key="s">
        <span class="row-label mono">槽 {{ s }}</span>
        <span
          v-for="(c, t) in row" :key="t" class="cell"
          :class="{ dim: t > stepper.step.value || (sel >= 0 && c?.id !== sel), held: c?.held, now: t === stepper.step.value }"
          :style="c ? { background: c.held ? 'transparent' : fill(c.id), borderColor: color(c.id) } : null"
          :title="c ? `R${c.id}${c.held ? ' 已结束, 槽位空转' : ''} · t=${t}` : '空闲'"
          @mouseenter="c && (sel = c.id)"
        >{{ c && c.head ? c.id : '' }}</span>
      </template>
    </div>
    <p class="lab-note info">
      <template v-if="selReq">R{{ selReq.id }}: 到达 t={{ selReq.arrive }}, 需要 {{ selReq.dur }} 步 (1 步 prefill + {{ selReq.dur - 1 }} 步 decode) ·
        开始 t={{ cur.start[selReq.id] }}, 结束 t={{ cur.finish[selReq.id] }} → 延迟 {{ cur.finish[selReq.id] - selReq.arrive }} 步
        (排队 {{ cur.start[selReq.id] - selReq.arrive }} 步)</template>
      <template v-else>悬停格子或点上面的 R 按钮, 高亮一条请求的全部格子。</template>
    </p>

    <template #stats>
      <div class="kv head"><span></span><span class="mono">静态 → 连续</span></div>
      <div class="kv"><span>槽位利用率</span><b :class="gapCls">{{ pct(sims.static.util) }} → {{ pct(sims.cont.util) }}</b></div>
      <div class="kv"><span>平均延迟 (步)</span><b :class="gapCls">{{ sims.static.lat.toFixed(1) }} → {{ sims.cont.lat.toFixed(1) }}</b></div>
      <div class="kv"><span>全部完成 (步)</span><b>{{ sims.static.T }} → {{ sims.cont.T }}</b></div>
      <div class="kv"><span>吞吐 (token/步)</span><b>{{ sims.static.thr.toFixed(2) }} → {{ sims.cont.thr.toFixed(2) }}</b></div>
      <p class="lab-note">
        连续批的全部秘密: 调度粒度从"一个 batch"变成"一步"。每步结束都检查谁做完了, 空出来的槽位立刻给 waiting 队首。
        真实引擎里"槽位"其实是 KV block 和 token 预算 (见 full_engine)。
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
import { mulberry32, range, sum } from '@/utils/labmath.js'

const N = 12
const PALETTE = ['var(--accent)', 'var(--left)', 'var(--eye)', 'var(--right)', 'var(--warn)', 'var(--danger)']
const color = (id) => PALETTE[id % PALETTE.length]
const fill = (id) => `color-mix(in srgb, ${color(id)} ${id % 12 < 6 ? 55 : 28}%, transparent)`
const pct = (x) => (x * 100).toFixed(0) + '%'

const mode = ref('cont')
const rate = ref(1)
const slots = ref(4)
const seed = ref(1)
const sel = ref(-1)

// 随机数只跟 seed 走: 拖到达率时每条请求的长度不变, 只是到达时刻整体伸缩
const draws = computed(() => {
  const rand = mulberry32(seed.value * 7919)
  return range(N).map(() => ({ gap: -Math.log(1 - rand()), dur: 2 + Math.floor(rand() ** 2 * 13) })) // 长度偏斜: 多数短, 少数很长
})
const reqs = computed(() => {
  let t = 0
  return draws.value.map((d, id) => { t += d.gap / rate.value; return { id, arrive: Math.floor(t), dur: d.dur } })
})

const simulate = (kind) => {
  const B = slots.value, R = reqs.value
  const grid = range(B).map(() => []), slotOf = Array(B).fill(null) // slotOf[s] = { id, left }
  const start = {}, finish = {}
  let next = 0, batchEnd = 0
  for (let t = 0; Object.keys(finish).length < N && t < 400; t++) {
    const free = range(B).filter((s) => !slotOf[s])
    // ★ 唯一的区别: 静态 batch 要等整批都结束才接人; 连续批每一步都把空槽补上
    const mayAdmit = kind === 'cont' || (free.length === B && t >= batchEnd)
    if (mayAdmit) for (const s of free) {
      if (next >= N || R[next].arrive > t) break
      slotOf[s] = { id: R[next].id, left: R[next].dur, head: true }
      start[R[next].id] = t
      next++
    }
    if (kind === 'static' && mayAdmit) batchEnd = t + Math.max(0, ...slotOf.filter(Boolean).map((x) => x.left))
    for (let s = 0; s < B; s++) {
      const x = slotOf[s]
      if (!x) { grid[s][t] = null; continue }
      grid[s][t] = { id: x.id, head: x.head, held: x.left <= 0 }
      x.head = false
      if (--x.left === 0) finish[x.id] = t + 1
      if (x.left <= 0 && (kind === 'cont' || t + 1 >= batchEnd)) slotOf[s] = null
    }
  }
  const T = grid[0].length
  const busy = sum(R.map((r) => r.dur))
  return { grid, T, start, finish, util: busy / (B * T), thr: busy / T, lat: sum(R.map((r) => finish[r.id] - r.arrive)) / N }
}
const sims = computed(() => ({ static: simulate('static'), cont: simulate('cont') }))
const cur = computed(() => sims.value[mode.value])
const selReq = computed(() => reqs.value.find((r) => r.id === sel.value))
const gapCls = computed(() => (mode.value === 'cont' ? 'good' : 'bad'))

const stepper = useStepper(() => cur.value.T, { interval: 300 })
watch(cur, (c) => { stepper.pause(); stepper.step.value = c.T - 1 }, { immediate: true })
</script>

<style scoped>
.row-label { font-size: 11px; color: var(--text-dim); align-self: center; }
.cell { min-width: 20px; }
.cell.held { border-style: dashed; }
.cell.now { outline: 2px solid var(--accent); outline-offset: 1px; }
.reqs .req { min-height: 28px; padding: 2px 8px; border-width: 2px; }
.info { margin-top: 10px; min-height: 3.4em; }
.kv.head { border: 0; padding: 0; font-size: 11px; color: var(--text-dim); }
</style>
