<!--
  交错 1F1B (对应 llm_train/m04_pipeline_parallel:device_order + simulate, 逐行移植)。
  只讲一件事: 每张卡拿 v 个不相邻的小 stage, micro-batch 在卡之间绕 v 圈 —— 气泡除以 v, 代价是 v 倍的 P2P 和更多在途激活。
-->
<template>
  <LabFrame
    title="交错 1F1B — 把气泡再除以 v"
    sub="每张卡不再拿连续的一大段层, 而是拿 v 个小段 (颜色深浅 = 第几段)。一个 micro-batch 要在所有卡上绕 v 圈。悬停任意一格, 同一个 micro-batch 的全部足迹会亮起来; 拖 v 看气泡怎么缩、在途激活怎么涨。"
    module="llm_train/m04"
    run="python -m llm_train.m04_pipeline_parallel.demo"
    :challenge="{
      ask: 'PP=4, M=8。v 从 1 调到 2, 气泡从 27.3% 降到多少? 卡 0 的在途激活 (折算成整 stage) 是升还是降?',
      answer: '气泡 = (PP−1)/(v·M+PP−1): 3/11 = 27.3% → 3/19 = 15.8%。直觉: 每个小段只有原来 1/v 的计算量, 流水线「灌满」和「排空」的那段空等时间也就缩成 1/v。但不是白来的: 卡 0 的在途激活峰值从 4 个 stage 升到 11 份 1/2-stage = 5.5 个 stage (warmup 更长); 每个 micro-batch 跨卡传激活的次数从 2(PP−1) 变成 2(PP·v−1), 约 v 倍。所以 Megatron 只在 PP 跨机、M 又受显存限制加不上去时才开交错。',
    }"
  >
    <template #controls>
      <LabSlider v-model="pp" label="流水线级数 PP" :min="2" :max="4" />
      <LabSlider v-model="groups" label="M (须为 PP 的倍数)" :min="1" :max="3" :format="(g) => g * pp" />
      <LabSlider v-model="v" label="每卡虚拟 stage 数 v" :min="1" :max="4" :format="(x) => (x === 1 ? '1 (普通 1F1B)' : x)" />
      <StepPlayer :stepper="stepper" :label="`t = ${stepper.step.value}`" />
    </template>

    <div class="cells sched" :style="{ gridTemplateColumns: `40px repeat(${sim.T}, 18px)` }" @mouseleave="hoverM = -1">
      <template v-for="(row, s) in sim.grid" :key="s">
        <span class="row-label mono">卡 {{ s }}</span>
        <span
          v-for="(c, t) in row" :key="t" class="cell"
          :class="[c && (c.op === 'F' ? 'f' : 'b'), { dim: t > stepper.step.value || (hoverM >= 0 && (!c || c.m !== hoverM)), trace: c && c.m === hoverM }]"
          :style="c ? { '--k': 0.25 + 0.6 * (c.chunk + 1) / v } : null"
          :title="c ? `${c.op}${c.m} · 卡${s} 的第 ${c.chunk} 段 (虚拟 stage ${c.chunk * pp + s})` : '气泡'"
          @mouseenter="hoverM = c ? c.m : -1"
        >{{ c && c.head ? c.op + c.m : '' }}</span>
      </template>
    </div>

    <template #stats>
      <div class="kv"><span>气泡 (模拟)</span><b :class="{ good: v > 1 }">{{ (sim.bubble * 100).toFixed(1) }}%</b></div>
      <div class="kv"><span>公式 (PP−1)/(v·M+PP−1)</span><b>{{ (formula * 100).toFixed(1) }}%</b></div>
      <div class="kv"><span>在途激活峰值 (份 1/{{ v }}-stage)</span><b>[{{ sim.peak.join(', ') }}]</b></div>
      <div class="kv"><span>折算成整 stage (卡 0)</span><b :class="{ bad: v > 1 }">{{ (sim.peak[0] / v).toFixed(1) }} <small>(v=1 时 {{ Math.min(pp, M) }})</small></b></div>
      <div class="kv"><span>每个 micro-batch 的 P2P 次数</span><b :class="{ bad: v > 1 }">{{ 2 * (pp * v - 1) }}</b></div>
      <p class="lab-note">
        时间轴 1 格 = 1/v 个 stage 的前向, B 占 2 格。★ 与普通 1F1B 唯一的区别是 warmup 更长:
        <code class="inline">(PP−1−s)·2 + (v−1)·PP</code> 个 F 之后才开始 1F1B 交替。
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
import { range } from '@/utils/labmath.js'

const pp = ref(4), groups = ref(2), v = ref(2), hoverM = ref(-1)
const M = computed(() => groups.value * pp.value)
const TF = 1, TB = 2

// 卡 s 按什么顺序执行自己的 M·v 个 F 和 M·v 个 B (Megatron 交错顺序)
const deviceOrder = (s, PP, Mn, V) => {
  const fwd = [], bwd = []
  for (let g = 0; g < Mn; g += PP) {
    for (let c = 0; c < V; c++) for (let i = 0; i < Math.min(PP, Mn - g); i++) fwd.push(['F', g + i, c])
    for (let c = V - 1; c >= 0; c--) for (let i = 0; i < Math.min(PP, Mn - g); i++) bwd.push(['B', g + i, c])
  }
  // ★ warmup = 允许多少个 micro-batch "只 F 没 B"; 之后严格一 F 一 B
  const warmup = Math.min(V === 1 ? PP - 1 - s : (PP - 1 - s) * 2 + (V - 1) * PP, fwd.length)
  const order = fwd.slice(0, warmup)
  for (let f = warmup, b = 0; b < bwd.length; b++) { if (f < fwd.length) order.push(fwd[f++]); order.push(bwd[b]) }
  return order
}

const sim = computed(() => {
  const PP = pp.value, Mn = M.value, V = v.value, last = PP * V - 1
  const orders = range(PP).map((s) => deviceOrder(s, PP, Mn, V))
  const end = new Map(), free = Array(PP).fill(0), ptr = Array(PP).fill(0)
  const grid = range(PP).map(() => []), live = Array(PP).fill(0), peak = Array(PP).fill(0)
  const key = (op, m, k) => `${op}:${m}:${k}`
  for (let guard = 0; ptr.some((p, s) => p < orders[s].length) && guard < 5000; guard++) {
    for (let s = 0; s < PP; s++) {
      if (ptr[s] === orders[s].length) continue
      const [op, m, c] = orders[s][ptr[s]], k = c * PP + s // 虚拟 stage k 住在卡 k % PP 上
      const dep = op === 'F' ? (k > 0 ? key('F', m, k - 1) : null) : k < last ? key('B', m, k + 1) : key('F', m, k)
      if (dep && !end.has(dep)) continue // 依赖还没被调度, 这张卡先等
      const t0 = Math.max(free[s], dep ? end.get(dep) : 0), dur = op === 'F' ? TF : TB
      for (let d = 0; d < dur; d++) grid[s][t0 + d] = { op, m, chunk: c, head: d === 0 }
      free[s] = t0 + dur
      end.set(key(op, m, k), free[s])
      live[s] += op === 'F' ? 1 : -1
      peak[s] = Math.max(peak[s], live[s])
      ptr[s]++
    }
  }
  const T = Math.max(...free)
  grid.forEach((r) => { for (let t = 0; t < T; t++) r[t] = r[t] || null })
  return { grid, T, peak, bubble: 1 - (Mn * V * (TF + TB)) / T }
})
const formula = computed(() => (pp.value - 1) / (v.value * M.value + pp.value - 1))

const stepper = useStepper(() => sim.value.T, { interval: 220 })
watch(sim, (s) => { stepper.pause(); stepper.step.value = s.T - 1 }, { immediate: true })
</script>

<style scoped>
.row-label { font-size: 11px; color: var(--text-dim); align-self: center; }
.sched .cell { min-width: 18px; font-size: 9px; color: var(--text); }
.cell.f { background: color-mix(in srgb, var(--accent) calc(var(--k) * 100%), transparent); border-color: var(--accent); }
.cell.b { background: color-mix(in srgb, var(--left) calc(var(--k) * 100%), transparent); border-color: var(--left); }
.cell.trace { outline: 2px solid var(--warn); outline-offset: -1px; }
small { font-size: 10px; color: var(--text-dim); font-weight: 400; }
</style>
