<!--
  流水线并行调度实验台 (对应 llm_train/m04_pipeline_parallel)。
  这个文件同时是"怎么写一个 lab"的范例:
    LabFrame 负责版式, LabSlider 负责滑杆, useStepper + StepPlayer 负责时间轴,
    纯计算放在 computed 里, 模板只管画。
-->
<template>
  <LabFrame
    title="流水线调度 — GPipe vs 1F1B"
    sub="每一行是一张卡 (stage), 每一格是一个时间槽。F = 前向, B = 反向, 空格 = 气泡 (这张卡在等)。拖滑杆改变卡数和 micro-batch 数, 按播放看时间表怎么被填满。"
    module="llm_train/m04"
    run="python -m llm_train.m04_pipeline_parallel.demo"
    :challenge="{
      ask: '把 micro-batch 数从 4 拖到 16, 气泡比例和 stage 0 的激活峰值分别怎么变? 两种调度有什么不同?',
      answer: '气泡比例 (PP-1)/(PP-1+M) 两种调度一样, 都随 M 增大而下降。区别在显存: GPipe 的 stage 0 要同时攥着全部 M 份激活, 峰值 = M; 1F1B 限制在途 micro-batch 不超过 PP-s, 峰值恒为 PP, 与 M 无关。所以 1F1B 省的是显存, 不是时间。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: mode === 'gpipe' }" @click="mode = 'gpipe'">GPipe (先全 F 再全 B)</button>
        <button type="button" :class="{ active: mode === '1f1b' }" @click="mode = '1f1b'">1F1B (一前一后交替)</button>
      </div>
      <LabSlider v-model="pp" label="流水线级数 PP" :min="2" :max="6" />
      <LabSlider v-model="m" label="micro-batch 数 M" :min="1" :max="16" />
      <LabSlider v-model="bCost" label="B 耗时 (F = 1)" :min="1" :max="3" unit="×" />
      <StepPlayer :stepper="stepper" :label="`t = ${stepper.step.value}`" />
    </template>

    <div class="cells" :style="{ gridTemplateColumns: `48px repeat(${sched.T}, 22px)` }">
      <template v-for="(row, s) in sched.grid" :key="s">
        <span class="row-label mono">卡 {{ s }}</span>
        <span
          v-for="(c, t) in row" :key="t"
          class="cell"
          :class="[c && (c.type === 'F' ? 'on' : 'ok'), { dim: t > stepper.step.value, now: t === stepper.step.value }]"
          :title="c ? `${c.type}${c.m} @ 卡${s}, t=${t}` : '气泡'"
        >{{ c && c.head ? c.type + c.m : '' }}</span>
      </template>
    </div>

    <template #stats>
      <div class="kv"><span>总时间槽</span><b>{{ sched.T }}</b></div>
      <div class="kv"><span>气泡比例</span><b>{{ (sched.bubble * 100).toFixed(1) }}%</b></div>
      <div class="kv">
        <span>各卡激活峰值</span>
        <b :class="mode === '1f1b' ? 'good' : 'bad'">[{{ sched.peaks.join(', ') }}]</b>
      </div>
      <div class="kv"><span>当前在途激活</span><b>[{{ liveNow.join(', ') }}]</b></div>
      <p class="lab-note">
        1F1B 的关键只有一行: 卡 s 上"已前向未反向"的 micro-batch 数不许超过 PP − s。
        少了这个上限, 调度就会退化成 GPipe, 激活峰值回到 M。
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

const mode = ref('1f1b')
const pp = ref(4)
const m = ref(8)
const bCost = ref(2)

// 离散事件模拟: 每个时间槽, 每张空闲的卡挑一个"依赖已满足"的任务开工
const sched = computed(() => {
  const PP = pp.value, M = m.value, FB = { F: 1, B: bCost.value }
  const fEnd = range(PP).map(() => Array(M).fill(Infinity)) // fEnd[s][k]: F_k 在卡 s 上的完成时刻
  const bEnd = range(PP).map(() => Array(M).fill(Infinity))
  const fNext = Array(PP).fill(0), bNext = Array(PP).fill(0) // 每张卡下一个要做的 F / B 编号
  const busyUntil = Array(PP).fill(0)
  const grid = range(PP).map(() => [])

  for (let t = 0; bNext.some((b) => b < M) && t < 400; t++) {
    for (let s = 0; s < PP; s++) {
      if (busyUntil[s] > t) continue
      const k = bNext[s], j = fNext[s]
      // B_k 的依赖: 下游卡已反传完 (最后一张卡则是自己的 F_k 做完)
      const bReady = k < M && (s === PP - 1 ? fEnd[s][k] : bEnd[s + 1][k]) <= t
      // F_j 的依赖: 上游卡已前向完
      const fReady = j < M && (s === 0 || fEnd[s - 1][j] <= t)
      // ★ 1F1B 的全部秘密: 在途 (已 F 未 B) 不超过 PP - s; GPipe 没有这个上限, 但要求 F 全做完才开始 B
      const inflight = fNext[s] - bNext[s]
      const canF = fReady && (mode.value === 'gpipe' || inflight < PP - s)
      const canB = bReady && (mode.value === '1f1b' || fNext[s] === M)

      const type = canB ? 'B' : canF ? 'F' : null
      if (!type) continue
      const idx = type === 'B' ? k : j
      for (let d = 0; d < FB[type]; d++) grid[s][t + d] = { type, m: idx, head: d === 0 }
      busyUntil[s] = t + FB[type]
      if (type === 'B') { bEnd[s][idx] = busyUntil[s]; bNext[s]++ } else { fEnd[s][idx] = busyUntil[s]; fNext[s]++ }
    }
  }

  const T = Math.max(...grid.map((r) => r.length))
  grid.forEach((r) => { for (let t = 0; t < T; t++) r[t] = r[t] || null })
  const work = M * (FB.F + FB.B)
  // live[s][t]: 时刻 t 卡 s 上攥着的激活份数 = 已开始的 F − 已完成的 B
  const live = range(PP).map((s) => range(T).map((t) =>
    fEnd[s].filter((e) => e - FB.F <= t).length - bEnd[s].filter((e) => e <= t).length))
  return { grid, T, bubble: 1 - work / T, peaks: live.map((r) => Math.max(...r)), live }
})

const stepper = useStepper(() => sched.value.T, { interval: 350 })
const liveNow = computed(() => sched.value.live.map((r) => r[stepper.step.value] ?? 0))
// 换参数后直接跳到末尾, 先看到完整时间表, 再按播放重看过程
watch(sched, (s) => { stepper.pause(); stepper.step.value = s.T - 1 }, { immediate: true })
</script>

<style scoped>
.row-label { font-size: 11px; color: var(--text-dim); align-self: center; }
.cell.now { outline: 2px solid var(--accent); outline-offset: 1px; }
</style>
