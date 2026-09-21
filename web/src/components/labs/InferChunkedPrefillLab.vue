<!-- Chunked prefill 实验台 (llm_infer/m06): token 预算 B 是 TBT 与 TTFT 之间的旋钮。代价模型与 Python CostModel 相同。 -->
<template>
  <LabFrame
    title="Chunked prefill — 用 token 预算封顶卡顿"
    sub="D 个用户正在逐 token 生成, 第 3 步来了一条 P token 的长 prompt。每根柱子是一步前向, 高度 = 这一步的耗时 (代价模型: 20 ms + 0.25 ms × batch 内 token 数, 不是实测)。下面的时间轴画出一个 decode 用户收到 token 的时刻 —— 间隔就是 TBT。"
    module="llm_infer/m06"
    run="python -m llm_infer.m06_chunked_prefill.demo"
    :challenge="{
      ask: '把 token 预算 B 从 128 拖到 1024, decode 用户的最大卡顿和新请求的 TTFT 各往哪边走? 有没有两头都好的 B?',
      answer: '没有。B 越小, 每步耗时上限 20 + 0.25·B 越低, decode 用户的最大 TBT 越小。但 B 越小, 长 prompt 就被切成更多块: 每块都要多付一次 20 ms 固定开销, 还要和 decode 分享预算。所以新请求的 TTFT 变长 (默认参数: 276 ms → 445 ms)。B 是延迟抖动和首 token 延迟之间的旋钮, 不是免费午餐。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: mode === 'whole' }" @click="mode = 'whole'">prefill 优先 (整段)</button>
        <button type="button" :class="{ active: mode === 'chunk' }" @click="mode = 'chunk'">分块混批 (Sarathi)</button>
      </div>
      <LabSlider v-model="budget" label="token 预算 B" :min="32" :max="1024" :step="32" />
      <LabSlider v-model="P" label="长 prompt 长度 P" :min="256" :max="2048" :step="128" />
      <LabSlider v-model="D" label="正在 decode 的用户 D" :min="1" :max="16" />
      <StepPlayer :stepper="stepper" :label="`第 ${stepper.step.value} 步`" />
    </template>

    <svg class="tl" :viewBox="`0 0 ${W} 264`" role="img" aria-label="每步耗时与 token 到达时间轴">
      <line :x1="0" :x2="W" :y1="yOf(capMs)" :y2="yOf(capMs)" class="cap" />
      <text :x="W - 4" :y="yOf(capMs) - 4" text-anchor="end" class="lbl">{{ mode === 'chunk' ? `预算上限 20 + 0.25×${budget} = ${capMs} ms` : `无上限: 整段 prefill ${capMs} ms` }}</text>
      <g
        v-for="(s, i) in cur.steps" :key="i" role="button" tabindex="0" class="bar"
        :class="{ dim: i > stepper.step.value, now: i === stepper.step.value }" :aria-label="`第 ${i} 步 ${s.ms} ms`"
        @click="stepper.step.value = i" @keydown.enter="stepper.step.value = i"
      >
        <rect :x="i * bw" y="0" :width="bw" height="180" fill="transparent" />
        <rect :x="i * bw + gap" :y="yOf(20)" :width="bw - 2 * gap" :height="180 - yOf(20)" class="fixed" />
        <rect :x="i * bw + gap" :y="yOf(20 + 0.25 * s.dec)" :width="bw - 2 * gap" :height="yOf(20) - yOf(20 + 0.25 * s.dec)" class="dec" />
        <rect :x="i * bw + gap" :y="yOf(s.ms)" :width="bw - 2 * gap" :height="yOf(20 + 0.25 * s.dec) - yOf(s.ms)" class="pre" />
      </g>
      <!-- 时间轴: 横坐标是真实时间 (ms) -->
      <line x1="0" :x2="W" y1="212" y2="212" class="axis" />
      <line :x1="xT(cur.maxGap.from)" :x2="xT(cur.maxGap.to)" y1="212" y2="212" class="maxgap" />
      <text :x="(xT(cur.maxGap.from) + xT(cur.maxGap.to)) / 2" y="229" text-anchor="middle" class="lbl bad">最大 TBT {{ cur.maxTbt.toFixed(0) }} ms</text>
      <circle v-for="(t, i) in cur.emits" :key="i" :cx="xT(t)" cy="212" r="3" class="tok" />
      <line :x1="xT(cur.arriveT)" :x2="xT(cur.firstT)" y1="244" y2="244" class="ttft" />
      <text :x="xT(cur.arriveT)" y="260" class="lbl">新请求 TTFT {{ cur.ttft.toFixed(0) }} ms</text>
      <text x="0" y="200" class="lbl">● decode 用户收到 token 的时刻 (ms)</text>
    </svg>
    <p class="legend">
      <span class="sw fixed" />固定开销 20 ms <span class="sw dec" />decode token <span class="sw pre" />prefill token · 点柱子看这一步
    </p>

    <template #stats>
      <div class="kv head"><span></span><span class="mono">整段 → 分块</span></div>
      <div class="kv"><span>decode 最大 TBT</span><b :class="mode === 'chunk' ? 'good' : 'bad'">{{ sims.whole.maxTbt.toFixed(0) }} → {{ sims.chunk.maxTbt.toFixed(0) }} ms</b></div>
      <div class="kv"><span>新请求 TTFT</span><b :class="mode === 'chunk' ? 'bad' : 'good'">{{ sims.whole.ttft.toFixed(0) }} → {{ sims.chunk.ttft.toFixed(0) }} ms</b></div>
      <div class="kv"><span>prefill 切成几块</span><b>1 → {{ sims.chunk.nChunks }}</b></div>
      <div class="kv"><span>第 {{ stepper.step.value }} 步</span><b>{{ now.ms.toFixed(1) }} ms</b></div>
      <p class="lab-note">
        这一步: {{ now.dec }} 个 decode token + {{ now.pre }} 个 prefill token → 20 + 0.25 × {{ now.dec + now.pre }} = {{ now.ms.toFixed(2) }} ms。
        分块之所以结果不变: 因果 attention 下, 前面 token 的 KV 不依赖后面的 token。
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

const W = 720, WARM = 3, TAIL = 4
const stepMs = (n) => 20 + 0.25 * n // 与 m06 CostModel 相同: decode 访存受限 ≈ 固定开销, prefill 计算受限 ∝ token 数

const mode = ref('chunk')
const budget = ref(128)
const P = ref(1024)
const D = ref(4)

const simulate = (chunked) => {
  const steps = []
  const push = (dec, pre) => steps.push({ dec, pre, ms: stepMs(dec + pre) })
  for (let i = 0; i < WARM; i++) push(D.value, 0)
  let left = P.value
  if (!chunked) { push(0, left); left = 0 } // prefill 优先: 这一步只跑整段 prefill, 所有 decode 停一步
  // ★ 分块: 每步先给每个 decode 用户 1 个 token, 剩下的预算 B − D 才切给 prefill
  while (left > 0) { const c = Math.min(budget.value - D.value, left); push(D.value, c); left -= c }
  const lastPrefill = steps.length - 1
  for (let i = 0; i < TAIL; i++) push(D.value + 1, 0) // 新请求也进入 decode
  let t = 0
  const ends = steps.map((s) => (t += s.ms))
  const emits = ends.filter((_, i) => steps[i].dec > 0) // decode 用户在每个含 decode 的步末收到 1 个 token
  const gaps = emits.slice(1).map((e, i) => ({ from: emits[i], to: e, d: e - emits[i] }))
  const maxGap = gaps.reduce((a, b) => (b.d > a.d ? b : a))
  const arriveT = ends[WARM - 1]
  return { steps, ends, emits, maxGap, maxTbt: maxGap.d, arriveT, firstT: ends[lastPrefill], ttft: ends[lastPrefill] - arriveT,
    total: t, nChunks: lastPrefill - WARM + 1 }
}
const sims = computed(() => ({ whole: simulate(false), chunk: simulate(true) }))
const cur = computed(() => sims.value[mode.value === 'chunk' ? 'chunk' : 'whole'])
const capMs = computed(() => (mode.value === 'chunk' ? stepMs(budget.value) : stepMs(P.value)))

const bw = computed(() => W / cur.value.steps.length)
const gap = computed(() => Math.min(2, bw.value * 0.15))
const yMax = computed(() => Math.max(capMs.value, 60) * 1.15)
const yOf = (ms) => 180 - (ms / yMax.value) * 170
const xT = (t) => (t / cur.value.total) * W

const stepper = useStepper(() => cur.value.steps.length, { interval: 250 })
const now = computed(() => cur.value.steps[stepper.step.value] || cur.value.steps[0])
watch(cur, (c) => { stepper.pause(); stepper.step.value = c.steps.length - 1 }, { immediate: true })
</script>

<style scoped>
.tl { min-width: 560px; } /* 窄屏: 图保持可读, 由 .lab-viz 横向滚动 */
.bar { cursor: pointer; outline: none; }
.bar.dim { opacity: 0.3; }
.bar.now rect:first-child, .bar:focus-visible rect:first-child { fill: var(--accent-soft); }
.fixed { fill: var(--border-strong); background: var(--border-strong); }
.dec { fill: var(--left); background: var(--left); }
.pre { fill: var(--warn); background: var(--warn); }
.cap { stroke: var(--danger); stroke-dasharray: 4 3; }
.axis { stroke: var(--border-strong); }
.maxgap { stroke: var(--danger); stroke-width: 4; }
.ttft { stroke: var(--warn); stroke-width: 3; }
.tok { fill: var(--left); }
.lbl { font-size: 12px; fill: var(--text-muted); }
.lbl.bad { fill: var(--danger); }
.legend { font-size: 11px; color: var(--text-dim); margin-top: 6px; }
.sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin: 0 4px 0 10px; }
.kv.head { border: 0; padding: 0; font-size: 11px; color: var(--text-dim); }
</style>
