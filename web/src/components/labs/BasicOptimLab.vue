<!--
  优化器轨迹实验台 (对应 llm_basic/optim.py:adam_step)。
  病态二次损失 f = ½(x² + κ·y²): y 方向陡, x 方向平。同一个 lr 下看 SGD / 动量 / Adam 怎么走。
-->
<template>
  <LabFrame
    title="优化器轨迹 — 同一个学习率, 三种走法"
    sub="等高线是一个又窄又长的山谷 (y 方向比 x 方向陡 κ 倍)。拖动图上的起点, 调 lr / β / κ, 按播放看三条轨迹。注意 Adam 的头几步: 两个坐标迈的步子几乎一样大, 与坡度无关。"
    module="llm_basic/optim.py"
    run="python llm_basic/optim.py"
    :challenge="{
      ask: '把 κ 拖到 40, lr 拖到 0.06。先猜: 三个优化器谁先发散? 再把 lr 调回 0.04, 谁在平缓的 x 方向走得最慢?',
      answer: 'SGD 先发散: 它在 y 方向每步乘 (1 − lr·κ), lr > 2/κ = 0.05 时绝对值大于 1, 越震越大。动量的上限是 2(1+β)/κ, 宽得多。Adam 每步位移 ≈ lr·m̂/√v̂, 大小约等于 lr, 根本不看曲率, 不会发散 (只会在谷底以 lr 为幅度抖)。lr 调小后 SGD 在 x 方向每步只缩 (1 − lr), 慢得要命 —— 这就是病态: lr 被最陡的方向卡死, 最平的方向就学不动。Adam 用 √v̂ 给每个坐标单独归一化, 等于每个参数有自己的学习率。',
    }"
  >
    <template #controls>
      <LabSlider v-model="lr" label="学习率 lr" :min="0.01" :max="0.4" :step="0.01" :format="(t) => t.toFixed(2)" />
      <LabSlider v-model="beta" label="动量 β (Adam 的 β₁)" :min="0" :max="0.99" :step="0.01" :format="(t) => t.toFixed(2)" />
      <LabSlider v-model="kappa" label="条件数 κ" :min="1" :max="50" />
      <StepPlayer :stepper="stepper" :label="`第 ${stepper.step.value} 步`" />
    </template>

    <svg ref="svg" viewBox="-3.2 -2.1 6.4 4.2" role="group" aria-label="损失等高线与优化轨迹">
      <ellipse v-for="c in LEVELS" :key="c" cx="0" cy="0" :rx="Math.sqrt(2 * c)" :ry="Math.sqrt(2 * c / kappa)" class="contour" />
      <line x1="-3.2" x2="3.2" y1="0" y2="0" class="axis" /><line y1="-2.1" y2="2.1" x1="0" x2="0" class="axis" />
      <g v-for="o in OPTS" :key="o.id">
        <polyline :points="poly(o.id)" class="path" :style="{ stroke: o.color }" />
        <circle :cx="at(o.id)[0]" :cy="-at(o.id)[1]" r="0.07" :style="{ fill: o.color }" />
      </g>
      <circle
        :cx="x0" :cy="-y0" r="0.13" class="start draggable" tabindex="0" role="slider" aria-label="拖动起点"
        :aria-valuetext="`${x0.toFixed(1)}, ${y0.toFixed(1)}`"
        @pointerdown="start($event, { svg, onMove })"
        @keydown.left="x0 = clamp(x0 - 0.1, -3, 3)" @keydown.right="x0 = clamp(x0 + 0.1, -3, 3)"
        @keydown.up.prevent="y0 = clamp(y0 + 0.1, -2, 2)" @keydown.down.prevent="y0 = clamp(y0 - 0.1, -2, 2)"
      />
    </svg>
    <div class="legend">
      <span v-for="o in OPTS" :key="o.id"><i :style="{ background: o.color }" />{{ o.label }}</span>
      <span><i class="s" />起点 (可拖)</span>
    </div>

    <template #stats>
      <div v-for="o in OPTS" :key="o.id" class="kv">
        <span>{{ o.label }} 当前 loss</span>
        <b :class="lossAt(o.id) < 1e-3 ? 'good' : lossAt(o.id) > f0 ? 'bad' : ''">{{ fmt(lossAt(o.id)) }}</b>
      </div>
      <div class="kv"><span>SGD 稳定上限 2/κ</span><b :class="lr < 2 / kappa ? 'good' : 'bad'">{{ (2 / kappa).toFixed(3) }}</b></div>
      <div class="kv"><span>动量稳定上限 2(1+β)/κ</span><b :class="lr < 2 * (1 + beta) / kappa ? 'good' : 'bad'">{{ (2 * (1 + beta) / kappa).toFixed(3) }}</b></div>
      <p class="lab-note">
        SGD 每步在 y 方向乘 (1 − lr·κ) = {{ (1 - lr * kappa).toFixed(2) }}, 在 x 方向乘 (1 − lr) = {{ (1 - lr).toFixed(2) }}。
        前者绝对值超过 1 就发散, 后者接近 1 就几乎不动 —— 一个 lr 没法同时伺候两个方向。
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
import { useDrag } from '@/composables/useDrag.js'
import { clamp } from '@/utils/labmath.js'

const N = 80, LEVELS = [0.05, 0.25, 0.75, 1.5, 3, 5]
const OPTS = [
  { id: 'sgd', label: 'SGD', color: 'var(--danger)' },
  { id: 'mom', label: '动量', color: 'var(--eye)' },
  { id: 'adam', label: 'Adam', color: 'var(--left)' },
]
const lr = ref(0.04), beta = ref(0.8), kappa = ref(20), x0 = ref(-2.6), y0 = ref(1.4)
const svg = ref(null)
const { start } = useDrag()
const onMove = ({ x, y }) => { x0.value = clamp(x, -3, 3); y0.value = clamp(-y, -2, 2) }

const f = ([x, y]) => 0.5 * (x * x + kappa.value * y * y)
const grad = ([x, y]) => [x, kappa.value * y]

const paths = computed(() => {
  const a = lr.value, b1 = beta.value, b2 = 0.999, eps = 1e-8
  const out = { sgd: [], mom: [], adam: [] }
  let w = [x0.value, y0.value]
  for (let t = 0; t <= N; t++) { out.sgd.push(w); const g = grad(w); w = [w[0] - a * g[0], w[1] - a * g[1]] }
  w = [x0.value, y0.value]; let v = [0, 0]
  for (let t = 0; t <= N; t++) { out.mom.push(w); const g = grad(w); v = v.map((vi, i) => b1 * vi + g[i]); w = w.map((wi, i) => wi - a * v[i]) }
  w = [x0.value, y0.value]; let m = [0, 0], s = [0, 0]
  for (let t = 1; t <= N + 1; t++) {
    out.adam.push(w); const g = grad(w)
    m = m.map((mi, i) => b1 * mi + (1 - b1) * g[i]); s = s.map((si, i) => b2 * si + (1 - b2) * g[i] * g[i])
    // ★ 偏差修正后按坐标归一化: 步长 ≈ lr, 与该方向的坡度无关
    w = w.map((wi, i) => wi - a * (m[i] / (1 - b1 ** t)) / (Math.sqrt(s[i] / (1 - b2 ** t)) + eps))
  }
  return out
})

const stepper = useStepper(() => N + 1, { interval: 120 })
watch([lr, beta, kappa, x0, y0], () => { stepper.pause(); stepper.step.value = N })
stepper.step.value = N
const at = (id) => paths.value[id][stepper.step.value] || paths.value[id][N]
// 发散时坐标会到 1e30, 画之前夹住, 否则 SVG 渲染异常
const poly = (id) => paths.value[id].slice(0, stepper.step.value + 1).map(([x, y]) => `${clamp(x, -50, 50).toFixed(3)},${clamp(-y, -50, 50).toFixed(3)}`).join(' ')
const lossAt = (id) => f(at(id))
const f0 = computed(() => f([x0.value, y0.value]))
const fmt = (t) => (t > 1e4 ? '发散' : t < 1e-3 ? t.toExponential(1) : t.toFixed(3))
</script>

<style scoped>
svg { background: var(--bg-elev); border-radius: var(--radius-sm); width: 100%; }
.contour { fill: none; stroke: var(--border-strong); stroke-width: 0.015; }
.axis { stroke: var(--border); stroke-width: 0.01; }
.path { fill: none; stroke-width: 0.03; stroke-linejoin: round; }
.start { fill: var(--accent); stroke: var(--text); stroke-width: 0.03; }
.start:focus-visible { outline: none; stroke-width: 0.07; }
.legend { display: flex; flex-wrap: wrap; gap: 14px; margin-top: 8px; font-size: 12px; color: var(--text-muted); }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; }
.legend i.s { background: var(--accent); }
</style>
