<!--
  gradcheck ε 实验台 (对应 llm_basic/gradcheck.py:numeric_grad)。
  只讲一件事: 数值梯度的误差是 U 形的 —— ε 太大截断误差, ε 太小舍入误差。
  曲线是浏览器里真算出来的 (JS number = float64; float32 用 Math.fround 逐步舍入模拟)。
-->
<template>
  <LabFrame
    title="gradcheck 的 ε — 为什么不是越小越好"
    sub="对同一个 softmax+CE 小网络的一个权重, 用有限差分估梯度, 再和解析梯度比相对误差。横轴 ε、纵轴误差都是对数。直接拖图上的竖线 (或滑杆) 选 ε; 切换差分方式、浮点精度, 再注入一个真 bug 看曲线长什么样。"
    module="llm_basic/gradcheck.py"
    run="python llm_basic/gradcheck.py"
    :challenge="{
      ask: '先猜: 切到 float32 后, U 形谷底往哪边移、最低误差变成多少量级? 这对 gradcheck 的写法意味着什么?',
      answer: '中心差分的总误差 ≈ ε² (截断) + δ/ε (舍入), δ 是浮点相对精度。float64 的 δ≈1e-16, 谷底在 ε≈δ^(1/3)≈1e-5, 最低误差约 1e-11; float32 的 δ≈6e-8, 谷底右移到 ε≈1e-3, 最低误差只有 1e-6 ~ 1e-5, 连 1e-6 的通过线都够不着 —— 已经分不清 “实现正确” 和 “有个小 bug”。所以 gradcheck.py 全程用 float64、ε=1e-5、中心差分; 而注入 bug 后曲线是一条和 ε 无关的高台, 这才是真错误的指纹。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: central }" @click="central = true">中心差分 (f(w+ε)−f(w−ε))/2ε</button>
        <button type="button" :class="{ active: !central }" @click="central = false">前向差分 (f(w+ε)−f(w))/ε</button>
      </div>
      <div class="row">
        <button type="button" :class="{ active: !f32 }" @click="f32 = false">float64</button>
        <button type="button" :class="{ active: f32 }" @click="f32 = true">float32</button>
        <button type="button" :class="{ active: bug }" :aria-pressed="bug" @click="bug = !bug">{{ bug ? '✓ 已注入 bug: backward 忘了减 onehot' : '注入 bug: backward 忘了减 onehot' }}</button>
      </div>
      <LabSlider v-model="k" label="步长 ε = 10^k" :min="-12" :max="-1" :step="0.25" :format="(t) => '1e' + t.toFixed(2)" />
    </template>

    <svg ref="svg" viewBox="0 0 560 300" role="img" aria-label="相对误差随 ε 变化的曲线">
      <g class="grid">
        <template v-for="e in [-16, -12, -8, -4, 0]" :key="'y' + e">
          <line :x1="X0" :x2="X1" :y1="py(e)" :y2="py(e)" /><text :x="X0 - 6" :y="py(e) + 4" class="yl">1e{{ e }}</text>
        </template>
        <template v-for="e in [-12, -10, -8, -6, -4, -2]" :key="'x' + e">
          <line :x1="px(e)" :x2="px(e)" :y1="Y0" :y2="Y1" /><text :x="px(e)" :y="Y1 + 14" class="xl">1e{{ e }}</text>
        </template>
      </g>
      <line :x1="X0" :x2="X1" :y1="py(-6)" :y2="py(-6)" class="pass" />
      <text :x="X1 - 4" :y="py(-6) - 4" class="passl">通过线 1e-6</text>
      <polyline :points="pts" class="curve" :class="{ bad: bug }" />
      <text :x="px(-11.6)" :y="Y0 + 12" class="ann">← 舍入误差 ∝ δ/ε</text>
      <text :x="px(-1.2)" :y="Y0 + 12" class="ann end">截断误差 ∝ ε{{ central ? '²' : '' }} →</text>
      <line :x1="px(-5)" :x2="px(-5)" :y1="Y1 - 6" :y2="Y1" class="tick" /><text :x="px(-5)" :y="Y1 + 28" class="xl tickl">gradcheck.py 默认 1e-5</text>
      <g class="draggable" tabindex="0" role="slider" aria-label="拖动选择 ε" :aria-valuenow="k"
        @pointerdown="start($event, { svg, onMove })" @keydown.left="k = clamp(k - 0.25, -12, -1)" @keydown.right="k = clamp(k + 0.25, -12, -1)">
        <line :x1="px(k)" :x2="px(k)" :y1="Y0" :y2="Y1" class="marker" />
        <rect :x="px(k) - 12" :y="Y0" width="24" :height="Y1 - Y0" fill="transparent" />
        <circle :cx="px(k)" :cy="py(Math.log10(cur.err))" r="7" class="dot" :class="cur.err < 1e-6 ? 'ok' : 'no'" />
      </g>
    </svg>

    <template #stats>
      <div class="kv"><span>解析梯度 {{ bug ? '(带 bug)' : '' }}</span><b>{{ ana.toFixed(8) }}</b></div>
      <div class="kv"><span>数值梯度 @ ε={{ eps.toExponential(1) }}</span><b>{{ cur.num.toFixed(8) }}</b></div>
      <div class="kv"><span>相对误差</span><b :class="cur.err < 1e-6 ? 'good' : 'bad'">{{ cur.err.toExponential(1) }}</b></div>
      <div class="kv"><span>这条曲线的最佳 ε / 最低误差</span><b>{{ best.eps.toExponential(0) }} / {{ best.err.toExponential(0) }}</b></div>
      <p class="lab-note">
        {{ bug ? '有 bug 时误差是一条 ≈ 常数的高台: 无论 ε 取多少都过不了线。ε 选得不好的 “假失败” 会随 ε 改善, 真 bug 不会。'
          : cur.err < 1e-6 ? '通过。注意谷底左右两侧斜率不同: 左边每缩小 10 倍 ε 误差涨 10 倍 (δ/ε), 右边按 ε 的幂次下降。'
          : '没过线 —— 但解析梯度其实是对的, 只是 ε 选错了。这就是 gradcheck 要固定 float64 + 1e-5 的原因。' }}
      </p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { useDrag } from '@/composables/useDrag.js'
import { clamp, range } from '@/utils/labmath.js'

const central = ref(true), f32 = ref(false), bug = ref(false), k = ref(-5)
const svg = ref(null)
const { start } = useDrag()

// 被检查的函数: 3 类 softmax + CE, 只扰动 W[0][0]。r = 每步运算后的舍入 (float32 用 fround)
const Wm = [[0.6, -0.4], [-0.3, 0.8], [0.2, 0.1]], xin = [1.3, -0.7], Y = 0
const loss = (w00, r) => {
  const z = Wm.map((row, i) => r(r((i === 0 ? w00 : row[0]) * xin[0]) + r(row[1] * xin[1])))
  const m = Math.max(...z)
  const lse = r(m + r(Math.log(r(z.reduce((s, t) => r(s + r(Math.exp(r(t - m)))), 0)))))
  return r(lse - z[Y])
}
const p0 = (() => { const z = Wm.map((r) => r[0] * xin[0] + r[1] * xin[1]); const e = z.map(Math.exp); return e[0] / (e[0] + e[1] + e[2]) })()
// ★ 正确: (p0 − 1)·x0; bug 版忘了减 onehot
const ana = computed(() => (bug.value ? p0 : p0 - 1) * xin[0])

const measure = (kk) => {
  const r = f32.value ? Math.fround : (t) => t
  const e = r(10 ** kk), w = r(Wm[0][0])
  const num = central.value ? (loss(r(w + e), r) - loss(r(w - e), r)) / (2 * e) : (loss(r(w + e), r) - loss(w, r)) / e
  const err = Math.abs(num - ana.value) / Math.max(1e-300, Math.abs(num) + Math.abs(ana.value))
  return { num, err: Math.max(err, 1e-16) }   // 恰好为 0 时压到图的地板上
}
const ks = range(45).map((i) => -12 + i * 0.25)
const curve = computed(() => ks.map((kk) => ({ k: kk, ...measure(kk) })))
const cur = computed(() => measure(k.value))
const eps = computed(() => 10 ** k.value)
const best = computed(() => { const b = curve.value.reduce((a, c) => (c.err < a.err ? c : a)); return { eps: 10 ** b.k, err: b.err } })

const X0 = 52, X1 = 548, Y0 = 14, Y1 = 262
const px = (kk) => X0 + ((kk + 12) / 11) * (X1 - X0)
const py = (le) => Y1 - ((clamp(le, -16, 1) + 16) / 17) * (Y1 - Y0)
const pts = computed(() => curve.value.map((c) => `${px(c.k).toFixed(1)},${py(Math.log10(c.err)).toFixed(1)}`).join(' '))
const onMove = ({ x }) => { k.value = clamp(Math.round((((x - X0) / (X1 - X0)) * 11 - 12) * 4) / 4, -12, -1) }
</script>

<style scoped>
.grid line { stroke: var(--border); stroke-width: 1; }
.yl { text-anchor: end; font-size: 10px; fill: var(--text-dim); }
.xl { text-anchor: middle; font-size: 10px; fill: var(--text-dim); }
.tick { stroke: var(--eye); stroke-width: 2; }
.tickl { fill: var(--eye); }
.pass { stroke: var(--left); stroke-dasharray: 5 4; }
.passl { text-anchor: end; font-size: 10px; fill: var(--left); }
.curve { fill: none; stroke: var(--accent); stroke-width: 2; }
.curve.bad { stroke: var(--danger); }
.ann { font-size: 10px; fill: var(--text-muted); }
.ann.end { text-anchor: end; }
.marker { stroke: var(--text-muted); stroke-dasharray: 3 3; }
.dot { stroke: var(--bg); stroke-width: 2; }
.dot.ok { fill: var(--left); }
.dot.no { fill: var(--danger); }
</style>
