<!--
  浮点数轴 (对应 llm_train/core/numerics.py + m06_mixed_precision)。
  只讲一件事: 一个浮点格式 = 一段可表示范围 + 一个相对精度; 梯度分布落在范围外的部分会变 0 或 Inf, loss scale 就是把分布整体平移进范围。
-->
<template>
  <LabFrame
    title="浮点数轴 — 梯度落在哪个格式的范围里?"
    sub="横轴是数值大小 (对数)。每条色带是一种格式能表示的范围: 实色 = normal, 浅色 = subnormal (精度递减)。色带以左全部变 0, 以右溢出。FP32 / BF16 的色带两端远远伸出画面 (真实范围写在行首)。灰色直方图是一批典型梯度 |g|。拖动黄色竖线看某个数在选中格式里的左右邻居; 拖 loss scale 把梯度平移进范围。"
    module="llm_train/m06"
    run="python -m llm_train.m06_mixed_precision.demo"
    :challenge="{
      ask: '选 FP16, loss scale = 1 时有多少梯度变成 0? 把 scale 调到多大最合适? 再切到 BF16 —— 还需要 loss scale 吗?',
      answer: 'FP16 最小只能到 2^-24 ≈ 6e-8, 这批梯度有一大截在它左边, 直接变 0。scale 调到 2^10~2^14 时下溢接近 0, 且还没有溢出。再大右尾就撞上 65504 变 Inf。动态 loss scaling 就是自动找这个窗口: 溢出就减半并跳过这一步, 连续多步正常就翻倍。BF16 的指数位和 FP32 一样多 (8 位), 范围覆盖整个梯度分布, 所以不需要 loss scale; 代价是尾数只剩 7 位, 相对精度比 FP16 粗 8 倍。',
    }"
  >
    <template #controls>
      <div class="row">
        <button v-for="(f, key) in FMT" :key="key" type="button" :class="{ active: fmt === key }" @click="fmt = key">{{ f.name }}</button>
      </div>
      <LabSlider v-model="logScale" label="loss scale" :min="0" :max="24" :format="(k) => '2^' + k" />
      <LabSlider v-model="gradMu" label="梯度典型量级" :min="-9" :max="-2" :step="0.5" :format="(v) => '1e' + v" />
    </template>

    <svg ref="svg" :viewBox="`0 0 ${W} ${H}`" role="img" aria-label="各浮点格式的可表示范围与梯度直方图">
      <g v-for="t in ticks" :key="t">
        <line :x1="px(t)" :x2="px(t)" :y1="HIST_H" :y2="H - 18" class="grid" />
        <text :x="px(t)" :y="H - 5" class="tick" text-anchor="middle">1e{{ t }}</text>
      </g>
      <rect
        v-for="(b, i) in hist" :key="i" :x="px(b.x0)" :width="Math.max(0.5, px(b.x1) - px(b.x0) - 0.5)"
        :y="HIST_H - b.h * (HIST_H - 8)" :height="b.h * (HIST_H - 8)"
        :class="['hbar', b.state]"
      />
      <g v-for="(f, key, i) in FMT" :key="key" class="band" :class="{ sel: fmt === key }" tabindex="0" role="button" :aria-label="`选中 ${f.name}`" @click="fmt = key" @keydown.enter="fmt = key">
        <rect :x="0" :y="rowY(i) - 2" :width="W" :height="ROW - 2" class="hit" />
        <rect :x="px(log10(f.minSub))" :width="px(log10(f.minNorm)) - px(log10(f.minSub))" :y="rowY(i) + 3" :height="ROW - 12" :fill="f.color" opacity="0.3" />
        <rect :x="px(log10(f.minNorm))" :width="px(log10(f.max)) - px(log10(f.minNorm))" :y="rowY(i) + 3" :height="ROW - 12" :fill="f.color" opacity="0.85" />
        <text :x="6" :y="rowY(i) + ROW / 2 - 2" class="fname">{{ f.name }}</text>
        <text :x="6" :y="rowY(i) + ROW / 2 + 9" class="frange">{{ f.minSub.toExponential(0) }} ~ {{ f.max.toExponential(1) }}</text>
      </g>
      <!-- ★ 可拖的数值 -->
      <g class="draggable" tabindex="0" role="slider" aria-label="被考察的数值" :aria-valuenow="logV"
        @pointerdown="start($event, { svg, onMove: ({ x }) => (logV = clamp(pxInv(x), LO, HI)) })"
        @keydown.right.prevent="logV = Math.min(HI, logV + 0.25)" @keydown.left.prevent="logV = Math.max(LO, logV - 0.25)">
        <rect :x="px(logV) - 10" :y="0" width="20" :height="H - 18" fill="transparent" />
        <line :x1="px(logV)" :x2="px(logV)" :y1="4" :y2="H - 18" class="probe" />
        <circle :cx="px(logV)" :cy="HIST_H" r="6" class="probe-dot" />
      </g>
    </svg>

    <template #stats>
      <div class="kv"><span>数值 x</span><b>{{ sci(value) }}</b></div>
      <div class="kv"><span>{{ F.name }} 左 / 右邻居</span><b class="nb">{{ sci(nb.lo) }} / {{ sci(nb.hi) }}</b></div>
      <div class="kv"><span>舍入后</span><b :class="nb.q === 0 || !isFinite(nb.q) ? 'bad' : 'good'">{{ nb.q === 0 ? '0 (下溢)' : isFinite(nb.q) ? sci(nb.q) : (F.saturate ? '饱和' : 'Inf (溢出)') }}</b></div>
      <div class="kv"><span>相对误差</span><b :class="nb.err > 0.05 ? 'bad' : ''">{{ nb.err >= 1 ? '100' : (nb.err * 100).toPrecision(2) }}%</b></div>
      <div class="kv"><span>梯度 × scale 后变 0</span><b :class="frac.under > 0.01 ? 'bad' : 'good'">{{ (frac.under * 100).toFixed(1) }}%</b></div>
      <div class="kv"><span>梯度 × scale 后溢出</span><b :class="frac.over > 0 ? 'bad' : 'good'">{{ (frac.over * 100).toFixed(1) }}%</b></div>
      <p class="lab-note">指数位决定色带多长 (范围), 尾数位决定邻居多密 (精度): {{ F.note }}</p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { useDrag } from '@/composables/useDrag.js'
import { clamp, mulberry32, randn, range } from '@/utils/labmath.js'

// 一个格式只由三个数决定 (同 core/numerics.py): 尾数位 m, 最大值, 最小 normal 指数 eMin
const mk = (name, m, max, eMin, color, note, saturate = false) =>
  ({ name, m, max, eMin, color, note, saturate, minNorm: 2 ** eMin, minSub: 2 ** (eMin - m) })
const FMT = {
  fp32: mk('FP32', 23, 3.4028e38, -126, 'var(--text-dim)', '8 位指数 + 23 位尾数, 基准。'),
  bf16: mk('BF16', 7, 3.3895e38, -126, 'var(--left)', '8 位指数 + 7 位尾数: 范围同 FP32, 精度约 0.4%。'),
  fp16: mk('FP16', 10, 65504, -14, 'var(--accent)', '5 位指数 + 10 位尾数: 精度 0.05%, 但小于 6e-8 就没了。'),
  e5m2: mk('FP8 E5M2', 2, 57344, -14, 'var(--eye)', '5 位指数 + 2 位尾数: 范围近似 FP16, 相对误差可到 ~12%; TE 配方用它装梯度。', true),
  e4m3: mk('FP8 E4M3', 3, 448, -6, 'var(--right)', '4 位指数 + 3 位尾数: 只有 ±448、最小 2e-3, 必须配 scale 才能用。', true),
}
const fmt = ref('fp16'), logScale = ref(0), gradMu = ref(-6), logV = ref(-5), svg = ref(null)
const { start } = useDrag()
const F = computed(() => FMT[fmt.value])
const log10 = Math.log10
const value = computed(() => 10 ** logV.value)

// ★ 找 x 在格式网格上的左右邻居: normal 区步长 = 2^(⌊log2 x⌋ − m), subnormal 区步长固定 2^(eMin − m)
const neighbours = (x, f) => {
  if (x > f.max) return { lo: f.max, hi: Infinity, q: f.saturate ? f.max : Infinity }
  const e = Math.max(Math.floor(Math.log2(x)), f.eMin)
  const stepSize = 2 ** (e - f.m)
  const lo = Math.floor(x / stepSize) * stepSize, hi = lo + stepSize
  return { lo, hi, q: x - lo < hi - x ? lo : hi }
}
const nb = computed(() => {
  const n = neighbours(value.value, F.value)
  const q = n.q === Infinity && F.value.saturate ? F.value.max : n.q
  return { ...n, err: isFinite(q) ? Math.abs(q - value.value) / value.value : 1 }
})

// 一批对数正态的梯度 (固定 seed, 拖滑杆不乱跳), 乘上 loss scale 后分桶
const samples = range(4000).map(((rand) => () => randn(rand))(mulberry32(7)))
const LO = -14, HI = 8, BINS = 110 // 只画梯度活动的区间; FP32/BF16 的色带两端伸出画面外 (真实范围写在行首)
const scaled = computed(() => samples.map((z) => gradMu.value + 1.3 * z + logScale.value * log10(2)))
const stateOf = (lg) => (lg > log10(F.value.max) ? 'over' : lg < log10(F.value.minSub / 2) ? 'under' : 'ok')
const hist = computed(() => {
  const counts = Array(BINS).fill(0)
  for (const lg of scaled.value) counts[clamp(Math.floor(((lg - LO) / (HI - LO)) * BINS), 0, BINS - 1)]++
  const top = Math.max(...counts)
  return counts.map((c, i) => {
    const x0 = LO + (i / BINS) * (HI - LO), x1 = LO + ((i + 1) / BINS) * (HI - LO)
    return { x0, x1, h: c / top, state: stateOf((x0 + x1) / 2) }
  }).filter((b) => b.h > 0)
})
const frac = computed(() => {
  const s = scaled.value.map(stateOf)
  return { under: s.filter((v) => v === 'under').length / s.length, over: s.filter((v) => v === 'over').length / s.length }
})

const W = 640, H = 250, HIST_H = 70, ROW = 30
const XL = 100
const px = (lg) => XL + ((clamp(lg, LO, HI) - LO) / (HI - LO)) * (W - XL - 10)
const pxInv = (x) => LO + ((x - XL) / (W - XL - 10)) * (HI - LO)
const rowY = (i) => HIST_H + 8 + i * ROW
const ticks = [-12, -8, -4, 0, 4, 8]
const sci = (v) => (v === 0 ? '0' : !isFinite(v) ? '∞' : v.toExponential(3))
</script>

<style scoped>
.grid { stroke: var(--border); }
.tick { font-size: 9px; fill: var(--text-dim); font-family: "SF Mono", Menlo, monospace; }
.fname { font-size: 10px; fill: var(--text-muted); font-family: "SF Mono", Menlo, monospace; }
.band { cursor: pointer; outline: none; }
.band .hit { fill: transparent; }
.band.sel .hit, .band:focus-visible .hit { fill: var(--accent-soft); }
.frange { font-size: 8px; fill: var(--text-dim); font-family: "SF Mono", Menlo, monospace; }
.band.sel .fname { fill: var(--text); font-weight: 600; }
.hbar { fill: var(--text-dim); }
.hbar.under, .hbar.over { fill: var(--danger); }
.probe { stroke: var(--warn); stroke-width: 2; }
.probe-dot { fill: var(--warn); stroke: var(--bg-card); stroke-width: 2; }
.nb { font-size: 12px !important; }
</style>
