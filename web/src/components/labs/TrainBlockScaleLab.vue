<!--
  Block scaling (对应 llm_train/m13_fp8_training + m15_fp4_microscaling + core/numerics.py:quant_blockwise)。
  只讲一件事: scale 由 amax 决定, 一个 outlier 会拖垮和它共享 scale 的所有元素 —— block 越小, 受害者越少。
-->
<template>
  <LabFrame
    title="Block scaling — 一个 outlier 会连累多少邻居?"
    sub="64 个元素的一行张量, 其中一个是 outlier (橙色)。上下拖动它的柱顶改变大小, 点别的柱子把 outlier 挪过去。下面两行色块是每个元素量化后的相对误差: 整张量共用一个 scale vs 每个 block 一个 scale。"
    module="llm_train/m13 · m15"
    run="python -m llm_train.m15_fp4_microscaling.demo"
    :challenge="{
      ask: '先选 FP8 E4M3: outlier 要多大, 整张量 scale 才开始把别的元素冲成 0? 再换 MXFP4 —— 为什么 FP4 的 block 必须小到 16~32?',
      answer: 'E4M3 自带约 2^15~2^17 的动态范围。outlier 在 1 万倍以内时, 整张量 scale 和 block scale 几乎打平 (m13 的训练消融里 4.54e-4 vs 4.61e-4)。到 10 万倍才有约 18% 的元素归零, 100 万倍几乎全灭。所以 block scaling 在 FP8 上是给极端 outlier 上的保险, 不是日常收益。FP4 E2M1 只有 ±{0.5,1,1.5,2,3,4,6}, 动态范围才 12 倍: outlier 只要比邻居大十几倍, 共享 scale 的邻居就大片归零, 所以 block 必须很小; 代价是每 block 多存 8 bit 的 scale (32 → +0.25 bit/元素, 16 → +0.5)。NVFP4 的 scale 带尾数, 能把 amax 精确对到 6; MXFP4 的 scale 只能是 2 的幂, amax/scale 落在 (6,8) 时最大值自己还会被饱和截断。',
    }"
  >
    <template #controls>
      <div class="row">
        <button v-for="(f, key) in RECIPES" :key="key" type="button" :class="{ active: recipe === key }" @click="recipe = key">{{ f.name }}</button>
        <button type="button" @click="seed++">换一组数</button>
      </div>
      <LabSlider v-model="logBlock" label="block 大小" :min="2" :max="6" :format="(k) => (k === 6 ? '64 (= 整张量)' : 2 ** k)" />
      <LabSlider v-model="logOut" label="outlier 倍数" :min="0" :max="6" :step="0.05" :format="(v) => '×' + fmtNum(Math.round(10 ** v))" />
    </template>

    <svg ref="svg" :viewBox="`0 0 ${W} ${H}`" role="img" aria-label="张量各元素的绝对值 (对数) 与 block scale">
      <rect v-for="b in nBlocks" :key="'bg' + b" :x="bx((b - 1) * block)" :width="block * BW" y="6" :height="PLOT" :class="['blk', { alt: b % 2 }]" />
      <g v-for="t in [-4, -2, 0, 2, 4]" :key="t">
        <line :x1="X0" :x2="W" :y1="py(t)" :y2="py(t)" class="grid" />
        <text :x="X0 - 4" :y="py(t) + 3" class="tick" text-anchor="end">1e{{ t }}</text>
      </g>
      <rect
        v-for="(v, i) in x" :key="i" :x="bx(i) + 1" :width="BW - 2" :y="py(log10(Math.abs(v)))" :height="Math.max(1, PLOT + 6 - py(log10(Math.abs(v))))"
        :class="['bar', { out: i === outIdx }]" tabindex="0" role="button" :aria-label="`把 outlier 放到第 ${i} 个元素`"
        @click="outIdx = i" @keydown.enter="outIdx = i"
      />
      <!-- 每个 block 的 "能表示的最小非零值" 线: 低于它的元素会变 0 -->
      <line v-for="(f, b) in blockRes.floors" :key="'fl' + b" :x1="bx(b * block)" :x2="bx((b + 1) * block)" :y1="py(log10(f))" :y2="py(log10(f))" class="floor" />
      <circle
        :cx="bx(outIdx) + BW / 2" :cy="py(log10(Math.abs(x[outIdx])))" r="7" class="handle draggable"
        tabindex="0" role="slider" aria-label="outlier 大小" :aria-valuenow="logOut"
        @pointerdown="start($event, { svg, onMove: ({ y }) => (logOut = clamp(Math.round((pyInv(y) - log10(BASE)) * 20) / 20, 0, 6)) })"
        @keydown.up.prevent="logOut = Math.min(6, logOut + 0.25)" @keydown.down.prevent="logOut = Math.max(0, logOut - 0.25)"
      />
      <g v-for="(row, ri) in [tensorRes, blockRes]" :key="ri">
        <text :x="X0 - 4" :y="PLOT + 30 + ri * 22" class="tick" text-anchor="end">{{ ri ? 'block' : '整张量' }}</text>
        <rect
          v-for="(e, i) in row.err" :key="i" :x="bx(i) + 0.5" :width="BW - 1" :y="PLOT + 18 + ri * 22" height="16" rx="1.5"
          :fill="e >= 1 ? 'var(--danger)' : heat(e * 2.5, 'var(--warn)')" class="errc"
        ><title>#{{ i }}: x = {{ x[i].toExponential(2) }}, Q(x) = {{ row.q[i].toExponential(2) }}, 误差 {{ (e * 100).toFixed(0) }}%</title></rect>
      </g>
    </svg>
    <p class="lab-note">红 = 被冲成 0 (误差 100%), 黄越深误差越大。红色虚线 = 该 block 里能表示的最小非零值的一半, 柱子低于它就归零。</p>

    <template #stats>
      <div class="kv"><span>整张量 scale: 归零元素</span><b :class="tensorRes.zeros ? 'bad' : 'good'">{{ tensorRes.zeros }} / 64</b></div>
      <div class="kv"><span>block scale: 归零元素</span><b :class="blockRes.zeros ? 'bad' : 'good'">{{ blockRes.zeros }} / 64</b></div>
      <div class="kv"><span>平均相对误差 (整张量 → block)</span><b :class="{ good: blockRes.mean < tensorRes.mean * 0.7 }">{{ pct(tensorRes.mean) }} → {{ pct(blockRes.mean) }}</b></div>
      <div class="kv"><span>每元素位宽 (含 scale)</span><b>{{ (R.bits + R.scaleBits / block).toFixed(2) }} bit</b></div>
      <p class="lab-note">★ {{ R.note }}</p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { useDrag } from '@/composables/useDrag.js'
import { clamp, fmtNum, heat, mulberry32, randn, range, sum } from '@/utils/labmath.js'

const NEL = 64, BASE = 0.02
// 与 core/numerics.py:fake_quant_float 同一个算法: 饱和 → normal 区按尾数舍入 → subnormal 区固定步长
const FLOAT = { e4m3: [3, 448, -6], e2m1: [1, 6, 0] }
const fq = (v, fmt) => {
  const [m, max, eMin] = FLOAT[fmt]
  const a = Math.min(Math.abs(v), max)
  if (a === 0) return 0
  const e = Math.max(Math.floor(Math.log2(a)), eMin)
  const q = Math.round(a / 2 ** (e - m)) * 2 ** (e - m)
  return Math.sign(v) * q
}
const RECIPES = {
  fp8: { name: 'FP8 E4M3', fmt: 'e4m3', bits: 8, scaleBits: 32, scale: (amax) => amax / 448, note: 'scale = amax / 448 (高精度存)。E4M3 自带 2^15 以上的动态范围: 没有极端 outlier 时整张量 scale 就够用, DeepSeek-V3 的 block 128 是为 outlier 买的保险。' },
  mxfp4: { name: 'MXFP4 (E8M0 scale)', fmt: 'e2m1', bits: 4, scaleBits: 8, scale: (amax) => 2 ** (Math.floor(Math.log2(amax)) - 2), note: 'scale = 2^(⌊log2 amax⌋−2), 只能是 2 的幂: amax/scale 落在 [4,8), 大于 6 的会被饱和。标准 block = 32。' },
  nvfp4: { name: 'NVFP4 (E4M3 scale)', fmt: 'e2m1', bits: 4, scaleBits: 8, scale: (amax, tmax) => { const ts = tmax / (448 * 6); return fq(amax / 6 / ts, 'e4m3') * ts }, note: 'block scale 本身是 E4M3 (带 3 位尾数), 再乘一个整张量的 FP32 scale: amax 几乎精确贴到 6。标准 block = 16。' },
}
const recipe = ref('mxfp4'), logBlock = ref(4), logOut = ref(1.5), outIdx = ref(21), seed = ref(3), svg = ref(null)
const { start } = useDrag()
const R = computed(() => RECIPES[recipe.value])
const block = computed(() => 2 ** logBlock.value)
const nBlocks = computed(() => NEL / block.value)
const log10 = Math.log10

const x = computed(() => {
  const rand = mulberry32(seed.value)
  return range(NEL).map((i) => (i === outIdx.value ? BASE * 10 ** logOut.value : randn(rand) * BASE || BASE))
})

// ★ 同一个量化器, 只是 "谁和谁共享 amax" 不同
const quantize = (xs, B, r) => {
  const tmax = Math.max(...xs.map(Math.abs))
  const q = [], floors = []
  for (let b0 = 0; b0 < xs.length; b0 += B) {
    const blk = xs.slice(b0, b0 + B)
    const s = r.scale(Math.max(...blk.map(Math.abs)), tmax) || 1
    floors.push(s * 2 ** (FLOAT[r.fmt][2] - FLOAT[r.fmt][0]) / 2)
    blk.forEach((v) => q.push(fq(v / s, r.fmt) * s))
  }
  const err = xs.map((v, i) => Math.min(1, Math.abs(q[i] - v) / Math.abs(v)))
  return { q, err, floors, zeros: q.filter((v) => v === 0).length, mean: sum(err) / err.length }
}
const tensorRes = computed(() => quantize(x.value, NEL, R.value))
const blockRes = computed(() => quantize(x.value, block.value, R.value))

const W = 640, PLOT = 170, H = PLOT + 66, X0 = 44, BW = (W - X0) / NEL
const LO = -5, HI = 4.6
const bx = (i) => X0 + i * BW
const py = (lg) => 6 + (1 - (clamp(lg, LO, HI) - LO) / (HI - LO)) * PLOT
const pyInv = (yy) => LO + (1 - (yy - 6) / PLOT) * (HI - LO)
const pct = (v) => (v * 100).toFixed(1) + '%'
</script>

<style scoped>
.blk { fill: transparent; }
.blk.alt { fill: var(--accent-soft); }
.grid { stroke: var(--border); }
.tick { font-size: 9px; fill: var(--text-dim); font-family: "SF Mono", Menlo, monospace; }
.bar { fill: var(--text-dim); cursor: pointer; outline: none; }
.bar:hover, .bar:focus-visible { fill: var(--accent); }
.bar.out { fill: var(--eye); }
.floor { stroke: var(--danger); stroke-dasharray: 3 2; stroke-width: 1.5; }
.handle { fill: var(--eye); stroke: var(--bg-card); stroke-width: 2; }
.errc { stroke: var(--border); stroke-width: 0.5; }
</style>
