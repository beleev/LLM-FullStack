<!--
  学习率调度 (对应 llm_train/m10_training_stability:warmup_cosine_lr / wsd_lr)。
  只讲一件事: cosine 的每一步 LR 都依赖总步数, 想加训就得从头来; WSD 的稳定段与总步数无关, 任何稳定段的 checkpoint 都能接着训。
-->
<template>
  <LabFrame
    title="Cosine vs WSD — 训到一半想加训怎么办?"
    sub="实线 = 按原计划 T₀ 步走的学习率; 虚线 = 假如一开始就知道要训 T₁ 步, 应该走的学习率。红色阴影 = 已经训过的步里, 实际 LR 与「本该用的」不一致的部分。拖动横轴上的三个手柄: warmup 结束点、原计划 T₀、新计划 T₁。"
    module="llm_train/m10"
    run="python -m llm_train.m10_training_stability.demo"
    :challenge="{
      ask: '原计划 1000 步, 训完发现 loss 还在降, 想加到 1600 步。cosine 和 WSD 各要回退多少步重训?',
      answer: 'cosine 的 LR = f(step / 总步数)。总步数一改, warmup 之后每一步的 LR 都变了。已经训过的 1000 步里, 只有 warmup 那一小段还「算数」。于是要么几乎从头重训, 要么硬着头皮把 LR 重新拉高 (re-warmup, loss 会先反弹)。WSD 的稳定段就是一条水平线, 与总步数无关。只要回到退火刚开始时的那个 checkpoint (第 900 步附近), 继续保持高 LR 训到 1440 步再退火即可。浪费的只有那约 100 步退火。这也是 MiniCPM / DeepSeek-V3 / Kimi 用 WSD 的原因: 同一条主干上随时可以分叉出一个退火分支来拿「成品」模型。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: mode === 'cosine' }" @click="mode = 'cosine'">Warmup + Cosine</button>
        <button type="button" :class="{ active: mode === 'wsd' }" @click="mode = 'wsd'">WSD (Warmup-Stable-Decay)</button>
      </div>
      <LabSlider v-model="decayFrac" label="WSD 退火占比" :min="0.05" :max="0.4" :step="0.05" :format="(v) => Math.round(v * 100) + '%'" />
      <LabSlider v-model="minRatio" label="最终 LR / 峰值 LR" :min="0" :max="0.3" :step="0.05" />
    </template>

    <svg ref="svg" :viewBox="`0 0 ${W} ${H}`" role="img" aria-label="学习率随训练步数的变化">
      <line :x1="X0" :x2="W - 10" :y1="py(0)" :y2="py(0)" class="axis" />
      <g v-for="t in [0, 500, 1000, 1500, 2000]" :key="t">
        <line :x1="px(t)" :x2="px(t)" :y1="py(0)" :y2="py(0) + 4" class="axis" />
        <text :x="px(t)" :y="py(0) + 15" class="tick" text-anchor="middle">{{ t }}</text>
      </g>
      <text :x="X0 - 6" :y="py(1) + 3" class="tick" text-anchor="end">峰值</text>
      <line :x1="X0" :x2="W - 10" :y1="py(1)" :y2="py(1)" class="gridl" />
      <polygon :points="gapPoly" class="gap" />
      <polyline :points="poly(t1)" class="ideal" />
      <polyline :points="poly(t0)" class="plan" />
      <g v-for="h in handles" :key="h.key" class="draggable" tabindex="0" role="slider" :aria-label="h.label" :aria-valuenow="h.v"
        @pointerdown="start($event, { svg, onMove: ({ x }) => h.set(Math.round(pxInv(x) / 10) * 10) })"
        @keydown.right.prevent="h.set(h.v + 10)" @keydown.left.prevent="h.set(h.v - 10)">
        <line :x1="px(h.v)" :x2="px(h.v)" :y1="py(1) - 8" :y2="py(0)" :stroke="h.color" stroke-dasharray="2 3" />
        <rect :x="px(h.v) - 9" :y="py(0) + 20" width="18" height="18" rx="4" :fill="h.color" />
        <text :x="px(h.v)" :y="py(0) + 52" class="hl" text-anchor="middle" :fill="h.color">{{ h.short }} {{ h.v }}</text>
      </g>
    </svg>

    <template #stats>
      <div class="kv"><span>已训 {{ t0 }} 步里仍「算数」的</span><b :class="keepFrac > 0.5 ? 'good' : 'bad'">{{ keep }} 步</b></div>
      <div class="kv"><span>加训到 {{ t1 }} 需要回退重训</span><b :class="keepFrac > 0.5 ? 'good' : 'bad'">{{ t0 - keep }} 步</b></div>
      <div class="kv"><span>第 {{ t0 }} 步: 实际 LR vs 本该用的</span><b>{{ lr(t0 - 1, t0).toFixed(2) }} vs {{ lr(t0 - 1, t1).toFixed(2) }}</b></div>
      <div class="kv"><span>LR 是否依赖总步数</span><b :class="mode === 'wsd' ? 'good' : 'bad'">{{ mode === 'wsd' ? '仅退火段' : '几乎每一步' }}</b></div>
      <p class="lab-note">
        ★ WSD 的稳定段 <code class="inline">return base_lr</code> 里没有 total_steps。
        warmup 的意义两者相同: Adam 的二阶矩估计在最初几百步很不准, 先用小 LR 走。
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

const mode = ref('wsd'), warm = ref(100), t0 = ref(1000), t1 = ref(1600), decayFrac = ref(0.1), minRatio = ref(0.1), svg = ref(null)
const { start } = useDrag()
const XMAX = 2000

// 与 Python 逐行相同 (base_lr = 1)
const lr = (step, total) => {
  const w = warm.value, mr = minRatio.value
  if (step < w) return (step + 1) / w
  if (mode.value === 'cosine') {
    const p = (step - w) / Math.max(1, total - w)
    return mr + (1 - mr) * 0.5 * (1 + Math.cos(Math.PI * p))
  }
  const decayStart = total - Math.floor(total * decayFrac.value)
  if (step < decayStart) return 1 // ★ 稳定段: 与 total 无关
  return 1 - (1 - mr) * ((step - decayStart) / Math.max(1, total - decayStart))
}

// 已训过的 [0, T0) 里, 有多少步的 LR 与 "按 T1 规划" 的一致 (从头数, 第一次不一致就停)
const keep = computed(() => {
  let s = 0
  while (s < t0.value && Math.abs(lr(s, t0.value) - lr(s, t1.value)) < 1e-9) s++
  return s
})
const keepFrac = computed(() => keep.value / t0.value)

const W = 620, H = 250, X0 = 40
const px = (s) => X0 + (s / XMAX) * (W - X0 - 14)
const pxInv = (x) => ((x - X0) / (W - X0 - 14)) * XMAX
const py = (v) => 22 + (1 - v) * (H - 90)
const pts = (total) => range(Math.floor(total / 5) + 1).map((i) => Math.min(i * 5, total - 1)).map((s) => [px(s), py(lr(s, total))])
const poly = (total) => pts(total).map((p) => p.join(',')).join(' ')
const gapPoly = computed(() => {
  const ss = range(Math.floor(t0.value / 5) + 1).map((i) => Math.min(i * 5, t0.value - 1))
  const top = ss.map((s) => `${px(s)},${py(lr(s, t1.value))}`)
  const bot = ss.map((s) => `${px(s)},${py(lr(s, t0.value))}`).reverse()
  return [...top, ...bot].join(' ')
})

const handles = computed(() => [
  { key: 'w', short: 'warmup', label: 'warmup 结束步', v: warm.value, color: 'var(--eye)', set: (v) => (warm.value = clamp(v, 10, t0.value - 100)) },
  { key: 't0', short: 'T₀', label: '原计划总步数', v: t0.value, color: 'var(--accent)', set: (v) => (t0.value = clamp(v, warm.value + 100, t1.value)) },
  { key: 't1', short: 'T₁', label: '新计划总步数', v: t1.value, color: 'var(--left)', set: (v) => (t1.value = clamp(v, t0.value, XMAX)) },
])
</script>

<style scoped>
.axis { stroke: var(--border-strong); }
.gridl { stroke: var(--border); stroke-dasharray: 2 4; }
.tick { font-size: 10px; fill: var(--text-dim); font-family: "SF Mono", Menlo, monospace; }
.plan { fill: none; stroke: var(--accent); stroke-width: 2.5; }
.ideal { fill: none; stroke: var(--left); stroke-width: 2; stroke-dasharray: 6 4; }
.gap { fill: var(--danger); opacity: 0.25; }
.hl { font-size: 10px; font-family: "SF Mono", Menlo, monospace; }
</style>
