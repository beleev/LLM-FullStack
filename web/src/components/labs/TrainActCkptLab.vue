<!--
  激活重算 (对应 llm_train/m07_activation_checkpointing:run)。
  只讲一件事: 峰值 ≈ L/k (常驻的段边界) + k (当前段重算出来的瞬时激活), k≈√L 时最小。
-->
<template>
  <LabFrame
    title="激活重算 — 段长 k 怎么选?"
    sub="上图: 峰值激活份数随段长 k 的变化, 左右拖动圆点改 k。下图: 一次前向 + 反向里每个激活的生死 —— 紫色 = 前向时存下的段边界, 橙色 = 反向时临时重算出来的, 用完即弃。"
    module="llm_train/m07"
    run="python -m llm_train.m07_activation_checkpointing.demo"
    :challenge="{
      ask: 'L = 36 层。k=1 (全存)、k=6、k=36 (只存输入) 三种, 峰值激活各是多少份? 哪个最省?',
      answer: 'k=1 是 37 份; k=36 只存了输入, 但反向时要把整段 36 层一次性重算出来, 峰值还是 37 份 —— 白白多算一遍前向却一点没省。k=6=√L 时常驻 7 个边界 + 段内瞬时 5 份 = 12 份, 最省。重算的计算代价几乎与 k 无关 (都是约多一次前向, +33% 以内), 所以 k 只需要对着显存选。',
    }"
  >
    <template #controls>
      <LabSlider v-model="L" label="层数 L" :min="4" :max="64" />
      <LabSlider v-model="k" label="段长 k (每 k 层存一次)" :min="1" :max="L" />
      <StepPlayer :stepper="stepper" :label="frame.label" />
    </template>

    <svg ref="svg" :viewBox="`0 0 ${W} ${H}`" role="img" aria-label="峰值激活随段长变化曲线">
      <line :x1="X0" :x2="W - 10" :y1="H - 24" :y2="H - 24" class="axis" />
      <line :x1="X0" :x2="X0" :y1="10" :y2="H - 24" class="axis" />
      <text :x="X0 - 6" :y="py(L + 1) + 4" class="tick" text-anchor="end">{{ L + 1 }}</text>
      <text :x="X0 - 6" :y="py(best.peak) + 4" class="tick" text-anchor="end">{{ best.peak }}</text>
      <text :x="W - 10" :y="H - 8" class="tick" text-anchor="end">段长 k →</text>
      <text :x="X0 + 4" :y="14" class="tick">峰值激活份数</text>
      <line :x1="px(Math.sqrt(L))" :x2="px(Math.sqrt(L))" :y1="14" :y2="H - 24" class="sqrt" />
      <text :x="px(Math.sqrt(L)) + 4" :y="26" class="sqrt-t">√L = {{ Math.sqrt(L).toFixed(1) }}</text>
      <polyline :points="curve.map((c) => `${px(c.k)},${py(c.peak)}`).join(' ')" class="curve" />
      <circle
        :cx="px(k)" :cy="py(cur.peak)" r="8" class="handle draggable" tabindex="0" role="slider"
        aria-label="段长 k" :aria-valuenow="k"
        @pointerdown="start($event, { svg, onMove: ({ x }) => (k = Math.round(clamp(pxInv(x), 1, L))) })"
        @keydown.right.prevent="k = Math.min(L, k + 1)" @keydown.left.prevent="k = Math.max(1, k - 1)"
      />
    </svg>

    <div class="cells acts" :style="{ gridTemplateColumns: `repeat(${Math.min(L + 1, 22)}, minmax(18px, 1fr))` }">
      <span
        v-for="i in L + 1" :key="i" class="cell"
        :class="[frame.live[i - 1] === 'ckpt' ? 'on' : frame.live[i - 1] === 'tmp' ? 'hot' : 'dim', { now: frame.at === i - 1 }]"
        :title="`a${i - 1}: 第 ${i - 1} 层的输入` + (frame.live[i - 1] ? '' : ' (此刻不在显存里)')"
      >{{ i - 1 }}</span>
    </div>

    <template #stats>
      <div class="kv"><span>此刻活着的激活</span><b>{{ frame.count }}</b></div>
      <div class="kv"><span>峰值 (k = {{ k }})</span><b :class="cur.peak === best.peak ? 'good' : cur.peak > L * 0.8 ? 'bad' : ''">{{ cur.peak }} 份</b></div>
      <div class="kv"><span>最优 k = {{ best.k }}</span><b class="good">{{ best.peak }} 份</b></div>
      <div class="kv"><span>前向层调用次数</span><b>{{ cur.fwd }} <small>(不重算 {{ L }})</small></b></div>
      <div class="kv"><span>额外计算 (F:B = 1:2)</span><b :class="{ bad: cur.extra > 0.3 }">+{{ (cur.extra * 100).toFixed(1) }}%</b></div>
      <p class="lab-note">
        ★ 峰值出现在反向刚重算完最后一段时: 所有段边界都还在, 再加上这一段的内部激活。
        梯度与不重算逐位相同 —— 换的只是"存"和"算"。
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
import { clamp, range } from '@/utils/labmath.js'

const L = ref(36), k = ref(6), svg = ref(null)
const { start } = useDrag()
watch(L, (v) => { if (k.value > v) k.value = v })

// 和 Python 的 run() 逐行对应: live 记录每个激活是 ckpt(前向存的) / tmp(重算出的) / 无
const simulate = (Ln, seg, wantFrames) => {
  const live = Array(Ln + 1).fill(null)
  let peak = 0, fwd = 0
  const frames = []
  const snap = (label, at) => {
    const count = live.filter(Boolean).length
    peak = Math.max(peak, count)
    if (wantFrames) frames.push({ label, at, live: [...live], count })
  }
  live[0] = 'ckpt'; snap('输入', 0)
  for (let i = 0; i < Ln; i++) {
    fwd++
    if ((i + 1) % seg === 0 || i + 1 === Ln) live[i + 1] = 'ckpt' // 只在段边界保存
    snap(`前向 层${i}`, i + 1)
  }
  for (let s0 = Math.floor((Ln - 1) / seg) * seg; s0 >= 0; s0 -= seg) {
    const end = Math.min(s0 + seg, Ln)
    for (let i = s0; i < end - 1; i++) { fwd++; live[i + 1] = 'tmp'; snap(`重算 层${i}`, i + 1) } // ★ 瞬时激活也占显存
    for (let i = end - 1; i >= s0; i--) { live[i + 1] = null; snap(`反向 层${i}`, i) }
  }
  return { peak, fwd, frames, extra: (fwd - Ln) / (3 * Ln) }
}

const curve = computed(() => range(L.value).map((i) => ({ k: i + 1, ...simulate(L.value, i + 1, false) })))
const best = computed(() => curve.value.reduce((a, b) => (b.peak < a.peak ? b : a)))
const cur = computed(() => simulate(L.value, k.value, true))
const stepper = useStepper(() => cur.value.frames.length, { interval: 160 })
const frame = computed(() => cur.value.frames[Math.min(stepper.step.value, cur.value.frames.length - 1)])
// 换参数后停在峰值那一帧: 先看到最挤的时刻
watch(cur, (c) => { stepper.pause(); stepper.step.value = c.frames.findIndex((f) => f.count === c.peak) }, { immediate: true })

const W = 460, H = 190, X0 = 40
const px = (kk) => X0 + ((kk - 1) / Math.max(1, L.value - 1)) * (W - X0 - 20)
const pxInv = (x) => 1 + ((x - X0) / (W - X0 - 20)) * (L.value - 1)
const py = (p) => 18 + (1 - p / (L.value + 1)) * (H - 46)
</script>

<style scoped>
.axis { stroke: var(--border-strong); }
.tick { font-size: 10px; fill: var(--text-dim); font-family: "SF Mono", Menlo, monospace; }
.curve { fill: none; stroke: var(--accent); stroke-width: 2; }
.sqrt { stroke: var(--left); stroke-dasharray: 4 3; }
.sqrt-t { font-size: 10px; fill: var(--left); }
.handle { fill: var(--warn); stroke: var(--bg-card); stroke-width: 2; }
.acts { margin-top: 12px; }
.acts .cell { min-width: 18px; }
.cell.now { outline: 2px solid var(--accent); outline-offset: 1px; }
small { font-size: 10px; color: var(--text-dim); font-weight: 400; }
</style>
