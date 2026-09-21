<!--
  Mamba 选择性扫描实验台 (对应 llm_models/layers/sparse/ssm.py:SelectiveSSM.forward)。
  只讲一件事: h_t = exp(Δ_t·A)·h_{t-1} + Δ_t·B·x_t 里的 Δ_t 由输入决定 ——
  Δ 大 = "忘掉过去, 记住这个"; Δ 小 = "这个 token 当没看见, 状态原样传下去"。
-->
<template>
  <LabFrame
    title="Mamba 选择性扫描 — Δ 决定记谁、忘谁"
    sub="一句话里只有 “7” 是要记住的, 其余是废话。上排柱子是每个 token 的 Δ_t: 直接上下拖。中间折线是隐状态 h_t, 下排是 “最终状态里每个 token 占了多少”。目标: 让 “7” 在最终状态里占比最高。"
    module="llm_models/layers/sparse/ssm.py"
    run="python -m llm_models.run_models.language_models.mamba.infer_mamba"
    :challenge="{
      ask: '先切到 “LTI (所有 token 共用一个 Δ)”, 怎么拖全局 Δ 都试一遍: “7” 的占比最高能到多少? 为什么? 再切回选择性, 只改两类 token 的 Δ 就让占比超过 90%。',
      answer: 'LTI 下 token j 对最终状态的贡献是 Δ·exp(−Δ·a·(T−1−j)): 越早的 token 被衰减的次数越多, 所以 “7” (后面还有 7 个 token) 的占比永远低于它后面的任何一个废话, 极限是 Δ→0 时人人平等的 1/10。线性时不变系统只能按 “距离现在多远” 加权, 不能按 “内容重不重要” 加权。Mamba 让 Δ_t = softplus(Linear(x_t)) 依赖输入: 给 “7” 一个大 Δ (写入强、同时 exp(Δ·A)≈0 把之前的状态清掉), 给废话 Δ≈0 (写入≈0 且 exp(Δ·A)≈1 状态原样通过)。一个标量同时控制写入门和遗忘门, 这就是 “选择性”。代价: 系统变成时变的, 不能再用卷积并行, 只能用并行 scan。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: selective }" @click="selective = true">选择性 (每个 token 自己的 Δ)</button>
        <button type="button" :class="{ active: !selective }" @click="selective = false">LTI (所有 token 共用一个 Δ, 如 S4)</button>
        <button type="button" @click="ideal">一键理想选择</button>
        <button type="button" @click="flat">全部重置为 0.5</button>
      </div>
      <LabSlider v-model="a" label="衰减速率 a (A = −a)" :min="0.2" :max="3" :step="0.1" :format="(t) => t.toFixed(1)" />
      <LabSlider v-if="!selective" v-model="dGlobal" label="全局 Δ" :min="0.02" :max="3" :step="0.02" :format="(t) => t.toFixed(2)" />
    </template>

    <svg ref="svg" viewBox="0 0 540 330" role="group" aria-label="选择性扫描">
      <text x="6" y="12" class="sec">Δ_t (可拖)</text>
      <text x="6" y="142" class="sec">隐状态 h_t</text>
      <text x="6" y="242" class="sec">最终状态 h_T 中的占比</text>
      <line :x1="X0" :x2="X1" :y1="DY1" :y2="DY1" class="base" /><line :x1="X0" :x2="X1" :y1="HY1" :y2="HY1" class="base" /><line :x1="X0" :x2="X1" :y1="CY1" :y2="CY1" class="base" />
      <g v-for="(tk, j) in TOKENS" :key="j">
        <rect :x="cx(j) - 14" :y="dy(sim.d[j])" width="28" :height="DY1 - dy(sim.d[j])" class="dbar" :class="{ key: j === KEY }" />
        <text :x="cx(j)" :y="dy(sim.d[j]) - 4" class="num">{{ sim.d[j].toFixed(2) }}</text>
        <g v-if="selective" class="draggable" tabindex="0" role="slider" :aria-label="`token ${tk} 的 Δ`" :aria-valuenow="deltas[j]"
          @pointerdown="start($event, { svg, onMove: ({ y }) => setD(j, y) })"
          @keydown.up.prevent="bump(j, 0.1)" @keydown.down.prevent="bump(j, -0.1)">
          <rect :x="cx(j) - 18" :y="DY0 - 6" width="36" :height="DY1 - DY0 + 12" fill="transparent" />
          <circle :cx="cx(j)" :cy="dy(sim.d[j])" r="6" class="handle" />
        </g>
        <rect :x="cx(j) - 14" :y="CY1 - sim.share[j] * (CY1 - CY0)" width="28" :height="sim.share[j] * (CY1 - CY0)" class="cbar" :class="{ key: j === KEY }" />
        <text :x="cx(j)" :y="CY1 - sim.share[j] * (CY1 - CY0) - 3" class="num">{{ (sim.share[j] * 100).toFixed(0) }}%</text>
        <text :x="cx(j)" y="322" class="tok" :class="{ key: j === KEY }">{{ tk }}</text>
      </g>
      <polyline :points="hLine" class="hline" />
      <circle v-for="(h, j) in sim.h" :key="'h' + j" :cx="cx(j)" :cy="hy(h)" r="3" class="hdot" />
    </svg>

    <template #stats>
      <div class="kv"><span>“7” 在最终状态的占比</span><b :class="sim.share[KEY] > 0.5 ? 'good' : 'bad'">{{ (sim.share[KEY] * 100).toFixed(0) }}%</b></div>
      <div class="kv"><span>“7” 的信号强度 (绝对贡献)</span><b>{{ sim.contrib[KEY].toFixed(3) }}</b></div>
      <div class="kv"><span>经过 “7” 时旧状态保留 exp(−Δ·a)</span><b>{{ Math.exp(-sim.d[KEY] * a).toFixed(2) }}</b></div>
      <div class="kv"><span>状态大小 (与序列长度无关)</span><b class="good">O(1)</b></div>
      <p class="lab-note">
        {{ selective ? '每个 Δ_t 同时干两件事: 写入强度 Δ_t·x_t, 和对旧状态的保留率 exp(−Δ_t·a)。废话 token 的 Δ 拖到 0 附近 = 既不写入也不衰减, 信息 “穿过” 它。'
          : 'LTI: 贡献只取决于离结尾多远, 最后一个 token 永远占大头。“7” 的占比上限 = 1/10 (Δ→0 时)。' }}
      </p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { useDrag } from '@/composables/useDrag.js'
import { clamp, sum } from '@/utils/labmath.js'

const TOKENS = ['密码', '是', '7', '嗯', '那个', '就是', '然后', '呢', '请', '回答'], KEY = 2, DMAX = 3
const selective = ref(true), a = ref(1), dGlobal = ref(0.5)
const deltas = ref(TOKENS.map(() => 0.5))
const svg = ref(null)
const { start } = useDrag()

const sim = computed(() => {
  const d = selective.value ? deltas.value : TOKENS.map(() => dGlobal.value)
  const h = []
  let cur = 0
  for (let t = 0; t < d.length; t++) { cur = Math.exp(-d[t] * a.value) * cur + d[t] * 1; h.push(cur) } // ★ h_t = Ā_t·h_{t-1} + Δ_t·B·x_t (B = x = 1)
  // token j 在 h_T 里的贡献 = 写入量 × 之后每一步的保留率
  const contrib = d.map((dj, j) => dj * Math.exp(-a.value * sum(d.slice(j + 1))))
  const tot = sum(contrib) || 1
  return { d, h, contrib, share: contrib.map((c) => c / tot) }
})

const X0 = 60, X1 = 530, DY0 = 22, DY1 = 120, HY0 = 150, HY1 = 222, CY0 = 256, CY1 = 306
const cx = (j) => X0 + 24 + j * ((X1 - X0 - 48) / (TOKENS.length - 1))
const dy = (v) => DY1 - (v / DMAX) * (DY1 - DY0)
const hMax = computed(() => Math.max(1, ...sim.value.h))
const hy = (v) => HY1 - (v / hMax.value) * (HY1 - HY0)
const hLine = computed(() => sim.value.h.map((h, j) => `${cx(j)},${hy(h).toFixed(1)}`).join(' '))
const setD = (j, y) => { deltas.value = deltas.value.map((v, i) => (i === j ? clamp(Math.round(((DY1 - y) / (DY1 - DY0)) * DMAX * 50) / 50, 0.02, DMAX) : v)) }
const bump = (j, by) => { deltas.value = deltas.value.map((v, i) => (i === j ? clamp(v + by, 0.02, DMAX) : v)) }
const ideal = () => { selective.value = true; deltas.value = TOKENS.map((_, j) => (j === KEY ? 3 : 0.02)) }
const flat = () => { deltas.value = TOKENS.map(() => 0.5) }
</script>

<style scoped>
.sec { font-size: 10px; fill: var(--text-dim); }
.base { stroke: var(--border-strong); }
.dbar { fill: color-mix(in srgb, var(--accent) 35%, transparent); stroke: var(--accent); }
.dbar.key, .cbar.key { fill: color-mix(in srgb, var(--eye) 45%, transparent); stroke: var(--eye); }
.cbar { fill: color-mix(in srgb, var(--text-dim) 35%, transparent); stroke: var(--text-dim); }
.handle { fill: var(--accent); stroke: var(--bg); stroke-width: 2; }
.draggable:focus-visible .handle { stroke: var(--text); }
.num { text-anchor: middle; font-size: 9px; fill: var(--text-muted); font-family: "SF Mono", Menlo, monospace; }
.tok { text-anchor: middle; font-size: 11px; fill: var(--text-muted); }
.tok.key { fill: var(--eye); font-weight: 700; }
.hline { fill: none; stroke: var(--left); stroke-width: 2; }
.hdot { fill: var(--left); }
</style>
