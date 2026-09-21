<!--
  初始化实验台: 为什么"刚建好的模型"第一步 loss 应该 ≈ ln V, 以及绑权重 + std=1 怎么把它炸到 250。
  对应 llm_models/utils/init.py: std=0.02 时 LLaMA 首步 CE 7.09, ln 1000 = 6.91。
-->
<template>
  <LabFrame
    title="初始化 — 第一步 loss 该是多少?"
    sub="一个什么都没学的模型应该均匀瞎猜, loss = ln V。拖 σ 看首步 CE 什么时候离开这条参考线。输入嵌入和输出头绑权重时, 每个 token 会给「自己」打一个 ≈ σ·d 的高分, 而正确答案几乎从来不是它自己。"
    module="llm_models/utils/init.py"
    run="python -m llm_models.run_models.language_models.llama.train_llama"
    :challenge="{
      ask: '不改 σ=1, 只把「绑权重」关掉, 首步 CE 会回到 ln V 吗? 为什么真实 LLaMA 不绑权重也不乘 √d 却没这个问题?',
      answer: '关掉绑权重后「给自己打高分」的那一项消失, 但其余 logit 的标准差仍是 σ·√d = 16, softmax 依旧极尖, CE 还是几十。根因是 logit 的尺度而不只是绑权重 —— 所以把 σ 降到 0.02 才治本 (logit 标准差 ≈ 0.3, 近似均匀)。真实 LLaMA 用小 σ 初始化, 两个问题都不存在。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: tied }" @click="tied = !tied">绑权重 lm_head = embedding: {{ tied ? '开' : '关' }}</button>
        <button type="button" @click="logSigma = Math.log10(0.02)">σ = 0.02 (本仓库用的值)</button>
        <button type="button" @click="logSigma = 0">σ = 1 (nn.Embedding 默认)</button>
        <button type="button" @click="seed++">换一组随机权重</button>
      </div>
      <LabSlider v-model="logSigma" label="嵌入初始化 σ" :min="-2.5" :max="0.3" :step="0.05" :format="(v) => (10 ** v).toPrecision(2)" />
      <LabSlider v-model="d" label="模型宽度 d" :min="32" :max="512" :step="32" />
      <LabSlider v-model="V" label="词表大小 V" :min="100" :max="2000" :step="100" />
    </template>

    <svg :viewBox="`0 0 ${W} ${H}`" role="img" aria-label="首步交叉熵随初始化标准差的变化曲线">
      <line :x1="PAD" :x2="W - 8" :y1="yOf(lnV)" :y2="yOf(lnV)" class="ref" />
      <text :x="W - 10" :y="yOf(lnV) - 5" class="lbl" text-anchor="end">ln V = {{ lnV.toFixed(2) }} (均匀瞎猜)</text>
      <polyline :points="curve" class="curve" />
      <!-- ★ 直接拖这个点改 σ: 手放在被解释的那个量上 -->
      <circle
        :cx="xOf(logSigma)" :cy="yOf(ce)" r="8" class="knob draggable" tabindex="0" role="slider"
        aria-label="拖动改变初始化标准差" :aria-valuenow="sigma"
        @pointerdown="drag.start($event, { svg: $event.target.ownerSVGElement, onMove: ({ x }) => (logSigma = clamp(sOf(x), -2.5, 0.3)) })"
        @keydown.left.prevent="logSigma = clamp(logSigma - 0.05, -2.5, 0.3)"
        @keydown.right.prevent="logSigma = clamp(logSigma + 0.05, -2.5, 0.3)"
      />
      <text v-for="t in [-2, -1, 0]" :key="t" :x="xOf(t)" :y="H - 6" class="lbl" text-anchor="middle">σ = {{ 10 ** t }}</text>
      <text :x="4" :y="14" class="lbl">首步 CE (对数轴)</text>
    </svg>

    <template #stats>
      <div class="kv"><span>首步 CE (蒙特卡洛)</span><b :class="ok ? 'good' : 'bad'">{{ ce.toFixed(2) }}</b></div>
      <div class="kv"><span>ln V</span><b>{{ lnV.toFixed(2) }}</b></div>
      <div class="kv"><span>「自己」的 logit ≈ σ·d</span><b>{{ tied ? (sigma * d).toFixed(1) : '—' }}</b></div>
      <div class="kv"><span>其余 logit 的标准差 σ·√d</span><b>{{ (sigma * Math.sqrt(d)).toFixed(2) }}</b></div>
      <p class="lab-note">
        {{ ok ? '✓ 接近 ln V: 模型从"均匀瞎猜"出发, loss 下降才说明它真的学到了东西。'
              : '✗ 远高于 ln V: 模型一上来就极度自信地猜错。这时"loss 在下降"只是在纠正初始化, 不代表学会了任务。' }}
      </p>
      <p class="lab-note">简化模型: 最后一层归一化后隐藏向量范数 ≈ √d, 且方向仍贴着输入 token 的嵌入 (残差连接)。它复现了仓库里的实测值: σ=1, d=256, V=1000 → ≈ 250; σ=0.02 → ≈ 7.1。</p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { useDrag } from '@/composables/useDrag.js'
import { clamp, mulberry32, randn, range } from '@/utils/labmath.js'

const logSigma = ref(0)
const d = ref(256)
const V = ref(1000)
const tied = ref(true)
const seed = ref(1)
const drag = useDrag()

const sigma = computed(() => 10 ** logSigma.value)
const lnV = computed(() => Math.log(V.value))

// 首步 CE 的蒙特卡洛估计: 目标 token 是随机的 (几乎从不等于输入 token 自己)
const ceAt = (sig, dd, vv, isTied, sd) => {
  const rand = mulberry32(sd * 7919)
  const spread = sig * Math.sqrt(dd) // 其余 logit ~ N(0, σ²·d): |h| ≈ √d, 每个嵌入分量 std = σ
  let total = 0
  const trials = 24
  for (let t = 0; t < trials; t++) {
    const logits = range(vv).map(() => spread * randn(rand))
    if (isTied) logits[0] = sig * dd        // ★ h · e_self = √d · |e_self| ≈ √d · σ√d = σ·d
    const target = logits[1]                // 正确答案是"别的某个 token"
    const m = Math.max(...logits)
    total += m + Math.log(logits.reduce((s, z) => s + Math.exp(z - m), 0)) - target
  }
  return total / trials
}
const ce = computed(() => ceAt(sigma.value, d.value, V.value, tied.value, seed.value))
const ok = computed(() => Math.abs(ce.value - lnV.value) < 0.5) // 与 train 脚本里的断言同一阈值

// ── 画图: x = log10 σ, y = log CE ─────────────────────────────────
const W = 520, H = 260, PAD = 34
const xOf = (ls) => PAD + ((ls + 2.5) / 2.8) * (W - PAD - 12)
const sOf = (x) => ((x - PAD) / (W - PAD - 12)) * 2.8 - 2.5
const yOf = (c) => H - 24 - (Math.log(clamp(c, 3, 1500) / 3) / Math.log(500)) * (H - 48)
const curve = computed(() =>
  range(57).map((i) => {
    const ls = -2.5 + i * 0.05
    return `${xOf(ls)},${yOf(ceAt(10 ** ls, d.value, V.value, tied.value, seed.value))}`
  }).join(' '))
</script>

<style scoped>
.ref { stroke: var(--left); stroke-dasharray: 5 4; stroke-width: 1.5; }
.curve { fill: none; stroke: var(--accent); stroke-width: 2; }
.knob { fill: var(--accent); stroke: var(--bg); stroke-width: 2; }
.knob:focus-visible { outline: none; stroke: var(--warn); stroke-width: 3; }
.lbl { fill: var(--text-dim); font-size: 11px; font-family: "SF Mono", Menlo, monospace; }
</style>
