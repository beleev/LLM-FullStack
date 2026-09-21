<!--
  蒸馏的 T² 梯度补偿实验台 (对应 llm_finetune/methods/distill.py: DistillLoss)。
  只讲一件事: 软标签 KL 对 student logits 的梯度是 (1/T)(q^T − p^T), T 大时 q^T − p^T 本身又 ∝ 1/T,
  所以整体按 1/T² 衰减; 不乘 T² 的话, 调高温度等于偷偷把蒸馏项关掉, α 的含义也跟着 T 漂移。
  极限有闭式: T·(q^T − p^T) → [(z_s − z̄_s) − (z_t − z̄_t)] / V, 图上虚线就是它。
-->
<template>
  <LabFrame
    title="蒸馏为什么要乘 T² — 温度一高, 软标签梯度就悄悄消失"
    sub="左边是温度 T 下 teacher (紫) 与 student (绿) 的分布, 点某个 token 把它设为硬标签。右边是软标签项对 student logits 的梯度范数随 T 的变化: 橙 = 不补偿, 绿 = 乘 T² 之后。拖 T, 看竖线扫过两条曲线。"
    module="llm_finetune/methods/distill.py"
    run="python -m llm_finetune.run_finetune.distill.train_distill"
    :challenge="{
      ask: '关掉「乘 T²」, α 固定 0.5, 把 T 从 1 拖到 8: 软标签项占总梯度的比例从多少掉到多少? 打开 T² 再拖一遍。',
      answer: '不补偿时软标签梯度范数大约按 1/T² 掉: T = 8 时只剩 T = 1 时的 2% 左右, 「α = 0.5」名义上五五开, 实际上几乎全是硬标签 CE 在训练 —— 你以为在蒸馏, 其实在做普通 SFT。乘上 T² 后, 软标签梯度随 T 增大趋于一个常数 (虚线): 它等于两边「去均值 logits 之差」除以 V —— 高温极限下蒸馏退化成 logits 回归 (Hinton 2015 的原话)。这样 T 只负责「暗知识展开多少」, α 只负责「软硬各占多少」, 两个超参才互不干扰。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: comp }" @click="comp = !comp">{{ comp ? '✓ 已乘 T²' : '✗ 未乘 T²' }}</button>
        <button type="button" @click="seed++">换一组 logits</button>
      </div>
      <LabSlider v-model="T" label="温度 T" :min="1" :max="10" :step="0.25" :format="(v) => v.toFixed(2)" />
      <LabSlider v-model="alpha" label="α (硬标签 CE 的权重)" :min="0" :max="1" :step="0.05" :format="(v) => v.toFixed(2)" />
      <LabSlider v-model="dist" label="student 离 teacher 多远" :min="0.2" :max="3" :step="0.1" :format="(v) => v.toFixed(1)" />
    </template>

    <svg viewBox="0 0 640 300" role="img" aria-label="软化分布与梯度范数曲线">
      <!-- 左: 温度 T 下的两个分布; 点击选硬标签 -->
      <text x="10" y="16" class="t">softmax(z/T), T = {{ T.toFixed(2) }}</text>
      <g
        v-for="(tk, i) in toks" :key="i" class="tok" role="button" tabindex="0"
        :aria-label="`把 token ${i} 设为硬标签`" @click="label = i" @keydown.enter="label = i"
      >
        <rect :x="14 + i * 40" y="24" width="36" height="236" :fill="label === i ? 'var(--accent-soft)' : 'transparent'" />
        <rect :x="17 + i * 40" :y="250 - tk.p * 210" width="14" :height="tk.p * 210" fill="var(--accent)" />
        <rect :x="33 + i * 40" :y="250 - tk.q * 210" width="14" :height="tk.q * 210" fill="var(--left)" />
        <text :x="32 + i * 40" y="274" text-anchor="middle" class="t">{{ label === i ? '标签' : 'tok' + i }}</text>
      </g>

      <!-- 右: ‖∇软标签‖ vs T -->
      <text :x="cx(1)" y="16" class="t">‖∂(软标签项)/∂z_student‖</text>
      <line :x1="cx(1)" :x2="cx(10)" :y1="cy(0)" :y2="cy(0)" stroke="var(--border-strong)" />
      <text v-for="t in [1, 2, 4, 6, 8, 10]" :key="t" :x="cx(t)" y="274" text-anchor="middle" class="t">{{ t }}</text>
      <text :x="cx(5.5)" y="292" text-anchor="middle" class="t">温度 T</text>
      <line :x1="cx(1)" :x2="cx(10)" :y1="cy(limit)" :y2="cy(limit)" stroke="var(--left)" stroke-dasharray="4 4" opacity="0.7" />
      <text :x="cx(10)" :y="cy(limit) + 14" text-anchor="end" class="t" fill="var(--left)">T→∞ 极限 {{ limit.toFixed(3) }}</text>
      <path :d="curve(false)" fill="none" stroke="var(--eye)" stroke-width="2" :opacity="comp ? 0.4 : 1" />
      <path :d="curve(true)" fill="none" stroke="var(--left)" stroke-width="2" :opacity="comp ? 1 : 0.4" />
      <line :x1="cx(T)" :x2="cx(T)" y1="24" :y2="cy(0)" stroke="var(--text-dim)" stroke-dasharray="2 3" />
      <circle :cx="cx(T)" :cy="cy(kdNorm(T, comp))" r="6" :fill="comp ? 'var(--left)' : 'var(--eye)'" stroke="var(--bg-card)" stroke-width="2" />
    </svg>

    <template #stats>
      <div class="kv"><span>‖∇CE‖ (硬标签)</span><b>{{ ceNorm.toFixed(3) }}</b></div>
      <div class="kv"><span>‖∇KD‖ 不补偿</span><b>{{ kdNorm(T, false).toFixed(4) }}</b></div>
      <div class="kv"><span>‖∇KD‖ × T²</span><b>{{ kdNorm(T, true).toFixed(4) }}</b></div>
      <div class="kv"><span>软标签占总梯度</span><b :class="share < 0.1 && alpha < 0.9 ? 'bad' : 'good'">{{ (share * 100).toFixed(1) }}%</b></div>
      <p class="lab-note">
        占比 = (1−α)·‖∇KD‖ / (α·‖∇CE‖ + (1−α)·‖∇KD‖)。α = {{ alpha.toFixed(2) }} 名义上给软标签 {{ ((1 - alpha) * 100).toFixed(0) }}%;
        实际占比偏离它多少, 就是 T² 补没补到位的度量。
      </p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { mulberry32, randn, range, softmax, sum } from '@/utils/labmath.js'

const V = 6
const T = ref(2), alpha = ref(0.5), dist = ref(1.5), seed = ref(1), comp = ref(false), label = ref(0)
const cx = (t) => 310 + (t - 1) * 34
const cy = (g) => 250 - Math.min(g / yMax.value, 1.05) * 200

const base = computed(() => {
  const rand = mulberry32(seed.value * 613)
  const zt = range(V).map((i) => (i === 0 ? 4 : randn(rand) * 1.5))     // teacher: tok0 明显最优, 其余有相对排序
  return { zt, noise: range(V).map(() => randn(rand)) }
})
const zt = computed(() => base.value.zt)
const zs = computed(() => zt.value.map((z, i) => z * 0.4 + dist.value * base.value.noise[i]))   // 学生: 还没学好的版本
const norm = (g) => Math.hypot(...g)

// ★ ∂/∂z_s KL(p^T ‖ q^T) = (1/T)·(q^T − p^T); 补偿后再乘 T²
const kdGrad = (t, c) => { const p = softmax(zt.value, t), q = softmax(zs.value, t); return q.map((v, i) => (v - p[i]) / t * (c ? t * t : 1)) }
const kdNorm = (t, c) => norm(kdGrad(t, c))
const ceNorm = computed(() => norm(softmax(zs.value).map((q, i) => q - (i === label.value ? 1 : 0))))   // ∂CE/∂z = q − onehot
// 闭式极限: softmax(z/T) ≈ 1/V + (z − z̄)/(V·T)
const limit = computed(() => {
  const c = (z) => { const m = sum(z) / V; return z.map((x) => x - m) }
  const a = c(zs.value), b = c(zt.value)
  return norm(a.map((x, i) => (x - b[i]) / V))
})
const yMax = computed(() => Math.max(kdNorm(1, true), limit.value, ...range(10).map((k) => kdNorm(k + 1, true))) * 1.05)
const curve = (c) => range(91).map((k) => { const t = 1 + k * 0.1; return `${k ? 'L' : 'M'}${cx(t).toFixed(1)},${cy(kdNorm(t, c)).toFixed(1)}` }).join(' ')

const toks = computed(() => { const p = softmax(zt.value, T.value), q = softmax(zs.value, T.value); return range(V).map((i) => ({ p: p[i], q: q[i] })) })
const share = computed(() => {
  const kd = (1 - alpha.value) * kdNorm(T.value, comp.value), ce = alpha.value * ceNorm.value
  return kd + ce === 0 ? 0 : kd / (kd + ce)
})
</script>

<style scoped>
.t { font-size: 11px; fill: var(--text-muted); font-family: "SF Mono", Menlo, monospace; }
.tok { cursor: pointer; }
.tok:focus-visible { outline: 2px solid var(--accent); }
</style>
