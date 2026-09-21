<!--
  重要性比率 + clip 实验台 (PPO / GRPO 的 clipped surrogate, DAPO 的 clip-higher)。
  只讲一件事: 目标 min(ρÂ, clip(ρ, 1−ε_low, 1+ε_high)Â) 在阴影区里是平的 —— 梯度为 0,
  这个 token 本轮不再被推。上界只拦"想继续涨"的 token, 而真正会撞到它的只有低概率 (探索) token。
-->
<template>
  <LabFrame
    title="ratio clip — 哪些 token 这一步已经不许再推了"
    sub="横轴是重要性比率 ρ = π_new / π_old (同一批样本做多轮内层更新时, ρ 会逐渐离开 1)。曲线是这个 token 的代理目标; 阴影区梯度为 0。拖动曲线上的点, 或者改两个 ε。"
    module="llm_finetune/methods/grpo.py"
    :challenge="{
      ask: '设 Â > 0。把 π_old 拖到最右 (≈0.89): 这个 token 能撞到 1+ε_high 的上界吗? 再拖到 0.01, 对称 ε = 0.2 时它一步最多涨到多少? 换成 DAPO 的 0.2 / 0.28 呢?',
      answer: 'π_new 不能超过 1, 所以 ρ ≤ 1/π_old。π_old ≈ 0.89 时 ρ 最大 1.12, 永远碰不到 1.2 的上界 —— 高概率 token 实际上不受上界约束, 想涨就涨。π_old = 0.01 的 token 却被卡在 0.012: 低概率的「探索 token」每轮最多涨 20%, 要翻身得连续很多轮都拿到正优势。结果是强者恒强, 熵一路塌缩。DAPO 的 clip-higher 只放宽上界 (0.28 → 0.0128), 给低概率的好 token 更多上升空间; 下界不动, 因为放宽下界会让 token 概率被一步压到接近 0, 同样塌缩。注意只有「做多轮内层更新」时 clip 才起作用: 第一轮 π_new = π_old, ρ 恒为 1。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: A > 0 }" @click="A = 1">Â &gt; 0 (好 token, 想推高)</button>
        <button type="button" :class="{ active: A < 0 }" @click="A = -1">Â &lt; 0 (坏 token, 想压低)</button>
        <button type="button" @click="preset(0.2, 0.2)">PPO/GRPO 对称 0.2</button>
        <button type="button" @click="preset(0.2, 0.28)">DAPO clip-higher 0.2/0.28</button>
      </div>
      <LabSlider v-model="eLo" label="ε_low" :min="0.05" :max="0.5" :step="0.01" :format="(v) => v.toFixed(2)" />
      <LabSlider v-model="eHi" label="ε_high" :min="0.05" :max="0.5" :step="0.01" :format="(v) => v.toFixed(2)" />
      <LabSlider v-model="logPi" label="π_old (旧策略给它的概率)" :min="-2" :max="-0.05" :step="0.05" :format="() => piOld.toFixed(3)" />
    </template>

    <svg ref="svgEl" viewBox="0 0 640 330" role="img" aria-label="clipped surrogate 目标随 ratio 变化">
      <!-- 梯度为 0 的区域 -->
      <rect :x="flat.x" y="20" :width="flat.w" height="290" fill="var(--warn)" opacity="0.14" />
      <text :x="flat.x + flat.w / 2" y="36" text-anchor="middle" class="t" fill="var(--warn)">梯度 = 0</text>
      <line x1="50" x2="622" :y1="sy(0)" :y2="sy(0)" stroke="var(--border-strong)" />
      <line :x1="sx(1)" :x2="sx(1)" y1="20" y2="310" stroke="var(--border-strong)" stroke-dasharray="3 3" />
      <text :x="sx(1)" y="325" text-anchor="middle" class="t">ρ = 1</text>
      <text :x="sx(1 - eLo)" y="325" text-anchor="end" class="t">1−ε_low</text>
      <text :x="sx(1 + eHi)" y="325" text-anchor="start" class="t">1+ε_high</text>
      <line v-for="e in [1 - eLo, 1 + eHi]" :key="e" :x1="sx(e)" :x2="sx(e)" y1="20" y2="310" stroke="var(--warn)" stroke-width="1" />
      <!-- 未裁剪的 ρÂ (虚线) 与裁剪后的目标 -->
      <line :x1="sx(0)" :y1="sy(0)" :x2="sx(RMAX)" :y2="sy(RMAX * A)" stroke="var(--text-dim)" stroke-dasharray="4 4" />
      <path :d="objPath" fill="none" stroke="var(--accent)" stroke-width="2.5" />
      <!-- π_new ≤ 1 → ρ ≤ 1/π_old -->
      <g v-if="rhoCap < RMAX">
        <rect :x="sx(rhoCap)" y="20" :width="sx(RMAX) - sx(rhoCap)" height="290" fill="var(--text-dim)" opacity="0.25" />
        <text :x="sx(rhoCap) + 4" y="300" class="t">ρ &gt; 1/π_old: 不可能 (π_new &gt; 1)</text>
      </g>
      <!-- ★ 手放在 ρ 上 -->
      <circle
        class="draggable" :cx="sx(rho)" :cy="sy(obj(rho))" r="10" :fill="grad ? 'var(--left)' : 'var(--warn)'" stroke="var(--bg-card)" stroke-width="2"
        tabindex="0" role="slider" aria-label="重要性比率 ρ" :aria-valuenow="rho" aria-valuemin="0" :aria-valuemax="RMAX"
        @pointerdown="start($event, { svg: svgEl, onMove: ({ x }) => setRho((x - 50) / 260) })"
        @keydown.right.prevent="setRho(rho + 0.05)" @keydown.left.prevent="setRho(rho - 0.05)"
      />
    </svg>

    <template #stats>
      <div class="kv"><span>ρ = π_new / π_old</span><b>{{ rho.toFixed(2) }}</b></div>
      <div class="kv"><span>π_new</span><b>{{ (rho * piOld).toFixed(4) }}</b></div>
      <div class="kv"><span>这个 token 的梯度 ∂L/∂ρ</span><b :class="grad ? 'good' : 'bad'">{{ grad ? (A > 0 ? '+1·|Â|' : '−1·|Â|') : '0 (被 clip)' }}</b></div>
      <div class="kv"><span>上界内 π 最多涨到</span><b>{{ Math.min(1, piOld * (1 + eHi)).toFixed(4) }}</b></div>
      <div class="kv"><span>上界碰得到吗</span><b :class="rhoCap > 1 + eHi ? '' : 'good'">{{ rhoCap > 1 + eHi ? '碰得到' : '碰不到 (1/π_old 不到 1+ε)' }}</b></div>
      <p class="lab-note">
        min(·) 让 clip 只在"对自己有利的方向"上生效: Â &gt; 0 时只封顶, Â &lt; 0 时只托底。
        走错方向的 token (例如 Â &gt; 0 但 ρ 已经掉到 0.5) 永远有梯度, 会被拉回来。
      </p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { useDrag } from '@/composables/useDrag.js'
import { clamp } from '@/utils/labmath.js'

const RMAX = 2.2
const sx = (r) => 50 + r * 260
const sy = (v) => 165 - v * 62

const A = ref(1), eLo = ref(0.2), eHi = ref(0.2), logPi = ref(-1), rho = ref(1.1)
const svgEl = ref(null)
const { start } = useDrag()

const preset = (lo, hi) => { eLo.value = lo; eHi.value = hi }
const piOld = computed(() => 10 ** logPi.value)
const rhoCap = computed(() => 1 / piOld.value)
const setRho = (r) => { rho.value = clamp(Math.round(r * 100) / 100, 0, Math.min(RMAX, rhoCap.value)) }
watch(rhoCap, () => setRho(rho.value))

// ★ PPO clipped surrogate: min(ρÂ, clip(ρ, 1−ε_low, 1+ε_high)·Â)
const obj = (r) => Math.min(r * A.value, clamp(r, 1 - eLo.value, 1 + eHi.value) * A.value)
// 梯度非零 ⇔ min 选中的是未裁剪的那一项
const grad = computed(() => (A.value > 0 ? rho.value <= 1 + eHi.value : rho.value >= 1 - eLo.value))
const flat = computed(() => (A.value > 0
  ? { x: sx(1 + eHi.value), w: sx(RMAX) - sx(1 + eHi.value) }
  : { x: sx(0), w: sx(1 - eLo.value) - sx(0) }))
const objPath = computed(() => [0, 1 - eLo.value, 1 + eHi.value, RMAX]
  .map((r, i) => `${i ? 'L' : 'M'}${sx(r).toFixed(1)},${sy(obj(r)).toFixed(1)}`).join(' '))
</script>

<style scoped>
.t { font-size: 11px; fill: var(--text-muted); font-family: "SF Mono", Menlo, monospace; }
</style>
