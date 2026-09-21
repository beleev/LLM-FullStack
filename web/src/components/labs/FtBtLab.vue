<!--
  Bradley–Terry 奖励模型实验台 (对应 llm_finetune/methods/reward_model.py: bradley_terry_loss)。
  只讲一件事: RM 只学"差", P(A≻B) = σ(r_A − r_B); 标注噪声 ε 决定了这个差最多学到多大 ——
  期望 loss 的最小点在 Δ* = ln((1−ε)/ε), 最小值 = 噪声的二元熵 H(ε)。闭式解与图上曲线的最低点一致。
-->
<template>
  <LabFrame
    title="Bradley–Terry — 奖励模型学到的只是「差」"
    sub="上面的数轴是 RM 给两条回答打的分: 绿色 A 是标注员 (大多数时候) 选的那条, 红色 B 是被拒的。拖动两个点。下图是「标注有 ε 概率标反」时的期望 loss 随分差 Δ = r_A − r_B 的变化。"
    module="llm_finetune/methods/reward_model.py"
    run="python -m llm_finetune.run_finetune.rm.train_rm"
    :challenge="{
      ask: '先把 ε 拖到 0, 一直把 A 往右拖: loss 有最小值吗? 再把 ε 调到 0.2, 找到 loss 最低的分差, 它是多少? 最后点「两个一起 +2」—— 什么变了?',
      answer: 'ε = 0 时 loss = −log σ(Δ) 单调下降, 最优分差是 +∞: 无噪声数据上 RM 会把分数无限拉开, 所以 RM 的分数没有量纲, 只能比大小。只要有一点标注噪声, 期望 loss = −(1−ε)·log σ(Δ) − ε·log σ(−Δ) 就变成一个碗, 最低点在 Δ* = ln((1−ε)/ε): ε = 0.2 → 1.386, ε = 0.1 → 2.197, ε = 0.5 → 0 (纯噪声, 学不到任何偏好)。而且此时 σ(Δ*) 恰好等于 1−ε —— 训练良好的 RM 输出的是校准过的「标注员一致率」。碗底的高度是二元熵 H(ε): 这是噪声数据上 loss 的下限, 训到这里就别再训了。两个一起平移: P 和 loss 完全不变 —— BT 模型对整体平移不可辨识, 这也是 GRPO/PPO 要做 baseline/归一化的原因之一。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" @click="shift(2)">两个一起 +2</button>
        <button type="button" @click="shift(-2)">两个一起 −2</button>
        <button type="button" :disabled="eps === 0" @click="rA = clamp(rB + dStar, -R, R)">把 A 放到最优分差</button>
      </div>
      <LabSlider v-model="eps" label="标注噪声 ε (标反的概率)" :min="0" :max="0.5" :step="0.01" :format="(v) => v.toFixed(2)" />
    </template>

    <svg ref="svgEl" viewBox="0 0 640 330" role="img" aria-label="两条回答的奖励分数与期望 loss 曲线">
      <!-- 上: 奖励数轴 -->
      <line :x1="rx(-R)" :x2="rx(R)" y1="60" y2="60" stroke="var(--border-strong)" />
      <text v-for="t in [-6, -3, 0, 3, 6]" :key="'r' + t" :x="rx(t)" y="88" text-anchor="middle" class="t">{{ t }}</text>
      <text x="8" y="20" class="t">RM 分数 r</text>
      <line :x1="rx(rB)" :x2="rx(rA)" y1="60" y2="60" :stroke="delta >= 0 ? 'var(--left)' : 'var(--danger)'" stroke-width="4" />
      <text :x="(rx(rA) + rx(rB)) / 2" y="40" text-anchor="middle" class="t">Δ = {{ delta.toFixed(2) }}</text>
      <g v-for="d in dots" :key="d.id">
        <circle
          class="draggable" :cx="rx(d.v)" cy="60" r="12" :fill="d.color" stroke="var(--bg-card)" stroke-width="2"
          tabindex="0" role="slider" :aria-label="`回答 ${d.id} 的分数`" :aria-valuenow="d.v" :aria-valuemin="-R" :aria-valuemax="R"
          @pointerdown="start($event, { svg: svgEl, onMove: ({ x }) => d.set((x - 320) / RS) })"
          @keydown.right.prevent="d.set(d.v + 0.25)" @keydown.left.prevent="d.set(d.v - 0.25)"
        />
        <text :x="rx(d.v)" y="64.5" text-anchor="middle" class="n">{{ d.id }}</text>
      </g>

      <!-- 下: 期望 loss vs Δ -->
      <text x="8" y="120" class="t" fill="var(--accent)">期望 loss(Δ) = −(1−ε)·log σ(Δ) − ε·log σ(−Δ)</text>
      <line :x1="dx(-DM)" :x2="dx(DM)" :y1="ly(0)" :y2="ly(0)" stroke="var(--border-strong)" />
      <text v-for="t in [-6, -3, 0, 3, 6]" :key="'d' + t" :x="dx(t)" y="322" text-anchor="middle" class="t">Δ={{ t }}</text>
      <path :d="lossPath" fill="none" stroke="var(--accent)" stroke-width="2" />
      <g v-if="eps > 0">
        <line :x1="dx(dStar)" :x2="dx(dStar)" :y1="ly(0)" :y2="ly(lossAt(dStar))" stroke="var(--left)" stroke-dasharray="3 3" />
        <circle :cx="dx(dStar)" :cy="ly(lossAt(dStar))" r="4" fill="var(--left)" />
        <text :x="dx(dStar)" :y="ly(lossAt(dStar)) - 8" text-anchor="middle" class="t" fill="var(--left)">Δ* = {{ dStar.toFixed(2) }}</text>
      </g>
      <circle :cx="dx(clamp(delta, -DM, DM))" :cy="ly(lossAt(delta))" r="7" fill="var(--eye)" stroke="var(--bg-card)" stroke-width="2" />
    </svg>

    <template #stats>
      <div class="kv"><span>P(A ≻ B) = σ(Δ)</span><b>{{ sigmoid(delta).toFixed(3) }}</b></div>
      <div class="kv"><span>期望 loss</span><b :class="gap < 0.01 ? 'good' : ''">{{ lossAt(delta).toFixed(3) }}</b></div>
      <div class="kv"><span>最优分差 ln((1−ε)/ε)</span><b>{{ eps === 0 ? '+∞' : dStar.toFixed(3) }}</b></div>
      <div class="kv"><span>loss 下限 = H(ε)</span><b>{{ floor.toFixed(3) }}</b></div>
      <p class="lab-note">
        梯度 ∂loss/∂Δ = σ(Δ) − (1−ε): 当 RM 预测的胜率等于标注员的真实一致率时梯度为 0。
        噪声越大, 学到的分差越扁 —— 下游 RL 拿到的奖励信号也就越弱。
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

const R = 7, RS = 42, DM = 8, LTOP = 3
const rx = (r) => 320 + r * RS
const dx = (d) => 320 + d * 36
const ly = (l) => 300 - Math.min(l, LTOP) / LTOP * 165

const rA = ref(1), rB = ref(-0.5), eps = ref(0.1)
const svgEl = ref(null)
const { start } = useDrag()

const snap = (v) => clamp(Math.round(v * 20) / 20, -R, R)
const dots = computed(() => [
  { id: 'B', v: rB.value, color: 'var(--danger)', set: (v) => (rB.value = snap(v)) },
  { id: 'A', v: rA.value, color: 'var(--left)', set: (v) => (rA.value = snap(v)) },
])
// 平移时两个点一起夹住, 保证分差不变
const shift = (s) => { const k = clamp(s, -R - Math.min(rA.value, rB.value), R - Math.max(rA.value, rB.value)); rA.value += k; rB.value += k }

const sigmoid = (z) => 1 / (1 + Math.exp(-z))
const softplus = (z) => Math.max(z, 0) + Math.log1p(Math.exp(-Math.abs(z)))
const delta = computed(() => rA.value - rB.value)
// ★ 标签以 1−ε 的概率说 A 赢, 以 ε 的概率说 B 赢; −log σ(z) = softplus(−z)
const lossAt = (d) => (1 - eps.value) * softplus(-d) + eps.value * softplus(d)
const dStar = computed(() => Math.log((1 - eps.value) / Math.max(eps.value, 1e-12)))
const H = (p) => (p <= 0 || p >= 1 ? 0 : -p * Math.log(p) - (1 - p) * Math.log(1 - p))
const floor = computed(() => H(eps.value))
const gap = computed(() => lossAt(delta.value) - floor.value)
const lossPath = computed(() => range(161).map((k) => { const d = -DM + k * 0.1; return `${k ? 'L' : 'M'}${dx(d).toFixed(1)},${ly(lossAt(d)).toFixed(1)}` }).join(' '))
</script>

<style scoped>
.t { font-size: 11px; fill: var(--text-muted); font-family: "SF Mono", Menlo, monospace; }
.n { font-size: 11px; fill: var(--bg-card); font-weight: 700; pointer-events: none; }
</style>
