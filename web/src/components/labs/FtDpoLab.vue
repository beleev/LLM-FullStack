<!--
  DPO loss 曲面实验台 (对应 llm_finetune/methods/dpo.py: DPOLoss)。
  只讲一件事: loss = -log σ(β·Δ), 它对 Δ 的梯度权重是 σ(-β·Δ) ——
  模型已经排对的偏好对几乎不产生梯度, 训练自动把力气花在排错的样本上。
  Δ = (log π_c − log π_r) − (log ref_c − log ref_r), 即 policy 相对 ref 的隐式奖励差。
-->
<template>
  <LabFrame
    title="DPO — 一条 logsigmoid 曲线上的 6 个偏好对"
    sub="横轴是隐式奖励差 Δ (policy 相对 ref, chosen 比 rejected 多涨了多少 log 概率)。上图是每个样本的 loss, 下图是它分到的梯度权重 σ(−βΔ)。左右拖动任意一个点, 或者改 β。"
    module="llm_finetune/methods/dpo.py"
    run="python -m llm_finetune.run_finetune.dpo.train_dpo"
    :challenge="{
      ask: '点「从 ref 出发」: 6 个点的 loss 都是多少? 为什么? 然后把一个点拖到 Δ = +8, 再把 β 从 0.1 拖到 1.0 —— 这个点的梯度权重怎么变, β 到底在控制什么?',
      answer: '训练第 0 步 policy = ref, 所有 Δ = 0, loss 恒为 −log σ(0) = ln 2 ≈ 0.693, 梯度权重恒为 0.5 —— 如果你的 DPO 初始 loss 不是 0.693, 说明 ref 和 policy 没对齐 (这是最好用的自检)。β 是「Δ 要涨到多大才算够」的尺子。β = 0.1 时 Δ = 8 的样本权重还有 σ(−0.8) ≈ 0.31, 模型会继续把它往右推, 离 ref 越来越远。β = 1 时同一个点权重只剩 σ(−8) ≈ 0.0003, 基本停手。所以 β 大 = 贴近 ref、保守; β 小 = 允许偏离 ref 很远去拉大差距。它就是 RLHF 目标里 KL 惩罚系数的化身。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" @click="deltas = deltas.map(() => 0)">从 ref 出发 (全部 Δ = 0)</button>
        <button type="button" @click="reseed">训练中途 (换一组)</button>
      </div>
      <LabSlider v-model="beta" label="β" :min="0.05" :max="1" :step="0.05" :format="(v) => v.toFixed(2)" />
    </template>

    <svg ref="svgEl" viewBox="0 0 640 330" role="img" aria-label="DPO loss 与梯度权重曲线">
      <!-- 排错区 (Δ<0) 底色 -->
      <rect :x="sx(-XM)" y="14" :width="sx(0) - sx(-XM)" height="290" fill="var(--danger)" opacity="0.06" />
      <text :x="sx(-XM) + 6" y="28" class="t" fill="var(--danger)">Δ &lt; 0: 排错了 (rejected 涨得更多)</text>
      <text :x="sx(XM) - 6" y="28" class="t" text-anchor="end" fill="var(--left)">Δ &gt; 0: 排对了</text>
      <line :x1="sx(0)" :x2="sx(0)" y1="14" y2="304" stroke="var(--border-strong)" stroke-dasharray="3 3" />

      <!-- 上: loss -->
      <path :d="lossPath" fill="none" stroke="var(--accent)" stroke-width="2" />
      <line :x1="sx(-XM)" :x2="sx(XM)" :y1="ly(Math.LN2)" :y2="ly(Math.LN2)" stroke="var(--border)" />
      <text :x="sx(XM) - 4" :y="ly(Math.LN2) - 4" class="t" text-anchor="end">ln 2 = 0.693</text>
      <text :x="sx(0) + 8" y="46" class="t" fill="var(--accent)">loss = −log σ(βΔ)</text>
      <!-- 下: 梯度权重 -->
      <path :d="wPath" fill="none" stroke="var(--eye)" stroke-width="2" />
      <line :x1="sx(-XM)" :x2="sx(XM)" :y1="wy(0)" :y2="wy(0)" stroke="var(--border)" />
      <text x="6" :y="wy(1) - 4" class="t" fill="var(--eye)">梯度权重 σ(−βΔ)</text>
      <text v-for="t in [-8, -4, 0, 4, 8]" :key="t" :x="sx(t)" y="322" class="t" text-anchor="middle">{{ t }}</text>

      <g v-for="(p, i) in pts" :key="i">
        <line :x1="sx(p.d)" :x2="sx(p.d)" :y1="ly(p.loss)" :y2="wy(p.w)" stroke="var(--text-dim)" stroke-width="0.7" />
        <circle :cx="sx(p.d)" :cy="wy(p.w)" r="5" fill="var(--eye)" />
        <!-- ★ 手放在 Δ 上: 拖动 = 这个偏好对被训练推着走 -->
        <circle
          class="draggable" :cx="sx(p.d)" :cy="ly(p.loss)" r="10"
          :fill="p.d >= 0 ? 'var(--left)' : 'var(--danger)'" stroke="var(--bg-card)" stroke-width="2"
          tabindex="0" role="slider" :aria-label="`偏好对 ${i + 1} 的 Δ`" :aria-valuenow="p.d" :aria-valuemin="-XM" :aria-valuemax="XM"
          @pointerdown="start($event, { svg: svgEl, onMove: ({ x }) => setD(i, (x - sx(0)) / SC) })"
          @keydown.right.prevent="setD(i, p.d + 0.5)" @keydown.left.prevent="setD(i, p.d - 0.5)"
        />
        <text :x="sx(p.d)" :y="ly(p.loss) + 3.5" text-anchor="middle" class="n">{{ i + 1 }}</text>
      </g>
    </svg>

    <template #stats>
      <div class="kv"><span>batch 平均 loss</span><b>{{ meanLoss.toFixed(3) }}</b></div>
      <div class="kv"><span>偏好准确率 (Δ &gt; 0)</span><b :class="acc === 1 ? 'good' : ''">{{ (acc * 100).toFixed(0) }}%</b></div>
      <div class="kv"><span>排错样本拿走的梯度</span><b :class="wrongShare > 0.5 ? 'bad' : ''">{{ (wrongShare * 100).toFixed(0) }}%</b></div>
      <div class="kv"><span>最"对"的样本的权重</span><b :class="minW < 0.05 ? 'good' : ''">{{ minW.toFixed(3) }}</b></div>
      <p class="lab-note">
        ∂loss/∂Δ = −β·σ(−βΔ)。权重 → 0 的样本等于自动退出训练;
        权重 → 1 的样本 (排得很错) 以满额梯度被纠正。这和分类里"只有分错的样本推动决策边界"是同一件事。
      </p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { useDrag } from '@/composables/useDrag.js'
import { clamp, mulberry32, randn, range, sum } from '@/utils/labmath.js'

const XM = 10, SC = 29, LMAX = 3.2
const sx = (d) => 320 + d * SC
const ly = (l) => 200 - Math.min(l, LMAX) / LMAX * 150     // loss 面板: y∈[50,200]
const wy = (w) => 300 - w * 70                             // 权重面板: y∈[230,300]

const beta = ref(0.3)
const seed = ref(1)
const svgEl = ref(null)
const { start } = useDrag()

const sample = (s) => { const rand = mulberry32(s * 4099); return range(6).map(() => clamp(Math.round((1.5 + randn(rand) * 4) * 2) / 2, -XM, XM)) }
const deltas = ref(sample(1))
const reseed = () => { seed.value++; deltas.value = sample(seed.value) }
const setD = (i, d) => { deltas.value = deltas.value.map((v, k) => (k === i ? clamp(Math.round(d * 10) / 10, -XM, XM) : v)) }

const sigmoid = (z) => 1 / (1 + Math.exp(-z))
const softplus = (z) => Math.max(z, 0) + Math.log1p(Math.exp(-Math.abs(z)))   // 数值稳定的 log(1+e^z)
const lossOf = (d) => softplus(-beta.value * d)            // ★ −log σ(βΔ) = softplus(−βΔ)
const wOf = (d) => sigmoid(-beta.value * d)                // ★ 梯度权重

const pts = computed(() => deltas.value.map((d) => ({ d, loss: lossOf(d), w: wOf(d) })))
const curve = (f, y) => range(101).map((k) => { const d = -XM + k * 0.2; return `${k ? 'L' : 'M'}${sx(d).toFixed(1)},${y(f(d)).toFixed(1)}` }).join(' ')
const lossPath = computed(() => curve(lossOf, ly))
const wPath = computed(() => curve(wOf, wy))

const meanLoss = computed(() => sum(pts.value.map((p) => p.loss)) / pts.value.length)
const acc = computed(() => pts.value.filter((p) => p.d > 0).length / pts.value.length)
const wrongShare = computed(() => sum(pts.value.filter((p) => p.d < 0).map((p) => p.w)) / sum(pts.value.map((p) => p.w)))
const minW = computed(() => Math.min(...pts.value.map((p) => p.w)))
</script>

<style scoped>
.t { font-size: 11px; fill: var(--text-muted); font-family: "SF Mono", Menlo, monospace; }
.n { font-size: 10px; fill: var(--bg-card); font-weight: 600; pointer-events: none; }
</style>
