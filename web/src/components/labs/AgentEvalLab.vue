<!--
  Agent 评测: pass@k vs pass^k (对应 llm_agent/m13_evals)。
  只讲一件事: 同一个单次成功率 p, "k 次里至少成一次"和"k 次全成"随 k 走向相反的两端。
-->
<template>
  <LabFrame
    title="pass@k vs pass^k — 能力上限, 还是可靠性?"
    sub="拖动左侧的圆点改变单次成功率 p, 点击曲线图上任意位置 (或拖滑杆) 选 k。下面的格子是真的按 p 抽样出来的: 每行一个任务、每格一次尝试, 行尾标出这一行算不算 pass@k / pass^k。"
    module="llm_agent/m13"
    run="python -m llm_agent.m13_evals.demo"
    :challenge="{
      ask: 'p = 0.9 听起来很可靠。k = 8 时 pass@k 和 pass^k 各是多少? 一个每天被调用 8 次的客服 agent, 用户感受到的是哪一个?',
      answer: 'pass@8 = 1 − 0.1⁸ ≈ 100%, pass^8 = 0.9⁸ ≈ 43%。pass@k 回答「多给几次机会, 它做不做得到」—— 适合有验证器、可以重试挑最优的场景 (写代码跑测试)。pass^k 回答「每一次都做对的概率」—— 动作不可撤销、没人复核的 agent (退款、发邮件、改库) 用户感受到的就是它。要把 pass^8 提到 90%, 单次 p 得到 98.7%: 可靠性是指数级昂贵的。',
    }"
  >
    <template #controls>
      <LabSlider v-model="p" label="单次成功率 p" :min="0.05" :max="0.99" :step="0.01" :format="(v) => (v * 100).toFixed(0) + '%'" />
      <LabSlider v-model="k" label="尝试次数 k" :min="1" :max="KMAX" />
      <div class="row"><button type="button" @click="seed++">换一组抽样</button></div>
    </template>

    <svg ref="svg" :viewBox="`0 0 ${W} ${H}`" role="img" aria-label="pass@k 与 pass^k 曲线" @click="pickK">
      <g class="grid">
        <line v-for="y in [0, 0.25, 0.5, 0.75, 1]" :key="y" :x1="X0" :x2="W - 10" :y1="py(y)" :y2="py(y)" />
        <text v-for="y in [0, 0.5, 1]" :key="'t' + y" :x="X0 - 6" :y="py(y) + 4" text-anchor="end">{{ y * 100 }}%</text>
        <text v-for="i in [1, 4, 8, 12, 16]" :key="'k' + i" :x="px(i)" :y="H - 6" text-anchor="middle">k={{ i }}</text>
      </g>
      <line class="cursor" :x1="px(k)" :x2="px(k)" :y1="py(1)" :y2="py(0)" />
      <polyline class="at" :points="curve((i) => 1 - (1 - p) ** i)" />
      <polyline class="hat" :points="curve((i) => p ** i)" />
      <circle class="dot at" :cx="px(k)" :cy="py(passAt)" r="4" />
      <circle class="dot hat" :cx="px(k)" :cy="py(passHat)" r="4" />
      <text class="lbl at" :x="W - 12" :y="py(1) + 14" text-anchor="end">pass@k = 1 − (1 − p)ᵏ</text>
      <text class="lbl hat" :x="W - 12" :y="py(0) - 8" text-anchor="end">pass^k = pᵏ</text>
      <circle
        class="handle draggable" :cx="px(1)" :cy="py(p)" r="8" tabindex="0" role="slider" aria-label="单次成功率 p"
        :aria-valuenow="p" aria-valuemin="0.05" aria-valuemax="0.99"
        @pointerdown="start($event, { svg, onMove: ({ y }) => (p = clamp(Math.round((1 - (y - Y0) / (H - Y0 - YB)) * 100) / 100, 0.05, 0.99)) })"
        @click.stop @keydown.up.prevent="p = clamp(+(p + 0.01).toFixed(2), 0.05, 0.99)" @keydown.down.prevent="p = clamp(+(p - 0.01).toFixed(2), 0.05, 0.99)"
      />
    </svg>

    <div class="trial-wrap">
      <div v-for="half in [0, 1]" :key="half" class="cells trials" :style="{ gridTemplateColumns: `repeat(${k}, 12px) 16px 16px` }">
        <template v-for="(row, t) in trials.slice(half * TASKS / 2, (half + 1) * TASKS / 2)" :key="t">
          <span v-for="(ok, a) in row" :key="a" class="cell" :class="ok ? 'ok' : 'bad'" />
          <span class="tag-at mono" :class="{ yes: row.some(Boolean) }">@</span>
          <span class="tag-hat mono" :class="{ yes: row.every(Boolean) }">^</span>
        </template>
      </div>
    </div>

    <template #stats>
      <div class="kv"><span>pass@{{ k }} (至少一次)</span><b class="good">{{ pct(passAt) }}</b></div>
      <div class="kv"><span>pass^{{ k }} (次次都对)</span><b :class="passHat < 0.5 ? 'bad' : ''">{{ pct(passHat) }}</b></div>
      <div class="kv"><span>{{ TASKS }} 个任务的抽样值</span><b>{{ pct(emp.at) }} / {{ pct(emp.hat) }}</b></div>
      <div class="kv"><span>要 pass^{{ k }} ≥ 90%, p 需</span><b>{{ pct(0.9 ** (1 / k)) }}</b></div>
      <p class="lab-note">只看最终答案还不够: 轨迹检查 (trajectory check) 会另外断言过程 —— 有没有调用不该调的工具、有没有跳过必须的确认步骤。答案对、过程违规, 同样算失败。</p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { useDrag } from '@/composables/useDrag.js'
import { clamp, mulberry32, range } from '@/utils/labmath.js'

const KMAX = 16, TASKS = 40
const W = 560, H = 230, X0 = 44, Y0 = 12, YB = 24
const p = ref(0.9), k = ref(8), seed = ref(1)
const svg = ref(null)
const { start } = useDrag()

const px = (i) => X0 + ((i - 1) / (KMAX - 1)) * (W - X0 - 16)
const py = (v) => Y0 + (1 - v) * (H - Y0 - YB)
const curve = (f) => range(KMAX).map((i) => `${px(i + 1)},${py(f(i + 1))}`).join(' ')
const pct = (v) => (v * 100).toFixed(1) + '%'

// ★ 全部的数学: 独立同分布的 k 次尝试
const passAt = computed(() => 1 - (1 - p.value) ** k.value)
const passHat = computed(() => p.value ** k.value)

// 抽样: 每个格子先抽一个固定的 u, 成功 = u < p。拖 p 时格子只会单调地翻面, 不会乱跳
const us = computed(() => { const rand = mulberry32(seed.value * 7919); return range(TASKS).map(() => range(KMAX).map(() => rand())) })
const trials = computed(() => us.value.map((row) => row.slice(0, k.value).map((u) => u < p.value)))
const emp = computed(() => ({
  at: trials.value.filter((r) => r.some(Boolean)).length / TASKS,
  hat: trials.value.filter((r) => r.every(Boolean)).length / TASKS,
}))

const pickK = (e) => {
  const pt = svg.value.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY
  const x = pt.matrixTransform(svg.value.getScreenCTM().inverse()).x
  k.value = clamp(Math.round(((x - X0) / (W - X0 - 16)) * (KMAX - 1)) + 1, 1, KMAX)
}
</script>

<style scoped>
svg { width: 100%; cursor: crosshair; }
.grid line { stroke: var(--border); stroke-width: 1; }
.grid text { font-size: 10px; fill: var(--text-dim); }
polyline { fill: none; stroke-width: 2; }
.at { stroke: var(--left); } .hat { stroke: var(--danger); }
.dot.at { fill: var(--left); } .dot.hat { fill: var(--danger); }
.lbl { font-size: 11px; stroke: none; } .lbl.at { fill: var(--left); } .lbl.hat { fill: var(--danger); }
.cursor { stroke: var(--accent); stroke-dasharray: 3 3; }
.handle { fill: var(--accent); stroke: var(--bg-card); stroke-width: 2; }
.trial-wrap { display: flex; flex-wrap: wrap; gap: 8px 28px; margin-top: 12px; }
.trials { gap: 2px; grid-auto-rows: 10px; align-content: start; }
.trials .cell { min-width: 0; width: 12px; height: 10px; border-radius: 2px; }
.tag-at, .tag-hat { font-size: 9px; line-height: 10px; color: var(--text-dim); opacity: 0.3; padding-left: 4px; }
.tag-at.yes { color: var(--left); opacity: 1; } .tag-hat.yes { color: var(--danger); opacity: 1; }
</style>
