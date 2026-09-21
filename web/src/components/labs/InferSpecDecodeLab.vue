<!--
  投机解码实验台 (对应 llm_infer/m07_speculative_decoding/speculative.py)。
  一件事: 每次 target 调用的期望产出 (1−α^(K+1))/(1−α) —— K 再大也被 1/(1−α) 封顶, draft 不是免费的。
-->
<template>
  <LabFrame
    title="投机解码 — draft 猜 K 个, target 一次验 K+1 个"
    sub="上图: 每次 target 调用的期望产出随 K 的变化 (可以直接拖曲线上的圆点改 K)。下方: 逐轮回放 draft → 验证 → 接受最长前缀 + 白送 1 个 token。"
    module="llm_infer/m07"
    run="python -m llm_infer.m07_speculative_decoding.demo"
    :challenge="{
      ask: 'α = 0.6 时把 K 从 4 拖到 10, 期望产出涨了多少? 再把 draft 成本 c 调到 0.2, 加速比最高的 K 是几? 如果 draft 模型很烂 (α≈0.1), 输出质量会变差吗?',
      answer: 'α=0.6 时 K=4 → 2.31, K=10 → 2.49, 上限 1/(1−α)=2.5: 第一个猜错后面全废, 长 draft 的尾巴几乎从不被用到。c=0.2 时加速比 E/(1+K·c) 在 K=2 就见顶 (1.40×), 再长反而更慢。draft 再烂也不改变输出: greedy 下只接受与 target argmax 相同的 token; 采样下以 min(1, p/q) 接受, 拒绝就从归一化残差 max(0, p−q) 重采样, 两步合起来恰好是 target 分布 p。烂 draft 只会让你更慢, 不会让你更错。',
    }"
  >
    <template #controls>
      <LabSlider v-model="K" label="draft 长度 K" :min="1" :max="10" />
      <LabSlider v-model="alpha" label="每 token 接受率 α" :min="0" :max="0.95" :step="0.05" :format="(t) => t.toFixed(2)" />
      <LabSlider v-model="cost" label="draft 成本 c (÷target)" :min="0" :max="0.5" :step="0.01" :format="(t) => t.toFixed(2)" />
      <div class="row">
        <button type="button" @click="seed++">换一组随机数</button>
        <StepPlayer :stepper="stepper" :label="`第 ${cur.round + 1} 轮 · ${PHASES[cur.phase]}`" />
      </div>
    </template>

    <svg ref="svg" viewBox="0 0 520 210" role="group" aria-label="期望产出与加速比随 K 的变化">
      <line x1="40" x2="510" y1="180" y2="180" class="axis" /><line x1="40" x2="40" y1="10" y2="180" class="axis" />
      <text v-for="k in 10" :key="k" :x="sx(k)" y="196" class="tick">{{ k }}</text>
      <text v-for="v in yTicks" :key="`y${v}`" x="34" :y="sy(v) + 3" class="tick end">{{ v }}</text>
      <text x="505" y="206" class="tick end">draft 长度 K</text>
      <line x1="40" x2="510" :y1="sy(1)" :y2="sy(1)" class="base" />
      <line v-if="cap <= yMax" x1="40" x2="510" :y1="sy(cap)" :y2="sy(cap)" class="cap" />
      <text x="508" :y="(cap <= yMax ? sy(cap) : 10) + 11" class="tick end warn">上限 1/(1−α) = {{ cap.toFixed(2) }}{{ cap > yMax ? ' (图外)' : '' }}</text>
      <polyline :points="pts(E)" class="curve e" />
      <polyline :points="pts(speed)" class="curve s" />
      <circle v-for="k in 10" :key="`d${k}`" :cx="sx(k)" :cy="sy(speed(k))" r="2.5" :class="['dot', { best: k === bestK }]" />
      <circle
        :cx="sx(K)" :cy="sy(E(K))" r="8" class="handle draggable" tabindex="0" role="slider" aria-label="拖动改变 K"
        :aria-valuenow="K" aria-valuemin="1" aria-valuemax="10"
        @pointerdown="start($event, { svg, onMove })"
        @keydown.left="K = clamp(K - 1, 1, 10)" @keydown.right="K = clamp(K + 1, 1, 10)"
      />
    </svg>
    <div class="legend">
      <span><i class="e" />期望 token / target 调用</span><span><i class="s" />加速比 E / (1 + K·c), 实心点 = 最优 K</span>
    </div>

    <div class="round" aria-live="polite">
      <span class="lbl mono">draft</span>
      <span v-for="(c, i) in slots" :key="i" class="cell" :class="c.cls" :title="c.tip">{{ c.text }}</span>
      <span class="msg">{{ msg }}</span>
    </div>

    <template #stats>
      <div class="kv"><span>期望产出 (公式)</span><b>{{ E(K).toFixed(2) }}</b></div>
      <div class="kv"><span>模拟 {{ ROUNDS }} 轮均值</span><b :class="Math.abs(sim.mean - E(K)) < 0.05 ? 'good' : 'bad'">{{ sim.mean.toFixed(2) }}</b></div>
      <div class="kv"><span>加速比 E/(1+K·c)</span><b :class="speed(K) > 1 ? 'good' : 'bad'">{{ speed(K).toFixed(2) }}×</b></div>
      <div class="kv"><span>省掉的 target 调用</span><b>{{ ((1 - 1 / E(K)) * 100).toFixed(0) }}%</b></div>
      <div class="kv"><span>回放至今: token / target 调用</span><b>{{ soFar.tokens }} / {{ soFar.calls }}</b></div>
      <p class="lab-note">
        每轮产出 = 接受数 n + 1。那个 "+1" 永远有: 拒绝时是 target 自己给的纠错 token, 全接受时是 bonus。
        所以最差也不比普通 decode 少出 token, 只是白付了 K 次 draft 的钱 (加速比 &lt; 1)。最优 K = {{ bestK }}。
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
import { argmax, clamp, mulberry32, range } from '@/utils/labmath.js'

const ROUNDS = 20000, SHOWN = 12, PHASES = ['draft 连猜', 'target 一次验证', '接受 / 纠错']
const K = ref(4), alpha = ref(0.7), cost = ref(0.1), seed = ref(1)
const svg = ref(null)
const { start } = useDrag()

// ★ 第一个拒绝之前的都接受: P(n ≥ i) = α^i → E[n+1] = Σ_{i=0..K} α^i = (1−α^(K+1))/(1−α)
const E = (k) => (alpha.value === 0 ? 1 : (1 - alpha.value ** (k + 1)) / (1 - alpha.value))
const speed = (k) => E(k) / (1 + k * cost.value)     // 一轮耗时 = 1 次 target + K 次 draft
const cap = computed(() => 1 / (1 - alpha.value))
const bestK = computed(() => argmax(range(10).map((i) => speed(i + 1))) + 1)

// 模拟: 每个槽位独立以 α 接受, 遇到第一个拒绝就停 (与 accept_greedy / accept_sampling 的循环同构)
const sim = computed(() => {
  const rand = mulberry32(seed.value * 7919 + 13)
  const accepts = []
  let total = 0
  for (let r = 0; r < ROUNDS; r++) {
    let n = 0
    while (n < K.value && rand() < alpha.value) n++
    total += n + 1
    if (r < SHOWN) accepts.push(n)
  }
  return { mean: total / ROUNDS, accepts }
})

const yMax = computed(() => Math.max(2, E(10) * 1.2))
const yTicks = computed(() => { const s = yMax.value > 6 ? 2 : 1; return range(Math.floor(yMax.value / s) + 1).map((i) => i * s) })
const sx = (k) => 40 + (k - 1) * (470 / 9)
const sy = (v) => 180 - (clamp(v, 0, yMax.value) / yMax.value) * 170
const pts = (f) => range(10).map((i) => `${sx(i + 1)},${sy(f(i + 1))}`).join(' ')
const onMove = ({ x }) => { K.value = clamp(Math.round((x - 40) / (470 / 9)) + 1, 1, 10) }

const stepper = useStepper(() => SHOWN * 3, { interval: 700 })
const cur = computed(() => ({ round: Math.floor(stepper.step.value / 3), phase: stepper.step.value % 3 }))
watch([K, alpha, seed], () => stepper.reset())

const slots = computed(() => {
  const n = sim.value.accepts[cur.value.round], ph = cur.value.phase, all = n === K.value
  const cells = range(K.value).map((i) => {
    if (ph === 0) return { text: `d${i + 1}`, cls: 'on', tip: 'draft 猜的 token' }
    if (ph === 1) return { text: `d${i + 1}`, cls: 'hot', tip: 'target 同一次 forward 里验证' }
    if (i < n) return { text: '✓', cls: 'ok', tip: '与 target 一致, 接受' }
    if (i === n) return { text: '✗→t', cls: 'bad', tip: '第一个拒绝: 换成 target 的纠错 token' }
    return { text: '—', cls: 'dim', tip: '前面已拒绝, 作废 (KV 被 truncate 回滚)' }
  })
  cells.push(ph === 0 ? { text: '', cls: 'dim', tip: '第 K+1 个槽位' }
    : ph === 1 ? { text: '+1', cls: 'hot', tip: '第 K+1 个槽位也一起算' }
      : { text: all ? 'bonus' : '—', cls: all ? 'ok' : 'dim', tip: all ? '全部接受: 再白送 1 个' : '用不上' })
  return cells
})
const msg = computed(() => {
  const n = sim.value.accepts[cur.value.round]
  if (cur.value.phase === 0) return `draft 自回归 ${K.value} 步 (成本 ${K.value}×c)`
  if (cur.value.phase === 1) return `target 1 次 forward 验 ${K.value + 1} 个槽位`
  return `接受 ${n} 个 + ${n === K.value ? 'bonus' : '纠错'} 1 个 = 本轮 ${n + 1} 个 token`
})
const soFar = computed(() => {
  const done = cur.value.round + (cur.value.phase === 2 ? 1 : 0)
  return { calls: done, tokens: sim.value.accepts.slice(0, done).reduce((a, n) => a + n + 1, 0) }
})
</script>

<style scoped>
/* 窄屏: 保住可读的最小宽度, 由 .lab-viz 横向滚动; 只有 .draggable 把手拦截触摸 */
svg { min-width: 460px; touch-action: pan-x pan-y; }
.axis { stroke: var(--border-strong); stroke-width: 1; }
.base { stroke: var(--text-dim); stroke-dasharray: 2 4; }
.cap { stroke: var(--warn); stroke-dasharray: 5 4; }
.tick { font-size: 10px; fill: var(--text-dim); text-anchor: middle; }
.tick.end { text-anchor: end; }
.tick.warn { fill: var(--warn); }
.curve { fill: none; stroke-width: 2; }
.curve.e { stroke: var(--accent); }
.curve.s { stroke: var(--left); stroke-dasharray: 4 3; }
.dot { fill: var(--bg-elev); stroke: var(--left); }
.dot.best { fill: var(--left); }
.handle { fill: var(--accent); stroke: var(--bg); stroke-width: 2; }
.legend { display: flex; flex-wrap: wrap; gap: 14px; font-size: 11px; color: var(--text-muted); margin: 6px 0 12px; }
.legend i { display: inline-block; width: 14px; height: 3px; margin-right: 5px; vertical-align: middle; }
.legend i.e { background: var(--accent); } .legend i.s { background: var(--left); }
.round { display: flex; flex-wrap: wrap; gap: 4px; align-items: center; }
.round .cell { min-width: 40px; height: 26px; }
.round .lbl { font-size: 11px; color: var(--text-dim); margin-right: 4px; }
.round .msg { font-size: 12px; color: var(--text-muted); margin-left: 8px; }
</style>
