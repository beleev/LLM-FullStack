<!--
  采样流水线实验台 (对应 llm_infer/m10_sampling/samplers.py:sample)。
  一件事: 采样参数不是一堆独立旋钮, 而是一条有先后顺序的流水线 —— 每一级看到的都是上一级处理过的分布。
-->
<template>
  <LabFrame
    title="采样流水线 — 重复惩罚 → 温度 → top-k → top-p → min-p"
    sub="上下文「今天天气真」的下一个 token。空心框 = 过滤前的概率, 实心 = 过滤并重新归一化后的概率; 被砍掉的 token 下面标着是哪一级砍的。点柱子 = 把它放进 / 移出「已出现过」的历史 (吃重复惩罚); 拖竖线 = 改 top-k。"
    module="llm_infer/m10"
    run="python -m llm_infer.m10_sampling.demo"
    :challenge="{
      ask: '设 top-p = 0.9, 记下存活个数; 然后把温度从 1.0 拖到 2.0。存活数变多还是变少? 再设 min-p = 0.1 做同样的事。如果把顺序改成「先 top-p 再除温度」, 结果还一样吗?',
      answer: '温度在过滤之前: T 变大 → 分布变平 → 凑够 0.9 的质量需要更多 token, top-p 留下的变多; min-p 的阈值是 min_p × p_max, 分布变平后 p_max 下降, 留下的也变多。两者都是「模型越犹豫留得越多」, 但 top-k 完全不看分布, 永远留 k 个。如果先过滤再除温度, 存活集合就与 T 无关了 —— 同一组参数在不同框架里结果不同, 多半就是顺序不同 (例如 min-p 放在 top-p 之前还是之后)。',
    }"
  >
    <template #controls>
      <LabSlider v-model="pen" label="① 重复惩罚" :min="1" :max="2" :step="0.05" :format="(t) => t.toFixed(2)" />
      <LabSlider v-model="T" label="② 温度 T" :min="0.2" :max="3" :step="0.1" :format="(t) => t.toFixed(1)" />
      <LabSlider v-model="topK" label="③ top-k (0 = 关)" :min="0" :max="16" />
      <LabSlider v-model="topP" label="④ top-p" :min="0.05" :max="1" :step="0.05" :format="(t) => t.toFixed(2)" />
      <LabSlider v-model="minP" label="⑤ min-p" :min="0" :max="0.5" :step="0.01" :format="(t) => t.toFixed(2)" />
      <div class="row">
        <button type="button" @click="seed++">换一组 logits</button>
        <button type="button" @click="hist = new Set()">清空历史</button>
      </div>
    </template>

    <svg ref="svg" :viewBox="`0 0 ${X0 + V * BW + 10} 250`" role="group" aria-label="按概率排序的 token 柱状图">
      <line :x1="X0" :x2="X0 + V * BW" y1="190" y2="190" class="axis" />
      <g v-for="(b, r) in bars" :key="b.i" tabindex="0" role="button" class="bar"
        :aria-label="`${b.tok}: ${hist.has(b.i) ? '移出' : '放进'}历史`" :aria-pressed="hist.has(b.i)"
        @click="toggle(b.i)" @keydown.enter="toggle(b.i)">
        <rect :x="X0 + r * BW" y="10" :width="BW" height="232" class="hit" />
        <rect :x="X0 + r * BW + 4" :y="190 - b.pre * SCALE" :width="BW - 8" :height="b.pre * SCALE" class="pre" :class="{ cut: b.by }" />
        <rect v-if="!b.by" :x="X0 + r * BW + 4" :y="190 - b.post * SCALE" :width="BW - 8" :height="b.post * SCALE" class="post" />
        <text :x="X0 + (r + 0.5) * BW" :y="184 - Math.max(b.pre, b.post) * SCALE" class="num">{{ (b.by ? b.pre : b.post).toFixed(2) }}</text>
        <text :x="X0 + (r + 0.5) * BW" y="206" class="tok" :class="{ h: hist.has(b.i) }">{{ b.tok }}</text>
        <text v-if="hist.has(b.i)" :x="X0 + (r + 0.5) * BW" y="220" class="tag h">已出现</text>
        <text v-if="b.by" :x="X0 + (r + 0.5) * BW" y="234" class="tag" :class="b.by">{{ NAMES[b.by] }}</text>
      </g>
      <g class="draggable" tabindex="0" role="slider" aria-label="拖动改变 top-k" :aria-valuenow="topK" aria-valuemin="0" aria-valuemax="16"
        @pointerdown="start($event, { svg, onMove })"
        @keydown.left="topK = clamp((topK || V) - 1, 1, V)" @keydown.right="topK = topK && topK < V - 1 ? topK + 1 : 0">
        <rect :x="cutX - 8" y="4" width="16" height="190" class="grab" />
        <line :x1="cutX" :x2="cutX" y1="14" y2="192" class="cutline" />
        <text :x="Math.min(cutX, X0 + V * BW - 22)" y="10" class="cutlbl">top-k = {{ topK || '关' }}</text>
      </g>
    </svg>

    <template #stats>
      <div class="kv"><span>存活 token</span><b>{{ alive.length }} / {{ V }}</b></div>
      <div class="kv"><span>保留的原始概率质量</span><b :class="mass < 0.5 ? 'bad' : 'good'">{{ (mass * 100).toFixed(1) }}%</b></div>
      <div class="kv"><span>熵: 过滤前 → 后 (nats)</span><b>{{ hPre.toFixed(2) }} → {{ hPost.toFixed(2) }}</b></div>
      <div class="kv"><span>top-1 概率 (最终)</span><b>{{ top1.toFixed(3) }}</b></div>
      <div class="kv"><span>各级砍掉</span><b class="sm">k {{ cutBy.k }} · p {{ cutBy.p }} · min-p {{ cutBy.m }}</b></div>
      <p class="lab-note">
        重复惩罚: 出现过的 token, 正 logit ÷ 惩罚、负 logit × 惩罚 (目标都是"让它变小")。它在温度之前, 所以 greedy 也受影响。
        top-p 留下"累计质量首次 ≥ p 的最小集合"; min-p 砍 p &lt; min_p·p_max 的。后面的级只看到前面留下的 token。
      </p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { useDrag } from '@/composables/useDrag.js'
import { clamp, entropy, mulberry32, randn, softmax, sum } from '@/utils/labmath.js'

const TOKS = ['好', '不错', '热', '冷', '棒', '糟', '的', '舒服', '闷', '差', '怪', '是', '吗', '了', '猫', '量子']
const BASE = [4.2, 3.6, 3.1, 2.8, 2.4, 1.9, 1.5, 1.2, 0.8, 0.4, 0, -0.5, -1, -1.5, -2.5, -3.5]
const NAMES = { k: 'top-k', p: 'top-p', m: 'min-p' }
const V = TOKS.length, X0 = 10, BW = 34
const pen = ref(1), T = ref(1), topK = ref(0), topP = ref(1), minP = ref(0), seed = ref(1)
const hist = ref(new Set())
const svg = ref(null)
const { start } = useDrag()
const toggle = (i) => { const s = new Set(hist.value); s.has(i) ? s.delete(i) : s.add(i); hist.value = s }

const logits = computed(() => { const rand = mulberry32(seed.value * 31 + 5); return BASE.map((z) => z + (seed.value > 1 ? 0.8 * randn(rand) : 0)) })

// ★ 与 samplers.sample 同序: rep penalty → /T → top-k → top-p → min-p; 每一级只在上一级的幸存者上重新 softmax
const pipe = computed(() => {
  let z = logits.value.map((v, i) => (hist.value.has(i) && pen.value !== 1 ? (v > 0 ? v / pen.value : v * pen.value) : v))
  z = z.map((v) => v / T.value)
  const pre = softmax(z), by = Array(V).fill('')
  const order = () => z.map((v, i) => i).filter((i) => z[i] > -Infinity).sort((a, b) => z[b] - z[a])
  const kill = (ids, tag) => ids.forEach((i) => { z[i] = -Infinity; by[i] = tag })
  if (topK.value > 0 && topK.value < V) kill(order().slice(topK.value), 'k')
  if (topP.value < 1) {
    const p = softmax(z), o = order()
    let cum = 0, keep = 0
    while (keep < o.length) { cum += p[o[keep++]]; if (cum >= topP.value - 1e-12) break }   // 首次 ≥ p 的那个也留下
    kill(o.slice(keep), 'p')
  }
  if (minP.value > 0) { const p = softmax(z), pm = Math.max(...p); kill(order().filter((i) => p[i] < minP.value * pm), 'm') }
  return { pre, post: softmax(z), by }
})

const bars = computed(() => TOKS.map((tok, i) => ({ i, tok, pre: pipe.value.pre[i], post: pipe.value.post[i], by: pipe.value.by[i] }))
  .sort((a, b) => b.pre - a.pre))
const alive = computed(() => bars.value.filter((b) => !b.by))
const mass = computed(() => sum(alive.value.map((b) => b.pre)))
const hPre = computed(() => entropy(pipe.value.pre)), hPost = computed(() => entropy(pipe.value.post))
const top1 = computed(() => Math.max(...pipe.value.post))
const cutBy = computed(() => Object.fromEntries(['k', 'p', 'm'].map((t) => [t, bars.value.filter((b) => b.by === t).length])))

// 纵轴按最大概率向上取整到 0.25 的倍数: 柱子够高, 又不会每动一下就缩放
const SCALE = computed(() => 165 / (Math.ceil(Math.max(...pipe.value.pre, ...pipe.value.post) * 4) / 4))
const cutX = computed(() => X0 + (topK.value || V) * BW)
const onMove = ({ x }) => { const k = clamp(Math.round((x - X0) / BW), 1, V); topK.value = k >= V ? 0 : k }
</script>

<style scoped>
/* 窄屏: 保住可读的最小宽度, 由 .lab-viz 横向滚动; 只有 .draggable 把手拦截触摸 */
svg { min-width: 540px; touch-action: pan-x pan-y; }
.axis { stroke: var(--border-strong); }
.bar { cursor: pointer; outline: none; }
.hit { fill: transparent; }
.bar:hover .hit, .bar:focus-visible .hit { fill: var(--accent-soft); }
.pre { fill: none; stroke: var(--text-dim); stroke-width: 1; }
.pre.cut { stroke-dasharray: 3 2; }
.post { fill: var(--accent); opacity: 0.85; }
.num { font-size: 8px; fill: var(--text-muted); text-anchor: middle; font-family: "SF Mono", Menlo, monospace; }
.tok { font-size: 11px; fill: var(--text); text-anchor: middle; }
.tok.h { fill: var(--warn); font-weight: 600; }
.tag { font-size: 8px; text-anchor: middle; fill: var(--text-dim); }
.tag.h { fill: var(--warn); }
.tag.k { fill: var(--danger); } .tag.p { fill: var(--eye); } .tag.m { fill: var(--right); }
.grab { fill: transparent; }
.cutline { stroke: var(--danger); stroke-width: 2; stroke-dasharray: 5 3; }
.cutlbl { font-size: 9px; fill: var(--danger); text-anchor: middle; }
.sm { font-size: 12px !important; }
</style>
