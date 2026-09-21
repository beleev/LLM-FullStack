<!--
  可学 attention sink 实验台 (对应 llm_models/layers/core/attention.py:GroupedQueryAttention 的 use_sink)。
  只讲一件事: softmax 强迫概率和为 1, 一个 head 即使 “没什么可看” 也得把 1 分完;
  GPT-OSS 给每个 head 加一个可学 logit 参与 softmax、算完就丢 —— 给 head 一个 “谁都不看” 的出口。
-->
<template>
  <LabFrame
    title="Attention sink — 给 softmax 一个 “弃权” 选项"
    sub="上排是 6 个 key 的注意力 logit: 直接上下拖每根柱子。下排是 softmax 之后的概率。最右一列是 sink: 它只是一个可学标量, 参与 softmax 分母, 但没有 value, 分到它头上的概率直接丢掉。"
    module="llm_models/layers/core/attention.py"
    run="python -m llm_models.run_models.moe.gpt_oss.infer_gpt_oss"
    :challenge="{
      ask: '点 “全都不相关”, 关掉 sink。先猜: 6 个 key 的概率各是多少、加起来多少? 这个 head 的输出是什么? 再打开 sink。',
      answer: '6 个 logit 都在 −3 左右, 但 softmax 只看相对大小: 每个 ≈ 1/6, 总和恒为 1。head 被迫输出 6 个无关 value 的平均 —— 满幅度的噪声, 照样加进残差流。没有 sink 的模型为了自救, 会学着把多余的概率倒到第一个 token 上 (StreamingLLM 观察到的现象), 所以滑窗一旦把开头 token 挤出 cache, 模型就崩。打开 sink (logit=0): 分母多了一项 e⁰=1, 而 6 个 key 合计才 6·e⁻³≈0.3, sink 吸走 77%, 真实 key 的概率和只剩 23% —— head 学会了 “小声说话”。有相关 key (logit=4) 时 e⁴≈55 远大于 1, sink 几乎不影响。它和滑窗是绝配: 不再依赖必须留在 cache 里的开头 token。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: useSink }" @click="useSink = true">use_sink = True (GPT-OSS)</button>
        <button type="button" :class="{ active: !useSink }" @click="useSink = false">标准 softmax</button>
        <button type="button" @click="logits = [...PRESET.hit]">预设: 有一个相关 key</button>
        <button type="button" @click="logits = [...PRESET.none]">预设: 全都不相关</button>
        <button type="button" @click="demo">预设: Python demo (3 个 0 分 key + sink = ln 3)</button>
      </div>
      <LabSlider v-if="useSink" v-model="sink" label="sink logit (可学)" :min="-4" :max="4" :step="0.1" :format="(t) => t.toFixed(1)" />
    </template>

    <svg ref="svg" viewBox="0 0 520 300" role="group" aria-label="logit 与概率">
      <text x="4" y="12" class="sec">logit (可拖)</text><text x="4" y="192" class="sec">softmax 概率</text>
      <line :x1="X0" :x2="X1" :y1="ly(0)" :y2="ly(0)" class="base" /><line :x1="X0" :x2="X1" :y1="PY1" :y2="PY1" class="base" />
      <g v-for="(l, j) in logits" :key="j">
        <rect :x="cx(j) - 18" :y="Math.min(ly(l), ly(0))" width="36" :height="Math.abs(ly(l) - ly(0))" class="lbar" />
        <text :x="cx(j)" :y="l >= 0 ? ly(l) - 10 : ly(l) + 18" class="num">{{ l.toFixed(1) }}</text>
        <g class="draggable" tabindex="0" role="slider" :aria-label="`key ${j} 的 logit`" :aria-valuenow="l"
          @pointerdown="start($event, { svg, onMove: ({ y }) => setL(j, y) })" @keydown.up.prevent="bump(j, 0.5)" @keydown.down.prevent="bump(j, -0.5)">
          <rect :x="cx(j) - 22" :y="LY0 - 4" width="44" :height="LY1 - LY0 + 8" fill="transparent" />
          <circle :cx="cx(j)" :cy="ly(l)" r="6" class="handle" />
        </g>
        <rect :x="cx(j) - 18" :y="pyy(probs[j])" width="36" :height="PY1 - pyy(probs[j])" class="pbar" />
        <text :x="cx(j)" :y="pyy(probs[j]) - 4" class="num">{{ (probs[j] * 100).toFixed(0) }}%</text>
        <text :x="cx(j)" y="292" class="tok">k{{ j }}</text>
      </g>
      <g v-if="useSink">
        <rect :x="cx(6) - 18" :y="Math.min(ly(sink), ly(0))" width="36" :height="Math.abs(ly(sink) - ly(0))" class="sbar" />
        <text :x="cx(6)" :y="sink >= 0 ? ly(sink) - 5 : ly(sink) + 12" class="num">{{ sink.toFixed(1) }}</text>
        <rect :x="cx(6) - 18" :y="pyy(sinkP)" width="36" :height="PY1 - pyy(sinkP)" class="sbar ghost" />
        <text :x="cx(6)" :y="pyy(sinkP) - 4" class="num">{{ (sinkP * 100).toFixed(0) }}% 丢弃</text>
        <text :x="cx(6)" y="292" class="tok s">sink</text>
      </g>
    </svg>

    <template #stats>
      <div class="kv"><span>真实 key 的概率和 (= 输出幅度)</span><b :class="keySum < 0.999 ? 'good' : ''">{{ (keySum * 100).toFixed(1) }}%</b></div>
      <div class="kv"><span>被 sink 吸走</span><b>{{ (sinkP * 100).toFixed(1) }}%</b></div>
      <div class="kv"><span>最大单 key 概率</span><b>{{ (Math.max(...probs) * 100).toFixed(1) }}%</b></div>
      <div class="kv"><span>最大 key logit − sink logit</span><b>{{ useSink ? (Math.max(...logits) - sink).toFixed(1) : '—' }}</b></div>
      <p class="lab-note">
        {{ useSink ? 'sink 像一个 “分数固定的空 key”: key 的 logit 明显高于它时它隐身, 都低于它时它把概率吸走。初始化为 0, 每个 head 自己学。'
          : '标准 softmax: 把所有 logit 一起往下拖, 概率纹丝不动 —— 它只认相对差, 表达不了 “这些我都不想看”。' }}
      </p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { useDrag } from '@/composables/useDrag.js'
import { clamp, softmax, sum } from '@/utils/labmath.js'

const PRESET = { hit: [-2, -1, 4, -1.5, -2, -1], none: [-3, -3.2, -2.8, -3.1, -2.9, -3] }
const useSink = ref(true), sink = ref(0), logits = ref([...PRESET.none])
const svg = ref(null)
const { start } = useDrag()

// ★ 与 Python 同: softmax(cat([scores, sink]))[..., :-1] —— sink 只进分母, 行和 < 1
const all = computed(() => softmax(useSink.value ? [...logits.value, sink.value] : logits.value))
const probs = computed(() => all.value.slice(0, 6))
const sinkP = computed(() => (useSink.value ? all.value[6] : 0))
const keySum = computed(() => sum(probs.value))

const X0 = 50, X1 = 515, LY0 = 22, LY1 = 170, PY0 = 204, PY1 = 274, LMAX = 6
const cx = (j) => X0 + 36 + j * 66
const ly = (v) => (LY0 + LY1) / 2 - (v / LMAX) * ((LY1 - LY0) / 2)
const pyy = (p) => PY1 - p * (PY1 - PY0)
const setL = (j, y) => { logits.value = logits.value.map((v, i) => (i === j ? clamp(Math.round((((LY0 + LY1) / 2 - y) / ((LY1 - LY0) / 2)) * LMAX * 10) / 10, -LMAX, LMAX) : v)) }
// infer_gpt_oss 里的算例: 3 个分数为 0 的 key + sink = ln 3 → 真实 key 合计正好一半 (另 3 个 key 压到 −6 ≈ 屏蔽)
const demo = () => { useSink.value = true; sink.value = Math.log(3); logits.value = [0, 0, 0, -6, -6, -6] }
const bump = (j, by) => { logits.value = logits.value.map((v, i) => (i === j ? clamp(v + by, -LMAX, LMAX) : v)) }
</script>

<style scoped>
.sec { font-size: 10px; fill: var(--text-dim); }
.base { stroke: var(--border-strong); }
.lbar { fill: color-mix(in srgb, var(--accent) 35%, transparent); stroke: var(--accent); }
.pbar { fill: color-mix(in srgb, var(--left) 45%, transparent); stroke: var(--left); }
.sbar { fill: color-mix(in srgb, var(--eye) 40%, transparent); stroke: var(--eye); }
.sbar.ghost { stroke-dasharray: 4 3; fill: color-mix(in srgb, var(--eye) 15%, transparent); }
.handle { fill: var(--accent); stroke: var(--bg); stroke-width: 2; }
.draggable:focus-visible .handle { stroke: var(--text); }
.num { text-anchor: middle; font-size: 10px; fill: var(--text-muted); font-family: "SF Mono", Menlo, monospace; }
.tok { text-anchor: middle; font-size: 11px; fill: var(--text-muted); }
.tok.s { fill: var(--eye); font-weight: 700; }
</style>
