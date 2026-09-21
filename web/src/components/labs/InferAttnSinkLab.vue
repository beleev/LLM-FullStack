<!-- Attention sink 实验台 (对应 llm_infer/m16_attention_sinks): 合成注意力 + 人为植入的 sink, 不是真模型的测量。 -->
<template>
  <LabFrame
    title="Attention sinks — 为什么滑动窗口不能扔掉开头几个 token"
    sub="这里的注意力是合成的: 我们人为在开头 4 个 token 上植入了 sink。训练过的 LLM 会自己长出这个现象, 随机权重的 TinyLM 没有, 所以不拿 TinyLM 演示。第一行: 最新 token 对整条流的注意力, 实心 = 还留在 cache 里 (sink + 最近窗口), 半透明 = 已被逐出。第二行: 只剩 cache 里的 token 时, softmax 被迫重新归一化后的注意力。"
    module="llm_infer/m16"
    run="python -m llm_infer.m16_attention_sinks.demo"
    :challenge="{
      ask: '把「保留的 sink 数」从 4 拖到 0 (纯滑动窗口), 丢失的注意力质量和幸存权重的放大倍数怎么变? 再把 sink 强度拖到 0 试试。',
      answer: 'softmax 的权重和恒为 1: 当前 token 没什么可看时, 多余的注意力被倒在开头几个 token 上。纯滑动窗口把这几个 token 逐出后, 分母突然少了一大块, 所有幸存权重被放大好几倍。注意力输出于是远离训练时见过的分布 → 困惑度爆炸。只要留住 4 个 sink, 分母基本不变, 丢掉的只是中间那些本来就没分到多少注意力的 token。sink 强度为 0 (没有 sink 现象) 时, 留不留开头就无所谓了 —— 这正是随机权重模型上看不到效果的原因。',
    }"
  >
    <template #controls>
      <LabSlider v-model="T" label="流长度 T" :min="16" :max="128" unit=" tok" />
      <LabSlider v-model="win" label="最近窗口 window" :min="4" :max="64" unit=" tok" />
      <LabSlider v-model="nSink" label="保留的 sink 数" :min="0" :max="8" />
      <LabSlider v-model="strength" label="sink 强度 (植入的 logit)" :min="0" :max="6" :step="0.5" />
      <div class="row">
        <button type="button" :class="{ active: reindex }" @click="reindex = true">位置按槽位重编号</button>
        <button type="button" :class="{ active: !reindex }" @click="reindex = false">沿用绝对位置</button>
        <button type="button" @click="seed++">换一组</button>
      </div>
    </template>

    <svg :viewBox="`0 0 ${W} ${2 * RH + 56}`" role="img" aria-label="逐出前后的注意力分布">
      <g v-for="(row, ri) in [m.full, m.renorm]" :key="ri" :transform="`translate(0, ${ri * (RH + 22) + 12})`">
        <text x="0" y="-2" class="ax">{{ ri ? '逐出后 (只在 cache 内重新归一化)' : '完整上下文的注意力' }}</text>
        <rect
          v-for="(p, j) in row" :key="j" :x="j * bw" :y="RH - (p / m.peak) * (RH - 10)" :width="Math.max(1, bw - 0.6)"
          :height="(p / m.peak) * (RH - 10)" :class="['bar', j < 4 ? 'sink' : '', { hl: j === hl }]" :opacity="m.kept[j] ? 1 : ri ? 0 : 0.25"
        />
        <line x1="0" :x2="W" :y1="RH" :y2="RH" class="base" />
      </g>
      <text x="0" :y="2 * RH + 50" class="ax">位置 0 (最老)</text>
      <text :x="W" :y="2 * RH + 50" class="ax" text-anchor="end">位置 {{ T - 1 }} (当前 query)</text>
    </svg>

    <p class="cap">cache 槽位 (共 {{ m.slots.length }} 个; 格内 = token 的绝对位置, 点一格看它喂给 RoPE 的位置):</p>
    <div class="slots">
      <button
        v-for="(abs, s) in m.slots" :key="s" type="button" class="cell slot"
        :class="[abs < 4 ? 'hot' : 'on', { hl: abs === hl }]" :title="`槽位 ${s} ← 绝对位置 ${abs}`"
        @click="hl = abs" @mouseenter="hl = abs"
      >{{ abs }}</button>
    </div>
    <p v-if="hlSlot >= 0" class="lab-note detail">
      绝对位置 <b>{{ hl }}</b> 的 token 现在住在槽位 <b>{{ hlSlot }}</b>; 喂给 RoPE 的位置 = <b>{{ reindex ? hlSlot : hl }}</b>,
      与当前 query 的距离 = {{ reindex ? m.slots.length - 1 - hlSlot : T - 1 - hl }}。
    </p>

    <template #stats>
      <div class="kv"><span>逐出丢掉的注意力质量</span><b :class="m.lost > 0.4 ? 'bad' : m.lost < 0.25 ? 'good' : ''">{{ (m.lost * 100).toFixed(1) }}%</b></div>
      <div class="kv"><span>幸存权重被放大</span><b :class="m.lost > 0.4 ? 'bad' : m.lost < 0.25 ? 'good' : ''">{{ (1 / (1 - m.lost)).toFixed(2) }}×</b></div>
      <div class="kv"><span>cache 条目 / 流长度</span><b class="good">{{ m.slots.length }} / {{ T }}</b></div>
      <div class="kv">
        <span>喂给 RoPE 的最大位置 (训练长度 {{ TRAIN }})</span>
        <b :class="maxPos >= TRAIN ? 'bad' : 'good'">{{ maxPos }}</b>
      </div>
      <p class="lab-note">
        放大倍数 = 1 / (1 − 丢失质量): 分母少了多少, 幸存的每个权重就等比例涨多少。<br />
        ★ 重编号: 位置 = cache 槽位 0..L−1, 永远不超过训练长度。为了每步都能按新槽位重转, <code class="inline">SinkCache</code>
        里存的是<b>未旋转 (pre-RoPE)</b> 的 K —— 和普通 KV cache 存 post-RoPE 的 K 正好相反, 代价是每步 O(L·D) 的重转。
        被逐出的中间内容是真的忘了: sink 只保证不崩, 不保证记得。
      </p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { mulberry32, randn, softmax, range, sum } from '@/utils/labmath.js'

const W = 640, RH = 90, TRAIN = 64
const T = ref(96), win = ref(16), nSink = ref(4), strength = ref(4), seed = ref(1)
const reindex = ref(true), hl = ref(0)

const m = computed(() => {
  const rand = mulberry32(seed.value), n = T.value
  const noise = range(128).map(() => 0.4 * randn(rand))                    // 先抽满 128 个, 拖 T 时旧 token 的噪声不变
  // 合成 logit: 开头 4 个 token 的 sink 加成 + 越近越高的 recency + 噪声
  const logits = range(n).map((j) => (j < 4 ? strength.value : 0) + 3 * Math.exp(-(n - 1 - j) / 6) + noise[j])
  const full = softmax(logits)
  const kept = range(n).map((j) => j < nSink.value || j >= n - win.value) // ★ SinkCache 留下的: 开头 n_sink 个 + 最近 window 个
  const lost = sum(full.filter((_, j) => !kept[j]))
  const renorm = full.map((p, j) => (kept[j] ? p / (1 - lost) : 0))       // 被逐出的 key 不存在了 → softmax 只在幸存者里归一
  return { full, kept, lost, renorm, peak: Math.max(...renorm), slots: range(n).filter((j) => kept[j]) }
})
const bw = computed(() => W / T.value)
const hlSlot = computed(() => m.value.slots.indexOf(hl.value))
const maxPos = computed(() => (reindex.value ? m.value.slots.length - 1 : T.value - 1))
</script>

<style scoped>
svg { min-width: 520px; } /* 窄屏: 图保持可读, 由 .lab-viz 横向滚动 */
.bar { fill: var(--accent); }
.bar.sink { fill: var(--warn); }
.bar.hl { fill: var(--text); }
.base { stroke: var(--border-strong); }
.ax { font-size: 11px; fill: var(--text-dim); }
.cap { font-size: 12px; color: var(--text-muted); margin: 12px 0 6px; }
.slots { display: flex; flex-wrap: wrap; gap: 3px; }
.slot { min-width: 26px; padding: 0 2px; cursor: pointer; font-size: 10px; min-height: 0; }
.slot.hl { outline: 2px solid var(--text); outline-offset: 1px; }
.detail { margin-top: 10px; }
.detail b { color: var(--text); }
</style>
