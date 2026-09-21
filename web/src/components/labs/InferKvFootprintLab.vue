<!-- KV 体积实验台 (对应 llm_infer/m18_kv_attention_variants): 两条公式 + 一条可拖的显存线。 -->
<template>
  <LabFrame
    title="KV 体积 — MHA / GQA / MQA / MLA 各占多少显存"
    sub="同一个 attention 数学, 只是「cache 里存什么」不同。上图是每 token 的 KV 字节; 下图每个小格 = 一条序列在所选上下文长度下的 KV, 竖线是留给 KV 的显存预算 —— 左右拖动它, 看每种结构能同时放下几条序列。"
    module="llm_infer/m18"
    run="python -m llm_infer.m18_kv_attention_variants.demo"
    :challenge="{
      ask: '选 DeepSeek-V3 预设: MLA 每 token 只有 68.6 KiB, 比 GQA-8 (LLaMA-3-8B 的 128 KiB) 还小, 可它有 128 个头。它靠什么做到的? 代价在哪?',
      answer: 'MLA 不缓存每头的 K/V, 只缓存一份低秩 latent c_kv (512 维) + 一份所有头共享的 RoPE key (64 维): (512+64)·61 层·2 字节 = 68.6 KiB, 比它自己的 MHA 版本 (3904 KiB) 小 56.9×。代价是 decode 时要把 latent 还原成各头的 K/V (或走 absorb 形式把 W_UK 吸进 q), 计算更多 —— 用算力换显存和带宽, 而 decode 恰好是带宽受限的。',
    }"
  >
    <template #controls>
      <div class="row">
        <button v-for="p in PRESETS" :key="p.name" type="button" :class="{ active: presetName === p.name }" @click="apply(p)">{{ p.name }}</button>
      </div>
      <LabSlider v-model="L" label="层数 n_layer" :min="8" :max="96" />
      <LabSlider v-model="hExp" label="query 头数 n_head" :min="3" :max="7" :format="(v) => 2 ** v" />
      <LabSlider v-model="kvExp" label="GQA 的 KV 头数" :min="0" :max="hExp" :format="(v) => 2 ** v" />
      <LabSlider v-model="dhExp" label="d_head" :min="6" :max="8" :format="(v) => 2 ** v" />
      <LabSlider v-model="dc" label="MLA latent d_c (+64 rope)" :min="128" :max="1024" :step="64" />
      <LabSlider v-model="ctxExp" label="上下文长度" :min="10" :max="17" :format="(v) => 2 ** v" unit=" tok" />
      <div class="row">
        <span class="hint">KV 精度:</span>
        <button v-for="d in DTYPES" :key="d.name" type="button" :class="{ active: nbytes === d.b }" @click="nbytes = d.b">{{ d.name }}</button>
      </div>
    </template>

    <div class="bars">
      <div v-for="v in variants" :key="v.name" class="bar-row" :class="{ native: v.name === native }">
        <span class="bar-name mono">{{ v.name }}<small>{{ v.what }}</small></span>
        <span class="bar-track"><span class="bar-fill" :style="{ width: (v.bytes / variants[0].bytes) * 100 + '%', background: v.color }" /></span>
        <span class="bar-val mono">{{ fmtKiB(v.bytes) }}</span>
      </div>
    </div>

    <svg ref="svg" :viewBox="`0 0 ${W} 190`" role="img" aria-label="每种结构在显存预算内能放几条序列">
      <g v-for="(v, i) in variants" :key="v.name" :transform="`translate(0, ${18 + i * 38})`">
        <text x="0" y="-4" class="ax">{{ v.name }} · 每条 {{ v.gib.toFixed(2) }} GiB · 放得下 {{ v.fit }} 条</text>
        <template v-if="v.gib * sx >= 3">
          <rect
            v-for="k in v.fit" :key="k" :x="(k - 1) * v.gib * sx + 0.5" y="0"
            :width="v.gib * sx - 1" height="18" :fill="v.color" opacity="0.75"
          />
        </template>
        <rect v-else x="0" y="0" :width="v.fit * v.gib * sx" height="18" :fill="v.color" opacity="0.75" />
        <rect v-if="!v.fit" x="0" y="0" :width="Math.min(W, v.gib * sx)" height="18" class="over" />
      </g>
      <line v-for="g in [0, 40, 80, 120, 160]" :key="g" :x1="g * sx" :x2="g * sx" y1="166" y2="172" class="tick" />
      <text v-for="g in [0, 40, 80, 120, 160]" :key="'t' + g" :x="Math.min(W - 2, Math.max(2, g * sx))" y="186" class="ax" :text-anchor="g === 0 ? 'start' : g === 160 ? 'end' : 'middle'">{{ g }} GiB</text>
      <g
        class="draggable" tabindex="0" role="slider" aria-label="KV 显存预算" :aria-valuenow="budget" aria-valuemin="4" aria-valuemax="160"
        @pointerdown="start($event, { svg, onMove: ({ x }) => (budget = clamp(Math.round(x / sx), 4, 160)) })"
        @keydown.left.prevent="budget = clamp(budget - 1, 4, 160)" @keydown.right.prevent="budget = clamp(budget + 1, 4, 160)"
      >
        <rect :x="budget * sx - 12" y="0" width="24" height="172" fill="transparent" />
        <line :x1="budget * sx" :x2="budget * sx" y1="0" y2="172" class="budget" />
        <circle :cx="budget * sx" cy="166" r="7" class="knob" />
        <text :x="budget * sx + (budget > 130 ? -10 : 10)" y="10" class="budget-t" :text-anchor="budget > 130 ? 'end' : 'start'">⇆ KV 预算 {{ budget }} GiB</text>
      </g>
    </svg>

    <template #stats>
      <div v-for="v in variants" :key="v.name" class="kv">
        <span>{{ v.name }} 最大并发</span>
        <b :class="v.fit === 0 ? 'bad' : v.fit === best ? 'good' : ''">{{ v.fit }} 条</b>
      </div>
      <div class="kv"><span>MLA 比 MHA 省</span><b class="good">{{ (variants[0].bytes / variants[3].bytes).toFixed(1) }}×</b></div>
      <p class="lab-note">
        ★ 每 token 字节 = 2 · n_kv · d_head · n_layer · bytes; MLA = (d_c + d_rope) · n_layer · bytes (K/V 共用 latent, 没有「2·」)。<br />
        并发上限 = ⌊预算 ÷ (每 token 字节 × 上下文)⌋。和 demo 对得上: 40 GiB、8192 上下文时 7B 形状的 MHA / GQA-8 / MQA = 10 / 40 / 320 条, DeepSeek-V3 MLA = 74 条。
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

const PRESETS = [
  { name: 'LLaMA-2-7B', L: 32, h: 5, kv: 3, dh: 7, dc: 512, native: 'MHA' },
  { name: 'LLaMA-3-8B', L: 32, h: 5, kv: 3, dh: 7, dc: 512, native: 'GQA' },
  { name: 'LLaMA-3-70B', L: 80, h: 6, kv: 3, dh: 7, dc: 512, native: 'GQA' },
  { name: 'DeepSeek-V3', L: 61, h: 7, kv: 3, dh: 7, dc: 512, native: 'MLA' },
]
const DTYPES = [{ name: 'fp16', b: 2 }, { name: 'int8', b: 1 }, { name: 'int4', b: 0.5 }]
const D_ROPE = 64, GIB = 2 ** 30, W = 640, sx = W / 160

const L = ref(32), hExp = ref(5), kvExp = ref(3), dhExp = ref(7), dc = ref(512)
const ctxExp = ref(13), nbytes = ref(2), budget = ref(40)
const presetName = ref('LLaMA-3-8B'), native = ref('GQA')
const apply = (p) => { L.value = p.L; hExp.value = p.h; kvExp.value = p.kv; dhExp.value = p.dh; dc.value = p.dc; presetName.value = p.name; native.value = p.native }

watch(hExp, (h) => { kvExp.value = Math.min(kvExp.value, h) }) // KV 头数不能超过 query 头数

const svg = ref(null)
const { start } = useDrag()

const variants = computed(() => {
  const H = 2 ** hExp.value, G = 2 ** Math.min(kvExp.value, hExp.value), dh = 2 ** dhExp.value, ctx = 2 ** ctxExp.value
  const gqa = (nkv) => 2 * nkv * dh * L.value * nbytes.value          // ★ 2 = K 和 V
  const rows = [
    { name: 'MHA', what: `${H} kv 头`, bytes: gqa(H), color: 'var(--danger)' },
    { name: 'GQA', what: `${G} kv 头`, bytes: gqa(G), color: 'var(--warn)' },
    { name: 'MQA', what: '1 kv 头', bytes: gqa(1), color: 'var(--left)' },
    { name: 'MLA', what: `latent ${dc.value}+${D_ROPE}`, bytes: (dc.value + D_ROPE) * L.value * nbytes.value, color: 'var(--accent)' },
  ]
  return rows.map((r) => ({ ...r, gib: (r.bytes * ctx) / GIB, fit: Math.floor((budget.value * GIB) / (r.bytes * ctx)) }))
})
const best = computed(() => Math.max(...variants.value.map((v) => v.fit)))
const fmtKiB = (b) => (b / 1024 >= 100 ? (b / 1024).toFixed(0) : (b / 1024).toFixed(1)) + ' KiB'
</script>

<style scoped>
svg { min-width: 520px; } /* 窄屏: 图保持可读, 由 .lab-viz 横向滚动 */
.hint { font-size: 12px; color: var(--text-muted); }
.bars { display: flex; flex-direction: column; gap: 6px; margin-bottom: 16px; }
.bar-row { display: grid; grid-template-columns: 96px 1fr 76px; gap: 8px; align-items: center; font-size: 12px; }
.bar-name { color: var(--text); }
.bar-name small { display: block; font-size: 10px; color: var(--text-dim); }
.bar-row.native .bar-name { color: var(--accent); }
.bar-track { height: 14px; background: var(--bg-elev); border: 1px solid var(--border); border-radius: 3px; overflow: hidden; }
.bar-fill { display: block; height: 100%; min-width: 2px; }
.bar-val { text-align: right; color: var(--text-muted); }
.ax { font-size: 11px; fill: var(--text-muted); }
.tick { stroke: var(--border-strong); }
.over { fill: none; stroke: var(--danger); stroke-dasharray: 4 3; }
.budget { stroke: var(--text); stroke-width: 2; }
.knob { fill: var(--text); }
.budget-t { font-size: 11px; fill: var(--text); }
</style>
