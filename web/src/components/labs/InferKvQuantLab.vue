<!--
  KV cache 量化分组实验台 (对应 llm_infer/m08_quantization/kv_quant.py:quantize_kv)。
  一件事: 同一个量化器, 只是"谁和谁共用一组 scale"不同 —— K 有固定离群通道所以按通道分组, V 没有所以按 token 分组 (KIVI)。
-->
<template>
  <LabFrame
    title="KV 量化 — per-tensor / per-token / per-channel 谁背离群值的锅"
    sub="最左是 16 个 token × 12 个通道的 K 矩阵 (一行 = 一个 token)。点某一列把它变成 / 取消「固定离群通道」(每个 token 在这一维都是大值)。右边三张是三种分组下的量化误差 |K − K̂|, 同一色标。悬停任意格子看它的 scale。"
    module="llm_infer/m08"
    run="python -m llm_infer.m08_quantization.demo"
    :challenge="{
      ask: 'K 型 + INT4: 三种分组谁误差最小? 先猜再看。然后切到 V 型 (没有固定离群通道, 但各 token 幅度差别大), 排名变了吗? 最后在 K 型下把比特数拖到 2。',
      answer: 'K 型: per-channel 最好。离群通道在所有 token 上都大, per-token 分组时每一行都含这几个离群值, 每行的 scale 都被撑大, 其余通道只剩一两个格点; 按通道分组则把离群值关在自己那一列, 别的列用各自的小 scale。V 型反过来: 没有固定离群列, 但 token 之间幅度差别大, 按行分组正好隔离大 token, 而且新 token 写入时可以独立量化, 对流式追加友好 —— 这就是 KIVI 的「K per-channel, V per-token」。比特越低差距越大: 真实 demo 里 INT4 的 K 误差 per-channel 0.026 vs per-token 0.18; 到 2 bit 时分错组基本不可用。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: kind === 'K' }" @click="setKind('K')">K 型 (固定离群通道)</button>
        <button type="button" :class="{ active: kind === 'V' }" @click="setKind('V')">V 型 (无固定离群通道)</button>
        <button type="button" @click="seed++">换一组</button>
      </div>
      <LabSlider v-model="bits" label="比特数" :min="2" :max="8" unit=" bit" />
      <LabSlider v-model="mag" label="离群通道幅度" :min="0" :max="20" :step="0.5" unit="×σ" />
    </template>

    <svg viewBox="0 0 572 205" role="group" aria-label="K 矩阵与三种分组的量化误差热力图" @mouseleave="hover = null">
      <g v-for="(p, k) in PANELS" :key="p.id" :transform="`translate(${8 + k * 142}, 0)`">
        <text x="0" y="10" class="cap" :class="{ best: k > 0 && k - 1 === bestI, worst: k > 0 && k - 1 === worstI }">{{ p.label }}{{ k > 0 ? ' ' + q.errs[k - 1].toFixed(3) : '' }}</text>
        <template v-if="k === 0">
          <rect v-for="c in D" :key="`h${c}`" :x="(c - 1) * CS" y="16" :width="CS - 1" height="9" tabindex="0" role="button"
            :aria-label="`通道 ${c - 1}: 切换离群`" :aria-pressed="outs.has(c - 1)" :class="['colhead', { on: outs.has(c - 1) }]"
            @click="toggle(c - 1)" @keydown.enter="toggle(c - 1)" />
        </template>
        <g v-for="t in T" :key="t">
          <rect v-for="c in D" :key="c" :x="(c - 1) * CS" :y="28 + (t - 1) * CS" :width="CS - 1" :height="CS - 1"
            :style="{ fill: k === 0 ? valColor(q.K[t - 1][c - 1]) : heat(Math.abs(q.K[t - 1][c - 1] - q.hats[k - 1][t - 1][c - 1]) / q.eMax, 'var(--danger)') }"
            :class="['c', { hov: hover && hover.t === t - 1 && hover.c === c - 1, clk: k === 0 }]"
            @mouseenter="hover = { t: t - 1, c: c - 1 }" @click="k === 0 && toggle(c - 1)" />
        </g>
      </g>
    </svg>
    <p class="read mono">{{ readout }}</p>

    <template #stats>
      <div v-for="(n, i) in NAMES" :key="n" class="kv">
        <span>{{ n }} 相对误差</span><b :class="i === bestI ? 'good' : i === worstI ? 'bad' : ''">{{ q.errs[i].toFixed(4) }}</b>
      </div>
      <div class="kv"><span>scale 组数 (tensor / token / channel)</span><b>1 / {{ T }} / {{ D }}</b></div>
      <div class="kv"><span>per-token ÷ per-channel 误差</span><b>{{ (q.errs[1] / q.errs[2]).toFixed(1) }}×</b></div>
      <p class="lab-note">
        ★ 三种分组只差"在哪个维度上统计 min/max": 全体 / 每行 / 每列。量化器本身 (非对称 RTN: q = round((x − lo)/scale)) 完全一样。
        相对误差 = ‖K − K̂‖_F / ‖K‖_F。V 型是玩具假设: 无固定离群列, 各 token 幅度服从对数正态。
      </p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { argmax, clamp, heat, mulberry32, randn, range } from '@/utils/labmath.js'

const T = 16, D = 12, CS = 11
const NAMES = ['per-tensor', 'per-token', 'per-channel']
const PANELS = [{ id: 'k', label: 'K (点列 = 切换离群通道)' }, ...NAMES.map((n) => ({ id: n, label: `${n} 误差` }))]
const kind = ref('K'), bits = ref(4), mag = ref(8), seed = ref(1), hover = ref(null)
const outs = ref(new Set([3, 9]))
const toggle = (c) => { const s = new Set(outs.value); s.has(c) ? s.delete(c) : s.add(c); outs.value = s }
const setKind = (k) => { kind.value = k; outs.value = new Set(k === 'K' ? [3, 9] : []) }

// 同一个非对称 min/max 量化器, 输入是"共用一组 scale 的那些数"
const quant = (xs, b) => {
  const lo = Math.min(...xs), L = 2 ** b - 1, scale = Math.max((Math.max(...xs) - lo) / L, 1e-4)
  return { scale, hat: xs.map((v) => clamp(Math.round((v - lo) / scale), 0, L) * scale + lo) }
}

const q = computed(() => {
  const r = mulberry32(seed.value * 53 + 11)
  const base = range(T).map(() => range(D).map(() => randn(r)))
  const rowS = range(T).map(() => Math.exp(randn(r) * 0.8)), sign = range(D).map(() => (r() < 0.5 ? -1 : 1))
  // K 型: 离群通道 = 每个 token 在这一维都是 ±mag 附近的大值; V 型: 每个 token 整行乘一个幅度
  const K = base.map((row, t) => row.map((v, c) => (kind.value === 'V' ? v * rowS[t] : v)
    + (outs.value.has(c) ? mag.value * sign[c] * (1 + 0.15 * base[t][(c + 1) % D]) : 0)))
  const all = quant(K.flat(), bits.value)
  const rows = K.map((row) => quant(row, bits.value))
  const cols = range(D).map((c) => quant(K.map((row) => row[c]), bits.value))
  const hats = [
    range(T).map((t) => all.hat.slice(t * D, (t + 1) * D)),   // axis=None
    rows.map((x) => x.hat),                                    // ★ axis=1: 每个 token 一组
    range(T).map((t) => cols.map((x) => x.hat[t])),            // ★ axis=0: 每个通道一组 (KIVI 对 K 的选择)
  ]
  const kf = K.flat(), hf = hats.map((H) => H.flat()), nk = Math.hypot(...kf)
  const errs = hf.map((h) => Math.hypot(...kf.map((v, i) => v - h[i])) / nk)
  const eMax = Math.max(1e-9, ...hf.flatMap((h) => h.map((v, i) => Math.abs(v - kf[i]))))
  const scales = (t, c) => [all.scale, rows[t].scale, cols[c].scale]
  return { K, hats, errs, eMax, scales, vMax: Math.max(...kf.map(Math.abs)) }
})
const bestI = computed(() => argmax(q.value.errs.map((e) => -e))), worstI = computed(() => argmax(q.value.errs))
const valColor = (v) => heat(Math.abs(v) / q.value.vMax, v >= 0 ? 'var(--accent)' : 'var(--right)')
const readout = computed(() => {
  if (!hover.value) return '悬停一个格子: 原值 → 三种分组下的反量化值 (以及它所在组的 scale)。scale 越大, 格点越粗。'
  const { t, c } = hover.value, s = q.value.scales(t, c)
  return `token ${t}, 通道 ${c}: 原值 ${q.value.K[t][c].toFixed(2)} → ` + NAMES.map((n, i) => `${n} ${q.value.hats[i][t][c].toFixed(2)} (scale ${s[i].toFixed(3)})`).join(' · ')
})
</script>

<style scoped>
/* 窄屏: 保住可读的最小宽度, 由 .lab-viz 横向滚动; 只有 .draggable 把手拦截触摸 */
svg { min-width: 540px; touch-action: pan-x pan-y; }
.cap { font-size: 9.5px; fill: var(--text-muted); }
.cap.best { fill: var(--left); font-weight: 600; }
.cap.worst { fill: var(--danger); }
.c { stroke: var(--border); stroke-width: 0.5; }
.c.clk { cursor: pointer; }
.c.hov { stroke: var(--text); stroke-width: 1.5; }
.colhead { fill: var(--bg-elev); stroke: var(--border-strong); stroke-width: 0.5; cursor: pointer; outline: none; }
.colhead.on { fill: var(--warn); }
.colhead:hover, .colhead:focus-visible { stroke: var(--accent); stroke-width: 1.5; }
.read { font-size: 11px; color: var(--text-muted); margin-top: 8px; line-height: 1.6; min-height: 3.2em; }
</style>
