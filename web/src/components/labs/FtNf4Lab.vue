<!--
  NF4 量化实验台 (对应 llm_finetune/methods/qlora.py: nf4_quantize)。
  只讲一件事: 权重近似正态分布, 所以 16 个码点应该按正态分位数摆 (中间密、两边疏), 而不是等间距;
  每个 block 用 absmax 缩放, 所以一个 outlier 会把整个 block 的码点一起"拉稀"。
  量化流程与 Python 完全一致: 分 block → 除以 absmax → 最近邻查码本 → 乘回 scale。
-->
<template>
  <LabFrame
    title="NF4 vs INT4 — 16 个码点该摆在哪"
    sub="灰色直方图是 512 个正态权重。中间一排小圆点是第 0 个 block 的权重, 上下两排刻度是这个 block 的 16 个 NF4 / INT4 码点 (刻度越高 = 被用得越多)。把红色的 outlier 往右拖, 看码点怎么被一起拉开。"
    module="llm_finetune/methods/qlora.py"
    run="python -m llm_finetune.run_finetune.qlora.train_qlora"
    :challenge="{
      ask: '先把 outlier 拖回 0: NF4 和 INT4 的「有效比特」各是多少? 再把 outlier 拖到 8σ、block 调到 256, 然后只把 block 缩到 16 —— RMSE 为什么大幅回落, 代价是什么?',
      answer: '没有 outlier、B=64 时 NF4 的 16 个码几乎被等概率使用, 熵约 3.9 bit, 4 bit 的容量基本吃满; INT4 等间距, 两端的码很少有人用, 熵只有 3.4–3.5 bit, RMSE 也高出约 20%。这就是「分位数码本适合正态权重」的含义: 权重密的地方码点也密。outlier 的伤害范围 = 它所在的那一个 block: absmax 被它撑大, 同 block 其余权重全挤进中间几个码点 (B=256、8σ 时 NF4 的 RMSE 从约 0.10 涨到 0.16, INT4 涨到 0.26)。block 越小被连累的权重越少, 但每个 block 要多存一个 FP32 scale: 4 + 32/B bit/参数, B=64 是 4.5 bit, B=16 已经 6 bit。QLoRA 的双重量化就是把这些 scale 再压到 8 bit 来抵消这部分开销。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: focus === 'nf4' }" @click="focus = 'nf4'">连线看 NF4</button>
        <button type="button" :class="{ active: focus === 'int4' }" @click="focus = 'int4'">连线看 INT4</button>
        <button type="button" @click="seed++">换一组权重</button>
      </div>
      <LabSlider v-model="out" label="outlier 大小" :min="0" :max="8" :step="0.1" unit="σ" :format="(v) => v.toFixed(1)" />
      <LabSlider v-model="logB" label="block 大小 B" :min="4" :max="8" :format="(v) => String(2 ** v)" />
    </template>

    <svg ref="svgEl" viewBox="0 0 640 300" role="img" aria-label="权重直方图与两套量化码点">
      <!-- 全体权重直方图 -->
      <rect v-for="(h, i) in hist" :key="'h' + i" :x="sx(-XMAX) + i * binW + 0.5" :y="92 - h * 80" :width="binW - 1" :height="h * 80" fill="var(--border-strong)" />
      <text x="8" y="16" class="t">全部 {{ N }} 个权重的分布 (单位 σ)</text>

      <!-- 两排码点 -->
      <g v-for="row in rows" :key="row.id" :opacity="focus === row.id ? 1 : 0.45">
        <text x="8" :y="row.y + (row.up ? -38 : 48)" class="t" :fill="row.color">{{ row.label }}</text>
        <line
          v-for="(lv, i) in row.levels" :key="i"
          :x1="sx(lv.v)" :x2="sx(lv.v)" :y1="row.y" :y2="row.y + (row.up ? -1 : 1) * (5 + 28 * lv.n / row.maxN)"
          :stroke="lv.n ? row.color : 'var(--text-dim)'" :stroke-width="lv.n ? 2.5 : 1"
        />
      </g>
      <!-- block 0 的权重 → 最近码点 -->
      <line
        v-for="(p, i) in blockPts" :key="'c' + i"
        :x1="sx(p.w)" :x2="sx(p.q)" y1="190" :y2="focusRow.y" :stroke="focusRow.color" stroke-width="0.6" opacity="0.55"
      />
      <line :x1="sx(-XMAX)" :x2="sx(XMAX)" y1="190" y2="190" stroke="var(--border)" />
      <circle v-for="(p, i) in blockPts" :key="'p' + i" :cx="sx(p.w)" cy="190" r="2.2" fill="var(--text-muted)" />
      <g v-for="t in [-8, -4, 0, 4, 8]" :key="'x' + t">
        <text :x="sx(t)" y="292" text-anchor="middle" class="t">{{ t }}σ</text>
      </g>
      <!-- ★ 可拖的 outlier: 它决定 block 0 的 absmax, 也就决定了 16 个码点铺多宽 -->
      <circle
        class="draggable" :cx="sx(outVal)" cy="190" r="9" fill="var(--danger)" stroke="var(--bg-card)" stroke-width="2"
        tabindex="0" role="slider" aria-label="outlier 大小" :aria-valuenow="out" aria-valuemin="0" aria-valuemax="8"
        @pointerdown="start($event, { svg: svgEl, onMove: ({ x }) => (out = Math.round(clamp((x - sx(0)) / SC, 0, 8) * 10) / 10) })"
        @keydown.right.prevent="out = clamp(out + 0.5, 0, 8)" @keydown.left.prevent="out = clamp(out - 0.5, 0, 8)"
      />
      <text :x="sx(outVal)" y="172" text-anchor="middle" class="t" fill="var(--danger)">{{ out < 0.05 ? '← 拖我' : 'outlier' }}</text>
    </svg>

    <template #stats>
      <div class="kv"><span>RMSE · NF4</span><b :class="res.nf4.rmse <= res.int4.rmse ? 'good' : 'bad'">{{ res.nf4.rmse.toFixed(4) }}</b></div>
      <div class="kv"><span>RMSE · INT4</span><b :class="res.int4.rmse < res.nf4.rmse ? 'good' : 'bad'">{{ res.int4.rmse.toFixed(4) }}</b></div>
      <div class="kv"><span>有效比特 NF4 / INT4</span><b>{{ res.nf4.bits.toFixed(2) }} / {{ res.int4.bits.toFixed(2) }}</b></div>
      <div class="kv"><span>block 0 用到的码点</span><b>{{ rows[0].used }} / {{ rows[1].used }}</b></div>
      <div class="kv"><span>存储 4 + 32/B</span><b>{{ (4 + 32 / B).toFixed(2) }} bit</b></div>
      <p class="lab-note">
        有效比特 = 16 个码的使用频率的熵。分位数码本让每个码被用到的概率 ≈ 1/16, 熵 → 4 bit;
        等间距码本的两端几乎没有权重落进去, 那几个码等于白占。
      </p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { useDrag } from '@/composables/useDrag.js'
import { clamp, entropy, mulberry32, randn, range } from '@/utils/labmath.js'

// 与 qlora.py 的 NF4_CODEBOOK 相同 (bitsandbytes 常数): 标准正态的 16 个等概率分位点, 归一化到 [-1, 1]
const NF4 = [-1, -0.6961928, -0.5250731, -0.3949175, -0.2844414, -0.1847734, -0.0910500, 0,
  0.0795803, 0.1609302, 0.2461123, 0.3379152, 0.4407098, 0.5626170, 0.7229568, 1]
const N = 512, XMAX = 8.6, SC = 300 / XMAX, OUT_IDX = 5
const sx = (v) => 320 + v * SC

const out = ref(0), logB = ref(6), seed = ref(1), focus = ref('nf4')
const svgEl = ref(null)
const { start } = useDrag()
const B = computed(() => 2 ** logB.value)

const weights = computed(() => {
  const rand = mulberry32(seed.value * 131)
  const w = range(N).map(() => randn(rand))
  if (out.value >= 0.05) w[OUT_IDX] = out.value
  return w
})
const outVal = computed(() => weights.value[OUT_IDX])

// 两种量化器: 输入已除以 absmax 的 x∈[-1,1], 返回 [码号, 反量化值]
const qNf4 = (x) => { let b = 0; NF4.forEach((c, i) => { if (Math.abs(c - x) < Math.abs(NF4[b] - x)) b = i }); return [b, NF4[b]] }
const qInt4 = (x) => { const q = clamp(Math.round(x * 7), -8, 7); return [q + 8, q / 7] }   // 对称 absmax INT4: 步长 absmax/7

const quantize = (w, qf) => {
  const deq = [], usage = Array(16).fill(0)
  for (let s = 0; s < w.length; s += B.value) {
    const blk = w.slice(s, s + B.value)
    const scale = Math.max(...blk.map(Math.abs), 1e-12)          // ★ 每个 block 一个 absmax scale
    for (const v of blk) { const [code, val] = qf(v / scale); usage[code]++; deq.push(val * scale) }
  }
  const rmse = Math.sqrt(deq.reduce((s, d, i) => s + (d - w[i]) ** 2, 0) / w.length)
  return { deq, rmse, bits: entropy(usage.map((u) => u / w.length)) / Math.LN2 }
}
const res = computed(() => ({ nf4: quantize(weights.value, qNf4), int4: quantize(weights.value, qInt4) }))

const block0 = computed(() => weights.value.slice(0, B.value))
const scale0 = computed(() => Math.max(...block0.value.map(Math.abs)))
const mkRow = (id, label, color, y, up, codes, deq) => {
  const levels = codes.map((c) => ({ v: c * scale0.value, n: 0 }))
  block0.value.forEach((_, i) => { const lv = levels.find((l) => Math.abs(l.v - deq[i]) < 1e-9); if (lv) lv.n++ })
  return { id, label, color, y, up, levels, used: levels.filter((l) => l.n).length, maxN: Math.max(...levels.map((l) => l.n), 1) }
}
const rows = computed(() => [
  mkRow('nf4', 'NF4 码点 (正态分位数)', 'var(--accent)', 150, true, NF4, res.value.nf4.deq),
  mkRow('int4', 'INT4 码点 (等间距)', 'var(--eye)', 230, false, range(16).map((i) => (i - 8) / 7), res.value.int4.deq),
])
const focusRow = computed(() => rows.value.find((r) => r.id === focus.value))
const blockPts = computed(() => block0.value.map((w, i) => ({ w, q: res.value[focus.value].deq[i] })).filter((_, i) => i !== OUT_IDX || out.value < 0.05))

const BINS = 86, binW = (2 * XMAX * SC) / BINS
const hist = computed(() => {
  const h = Array(BINS).fill(0)
  weights.value.forEach((w) => { h[clamp(Math.floor((w + XMAX) / (2 * XMAX) * BINS), 0, BINS - 1)]++ })
  const m = Math.max(...h)
  return h.map((c) => c / m)
})
</script>

<style scoped>
.t { font-size: 11px; fill: var(--text-muted); font-family: "SF Mono", Menlo, monospace; }
</style>
