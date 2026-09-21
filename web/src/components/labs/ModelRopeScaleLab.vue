<!--
  RoPE 长度外推实验台 (对应 llm_models/layers/core/position_encoding.py:scaled_inv_freq)。
  只讲一件事: 低频维度在训练长度内连一圈都没转完, 位置一超长它们就看到 “没见过的角度”;
  PI / NTK-aware / YaRN 是三种把角度压回训练范围的办法, 代价各不相同。公式与 Python 逐项一致。
-->
<template>
  <LabFrame
    title="YaRN / NTK — 每个频率该不该被压缩"
    sub="每根柱子是 RoPE 的一个频率 (d_head=64 → 32 对维度, 与 infer_rope_scaling 同设定), 高度 = 波长 (转一圈要多少个 token, 对数轴)。虚线 = 训练长度 2048。拖底部的位置标尺把 token 放到 2048 之外, 红柱 = 这个维度此刻的旋转角训练时从没出现过。点一根柱子, 右边表盘显示它见过的角度范围。"
    module="llm_models/layers/core/position_encoding.py"
    run="python -m llm_models.run_models.foundation.rope_scaling.infer_rope_scaling"
    :challenge="{
      ask: '扩展倍数拖到 8, 位置 m 拖到目标长度 16384。依次切换四种方案, 先猜: 哪种没有红柱但 “相邻 token 分辨率” 最差? 哪种分辨率满分却还剩红柱?',
      answer: 'PI (位置内插) 把所有频率一律 ÷8: 角度全部回到训练范围, 没有红柱。但最高频也被 ÷8, 相邻两个 token 的转角差只剩 1/8。模型于是分不清 “紧挨着” 和 “隔几个”, 必须微调才能恢复。NTK-aware 改的是 base: 最高频不动 (分辨率 100%), 最低频恰好 ÷8, 但中间那些 “没转满一圈” 的维度只被压了 8^(2i/(d−2)) < 8 倍, 仍然越界 —— 还剩红柱。YaRN 按 “训练期转了几圈” 分段: 转够 32 圈的高频原样外推, 不足 1 圈的低频按 PI ÷s, 中间线性过渡, 两头都保住。再乘 mscale = 0.1·ln s + 1, 补偿长上下文下 softmax 变平。',
    }"
  >
    <template #controls>
      <div class="row">
        <button v-for="m in MODES" :key="m.id" type="button" :class="{ active: mode === m.id }" @click="mode = m.id">{{ m.label }}</button>
      </div>
      <LabSlider v-model="s" label="扩展倍数 s" :min="1" :max="16" unit="×" />
      <LabSlider v-model="pos" label="token 位置 m" :min="0" :max="PMAX" :step="64" />
    </template>

    <div class="wrap">
      <svg ref="svg" viewBox="0 0 520 330" class="chart" role="group" aria-label="各频率波长">
        <template v-for="e in [1, 2, 3, 4, 5, 6]" :key="e">
          <line :x1="X0" :x2="X1" :y1="py(e)" :y2="py(e)" class="grid" /><text :x="X0 - 5" :y="py(e) + 3" class="yl">1e{{ e }}</text>
        </template>
        <g v-for="f in freqs" :key="f.i" class="barg" tabindex="0" role="button" :aria-label="`频率 ${f.i}`" :aria-pressed="sel === f.i" @click="sel = f.i" @keydown.enter="sel = f.i">
          <rect :x="bx(f.i)" :y="py(f.lw)" :width="BW" :height="Y1 - py(f.lw)" class="bar" :class="{ ood: f.ood, sel: sel === f.i }" />
          <line :x1="bx(f.i) - 2" :x2="bx(f.i) + BW + 2" :y1="py(f.lw0)" :y2="py(f.lw0)" class="orig" />
          <text v-if="f.i % 4 === 0 || f.i === NF - 1" :x="bx(f.i) + BW / 2" :y="Y1 + 12" class="xl">{{ f.i }}</text>
        </g>
        <line :x1="X0" :x2="X1" :y1="py(Math.log10(L))" :y2="py(Math.log10(L))" class="train" />
        <text :x="X0 + 4" :y="py(Math.log10(L)) - 4" class="tl">训练长度 L = 2048</text>
        <line v-if="s > 1" :x1="X0" :x2="X1" :y1="py(Math.log10(L * s))" :y2="py(Math.log10(L * s))" class="target" />
        <text v-if="s > 1" :x="X0 + 4" :y="py(Math.log10(L * s)) - 4" class="tg">目标长度 {{ L * s }}</text>
        <text :x="X0" :y="Y1 + 26" class="cap">频率编号 i (左 = 高频管局部顺序, 右 = 低频管远距离) · 横线 = 原始波长</text>
        <!-- 位置标尺: 直接拖 -->
        <line :x1="X0" :x2="X1" y1="306" y2="306" class="ruler" />
        <rect :x="X0" y="302" :width="px(L) - X0" height="8" class="seen" />
        <text :x="px(L)" y="326" class="xl">2048</text><text :x="px(L * s)" y="326" class="xl" v-if="s > 1">{{ L * s }}</text>
        <g class="draggable" tabindex="0" role="slider" aria-label="拖动 token 位置" :aria-valuenow="pos"
          @pointerdown="start($event, { svg, onMove })" @keydown.left="pos = clamp(pos - 256, 0, PMAX)" @keydown.right="pos = clamp(pos + 256, 0, PMAX)">
          <rect :x="px(pos) - 14" y="290" width="28" height="32" fill="transparent" />
          <circle :cx="px(pos)" cy="306" r="8" class="handle" :class="{ bad: nOod > 0 }" />
        </g>
      </svg>

      <svg viewBox="-70 -78 140 160" class="dial" role="img" :aria-label="`频率 ${sel} 的角度表盘`">
        <circle r="50" class="ring" />
        <path :d="arc" class="seenarc" />
        <line x1="0" y1="0" :x2="50 * Math.cos(ang)" :y2="-50 * Math.sin(ang)" class="hand" :class="{ bad: cur.ood }" />
        <circle :cx="50 * Math.cos(ang)" :cy="-50 * Math.sin(ang)" r="5" class="tip" :class="{ bad: cur.ood }" />
        <text y="-64" class="dt">频率 {{ sel }} · 训练期转 {{ cur.turns < 10 ? cur.turns.toFixed(2) : cur.turns.toFixed(0) }} 圈</text>
        <text y="76" class="dt">{{ cur.ood ? '当前角度从未见过' : '当前角度在见过的范围内' }}</text>
      </svg>
    </div>

    <template #stats>
      <div class="kv"><span>位置 m={{ pos }} 处越界维度</span><b :class="nOod ? 'bad' : 'good'">{{ nOod }} / {{ NF }}</b></div>
      <div class="kv"><span>相邻 token 分辨率 θ′₀/θ₀</span><b :class="hi < 0.6 ? 'bad' : 'good'">{{ (hi * 100).toFixed(0) }}%</b></div>
      <div class="kv"><span>完全没被改动的频率数</span><b>{{ untouched }} / {{ NF }}</b></div>
      <div class="kv"><span>最低频被压缩倍数</span><b>{{ lo.toFixed(2) }}×</b></div>
      <div class="kv"><span>温度补偿 mscale</span><b>{{ mscale.toFixed(4) }}</b></div>
      <p class="lab-note">{{ MODES.find((m) => m.id === mode).note }}</p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { useDrag } from '@/composables/useDrag.js'
import { clamp, range } from '@/utils/labmath.js'

const D = 64, NF = D / 2, BASE = 10000, L = 2048, PMAX = 32768, BETA_FAST = 32, BETA_SLOW = 1 // 与 infer_rope_scaling 同设定
const MODES = [
  { id: 'none', label: '不缩放 (直接外推)', note: '直接外推: 高频维度转过无数圈, 什么角度都见过, 没事。低频维度训练时只走过一小段弧, 超长后走到弧外, 注意力分数失真, 困惑度爆炸。' },
  { id: 'pi', label: 'PI 位置内插', note: 'PI: θ′ = θ/s, 等价于把位置 m 压成 m/s。所有角度回到训练范围, 但高频也被压, 局部顺序信息被抹糊。' },
  { id: 'ntk', label: 'NTK-aware', note: "NTK-aware: base′ = base·s^(d/(d−2))。θ₀ 不变、θ_last 恰好 ÷s, 中间按指数过渡 —— 免微调就能用, 但中段低频压得不够。" },
  { id: 'yarn', label: 'YaRN', note: 'YaRN (NTK-by-parts): 按训练期圈数 r 分段, r>32 不动, r<1 按 PI ÷s, 中间线性混合; 另乘 mscale 调 softmax 温度。' },
]
const mode = ref('none'), s = ref(4), pos = ref(8192), sel = ref(26)
const svg = ref(null)
const { start } = useDrag()

// ★ 与 scaled_inv_freq 同式
const theta0 = range(D / 2).map((i) => BASE ** (-(2 * i) / D))
const theta = computed(() => theta0.map((t, i) => {
  if (mode.value === 'none' || s.value === 1) return t
  if (mode.value === 'pi') return t / s.value
  if (mode.value === 'ntk') return (BASE * s.value ** (D / (D - 2))) ** (-(2 * i) / D)
  const r = (L * t) / (2 * Math.PI)
  const g = clamp((r - BETA_SLOW) / (BETA_FAST - BETA_SLOW), 0, 1)
  return ((1 - g) * t) / s.value + g * t
}))
const mscale = computed(() => (mode.value === 'yarn' && s.value > 1 ? 0.1 * Math.log(s.value) + 1 : 1))

const freqs = computed(() => theta.value.map((t, i) => {
  const seen = L * theta0[i]                       // 训练期转过的最大弧度
  return {
    i, lw: Math.log10((2 * Math.PI) / t), lw0: Math.log10((2 * Math.PI) / theta0[i]),
    turns: seen / (2 * Math.PI), seen,
    ood: seen < 2 * Math.PI && pos.value * t > seen * 1.0001, // 没转满一圈, 且当前角度超出见过的弧
  }
}))
const nOod = computed(() => freqs.value.filter((f) => f.ood).length)
const hi = computed(() => theta.value[0] / theta0[0])
const lo = computed(() => theta0[D / 2 - 1] / theta.value[D / 2 - 1])
const cur = computed(() => freqs.value[sel.value])
const ang = computed(() => (pos.value * theta.value[sel.value]) % (2 * Math.PI))
const arc = computed(() => {
  const a = Math.min(cur.value.seen, 2 * Math.PI - 1e-3)
  return `M 50 0 A 50 50 0 ${a > Math.PI ? 1 : 0} 0 ${(50 * Math.cos(a)).toFixed(2)} ${(-50 * Math.sin(a)).toFixed(2)}`
})

const X0 = 40, X1 = 510, Y0 = 12, Y1 = 250, BW = 10
const py = (lg) => Y1 - ((clamp(lg, 0.5, 6) - 0.5) / 5.5) * (Y1 - Y0)
const bx = (i) => X0 + 6 + i * ((X1 - X0 - 12) / NF)
// YaRN 下 γ=1 (训练期转够 32 圈) 的高频维度: 完全不动
const untouched = computed(() => theta.value.filter((t, i) => Math.abs(t / theta0[i] - 1) < 1e-9).length)
const px = (m) => X0 + (m / PMAX) * (X1 - X0)
const onMove = ({ x }) => { pos.value = clamp(Math.round((((x - X0) / (X1 - X0)) * PMAX) / 64) * 64, 0, PMAX) }
</script>

<style scoped>
.wrap { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; }
.chart { flex: 1 1 380px; min-width: 0; }
.dial { flex: 0 0 150px; width: 150px; }
.grid { stroke: var(--border); }
.yl { text-anchor: end; font-size: 9px; fill: var(--text-dim); }
.xl { text-anchor: middle; font-size: 9px; fill: var(--text-dim); }
.cap { font-size: 10px; fill: var(--text-dim); }
.barg { cursor: pointer; outline: none; }
.bar { fill: color-mix(in srgb, var(--left) 45%, transparent); stroke: var(--left); }
.bar.ood { fill: color-mix(in srgb, var(--danger) 45%, transparent); stroke: var(--danger); }
.bar.sel, .barg:focus-visible .bar { stroke: var(--accent); stroke-width: 2.5; }
.orig { stroke: var(--text-muted); stroke-width: 1.5; }
.train { stroke: var(--eye); stroke-dasharray: 5 4; }
.tl { text-anchor: start; font-size: 10px; fill: var(--eye); }
.target { stroke: var(--accent); stroke-dasharray: 2 3; }
.tg { text-anchor: start; font-size: 10px; fill: var(--accent); }
.ruler { stroke: var(--border-strong); stroke-width: 2; }
.seen { fill: color-mix(in srgb, var(--left) 50%, transparent); }
.handle { fill: var(--left); stroke: var(--bg); stroke-width: 2; }
.handle.bad { fill: var(--danger); }
.ring { fill: none; stroke: var(--border-strong); }
.seenarc { fill: none; stroke: var(--left); stroke-width: 7; stroke-linecap: round; }
.hand { stroke: var(--left); stroke-width: 2; }
.hand.bad, .tip.bad { stroke: var(--danger); fill: var(--danger); }
.tip { fill: var(--left); }
.dt { text-anchor: middle; font-size: 9px; fill: var(--text-muted); }
</style>
