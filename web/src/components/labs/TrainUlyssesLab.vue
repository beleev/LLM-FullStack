<!--
  Ulysses 序列并行 (对应 llm_train/m16_ulysses_sequence_parallel)。
  只讲一件事: 一次 all-to-all 把 "按序列切" 换成 "按头切", 每卡看到完整序列但只算 H/P 个头; 通信随 P 下降, 但 P 不能超过头数。
-->
<template>
  <LabFrame
    title="Ulysses — all-to-all 把「切序列」换成「切头」"
    sub="格子 = 激活张量的一块 (行 = 第几段序列, 列 = 第几个注意力头), 颜色 = 此刻在哪张卡上。单步播放看两次 all-to-all 怎么换切法; 悬停一个格子看它走哪条链路, 点右侧图例只看某一张卡。"
    module="llm_train/m16"
    run="python -m llm_train.m16_ulysses_sequence_parallel.demo"
    :challenge="{
      ask: 'H = 8 个头, 把 P 从 4 拖到 8、再拖到 6, 分别发生什么? P = 8 时 Ring 的通信量是 Ulysses 的几倍?',
      answer: 'P=8 时每卡只分到 1 个头, 已经到顶; P=6 时 8 个头分不匀, Ulysses 直接不可用 (GQA 模型 KV 头更少, 限制更紧)。通信量上, Ulysses 每卡发 4(P−1)/P² 份完整张量, 随 P 增大而下降; Ring 发 2(P−1)/P 份, 趋近于 2 不再下降 —— P=8 时是 1.75 vs 0.44, Ring 是 4 倍。但 all-to-all 要求任意两卡之间都有带宽, 适合机内 NVLink; Ring 只和邻居说话, 适合跨机。所以实战常把两者叠起来: 机内 Ulysses × 机间 Ring。',
    }"
  >
    <template #controls>
      <LabSlider v-model="P" label="序列并行度 P" :min="1" :max="8" />
      <LabSlider v-model="Hn" label="注意力头数 H" :min="2" :max="16" />
      <StepPlayer :stepper="stepper" :label="phases[phase].short" />
    </template>

    <p class="phase" :class="{ bad: !valid }">{{ valid ? phases[phase].text : `✗ ${Hn} 个头没法平均分给 ${P} 张卡 —— Ulysses 要求 H 能被 P 整除 (Ring Attention 没有这个限制)。` }}</p>
    <div class="cells grid" :style="{ gridTemplateColumns: `58px repeat(${Hn}, minmax(20px, 1fr))` }">
      <span />
      <span v-for="h in Hn" :key="'h' + h" class="head mono">h{{ h - 1 }}</span>
      <template v-for="s in P" :key="s">
        <span class="head mono">序列段 {{ s - 1 }}</span>
        <span
          v-for="h in Hn" :key="h" class="cell"
          :class="{ dim: focus >= 0 && owner(s - 1, h - 1) !== focus, moved: valid && moving && s - 1 !== headRank(h - 1), link: sameLink(s - 1, h - 1) }"
          :style="{ background: valid || !headPhase ? heat(0.55, COLORS[owner(s - 1, h - 1)]) : 'transparent' }"
          :title="`序列段 ${s - 1} · 头 ${h - 1}: 此刻在卡 ${owner(s - 1, h - 1)}`"
          @mouseenter="hover = { s: s - 1, h: h - 1 }" @mouseleave="hover = null"
        >{{ valid || !headPhase ? owner(s - 1, h - 1) : '?' }}</span>
      </template>
    </div>
    <div class="row legend">
      <button v-for="r in P" :key="r" type="button" :class="{ active: focus === r - 1 }" @click="focus = focus === r - 1 ? -1 : r - 1">
        <i :style="{ background: COLORS[r - 1] }" />卡 {{ r - 1 }}
      </button>
      <span v-if="hover && valid" class="hint mono">段{{ hover.s }}·头{{ hover.h }}: 卡{{ hover.s }} ⇄ 卡{{ headRank(hover.h) }}{{ hover.s === headRank(hover.h) ? ' (不用动)' : '' }}</span>
    </div>

    <svg :viewBox="`0 0 ${W} ${CH}`" role="img" aria-label="每卡通信量随并行度的变化" class="chart">
      <line :x1="X0" :x2="W - 8" :y1="cy(0)" :y2="cy(0)" class="axis" />
      <line :x1="X0" :x2="W - 8" :y1="cy(2)" :y2="cy(2)" class="gridl" />
      <text :x="X0 - 4" :y="cy(2) + 3" class="tick" text-anchor="end">2×</text>
      <text :x="X0 - 4" :y="cy(0) + 3" class="tick" text-anchor="end">0</text>
      <text v-for="p in 8" :key="p" :x="cx(p)" :y="CH - 4" class="tick" text-anchor="middle" :class="{ cur: p === P }">P={{ p }}</text>
      <polyline :points="range(8).map((i) => `${cx(i + 1)},${cy(ringVol(i + 1))}`).join(' ')" class="ring" />
      <polyline :points="range(8).map((i) => `${cx(i + 1)},${cy(ulyVol(i + 1))}`).join(' ')" class="uly" />
      <circle v-for="p in 8" :key="'u' + p" :cx="cx(p)" :cy="cy(ulyVol(p))" r="4" :class="['uly-dot', { no: Hn % p !== 0 }]" />
      <circle :cx="cx(P)" :cy="cy(ringVol(P))" r="5" class="ring-dot" />
      <text :x="X0 + 150" :y="12" class="lab ring-t">Ring 2(P−1)/P</text>
      <text :x="X0 + 150" :y="25" class="lab uly-t">Ulysses 4(P−1)/P² (空心 = H 除不尽, 不可用)</text>
    </svg>

    <template #stats>
      <div class="kv"><span>每卡负责的头</span><b :class="valid ? '' : 'bad'">{{ valid ? Hn / P : '除不尽' }}</b></div>
      <div class="kv"><span>Ulysses 每卡发送</span><b :class="valid ? 'good' : 'bad'">{{ valid ? ulyVol(P).toFixed(2) + '×' : '不可用' }}</b></div>
      <div class="kv"><span>Ring 每卡发送</span><b>{{ ringVol(P).toFixed(2) }}×</b></div>
      <div class="kv"><span>Ring / Ulysses</span><b>{{ valid && P > 1 ? (ringVol(P) / ulyVol(P)).toFixed(1) + ' 倍' : '—' }}</b></div>
      <p class="lab-note">
        单位 = 一份完整的 [T, H, d] 张量。★ Ulysses 做 4 次 all-to-all (Q、K、V 各一次 + 输出一次), 每次每卡只发自己那 1/P 里不属于自己的 (P−1)/P;
        Ring 要把 K、V 两份 1/P 的块绕环传 P−1 轮。
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
import { heat, range } from '@/utils/labmath.js'

const COLORS = ['var(--accent)', 'var(--left)', 'var(--eye)', 'var(--right)', 'var(--warn)', 'var(--danger)', 'var(--code-fn)', 'var(--text-dim)']
const P = ref(4), Hn = ref(8), focus = ref(-1), hover = ref(null)
const valid = computed(() => Hn.value % P.value === 0)
watch([P, Hn], () => { hover.value = null; focus.value = -1 }) // 网格形状变了, 旧的悬停/聚焦下标作废

const phases = [
  { short: '① 按序列切', text: '① 注意力之外: 卡 r 持有第 r 段序列的全部头 [T/P, H, d]。LayerNorm / MLP 都逐 token 算, 不需要通信。' },
  { short: '② all-to-all', text: '② 第一次 all-to-all (Q/K/V 各一次): 卡 s 把自己的第 j 组头发给卡 j。带边框的格子 = 正在换卡。' },
  { short: '③ 本地注意力', text: '③ 现在卡 j 持有完整序列的第 j 组头 [T, H/P, d]: 直接跑普通 FlashAttention, 因果 mask 天然均衡, 不需要 zigzag。' },
  { short: '④ all-to-all 换回', text: '④ 第二次 all-to-all: 把输出按序列段发回去, 恢复 [T/P, H, d], 接着算 MLP。' },
]
const stepper = useStepper(() => phases.length, { interval: 1600 })
const phase = computed(() => stepper.step.value)
const headPhase = computed(() => phase.value === 1 || phase.value === 2)
const moving = computed(() => phase.value === 1 || phase.value === 3)

// ★ seq_to_head: 头 h 属于第 ⌊h / (H/P)⌋ 组, all-to-all 之后这一列整个搬到那张卡
const headRank = (h) => Math.floor(h / (Hn.value / P.value))
const owner = (s, h) => (headPhase.value && valid.value ? headRank(h) : s)
const sameLink = (s, h) => valid.value && hover.value && hover.value.s === s && headRank(hover.value.h) === headRank(h)

// 每卡发送量, 单位 = 一份完整 [T,H,d]; 与 Python 里 assert 过的公式相同
const ulyVol = (p) => (4 * (p - 1)) / (p * p)
const ringVol = (p) => (2 * (p - 1)) / p

const W = 520, CH = 130, X0 = 30
const cx = (p) => X0 + 24 + ((p - 1) / 7) * (W - X0 - 60)
const cy = (v) => 14 + (1 - v / 2.2) * (CH - 34)
</script>

<style scoped>
.phase { font-size: 12px; color: var(--text-muted); line-height: 1.6; min-height: 3.2em; margin-bottom: 8px; }
.phase.bad { color: var(--danger); }
.head { font-size: 10px; color: var(--text-dim); align-self: center; text-align: center; }
.grid .cell { min-width: 20px; height: 26px; color: var(--text); }
.cell.moved { outline: 2px solid var(--warn); outline-offset: -2px; }
.cell.link { outline: 2px solid var(--text); outline-offset: -2px; }
.legend { margin-top: 10px; display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.legend button { font-size: 11px; min-height: 26px; padding: 2px 8px; }
.legend i { display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin-right: 5px; }
.hint { font-size: 11px; color: var(--text-muted); }
.chart { margin-top: 12px; }
.axis { stroke: var(--border-strong); }
.gridl { stroke: var(--border); stroke-dasharray: 2 4; }
.tick { font-size: 9px; fill: var(--text-dim); font-family: "SF Mono", Menlo, monospace; }
.tick.cur { fill: var(--accent); font-weight: 700; }
.ring { fill: none; stroke: var(--eye); stroke-width: 2; }
.uly { fill: none; stroke: var(--left); stroke-width: 2; }
.uly-dot { fill: var(--left); }
.uly-dot.no { fill: var(--bg-card); stroke: var(--left); stroke-width: 1.5; }
.ring-dot { fill: var(--eye); }
.lab { font-size: 10px; }
.ring-t { fill: var(--eye); }
.uly-t { fill: var(--left); }
</style>
