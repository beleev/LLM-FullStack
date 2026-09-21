<!--
  ZeRO 显存账 (对应 llm_train/m05_zero_fsdp)。
  只讲一件事: 混合精度 Adam 每参数 16 字节 = 2 + 2 + 12, ZeRO-1/2/3 依次把 12 / 2 / 2 这三块除以 N。
-->
<template>
  <LabFrame
    title="ZeRO 显存账 — 2 + 2 + 12 字节, 先切哪一块?"
    sub="四根柱子是同一个模型在 DDP / ZeRO-1 / ZeRO-2 / ZeRO-3 下每张卡的常驻显存。上下拖动红色虚线设定单卡显存上限, 看哪一级开始放得下; 点柱子选中一级, 悬停色块看它是什么。"
    module="llm_train/m05"
    run="python -m llm_train.m05_zero_fsdp.demo"
    :challenge="{
      ask: '7B 模型、80 GB 的卡。卡数从 8 加到 64, ZeRO-1 的单卡显存能降到多少? 为什么再加卡也没用?',
      answer: 'ZeRO-1 只切 12Ψ 的优化器状态: 4Ψ + 12Ψ/N。7B 时 4Ψ = 28 GB 是切不掉的地板, N=8 是 38.5 GB, N=64 是 29.3 GB, N→∞ 也只到 28 GB。想继续降只能把梯度 (ZeRO-2) 和参数 (ZeRO-3/FSDP) 也切掉 —— 代价是 ZeRO-3 每层前向、反向各多一次 all-gather, 通信量约 1.5×。',
    }"
  >
    <template #controls>
      <LabSlider v-model="sizeIdx" label="模型参数量 Ψ" :min="0" :max="sizes.length - 1" :format="(i) => sizes[i] + 'B'" />
      <LabSlider v-model="logN" label="数据并行卡数 N" :min="0" :max="10" :format="(k) => 2 ** k" />
      <div class="row">
        <button v-for="(s, i) in stages" :key="s.name" type="button" :class="{ active: stage === i }" @click="stage = i">{{ s.name }}</button>
      </div>
    </template>

    <svg ref="svg" :viewBox="`0 0 ${W} ${H}`" role="img" aria-label="各 ZeRO 级别的单卡显存堆叠柱状图">
      <g v-for="t in [0, 50, 100, 150, 200]" :key="t">
        <line :x1="X0" :x2="W - 8" :y1="y(t)" :y2="y(t)" class="grid" />
        <text :x="X0 - 6" :y="y(t) + 4" class="tick" text-anchor="end">{{ t }}</text>
      </g>
      <text :x="4" :y="12" class="tick">GB / 卡</text>
      <g
        v-for="(b, i) in bars" :key="i" class="bar" :class="{ sel: stage === i }"
        tabindex="0" role="button" :aria-label="`选中 ${stages[i].name}`"
        @click="stage = i" @keydown.enter="stage = i"
      >
        <rect :x="bx(i) - 8" :y="8" :width="BW + 16" :height="H - 34" class="hit" />
        <rect
          v-for="seg in b.segs" :key="seg.key"
          :x="bx(i)" :y="y(seg.top)" :width="BW" :height="Math.max(0, y(seg.bot) - y(seg.top))"
          :fill="seg.color" :class="{ transient: seg.key === 'gather' }"
          @mouseenter="hover = { ...seg, stage: stages[i].name }" @mouseleave="hover = null"
        />
        <text :x="bx(i) + BW / 2" :y="Math.max(20, y(Math.min(b.total, YMAX)) - 6)" class="val" text-anchor="middle" :class="b.total <= limit ? 'fit' : 'oom'">
          {{ b.total > YMAX ? '↑ ' : '' }}{{ gb(b.total) }}
        </text>
        <text :x="bx(i) + BW / 2" :y="H - 8" class="lbl" text-anchor="middle">{{ stages[i].name }}</text>
      </g>
      <!-- ★ 可拖的显存上限线 -->
      <g class="draggable" tabindex="0" role="slider" aria-label="单卡显存上限" :aria-valuenow="limit"
        @pointerdown="start($event, { svg, onMove: ({ y: py }) => (limit = Math.round(clamp(yInv(py), 16, 192))) })"
        @keydown.up.prevent="limit = Math.min(192, limit + 4)" @keydown.down.prevent="limit = Math.max(16, limit - 4)">
        <rect :x="X0" :y="y(limit) - 9" :width="W - X0 - 8" height="18" fill="transparent" />
        <line :x1="X0" :x2="W - 8" :y1="y(limit)" :y2="y(limit)" class="limit" />
        <text :x="W - 10" :y="y(limit) - 5" class="limit-t" text-anchor="end">⇅ 单卡上限 {{ limit }} GB</text>
      </g>
    </svg>
    <p class="lab-note">
      <template v-if="hover">{{ hover.stage }} · {{ hover.label }}: {{ gb(hover.bot === undefined ? 0 : hover.top - hover.bot) }} —— {{ hover.desc }}</template>
      <template v-else>
        <i class="sw" style="background: var(--accent)" />fp16 参数 2Ψ
        <i class="sw" style="background: var(--eye)" />fp16 梯度 2Ψ
        <i class="sw" style="background: var(--right)" />fp32 master + Adam m + v = 12Ψ
        <i class="sw tr" />ZeRO-3 瞬时 all-gather 的一层 (按 {{ LAYERS }} 层估)
      </template>
    </p>

    <template #stats>
      <div class="kv"><span>{{ stages[stage].name }} 单卡常驻</span><b :class="cur.total <= limit ? 'good' : 'bad'">{{ gb(cur.total) }}</b></div>
      <div class="kv"><span>公式</span><b class="formula">{{ stages[stage].formula }}</b></div>
      <div class="kv"><span>放得进 {{ limit }} GB 的最低级别</span><b :class="firstFit < 0 ? 'bad' : 'good'">{{ firstFit < 0 ? '都放不下' : stages[firstFit].name }}</b></div>
      <div class="kv"><span>每步通信量 (相对 DDP)</span><b :class="stage === 3 ? 'bad' : ''">{{ stages[stage].comm }}</b></div>
      <p class="lab-note">
        这里只算模型状态, 激活另算 (见下面的重算实验台)。ZeRO-1 的地板是 4Ψ, ZeRO-2 是 2Ψ, 只有 ZeRO-3 才随 N 无限下降:
        用到哪层 gather 哪层, 算完立刻 free, 所以瞬时只多出一层的完整参数。
        ZeRO-2 不比 DDP 贵, 因为 all-reduce 本来就 = reduce-scatter + all-gather (ZeRO-1 若朴素地用 all-reduce + all-gather 实现会是 1.5×, DeepSpeed 改用 reduce-scatter 后持平)。
      </p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { useDrag } from '@/composables/useDrag.js'
import { clamp } from '@/utils/labmath.js'

const sizes = [1, 3, 7, 13, 34, 70, 175, 405]
const LAYERS = 32
const stages = [
  { name: 'DDP', formula: '16Ψ', comm: '1×' },
  { name: 'ZeRO-1', formula: '4Ψ + 12Ψ/N', comm: '1×' },
  { name: 'ZeRO-2', formula: '2Ψ + 14Ψ/N', comm: '1×' },
  { name: 'ZeRO-3', formula: '16Ψ/N', comm: '≈1.5×' },
]
const sizeIdx = ref(2), logN = ref(3), stage = ref(1), limit = ref(80), hover = ref(null), svg = ref(null)
const { start } = useDrag()

const W = 460, H = 300, X0 = 44, BW = 62, YMAX = 200
const y = (g) => 20 + (1 - Math.min(g, YMAX) / YMAX) * (H - 50)
const yInv = (py) => (1 - (py - 20) / (H - 50)) * YMAX
const bx = (i) => X0 + 22 + i * (BW + 38)
const gb = (g) => (g >= 100 ? g.toFixed(0) : g.toFixed(1)) + ' GB'

// ★ 全部显存账就这一张表: 每块多少字节/参数, 从第几级开始除以 N。1 GB = 1e9 字节, 所以 1B 参数 × 1 字节 = 1 GB
const parts = [
  { key: 'p', label: 'fp16 参数', bytes: 2, shardFrom: 3, color: 'var(--accent)', desc: '前向反向都要用, 只有 ZeRO-3/FSDP 才切它' },
  { key: 'g', label: 'fp16 梯度', bytes: 2, shardFrom: 2, color: 'var(--eye)', desc: 'ZeRO-2 起用 reduce-scatter 同步, 每卡只留自己那片' },
  { key: 'o', label: 'fp32 master + m + v', bytes: 12, shardFrom: 1, color: 'var(--right)', desc: 'Adam 逐元素更新, 每卡只更新自己那 1/N, 结果逐位相同' },
]
const bars = computed(() => {
  const psi = sizes[sizeIdx.value], N = 2 ** logN.value
  return stages.map((_, s) => {
    let acc = 0
    const segs = parts.map((p) => {
      const size = (p.bytes * psi) / (s >= p.shardFrom ? N : 1)
      const seg = { ...p, bot: acc, top: acc + size }
      acc += size
      return seg
    })
    const total = acc
    if (s === 3 && N > 1) {
      const extra = (2 * psi) / LAYERS * (1 - 1 / N) // 一层完整 fp16 参数里不属于自己的那部分
      segs.push({ key: 'gather', label: '瞬时 all-gather', color: 'var(--accent)', bot: acc, top: acc + extra, desc: '用到哪层 gather 哪层, 算完立刻 free; 不计入常驻' })
    }
    return { segs, total }
  })
})
const cur = computed(() => bars.value[stage.value])
const firstFit = computed(() => bars.value.findIndex((b) => b.total <= limit.value))
</script>

<style scoped>
.grid { stroke: var(--border); stroke-width: 1; }
.tick, .lbl { font-size: 10px; fill: var(--text-dim); font-family: "SF Mono", Menlo, monospace; }
.lbl { font-size: 11px; fill: var(--text-muted); }
.bar { cursor: pointer; outline: none; }
.bar .hit { fill: transparent; stroke: transparent; rx: 6; }
.bar.sel .hit, .bar:focus-visible .hit { fill: var(--accent-soft); stroke: var(--accent); }
.bar.sel .lbl { fill: var(--text); font-weight: 600; }
.transient { opacity: 0.35; stroke: var(--accent); stroke-dasharray: 3 2; }
.val { font-size: 11px; font-family: "SF Mono", Menlo, monospace; }
.val.fit { fill: var(--left); }
.val.oom { fill: var(--danger); }
.limit { stroke: var(--danger); stroke-width: 2; stroke-dasharray: 6 4; }
.limit-t { font-size: 11px; fill: var(--danger); }
.formula { font-size: 13px !important; }
.sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin: 0 4px 0 10px; vertical-align: -1px; }
.sw:first-child { margin-left: 0; }
.sw.tr { background: var(--accent); opacity: 0.35; outline: 1px dashed var(--accent); }
</style>
