<!-- FlashAttention 实验台 (llm_infer/m11): 按 tile 走 N×N 分数矩阵, 跟着一行 query 看 online softmax 的 m / l 怎么更新。 -->
<template>
  <LabFrame
    title="FlashAttention — 分块 + online softmax"
    sub="左边是 N×N 的注意力分数矩阵 (因果, 右上角被 mask)。它从不完整存在: 每次只有一个 tile (橙色) 进 SRAM。点左侧的 q 行号跟踪一行 query: 它只靠「运行最大值 m」和「运行分母 l」两个数, 就能在看完最后一个 tile 时得到和整行 softmax 完全相同的结果。"
    module="llm_infer/m11"
    run="python -m llm_infer.m11_flash_attention.demo"
    :challenge="{
      ask: '单步播放, 盯着右下角的更新记录: 什么时候旧的 l 会被乘上一个小于 1 的数? 如果不乘会怎样?',
      answer: '当新 tile 里出现了比 m 更大的分数时。l 里存的是 Σexp(s − m_old), 基准换成 m_new 后, 旧的每一项都要乘 exp(m_old − m_new) < 1 才和新项在同一个基准下。不乘的话旧 key 的权重会被高估; 而如果干脆不减最大值, exp 在 fp16 下 s > 11 就溢出。online softmax 的全部内容就是这一行重缩放 —— 它让 softmax 变成可以分块累加的量, 于是 N×N 矩阵不用落地, 显存从 O(N²) 降到 O(块²)。',
    }"
  >
    <template #controls>
      <LabSlider v-model="N" label="序列长度 N" :min="8" :max="16" />
      <LabSlider v-model="bsIdx" label="块大小" :min="0" :max="3" :format="(v) => 2 ** v" />
      <div class="row"><button type="button" @click="seed++">换一组分数</button></div>
      <StepPlayer :stepper="stepper" :label="`tile ${stepper.step.value + 1} / ${tiles.length}`" />
    </template>

    <div class="wrap">
      <div class="cells" :style="{ gridTemplateColumns: `34px repeat(${N}, 22px)` }">
        <template v-for="i in N" :key="i">
          <button type="button" class="rowbtn mono" :class="{ active: row === i - 1 }" :aria-pressed="row === i - 1" @click="row = i - 1">q{{ i - 1 }}</button>
          <span v-for="j in N" :key="j" class="cell" :class="cellCls(i - 1, j - 1)" :title="j <= i ? `s[${i - 1}][${j - 1}] = ${S[i - 1][j - 1].toFixed(2)}` : 'mask'">
            {{ j <= i && row === i - 1 ? Math.round(S[i - 1][j - 1]) : '' }}
          </span>
        </template>
      </div>
      <div class="side">
        <svg :viewBox="`0 0 ${BW} 110`" role="img" aria-label="跟踪行的部分 softmax 与真实 softmax">
          <g v-for="j in rowLen" :key="j">
            <rect :x="(j - 1) * barW + 1" :y="100 - truth[j - 1] * barH" :width="barW - 2" :height="truth[j - 1] * barH" class="truth" />
            <rect v-if="j <= state.seen" :x="(j - 1) * barW + 3" :y="100 - Math.min(1, partial[j - 1]) * barH" :width="barW - 6" :height="Math.min(1, partial[j - 1]) * barH" class="part" />
          </g>
          <line x1="0" :x2="BW" y1="100" y2="100" class="axis" />
        </svg>
        <p class="legend"><span class="sw truth" />整行 softmax (真值) <span class="sw part" />已见 key 上的 exp(s−m)/l</p>
        <ol class="log mono">
          <li v-for="(u, i) in state.log" :key="i" :class="{ hot: u.rescale < 1, cur: i === state.log.length - 1 }">
            K块{{ u.b }}: m {{ fmt(u.mOld) }}→{{ fmt(u.m) }} · l = {{ fmt(u.lOld) }}×{{ u.rescale.toFixed(3) }} + {{ fmt(u.add) }} = {{ fmt(u.l) }}{{ u.rescale < 1 ? ' ★重缩放' : '' }}
          </li>
          <li v-if="!state.log.length">q{{ row }} 所在的 Q 块还没轮到。</li>
        </ol>
      </div>
    </div>

    <template #stats>
      <div class="kv"><span>SRAM 工作集 vs 整矩阵</span><b class="good">{{ bs * bs }} vs {{ N * N }} ({{ (N * N / (bs * bs)).toFixed(0) }}×)</b></div>
      <div class="kv"><span>q{{ row }} 的运行最大值 m</span><b>{{ state.log.length ? fmt(state.m) : '—' }}</b></div>
      <div class="kv"><span>q{{ row }} 的运行分母 l</span><b>{{ state.log.length ? fmt(state.l) : '—' }}</b></div>
      <div class="kv"><span>lse = m + ln l</span><b>{{ state.log.length ? fmt(state.m + Math.log(state.l)) : '—' }}</b></div>
      <div class="kv"><span>全部行: max|online − 整块|</span><b class="good">{{ maxDiff === 0 ? '0' : maxDiff.toExponential(1) }}</b></div>
      <p class="lab-note">
        每个 tile 只进 SRAM 一次, 共 {{ tiles.length }} 个 (因果: 整块被 mask 的 tile 直接跳过)。
        真实 kernel 还同步累计输出 O (同样乘 exp(m_old − m_new)), 并把 lse 存下来供反向和分段合并 (ring attention / chunked prefill) 使用。
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
import { mulberry32, randn, range, softmax } from '@/utils/labmath.js'

const BW = 320
const fmt = (x) => (x === -Infinity ? '−∞' : x.toFixed(2))

const N = ref(12)
const bsIdx = ref(1)
const seed = ref(4)
const row = ref(7)
const bs = computed(() => 2 ** bsIdx.value)
watch(N, (n) => { if (row.value > n - 1) row.value = n - 1 })

// 固定生成 16×16 再裁剪: 拖 N 时已有的分数不变
const S = computed(() => { const rand = mulberry32(seed.value * 31337); return range(16).map(() => range(16).map(() => randn(rand) * 2.5)) })
// tile 顺序同 FlashAttention-2: 外层 Q 块, 内层 K 块; 因果下 K 块起点 > Q 块终点的 tile 整块被 mask, 跳过
const tiles = computed(() => {
  const nb = Math.ceil(N.value / bs.value)
  return range(nb).flatMap((a) => range(a + 1).map((b) => ({ a, b })))
})

// 行 i 在吃掉 K 块 [b·bs, …] 之后的 (m, l): online softmax 的一步
const update = (i, b, m, l) => {
  const keys = range(bs.value).map((j) => b * bs.value + j).filter((j) => j <= i && j < N.value)
  const mNew = Math.max(m, ...keys.map((j) => S.value[i][j]))
  const rescale = m === -Infinity ? 0 : Math.exp(m - mNew) // ★ 基准从 m 换成 mNew: 旧的和先乘 exp(m_old − m_new)
  const add = keys.reduce((a, j) => a + Math.exp(S.value[i][j] - mNew), 0)
  return { b, mOld: m, lOld: l, m: mNew, rescale: m === -Infinity ? 1 : rescale, add, l: l * rescale + add, seen: keys[keys.length - 1] + 1 }
}
const state = computed(() => {
  let m = -Infinity, l = 0, seen = 0
  const log = []
  tiles.value.slice(0, stepper.step.value + 1).filter((t) => t.a === Math.floor(row.value / bs.value)).forEach((t) => {
    const u = update(row.value, t.b, m, l); log.push(u); ({ m, l, seen } = u)
  })
  return { m, l, seen, log }
})
const rowLen = computed(() => row.value + 1)
const barW = computed(() => BW / rowLen.value)
const truth = computed(() => softmax(S.value[row.value].slice(0, rowLen.value)))
const barH = 95 // 纵轴固定 0..1: 部分 softmax 在只见过少数 key 时会高于真值, 看它怎么被后来的 key 稀释
const partial = computed(() => range(state.value.seen).map((j) => Math.exp(S.value[row.value][j] - state.value.m) / state.value.l))
// 所有行走完全部 tile 后, 与整行 softmax 的最大差
const maxDiff = computed(() => Math.max(...range(N.value).map((i) => {
  let m = -Infinity, l = 0
  for (let b = 0; b * bs.value <= i; b++) ({ m, l } = update(i, b, m, l))
  const full = softmax(S.value[i].slice(0, i + 1))
  return Math.max(...full.map((p, j) => Math.abs(p - Math.exp(S.value[i][j] - m) / l)))
})))

const stepper = useStepper(() => tiles.value.length, { interval: 500 })
watch(tiles, (t) => { stepper.pause(); stepper.step.value = Math.floor(t.length / 2) }, { immediate: true })
const cellCls = (i, j) => {
  if (j > i) return 'dim'
  const idx = tiles.value.findIndex((t) => t.a === Math.floor(i / bs.value) && t.b === Math.floor(j / bs.value))
  return [idx === stepper.step.value ? 'hot' : idx < stepper.step.value ? 'on' : '', { follow: i === row.value }]
}
</script>

<style scoped>
.wrap { display: flex; flex-wrap: wrap; gap: 16px; align-items: flex-start; }
.side { flex: 1 1 220px; min-width: 0; }
.side svg { max-width: 360px; }
.rowbtn { min-height: 22px; height: 22px; padding: 0; font-size: 10px; }
.cell.follow { border-color: var(--text); font-size: 9px; }
.truth { fill: none; stroke: var(--text-muted); background: transparent; border: 1px solid var(--text-muted); }
.part { fill: var(--accent); background: var(--accent); }
.axis { stroke: var(--border-strong); }
.legend { font-size: 11px; color: var(--text-dim); margin: 4px 0 8px; }
.sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin: 0 4px 0 8px; box-sizing: border-box; }
.log { list-style: none; padding: 0; margin: 0; font-size: 11px; color: var(--text-muted); line-height: 1.7; }
.log .hot { color: var(--warn); }
.log .cur { color: var(--text); font-weight: 600; }
.log .hot.cur { color: var(--warn); }
</style>
