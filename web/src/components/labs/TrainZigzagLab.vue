<!--
  Zigzag 序列切分 (对应 llm_train/m12_sequence_parallel:shard_positions + ring_attention 的 work 矩阵)。
  只讲一件事: 因果 mask 下连续切分会让最后一张卡每轮都满载、第一张卡几乎闲着; 一头一尾配对 (zigzag) 后每卡每轮工作量相同。
-->
<template>
  <LabFrame
    title="Zigzag — 因果注意力下谁在等谁?"
    sub="左: 32 个 token 的因果 mask (行 = query, 列 = key), 颜色 = 这个 q·k 由哪张卡算; 亮的格子 = 环上当前这一轮在算的。右: 每轮每卡的工作量 (q·k 对数), 每轮耗时由最忙的那张卡决定 (橙框)。切换切法, 单步播放, 点卡号只看一张卡。"
    module="llm_train/m12"
    run="python -m llm_train.m12_sequence_parallel.demo"
    :challenge="{
      ask: 'D = 4, 连续切分。因果 mask 省掉了将近一半的计算, 墙钟时间也省一半吗? 换成 zigzag 再看。',
      answer: '没有。连续切分时, 卡 3 拿的是序列最后一段, 每一轮收到的 KV 块全在它的过去, 每轮都要算满 64 对。卡 0 除了第 0 轮以外全部跳过。每轮的耗时 = 最忙的卡, 所以墙钟 = 36 + 64×3 = 228, 相比不用因果性的 256 只省了 11%。zigzag 把序列切成 2D 段, 卡 r 拿第 r 段和第 2D−1−r 段 (一头一尾), 于是每张卡每一轮都恰好是 32 对 (第 0 轮 36), 墙钟 132, 快 1.73 倍。总计算量、通信量完全没变 —— 变的只是负载均衡。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: !zigzag }" @click="zigzag = false">连续切分</button>
        <button type="button" :class="{ active: zigzag }" @click="zigzag = true">Zigzag (一头一尾配对)</button>
        <span class="sep" />
        <button v-for="d in [2, 4, 8]" :key="d" type="button" :class="{ active: D === d }" @click="D = d">D = {{ d }} 卡</button>
      </div>
      <StepPlayer :stepper="stepper" :label="`第 ${round} 轮`" />
    </template>

    <div class="split">
      <svg :viewBox="`0 0 ${T * CS + 2} ${T * CS + 14}`" role="img" aria-label="因果 mask 的分工" class="mask">
        <rect v-for="p in T" :key="'o' + p" :x="(p - 1) * CS + 1" y="0" :width="CS - 1" height="6" :fill="COLORS[ownerOf[p - 1]]" :opacity="focus < 0 || focus === ownerOf[p - 1] ? 1 : 0.15" />
        <rect
          v-for="c in cellsList" :key="c.id" :x="c.k * CS + 1" :y="c.q * CS + 12" :width="CS - 1" :height="CS - 1"
          :fill="COLORS[c.rq]" :opacity="(focus >= 0 && focus !== c.rq) ? 0.06 : c.step === round ? 1 : c.step < round ? 0.35 : 0.1"
        />
      </svg>

      <div class="side">
        <div class="cells work" :style="{ gridTemplateColumns: `34px repeat(${D}, minmax(26px, 1fr))` }">
          <span />
          <button v-for="r in D" :key="r" type="button" class="rk mono" :class="{ active: focus === r - 1 }" @click="focus = focus === r - 1 ? -1 : r - 1">
            <i :style="{ background: COLORS[r - 1] }" />卡{{ r - 1 }}
          </button>
          <template v-for="(row, s) in sim.work" :key="s">
            <span class="rl mono">轮 {{ s }}</span>
            <span
              v-for="(w, r) in row" :key="r" class="cell"
              :class="{ slow: w === sim.rowMax[s] && w > 0, dim: s > round, now: s === round }"
              :style="{ background: heat(w / sim.block * 0.7, COLORS[r]) }"
              :title="`第 ${s} 轮, 卡 ${r}: 处理来自卡 ${(r - s + D) % D} 的 KV 块, ${w} 对 q·k`"
            >{{ w }}</span>
          </template>
          <span class="rl mono">合计</span>
          <span v-for="(w, r) in sim.perRank" :key="'t' + r" class="cell tot">{{ w }}</span>
        </div>
        <p class="lab-note">卡 r 持有的 token: <span class="mono">{{ focus >= 0 ? posText(focus) : '点上面的卡号查看' }}</span></p>
      </div>
    </div>

    <template #stats>
      <div class="kv"><span>每卡总工作量 max / min</span><b :class="sim.imb < 1.01 ? 'good' : 'bad'">{{ Math.max(...sim.perRank) }} / {{ Math.min(...sim.perRank) }}</b></div>
      <div class="kv"><span>墙钟 Σ(每轮最忙的卡)</span><b :class="zigzag ? 'good' : 'bad'">{{ sim.wall }}</b></div>
      <div class="kv"><span>不利用因果性的墙钟</span><b>{{ D * sim.block }}</b></div>
      <div class="kv"><span>总计算量 (两种切法相同)</span><b>{{ sim.total }} 对</b></div>
      <div class="kv"><span>zigzag 相对连续切分</span><b class="good">快 {{ speedup.toFixed(2) }}×</b></div>
      <p class="lab-note">★ 同步的环上, 每一轮所有卡要等最慢的那张。省下的计算只有摊平到每张卡上, 才会变成省下的时间。</p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import StepPlayer from '@/components/lab/StepPlayer.vue'
import { useStepper } from '@/composables/useStepper.js'
import { heat, range, sum } from '@/utils/labmath.js'

const COLORS = ['var(--accent)', 'var(--left)', 'var(--eye)', 'var(--right)', 'var(--warn)', 'var(--danger)', 'var(--code-fn)', 'var(--text-dim)']
const T = 32, CS = 9
const D = ref(4), zigzag = ref(false), focus = ref(-1)

// ★ 与 Python 的 shard_positions 相同: zigzag 把序列切 2D 段, 卡 r 拿第 r 段和第 2D-1-r 段
const positions = (world, zz) => {
  if (!zz) return range(world).map((r) => range(T / world).map((i) => (r * T) / world + i))
  const c = T / (2 * world)
  return range(world).map((r) => [...range(c).map((i) => r * c + i), ...range(c).map((i) => (2 * world - 1 - r) * c + i)])
}
// work[step][rank] = 本轮卡 rank 要算的 q·k 对数 (q 位置 ≥ k 位置); 第 step 轮卡 r 手里是卡 (r-step)%D 的 KV 块
const simulate = (world, zz) => {
  const pos = positions(world, zz)
  const work = range(world).map((s) => range(world).map((r) => {
    const kv = pos[(r - s + world) % world]
    return sum(pos[r].map((qp) => kv.filter((kp) => kp <= qp).length))
  }))
  const rowMax = work.map((row) => Math.max(...row))
  const perRank = range(world).map((r) => sum(work.map((row) => row[r])))
  return { pos, work, rowMax, perRank, wall: sum(rowMax), total: sum(perRank), block: (T / world) ** 2, imb: Math.max(...perRank) / Math.min(...perRank) }
}
const sim = computed(() => simulate(D.value, zigzag.value))
const speedup = computed(() => simulate(D.value, false).wall / simulate(D.value, true).wall)

const ownerOf = computed(() => { const o = []; sim.value.pos.forEach((ps, r) => ps.forEach((p) => (o[p] = r))); return o })
const cellsList = computed(() => {
  const o = ownerOf.value, out = []
  for (let q = 0; q < T; q++) for (let k = 0; k <= q; k++) out.push({ id: q * T + k, q, k, rq: o[q], step: (o[q] - o[k] + D.value) % D.value })
  return out
})
const stepper = useStepper(() => D.value, { interval: 1100 })
const round = computed(() => Math.min(stepper.step.value, D.value - 1))
watch(D, () => { stepper.reset(); focus.value = -1 })
const posText = (r) => { const p = sim.value.pos[r]; return zigzag.value ? `${p[0]}–${p[p.length / 2 - 1]} 和 ${p[p.length / 2]}–${p[p.length - 1]}` : `${p[0]}–${p[p.length - 1]}` }
</script>

<style scoped>
.sep { width: 12px; }
.split { display: grid; grid-template-columns: minmax(180px, 300px) minmax(0, 1fr); gap: 16px; align-items: start; }
.mask { width: 100%; }
.side { min-width: 0; overflow-x: auto; }
.work .cell { height: 28px; min-width: 26px; font-size: 11px; color: var(--text); }
.rl { font-size: 10px; color: var(--text-dim); align-self: center; }
.rk { font-size: 9px; min-height: 24px; padding: 2px 0; white-space: nowrap; }
.rk i { display: inline-block; width: 8px; height: 8px; border-radius: 2px; margin-right: 2px; }
.cell.slow { outline: 2px solid var(--warn); outline-offset: -2px; }
.cell.now { border-color: var(--text); }
.cell.tot { background: var(--bg-elev); font-weight: 600; }
@media (max-width: 720px) { .split { grid-template-columns: 1fr; } }
</style>
