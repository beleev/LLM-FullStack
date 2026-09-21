<!--
  batch 等价实验台 (对应 llm_train/m01_gradient_accumulation + m02_data_parallel)。
  只讲一件事: 梯度累积和 DDP 算出来的是同一个梯度, 区别只在拿时间换显存还是拿卡换时间。
  唯一能打破等价的是"各卡样本数不等" —— 那一个开关比其余控件加起来都重要。
-->
<template>
  <LabFrame
    title="一个 batch, 三种切法 — 梯度都是同一个"
    sub="24 个样本 = 24 个小格。拖卡数和 micro-batch 数换切法, 按播放: 格子一批批点亮, 每张卡右边的梯度缓冲区跟着涨, 最后一次 all-reduce 让所有卡握住同一个数。右边四个数字是你为这种切法付的账。"
    module="llm_train/m01+m02"
    run="python -m llm_train.m02_data_parallel.demo"
    :challenge="{
      ask: '把卡数从 1 拖到 8, 右边四个数字分别怎么变? 再把 micro-batch 数从 1 拖到 4 呢? 最后打开「各卡样本数不等」, 哪个数字先崩?',
      answer: '加卡: 激活峰值和步时间都除以卡数, 与基线的差一直贴在 1e-17 (浮点求和顺序), 只有通信从 0 涨到 2(P−1)/P×梯度字节 —— DDP 的全部买卖就是拿通信换时间。加 micro-batch: 激活峰值除以 K, 步时间一点没省, 通信也没变 —— 累积换的是显存, 不是速度。打开不等分: |Δ| 从 1e-17 跳到 1e-1。all-reduce(mean) 做的是「各卡局部均值的简单平均」, 只有每卡样本数相同时它才等于全局均值; 顺带步时间也被最大的那张卡拖长了。真实框架要么用 DistributedSampler 把尾巴补齐或丢掉, 要么把每卡 loss 先乘 n_r·P/N 再同步。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: mode === 'single' }" @click="preset(0, 1)">单卡整 batch</button>
        <button type="button" :class="{ active: mode === 'accum' }" @click="preset(0, 4)">梯度累积</button>
        <button type="button" :class="{ active: mode === 'ddp' }" @click="preset(3, 1)">DDP</button>
        <button type="button" :class="{ active: mode === 'both' }" @click="preset(3, 2)">累积 + DDP</button>
        <span class="sep" />
        <button type="button" :class="{ active: !uneven }" :disabled="P === 1" @click="uneven = false">各卡等分</button>
        <button type="button" class="warnbtn" :class="{ active: uneven }" :disabled="P === 1" @click="uneven = true">各卡样本数不等</button>
      </div>
      <LabSlider v-model="pIdx" label="卡数 P" :min="0" :max="5" :format="(i) => PS[i]" />
      <LabSlider v-model="k" label="每卡 micro-batch 数 K" :min="1" :max="kMax" />
      <StepPlayer :stepper="stepper" :label="phase" />
    </template>

    <div class="cells beq" :style="{ gridTemplateColumns: `46px repeat(${maxCols}, minmax(18px, 1fr)) 76px` }">
      <template v-for="(row, r) in rows" :key="r">
        <span class="row-label mono">{{ P === 1 ? '单卡' : '卡 ' + r }}</span>
        <span
          v-for="(c, ci) in row" :key="ci"
          class="cell" :class="cellClass(c)"
          :title="c ? `样本 #${c.gi} · 卡 ${r} 的第 ${c.m + 1} 个 micro-batch · 这一个样本的梯度 ${g[c.gi].toFixed(3)}` : ''"
          @mouseenter="c && (hover = { ...c, r })" @mouseleave="hover = null"
        >{{ c && K > 1 ? c.m + 1 : '' }}</span>
        <span class="buf mono" :class="{ done: reduced }">{{ bufs[r].toFixed(3) }}</span>
      </template>
    </div>

    <div v-if="P > 1" class="ar" :class="{ live: reduced }">
      <span class="mono">all-reduce(mean) · {{ reduced ? `${P} 张卡现在握着同一个梯度 ${synced.toFixed(3)}` : '等各卡累积完再同步' }}</span>
    </div>

    <p class="lab-note msg">
      <template v-if="hover">
        样本 #{{ hover.gi }} 在卡 {{ hover.r }} 的第 {{ hover.m + 1 }} 个 micro-batch 里, 它自己的梯度是 {{ g[hover.gi].toFixed(3) }}。
      </template>
      <template v-else>{{ phaseText }}</template>
      <br />
      <span class="mono dimtext">单卡整 batch 基线 = {{ baseline.toFixed(6) }}　当前方案同步后 = {{ synced.toFixed(6) }}</span>
    </p>

    <template #stats>
      <div class="kv">
        <span>assert |Δ| &lt; 1e-6</span>
        <b :class="equiv ? 'good' : 'bad'">{{ delta.toExponential(2) }} {{ equiv ? '✓' : '✗' }}</b>
      </div>
      <div class="kv"><span>激活峰值 (样本/次)</span><b :class="{ good: peak < 24 }">{{ peak }} · {{ (peak / 24).toFixed(2) }}×</b></div>
      <div class="kv"><span>步时间 (每卡串行样本)</span><b :class="{ good: slow < 24 }">{{ slow }} · {{ (slow / 24).toFixed(2) }}×</b></div>
      <div class="kv"><span>每步通信 / 卡</span><b :class="P > 1 ? 'bad' : 'good'">{{ P > 1 ? fmtBytes(commBytes) : '0 B' }}</b></div>
      <p class="lab-note">
        ★ 等分时 all-reduce(mean) 恰好等于全局均值, 所以 |Δ| 只剩浮点求和顺序的零头
        (m01/m02 用 float32, 打印出来是 2.98e-08 和 3.73e-09)。不等分时它偏了整整 15 个数量级。<br />
        通信按 Ψ = 1.3B 参数、fp32 梯度算: Ψ × 4B × 2(P−1)/P, 上限 2Ψ × 4B ——
        公式和「通信与 full_loop」那一章的 ring all-reduce 是同一个。
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
import { fmtBytes, mulberry32, randn, range, sum } from '@/utils/labmath.js'

const N = 24                     // 全局 batch: 24 个样本
const PS = [1, 2, 3, 4, 6, 8]    // 只取能整除 24 的卡数
const PSI = 1.3e9                // 参数量, 用来把通信量换算成看得见的字节

const pIdx = ref(3)
const k = ref(1)
const uneven = ref(false)
const hover = ref(null)

const P = computed(() => PS[pIdx.value])
const K = computed(() => k.value)
const mode = computed(() => (P.value === 1 ? (K.value === 1 ? 'single' : 'accum') : K.value === 1 ? 'ddp' : 'both'))
const preset = (pi, kk) => { pIdx.value = pi; k.value = kk; if (pi === 0) uneven.value = false }

// 每个样本自己的梯度 (标量代表整棵梯度树): 固定 seed, 拖滑杆时数字不乱跳
const g = computed(() => { const r = mulberry32(7); return range(N).map(() => randn(r)) })
const baseline = computed(() => sum(g.value) / N)   // ★ 单卡整 batch 的答案, 一切都要跟它对

// 各卡拿几个样本。不等分时把 0 号卡撑大, 其余等量缩小, 总数仍是 N
const sizes = computed(() => {
  const p = P.value
  if (p === 1) return [N]
  const base = N / p
  if (!uneven.value) return Array(p).fill(base)
  const d = Math.max(1, Math.floor(base / 3))
  return range(p).map((r) => (r === 0 ? base + d * (p - 1) : base - d))
})
const kMax = computed(() => Math.min(6, Math.min(...sizes.value)))
watch(kMax, (m) => { if (k.value > m) k.value = m }, { immediate: true })

// 每卡: 把自己的样本切成 K 个 micro-batch, 各自算均值, 再按 n_k/n_rank 加权累进缓冲区
const plan = computed(() => {
  const kk = K.value
  let off = 0
  return sizes.value.map((n) => {
    const micros = range(kk).map((i) => {
      const a = off + Math.floor((n * i) / kk)
      const idx = range(off + Math.floor((n * (i + 1)) / kk) - a).map((j) => a + j)
      const mean = idx.length ? sum(idx.map((j) => g.value[j])) / idx.length : 0
      return { idx, n: idx.length, mean, w: idx.length / n }   // ★ 权重是样本数占比, 不是 1/K
    })
    off += n
    return { n, micros, local: sum(micros.map((m) => m.w * m.mean)) }
  })
})

// all-reduce(mean) = 各卡局部均值的简单平均。等分时 == baseline, 不等分时不是
const synced = computed(() => sum(plan.value.map((r) => r.local)) / P.value)
const delta = computed(() => Math.abs(synced.value - baseline.value))
const equiv = computed(() => delta.value < 1e-6)

const frames = computed(() => K.value + (P.value > 1 ? 1 : 0) + 1)
const stepper = useStepper(frames, { interval: 700 })
const f = computed(() => Math.min(stepper.step.value, frames.value - 1))
const doneK = computed(() => Math.min(f.value, K.value))
const active = computed(() => (f.value >= 1 && f.value <= K.value ? f.value - 1 : -1))
const reduced = computed(() => P.value > 1 && f.value > K.value)

const maxCols = computed(() => Math.max(...sizes.value))
const rows = computed(() => plan.value.map((r) => {
  const cells = r.micros.flatMap((m, mi) => m.idx.map((gi) => ({ gi, m: mi })))
  while (cells.length < maxCols.value) cells.push(null)
  return cells
}))
const bufs = computed(() => plan.value.map((r) =>
  (reduced.value ? synced.value : sum(r.micros.slice(0, doneK.value).map((m) => m.w * m.mean)))))

const cellClass = (c) => {
  if (!c) return 'pad'
  if (reduced.value || c.m < active.value) return 'ok'
  if (c.m === active.value) return 'on now'
  return 'dim'
}

const peak = computed(() => Math.max(...plan.value.flatMap((r) => r.micros.map((m) => m.n))))
const slow = computed(() => Math.max(...sizes.value))
const commBytes = computed(() => (PSI * 4 * 2 * (P.value - 1)) / P.value)

const phase = computed(() => (reduced.value ? 'all-reduce' : f.value === 0 ? '未开始' : `micro ${f.value}/${K.value}`))
const phaseText = computed(() => {
  if (f.value === 0) return '每卡的梯度缓冲区都是 0。按播放, 或点 ▶ 单步。'
  if (reduced.value) return `all-reduce(mean): 每卡把自己那份梯度发出去再收回平均值。${P.value} 张卡现在逐位相同, 可以各自 optimizer.step()。`
  const n = plan.value.map((r) => r.micros[active.value].n).join(' / ')
  return `第 ${f.value} 个 micro-batch: 各卡分别前向反向 ${n} 个样本, 梯度乘 n_k/n_rank 后累进缓冲区 (缓冲区数字在右侧)。`
})

watch([P, K, uneven], () => { stepper.pause(); stepper.step.value = frames.value - 1 }, { immediate: true })
</script>

<style scoped>
.row-label { font-size: 11px; color: var(--text-dim); align-self: center; }
.beq .cell { height: 24px; font-size: 9px; }
.cell.pad { border: none; background: none; }
.cell.now { outline: 2px solid var(--accent); outline-offset: 1px; }
.buf {
  align-self: center; text-align: right; font-size: 12px;
  color: var(--text-muted); font-variant-numeric: tabular-nums; padding-left: 6px;
}
.buf.done { color: var(--left); }
.ar {
  margin-top: 8px; padding: 5px 10px; border-radius: var(--radius-sm); font-size: 11px;
  border: 1px dashed var(--border-strong); color: var(--text-dim); overflow: hidden;
}
.ar.live { border: 1px solid var(--left); color: var(--text); background: color-mix(in srgb, var(--left) 14%, transparent); }
.ar.live span { display: inline-block; animation: sweep 0.7s ease-out; }
@keyframes sweep { from { transform: translateX(-24px); opacity: 0.2; } to { transform: none; opacity: 1; } }
.sep { width: 1px; height: 18px; background: var(--border-strong); }
.warnbtn.active { border-color: var(--danger); color: var(--danger); }
.msg { margin-top: 12px; min-height: 3.4em; }
.dimtext { color: var(--text-dim); font-size: 11px; }
</style>
