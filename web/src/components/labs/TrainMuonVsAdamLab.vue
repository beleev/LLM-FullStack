<!--
  Muon vs AdamW (对应 llm_train/m14_muon_optimizer:run, 缩小到 16×16 在浏览器里真跑 150 步)。
  只讲一件事: Adam 的逐元素缩放只能修 "与坐标轴对齐" 的病态; 把同一个病态问题旋转一下, Adam 就修不动了, 而 Muon 的正交化与基无关。
-->
<template>
  <LabFrame
    title="Muon vs AdamW — 把病态方向转一下会怎样?"
    sub="病态的矩阵回归: 输入各主轴的尺度从 1 到 0.01 (条件数 1e4)。两种优化器各扫 5 个学习率、各跑 150 步 (真的在你的浏览器里跑)。切换「病态方向是否与坐标轴对齐」, 再点下面的学习率格子看每条 loss 曲线。"
    module="llm_train/m14"
    run="python -m llm_train.m14_muon_optimizer.demo"
    :challenge="{
      ask: '先看「轴对齐」: 谁赢? 再切到「随机旋转」—— 问题的难度 (条件数) 一点没变, 为什么 AdamW 的最好成绩明显变差, Muon 几乎不受影响?',
      answer: '轴对齐时, 每个输入坐标的梯度量级各不相同, Adam 的 1/√v 正好逐坐标把它们拉平。这是 Adam 的主场, 它赢 (换几组数据看, 从小胜到大胜都有)。旋转之后, 病态方向变成了所有坐标的线性组合。每个坐标上的梯度都混着大小尺度, 逐元素的缩放无从下手。Newton–Schulz 则作用在奇异值上: 旋转只改变奇异向量、不改变奇异值, 所以 Muon 的轨迹几乎不变。真实网络的权重没有理由与坐标轴对齐, 这就是 Muon 在 LLM 预训练上省算力的来源。(Python demo 用 32×32: 旋转时 Muon 好 9.2×, 轴对齐时 Adam 好 1.9×。)',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: !rotate }" @click="rotate = false">病态方向与坐标轴对齐</button>
        <button type="button" :class="{ active: rotate }" @click="rotate = true">随机旋转 (不对齐)</button>
        <button type="button" @click="seed++">换一组数据</button>
      </div>
      <div v-for="o in OPTS" :key="o.key" class="row lr-row">
        <span class="oname" :style="{ color: o.color }">{{ o.name }}</span>
        <button
          v-for="(lr, i) in LRS" :key="lr" type="button" class="chip mono" :class="{ active: pick[o.key] === i, best: sweep.best[o.key] === i }"
          :title="`lr = ${lr}, 最终 loss ${sweep[o.key][i].final.toExponential(2)}`" @click="pick[o.key] = i"
        >lr {{ lr }}<br /><small>{{ sweep[o.key][i].final.toExponential(1) }}</small></button>
        <button type="button" class="chip" @click="pick[o.key] = sweep.best[o.key]">选最好</button>
      </div>
    </template>

    <svg :viewBox="`0 0 ${W} ${H}`" role="img" aria-label="loss 随训练步数的变化 (对数纵轴)">
      <g v-for="t in [-6, -5, -4, -3, -2, -1, 0, 1]" :key="t">
        <line :x1="X0" :x2="W - 8" :y1="py(t)" :y2="py(t)" class="grid" />
        <text :x="X0 - 5" :y="py(t) + 3" class="tick" text-anchor="end">1e{{ t }}</text>
      </g>
      <text :x="W - 8" :y="H - 4" class="tick" text-anchor="end">训练步数 → 150</text>
      <polyline v-for="o in OPTS" :key="o.key" :points="curve(o.key)" fill="none" :stroke="o.color" stroke-width="2" />
      <g v-for="o in OPTS" :key="'e' + o.key">
        <circle :cx="px(STEPS - 1)" :cy="py(Math.log10(cur(o.key).final))" r="4" :fill="o.color" />
        <text :x="px(STEPS - 1) - 8" :y="py(Math.log10(cur(o.key).final)) - 7" class="endl" text-anchor="end" :fill="o.color">{{ o.name }} {{ cur(o.key).final.toExponential(1) }}</text>
      </g>
    </svg>

    <template #stats>
      <div class="kv"><span>AdamW 最好 (扫 5 个 lr)</span><b>{{ bestOf('adam').toExponential(2) }}</b></div>
      <div class="kv"><span>Muon 最好 (扫 5 个 lr)</span><b>{{ bestOf('muon').toExponential(2) }}</b></div>
      <div class="kv"><span>谁赢</span><b :class="ratio > 1 ? 'good' : 'bad'">{{ ratio > 1 ? `Muon 好 ${ratio.toFixed(1)}×` : `AdamW 好 ${(1 / ratio).toFixed(1)}×` }}</b></div>
      <div class="kv"><span>优化器状态 (每个矩阵)</span><b>2 份 vs 1 份</b></div>
      <p class="lab-note">
        ★ 公平比较 = 各自调到最好的学习率再比。格子里的小字是该学习率的最终 loss, 描边的是各自最好的那个。
        两边的 bias 都用 Adam (Muon 只管 2-D 矩阵); 学习率线性退火到 0, 与 Python demo 相同。
      </p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, reactive, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import { mulberry32, randn, range } from '@/utils/labmath.js'

const D = 16, B = 32, STEPS = 150, LRS = [0.03, 0.1, 0.3, 1, 3]
const OPTS = [{ key: 'adam', name: 'AdamW', color: 'var(--eye)' }, { key: 'muon', name: 'Muon', color: 'var(--left)' }]
const rotate = ref(true), seed = ref(1), pick = reactive({ adam: 3, muon: 2 })

// 扁平 Float64Array 上的矩阵乘: C[n×m] = A[n×k] · B[k×m] (tA/tB = 是否先转置)
const mm = (A, Bm, n, k, m, tA = false, tB = false) => {
  const C = new Float64Array(n * m)
  for (let i = 0; i < n; i++) for (let l = 0; l < k; l++) {
    const a = tA ? A[l * n + i] : A[i * k + l]
    if (a !== 0) for (let j = 0; j < m; j++) C[i * m + j] += a * (tB ? Bm[j * k + l] : Bm[l * m + j])
  }
  return C
}
// ★ 与 Python 的 newton_schulz 相同 (方阵, 不需要转置分支)
const newtonSchulz = (G) => {
  const a = 3.4445, b = -4.775, c = 2.0315
  const nrm = Math.sqrt(G.reduce((s, v) => s + v * v, 0)) + 1e-7 // Frobenius 归一 → 奇异值全部 ≤ 1
  let X = G.map((v) => v / nrm)
  for (let s = 0; s < 5; s++) {
    const A = mm(X, X, D, D, D, false, true), AA = mm(A, A, D, D, D)
    const P = A.map((v, i) => b * v + c * AA[i]), PX = mm(P, X, D, D, D)
    X = X.map((v, i) => a * v + PX[i])
  }
  return X
}
const adam = (p, g, m, v, t, lr) => {
  const c1 = 1 - 0.9 ** t, c2 = 1 - 0.999 ** t
  for (let i = 0; i < p.length; i++) {
    m[i] = 0.9 * m[i] + 0.1 * g[i]; v[i] = 0.999 * v[i] + 0.001 * g[i] * g[i]
    p[i] -= lr * (m[i] / c1) / (Math.sqrt(v[i] / c2) + 1e-8)
  }
}

const runOne = (opt, lr, rot, sd) => {
  const rand = mulberry32(sd * 1009 + 14)
  const scale = range(D).map((i) => 10 ** ((-2 * i) / (D - 1))) // 主轴标准差 1 → 0.01
  // 随机正交矩阵 Q (Gram–Schmidt); 不旋转时为单位阵
  const Q = new Float64Array(D * D)
  for (let i = 0; i < D; i++) {
    const v = range(D).map(() => randn(rand))
    for (let p = 0; p < i; p++) { let d = 0; for (let j = 0; j < D; j++) d += v[j] * Q[p * D + j]; for (let j = 0; j < D; j++) v[j] -= d * Q[p * D + j] }
    const n = Math.hypot(...v); for (let j = 0; j < D; j++) Q[i * D + j] = rot ? v[j] / n : i === j ? 1 : 0
  }
  const Wt = Float64Array.from({ length: D * D }, () => randn(rand)), bt = Float64Array.from({ length: D }, () => randn(rand))
  const Wm = new Float64Array(D * D), bias = new Float64Array(D), buf = new Float64Array(D * D)
  const mW = new Float64Array(D * D), vW = new Float64Array(D * D), mb = new Float64Array(D), vb = new Float64Array(D)
  const losses = []
  for (let t = 1; t <= STEPS; t++) {
    const raw = Float64Array.from({ length: B * D }, (_, i) => randn(rand) * scale[i % D])
    const x = mm(raw, Q, B, D, D)
    const diff = mm(x, Wm, B, D, D), target = mm(x, Wt, B, D, D)
    let loss = 0
    for (let i = 0; i < B * D; i++) { diff[i] += bias[i % D] - target[i] - bt[i % D]; loss += diff[i] * diff[i] }
    losses.push(loss / (B * D))
    const k = 2 / (B * D), gW = mm(x, diff, D, B, D, true).map((v) => v * k), gb = new Float64Array(D)
    for (let i = 0; i < B * D; i++) gb[i % D] += diff[i] * k
    const lrT = lr * (1 - t / STEPS)
    if (opt === 'muon') {
      for (let i = 0; i < D * D; i++) buf[i] = 0.95 * buf[i] + gW[i]
      const O = newtonSchulz(gW.map((g, i) => g + 0.95 * buf[i])), s = lrT * 0.2 * Math.sqrt(D)
      for (let i = 0; i < D * D; i++) Wm[i] -= s * O[i]
    } else adam(Wm, gW, mW, vW, t, lrT)
    adam(bias, gb, mb, vb, t, 0.05 * (1 - t / STEPS))
  }
  return { losses, final: losses[STEPS - 1] }
}

// 只在切换旋转 / 换数据时重算 (2 × 5 × 150 步)
const sweep = computed(() => {
  const out = { best: {} }
  for (const o of OPTS) {
    out[o.key] = LRS.map((lr) => runOne(o.key, lr, rotate.value, seed.value))
    out.best[o.key] = out[o.key].reduce((bi, r, i, arr) => (r.final < arr[bi].final ? i : bi), 0)
  }
  return out
})
const cur = (key) => sweep.value[key][pick[key]]
const bestOf = (key) => sweep.value[key][sweep.value.best[key]].final
const ratio = computed(() => bestOf('adam') / bestOf('muon'))

const W = 560, H = 250, X0 = 40, LO = -6, HI = 1.5
const px = (s) => X0 + (s / (STEPS - 1)) * (W - X0 - 12)
const py = (lg) => 10 + (1 - (Math.min(HI, Math.max(LO, lg)) - LO) / (HI - LO)) * (H - 30)
const curve = (key) => cur(key).losses.map((l, s) => `${px(s)},${py(Math.log10(isFinite(l) ? l : 1e9))}`).join(' ')
</script>

<style scoped>
.lr-row { gap: 6px; }
.oname { font-size: 12px; font-weight: 600; min-width: 52px; }
.chip { min-height: 36px; padding: 2px 8px; font-size: 11px; line-height: 1.25; }
.chip small { font-size: 9px; opacity: 0.8; }
.chip.best { box-shadow: 0 0 0 2px var(--left); }
.grid { stroke: var(--border); }
.tick { font-size: 9px; fill: var(--text-dim); font-family: "SF Mono", Menlo, monospace; }
.endl { font-size: 10px; font-family: "SF Mono", Menlo, monospace; }
</style>
