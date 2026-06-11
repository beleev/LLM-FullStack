<template>
  <div class="lab card">
    <div class="lab-head">
      <h3>MoE 路由实验台 — 倾斜、溢出与 all-to-all</h3>
      <p class="lab-sub">
        64 个 token 经 top-1 路由发往 E 个专家（均分在 D 张卡上）。路由越倾斜，
        热点卡越忙、容量溢出丢的 token 越多。对应 llm_train/m11_expert_parallel/demo.py。
      </p>
    </div>

    <div class="lab-controls">
      <div class="mode-row">
        <button
          v-for="d in [2, 4]" :key="d"
          :class="{ active: D === d }"
          @click="D = d"
        >D = {{ d }} 卡</button>
        <button @click="skew = 0">一键均衡 (aux loss)</button>
      </div>
      <div class="ctl">
        <label>专家数 E</label>
        <input type="range" min="4" max="16" step="4" v-model.number="E" />
        <span class="val mono">{{ E }}</span>
      </div>
      <div class="ctl">
        <label>容量因子 cf</label>
        <input type="range" min="1" max="2" step="0.25" v-model.number="cf" />
        <span class="val mono">{{ cf.toFixed(2) }}</span>
      </div>
      <div class="ctl">
        <label>路由倾斜度 skew</label>
        <input type="range" min="0" max="3" step="0.5" v-model.number="skew" />
        <span class="val mono">{{ skew.toFixed(1) }}</span>
      </div>
      <p v-if="E % D !== 0" class="warn mono">E={{ E }} 不能被 D={{ D }} 整除, 专家无法均分到卡</p>
    </div>

    <div class="lab-body">
      <div class="chart-wrap">
        <div class="chart">
          <div class="cap-line" :style="{ bottom: capPct + '%' }">
            <span class="mono">capacity = {{ capacity }}</span>
          </div>
          <div class="col" v-for="(c, e) in route.counts" :key="e">
            <div class="seg over" :style="{ height: hPct(Math.max(0, c - capacity)) }" />
            <div class="seg" :style="{ height: hPct(Math.min(c, capacity)), background: devColors[devOf(e)] }" />
          </div>
        </div>
        <div class="xlabs">
          <span class="mono" v-for="e in E" :key="e">e{{ e - 1 }}</span>
        </div>
      </div>

      <div class="lab-stats">
        <div class="stat">
          <span class="num mono">{{ imbalance }}x</span>
          <span class="cap">不均衡度 max/mean</span>
        </div>
        <div class="stat">
          <span class="num mono" :class="{ bad: dropped > 0 }">{{ dropped }}/64</span>
          <span class="cap">容量溢出丢弃的 token</span>
        </div>
        <div class="a2a">
          <span class="cap">all-to-all 发送矩阵 (行 = 源卡, 列 = 目标卡)</span>
          <div class="a2a-grid" :style="{ gridTemplateColumns: `repeat(${D}, 36px)` }">
            <template v-for="(row, s) in dispatch" :key="s">
              <span v-for="(v, t) in row" :key="t" class="a2a-cell mono" :style="a2aStyle(v)">{{ v }}</span>
            </template>
          </div>
        </div>
        <div class="legend">
          <span class="leg mono" v-for="d in D" :key="d">
            <i :style="{ background: devColors[d - 1] }" />卡{{ d - 1 }}
          </span>
        </div>
        <p class="note">
          aux loss（或 DeepSeek-V3 的 bias 调节）等价于把 skew 拉回 0 ——
          拖动倾斜度滑杆体会路由被"推平"后丢弃归零。
        </p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'

const N = 64
const D = ref(4)
const E = ref(8)
const cf = ref(1.25)
const skew = ref(1.5)
const devColors = ['var(--accent)', 'var(--left)', 'var(--eye)', 'var(--right)']

function mulberry32(seed) {
  return function () {
    let t = (seed += 0x6d2b79f5)
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

const devOf = (e) => Math.floor(e / (E.value / D.value))

// top-1 路由: score_e = skew·lin_e + noise, lin 从 +1 线性递减到 -1 (专家 0 天然最热)
const route = computed(() => {
  const counts = new Array(E.value).fill(0)
  const expertOf = []
  for (let n = 0; n < N; n++) {
    const rand = mulberry32(n * 1000 + 7)
    let best = 0
    let bestScore = -Infinity
    for (let e = 0; e < E.value; e++) {
      const lin = 1 - (2 * e) / (E.value - 1)
      const score = skew.value * lin + (rand() - 0.5) * 2.5
      if (score > bestScore) { bestScore = score; best = e }
    }
    expertOf.push(best)
    counts[best] += 1
  }
  return { counts, expertOf }
})

const capacity = computed(() => Math.ceil((N / E.value) * cf.value))
const dropped = computed(() =>
  route.value.counts.reduce((s, c) => s + Math.max(0, c - capacity.value), 0))
const imbalance = computed(() =>
  (Math.max(...route.value.counts) / (N / E.value)).toFixed(2))

// 发送矩阵: token n 的源卡 = floor(n / (N/D)), 目标卡 = 其专家所在卡
const dispatch = computed(() => {
  const M = Array.from({ length: D.value }, () => new Array(D.value).fill(0))
  route.value.expertOf.forEach((e, n) => {
    M[Math.floor(n / (N / D.value))][devOf(e)] += 1
  })
  return M
})

const scaleMax = computed(() => Math.max(capacity.value, ...route.value.counts) * 1.15)
const capPct = computed(() => ((capacity.value / scaleMax.value) * 100).toFixed(1))
const hPct = (v) => `${((v / scaleMax.value) * 100).toFixed(1)}%`
const a2aStyle = (v) => {
  const max = Math.max(1, ...dispatch.value.flat())
  const pct = Math.round((v / max) * 80)
  return { background: `color-mix(in srgb, var(--accent) ${pct}%, transparent)` }
}
</script>

<style scoped>
.lab { margin-bottom: 16px; }
.lab-head h3 { margin-bottom: 4px; }
.lab-sub { color: var(--text-muted); font-size: 13px; line-height: 1.6; margin-bottom: 14px; }

.lab-controls { display: flex; flex-direction: column; gap: 8px; margin-bottom: 14px; }
.mode-row { display: flex; flex-wrap: wrap; gap: 8px; }
.mode-row button { font-size: 12px; }
.ctl { display: grid; grid-template-columns: 110px 1fr 56px; gap: 10px; align-items: center; }
.ctl label { font-size: 12px; color: var(--text-muted); }
.ctl .val { font-size: 12px; text-align: right; color: var(--accent); }
.warn { font-size: 12px; color: var(--warn); }

.lab-body { display: grid; grid-template-columns: minmax(220px, 320px) 1fr; gap: 18px; align-items: start; }

.chart-wrap { background: var(--code-bg); border-radius: var(--radius-sm); padding: 10px; }
.chart { position: relative; height: 180px; display: flex; align-items: flex-end; gap: 4px; }
.col { flex: 1; display: flex; flex-direction: column; justify-content: flex-end; height: 100%; }
.seg { border-radius: 2px 2px 0 0; transition: height 0.2s; }
.seg.over { background: var(--danger); }
.cap-line { position: absolute; left: 0; right: 0; border-top: 2px dashed var(--warn); z-index: 1; }
.cap-line span { position: absolute; right: 0; top: -16px; font-size: 10px; color: var(--warn); }
.xlabs { display: flex; gap: 4px; margin-top: 4px; }
.xlabs span { flex: 1; text-align: center; font-size: 10px; color: var(--text-dim); }

.lab-stats { display: flex; flex-direction: column; gap: 10px; }
.stat { display: flex; align-items: baseline; gap: 10px; }
.stat .num { font-size: 20px; color: var(--text); }
.stat .num.bad { color: var(--danger); }
.stat .cap { font-size: 12px; color: var(--text-muted); }

.a2a { display: flex; flex-direction: column; gap: 6px; }
.a2a .cap { font-size: 12px; color: var(--text-muted); }
.a2a-grid { display: grid; gap: 3px; }
.a2a-cell { height: 28px; display: flex; align-items: center; justify-content: center; font-size: 11px; color: var(--text); border: 1px solid var(--border); border-radius: 3px; }

.legend { display: flex; gap: 12px; flex-wrap: wrap; }
.leg { display: inline-flex; align-items: center; gap: 5px; font-size: 11px; color: var(--text-muted); }
.leg i { width: 10px; height: 10px; border-radius: 2px; }

.note { font-size: 12px; color: var(--text-dim); line-height: 1.7; border-left: 2px solid var(--accent-soft); padding-left: 10px; }

@media (max-width: 960px) {
  .lab-body { grid-template-columns: 1fr; }
}
</style>
