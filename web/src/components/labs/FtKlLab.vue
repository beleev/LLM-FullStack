<!--
  forward KL vs reverse KL 实验台 —— off-policy 蒸馏与 on-policy 蒸馏的核心直觉。
  只讲一件事: 同一个"容量不够"的学生 (单峰), 最小化 KL(p‖q) 会摊开盖住老师的所有峰 (mode-covering),
  最小化 KL(q‖p) 会缩进其中一个峰 (mode-seeking)。期望在谁的样本上取, 决定了惩罚落在哪。
  两个 KL 都在网格上数值积分; forward 的最优解用闭式 (矩匹配) 给出, 与数值结果一致。
-->
<template>
  <LabFrame
    title="forward KL vs reverse KL — 学生该盖住所有峰, 还是钻进一个峰"
    sub="紫色是双峰的 teacher 分布 p, 绿色是只能单峰的 student q。拖绿色峰顶: 左右 = 均值 μ, 上下 = 宽度 σ (越高越窄)。下方小图是被选中的那个 KL 的逐点被积函数 —— 惩罚来自哪里一目了然。"
    module="llm_finetune/methods/on_policy_distill.py"
    :challenge="{
      ask: '先点「最小化 forward KL」, 再点「最小化 reverse KL」。两个最优学生分别落在哪? 各自的「垃圾样本率」是多少? 哪一个更像你希望小模型在生成时的表现?',
      answer: 'forward KL = E_{x~p}[log p/q], 期望在 teacher 的样本上取。只要 teacher 有质量而 student 没有, log(p/q) 就爆炸。所以学生被迫摊开盖住两个峰。代价是把大量概率放在两峰之间的低谷里 —— 那里 teacher 认为几乎不可能, 垃圾样本率很高。这就是离线蒸馏 / SFT 的行为: 在 teacher 写的数据上做 MLE。reverse KL = E_{x~q}[log q/p], 期望在 student 自己的样本上取。student 没去的地方完全不罚, 去了 teacher 不认可的地方重罚。于是它缩进一个峰: 样样像 teacher, 但放弃了另一种答法。on-policy 蒸馏 (学生采样、teacher 逐 token 打分) 优化的正是它: 容量小的学生宁可少会一点, 也不要胡说。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: view === 'f' }" @click="view = 'f'">看 forward KL(p‖q) 的被积函数</button>
        <button type="button" :class="{ active: view === 'r' }" @click="view = 'r'">看 reverse KL(q‖p) 的被积函数</button>
      </div>
      <div class="row">
        <button type="button" @click="fitForward">最小化 forward KL (闭式: 矩匹配)</button>
        <button type="button" @click="fitReverse">最小化 reverse KL (数值下降, 多起点)</button>
      </div>
      <LabSlider v-model="sep" label="teacher 两峰间距" :min="1" :max="6" :step="0.2" :format="(v) => v.toFixed(1)" />
      <LabSlider v-model="w" label="左峰的质量" :min="0.1" :max="0.9" :step="0.05" :format="(v) => v.toFixed(2)" />
    </template>

    <svg ref="svgEl" viewBox="0 0 640 340" role="img" aria-label="teacher 与 student 的概率密度">
      <path :d="area(pPdf, py)" fill="var(--accent)" opacity="0.18" />
      <path :d="line(pPdf, py)" fill="none" stroke="var(--accent)" stroke-width="2" />
      <path :d="line(qPdf, py)" fill="none" stroke="var(--left)" stroke-width="2" />
      <line x1="20" x2="620" :y1="py(0)" :y2="py(0)" stroke="var(--border-strong)" />
      <text x="24" y="18" class="t" fill="var(--accent)">teacher p (双峰)</text>
      <text x="24" y="34" class="t" fill="var(--left)">student q = N(μ, σ²)</text>
      <!-- ★ 手放在学生分布上 -->
      <circle
        class="draggable" :cx="sx(mu)" :cy="py(peak)" r="10" fill="var(--left)" stroke="var(--bg-card)" stroke-width="2"
        tabindex="0" role="slider" aria-label="student 均值, 方向键左右改 μ 上下改 σ" :aria-valuenow="mu" aria-valuemin="-5" aria-valuemax="5"
        @pointerdown="start($event, { svg: svgEl, onMove: drag })"
        @keydown.right.prevent="mu = clamp(mu + 0.2, -5, 5)" @keydown.left.prevent="mu = clamp(mu - 0.2, -5, 5)"
        @keydown.up.prevent="sigma = clamp(sigma - 0.1, SMIN, SMAX)" @keydown.down.prevent="sigma = clamp(sigma + 0.1, SMIN, SMAX)"
      />

      <!-- 被积函数: 正的部分是惩罚 -->
      <text x="24" y="252" class="t">{{ view === 'f' ? 'p(x)·log(p/q) — 在 teacher 有质量的地方结算' : 'q(x)·log(q/p) — 只在 student 自己去的地方结算' }}</text>
      <path :d="area(integrand, iy)" :fill="view === 'f' ? 'var(--accent)' : 'var(--left)'" opacity="0.45" />
      <line x1="20" x2="620" :y1="iy(0)" :y2="iy(0)" stroke="var(--border-strong)" />
    </svg>

    <template #stats>
      <div class="kv"><span>forward KL(p‖q)</span><b :class="{ good: view === 'f' }">{{ kl.f.toFixed(3) }}</b></div>
      <div class="kv"><span>reverse KL(q‖p)</span><b :class="{ good: view === 'r' }">{{ kl.r.toFixed(3) }}</b></div>
      <div class="kv"><span>垃圾样本率</span><b :class="junk > 0.15 ? 'bad' : 'good'">{{ (junk * 100).toFixed(1) }}%</b></div>
      <div class="kv"><span>漏掉的 teacher 质量</span><b :class="miss > 0.15 ? 'bad' : 'good'">{{ (miss * 100).toFixed(1) }}%</b></div>
      <div class="kv"><span>μ / σ</span><b>{{ mu.toFixed(2) }} / {{ sigma.toFixed(2) }}</b></div>
      <p class="lab-note">
        垃圾样本率 = student 落在 teacher 密度 &lt; 5% 峰值处的概率 (生成时"胡说"的比例);
        漏掉的质量 = teacher 的样本落在 student 密度 &lt; 5% teacher 峰值处的比例 (学生"不会"的答法)。
      </p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { useDrag } from '@/composables/useDrag.js'
import { clamp, range, sum } from '@/utils/labmath.js'

const S = 0.6, SMIN = 0.5, SMAX = 4, DX = 0.05, XS = range(321).map((i) => -8 + i * DX)
const SQ = Math.sqrt(2 * Math.PI)
const sx = (x) => 320 + x * 37.5
const py = (d) => 220 - d / 0.8 * 190          // 密度面板: y∈[30,220]
const iy = (v) => 300 - clamp(v, -0.12, 0.35) / 0.35 * 40
const logN = (x, m, s) => -((x - m) ** 2) / (2 * s * s) - Math.log(s * SQ)

const sep = ref(4), w = ref(0.5), mu = ref(0), sigma = ref(1.2), view = ref('f')
const svgEl = ref(null)
const { start } = useDrag()

// teacher 的 log 密度用 logsumexp, 避免尾部下溢成 log(0)
const logP = computed(() => XS.map((x) => {
  const a = Math.log(w.value) + logN(x, -sep.value / 2, S), b = Math.log(1 - w.value) + logN(x, sep.value / 2, S)
  const m = Math.max(a, b)
  return m + Math.log(Math.exp(a - m) + Math.exp(b - m))
}))
const logQ = computed(() => XS.map((x) => logN(x, mu.value, sigma.value)))
const pPdf = computed(() => logP.value.map(Math.exp))
const qPdf = computed(() => logQ.value.map(Math.exp))
const peak = computed(() => 1 / (sigma.value * SQ))

const klOf = (lq) => {
  const q = lq.map(Math.exp)
  return {
    f: sum(pPdf.value.map((p, i) => p * (logP.value[i] - lq[i]))) * DX,   // ★ 期望在 p 上取
    r: sum(q.map((v, i) => v * (lq[i] - logP.value[i]))) * DX,            // ★ 期望在 q 上取
  }
}
const kl = computed(() => klOf(logQ.value))
const integrand = computed(() => (view.value === 'f'
  ? pPdf.value.map((p, i) => p * (logP.value[i] - logQ.value[i]))
  : qPdf.value.map((q, i) => q * (logQ.value[i] - logP.value[i]))))

const thr = computed(() => 0.05 * Math.max(...pPdf.value))
const junk = computed(() => sum(qPdf.value.filter((_, i) => pPdf.value[i] < thr.value)) * DX)
const miss = computed(() => sum(pPdf.value.filter((_, i) => qPdf.value[i] < thr.value)) * DX)

const drag = ({ x, y }) => {
  mu.value = clamp(Math.round((x - 320) / 37.5 * 20) / 20, -5, 5)
  const h = clamp((220 - y) / 190 * 0.8, 1 / (SMAX * SQ), 1 / (SMIN * SQ))   // 峰高 ↔ σ = 1/(h√2π)
  sigma.value = Math.round(100 / (h * SQ)) / 100
}
// forward KL 在高斯族内的最优解 = 矩匹配 (闭式)
const fitForward = () => {
  const m = (sep.value / 2) * (1 - 2 * w.value)
  mu.value = m
  sigma.value = clamp(Math.sqrt(S * S + (sep.value / 2) ** 2 - m * m), SMIN, SMAX)
  view.value = 'f'
}
// reverse KL 没有闭式, 而且有多个局部极小 (盖住两峰的宽高斯也是一个): 从当前位置和两个峰各下降一次, 取最小的
const fitReverse = () => {
  const f = (mm, ss) => klOf(XS.map((x) => logN(x, mm, ss))).r
  const descend = (m, s) => {
    for (let step = 0.4; step > 0.004; step /= 2) {
      for (let it = 0; it < 12; it++) {
        const cands = [[m, s], [m + step, s], [m - step, s], [m, clamp(s + step, SMIN, SMAX)], [m, clamp(s - step, SMIN, SMAX)]]
        const best = cands.reduce((a, c) => (f(...c) < f(...a) ? c : a))
        if (best[0] === m && best[1] === s) break
        ;[m, s] = best
      }
    }
    return [m, s, f(m, s)]
  }
  const best = [[mu.value, sigma.value], [-sep.value / 2, S], [sep.value / 2, S]].map((st) => descend(...st)).reduce((a, c) => (c[2] < a[2] - 1e-6 ? c : a))
  mu.value = clamp(Math.round(best[0] * 100) / 100, -5, 5); sigma.value = Math.round(best[1] * 100) / 100; view.value = 'r'
}

const pts = (ys, fy) => ys.map((v, i) => `${sx(XS[i]).toFixed(1)},${fy(v).toFixed(1)}`)
const line = (ys, fy) => 'M' + pts(ys, fy).join(' L')
const area = (ys, fy) => `M${sx(-8)},${fy(0)} L` + pts(ys, fy).join(' L') + ` L${sx(8)},${fy(0)} Z`
</script>

<style scoped>
.t { font-size: 11px; fill: var(--text-muted); font-family: "SF Mono", Menlo, monospace; }
</style>
