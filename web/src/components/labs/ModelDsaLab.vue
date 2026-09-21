<!--
  DSA 稀疏选择实验台 (对应 llm_models/layers/core/attention.py:LightningIndexer / MultiHeadLatentSparseAttention)。
  只讲一件事: 便宜的 indexer 先给所有 key 打分, 只留 top-k 交给昂贵的主注意力;
  indexer 靠 KL(主注意力 ‖ softmax(indexer 分数)) 学会 “和主注意力排序一致”, 对齐得越好, k 就能越小。
-->
<template>
  <LabFrame
    title="DSA — 先粗选 top-k, 再精算注意力"
    sub="一个 query 面对 48 个历史 key。上排 = 主注意力真正的概率 (稠密算出来的 “标准答案”), 下排 = indexer 的打分。亮色 = 被 top-k 选中; 红色 = 主注意力很看重、却被 indexer 漏掉的 key。悬停任一列上下对照; 点右下曲线上的点直接设 k。"
    module="llm_models/layers/core/attention.py"
    run="python -m llm_models.run_models.moe.deepseek_v3_2.train_deepseek_v3_2"
    :challenge="{
      ask: '把 “indexer 对齐程度” 拖到 0 (没训练过), k = 8: 输出误差多大? 再拖到 1。先猜: 对齐好之后, 要把误差压到 5% 以内, k 需要多大?',
      answer: '没对齐的 indexer 等于随机选 8 个 key: 主注意力质量大头都被漏掉, 输出误差接近 100%, 稀疏化直接毁掉模型。对齐之后 top-8 就能盖住 90% 以上的注意力质量 —— 因为 softmax 注意力本来就是长尾的, 绝大多数 key 的权重接近 0, 算它们纯属浪费。这就是 DSA 的赌注: 注意力天然稀疏, 只要有个便宜的东西能猜中哪几个重要。top-k 不可导, indexer 从 LM loss 拿不到梯度, 所以单独用 KL 对齐 loss 训练, 并且先 dense warmup (主注意力看全部, 只训 indexer), 再切到稀疏。复杂度从 O(T²) 降到 O(T·k) + 一个很便宜的 O(T²) 打分。',
    }"
  >
    <template #controls>
      <LabSlider v-model="k" label="top-k" :min="1" :max="S" />
      <LabSlider v-model="align" label="indexer 对齐程度" :min="0" :max="1" :step="0.05" :format="(t) => t.toFixed(2)" />
      <div class="row"><button type="button" @click="seed++">换一个 query</button></div>
    </template>

    <div class="rows" @mouseleave="hov = -1">
      <p class="cap">主注意力概率 p (标准答案)</p>
      <div class="strip">
        <div v-for="j in S" :key="j" class="c" :class="cls(j - 1)" @mouseenter="hov = j - 1"><div :style="{ height: (sim.p[j - 1] / pMax) * 100 + '%' }" /></div>
      </div>
      <p class="cap">indexer 分数 softmax(I) —— 只用来排序选 top-{{ k }}</p>
      <div class="strip">
        <div v-for="j in S" :key="j" class="c idx" :class="cls(j - 1)" @mouseenter="hov = j - 1"><div :style="{ height: (sim.qi[j - 1] / qMax) * 100 + '%' }" /></div>
      </div>
      <p class="cap mono">{{ hov < 0 ? '悬停某一列查看' : `key ${hov}: p = ${sim.p[hov].toFixed(3)}, indexer 排名第 ${sim.rank[hov] + 1}, ${sim.rank[hov] < k ? '选中' : '未选中'}` }}</p>
    </div>

    <svg viewBox="0 0 520 130" role="group" aria-label="输出误差随 k 变化, 点击设置 k">
      <line x1="34" x2="514" :y1="ey(0.05)" :y2="ey(0.05)" class="ok" /><text x="514" :y="ey(0.05) - 3" class="lg">5% 误差线</text>
      <text x="30" :y="ey(1) + 8" class="yl">100%</text><text x="30" :y="ey(0) + 3" class="yl">0</text>
      <polyline :points="sim.curve.map((e, i) => `${ex(i + 1)},${ey(e).toFixed(1)}`).join(' ')" class="curve" />
      <circle
        v-for="(e, i) in sim.curve" :key="i" :cx="ex(i + 1)" :cy="ey(e)" :r="i + 1 === k ? 6 : 3.5" class="pt" :class="{ now: i + 1 === k }"
        tabindex="0" role="button" :aria-label="`设 k = ${i + 1}`" @click="k = i + 1" @keydown.enter="k = i + 1"
      />
      <text x="274" y="126" class="xl">k (点任意点设置) → 输出相对误差 ‖o_稀疏 − o_稠密‖ / ‖o_稠密‖</text>
    </svg>

    <template #stats>
      <div class="kv"><span>选中 key 覆盖的注意力质量</span><b :class="sim.mass > 0.9 ? 'good' : 'bad'">{{ (sim.mass * 100).toFixed(1) }}%</b></div>
      <div class="kv"><span>输出相对误差</span><b :class="sim.curve[k - 1] < 0.05 ? 'good' : 'bad'">{{ (sim.curve[k - 1] * 100).toFixed(1) }}%</b></div>
      <div class="kv"><span>对齐 loss KL(p ‖ softmax(I))</span><b>{{ sim.kl.toFixed(3) }}</b></div>
      <div class="kv"><span>主注意力计算量 k / S</span><b class="good">{{ ((k / S) * 100).toFixed(0) }}%</b></div>
      <p class="lab-note">KL 对齐 loss 只要求 “分布形状像”, 真正影响结果的是排序: top-k 里有没有包含 p 最大的那几个 key。</p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { mulberry32, randn, softmax, range, sum } from '@/utils/labmath.js'

const S = 48, DV = 4
const k = ref(8), align = ref(1), seed = ref(2), hov = ref(-1)

const sim = computed(() => {
  const r = mulberry32(seed.value * 31)
  // 真实注意力 logit: 大部分是噪声, 少数几个 key 明显相关 (注意力天然长尾)
  const logit = range(S).map(() => randn(r) * 0.8)
  range(5).forEach(() => { logit[Math.floor(r() * S)] += 3 + r() * 2 })
  const noise = range(S).map(() => randn(r) * 1.6)
  const vals = range(S).map(() => range(DV).map(() => randn(r)))
  const p = softmax(logit)
  // indexer 分数: align=1 与主注意力 logit 一致 (KL→0), align=0 纯噪声
  const I = logit.map((l, j) => align.value * l + (1 - align.value) * noise[j])
  const qi = softmax(I)
  const order = range(S).sort((a, b) => I[b] - I[a])
  const rank = Array(S); order.forEach((j, i) => { rank[j] = i })
  const out = (w) => range(DV).map((e) => sum(w.map((wj, j) => wj * vals[j][e])))
  const dense = out(p), nd = Math.hypot(...dense)
  // ★ 稀疏输出: 主注意力只在 top-k 上重新 softmax (其余位置 mask 成 -inf)
  const curve = range(S).map((i) => {
    const o = out(softmax(logit.map((l, j) => (rank[j] <= i ? l : -Infinity))))
    return Math.hypot(...o.map((x, e) => x - dense[e])) / nd
  })
  const kl = sum(p.map((pj, j) => (pj > 0 ? pj * Math.log(pj / qi[j]) : 0)))
  return { p, qi, rank, curve, kl, mass: sum(p.filter((_, j) => rank[j] < k.value)) }
})
const pMax = computed(() => Math.max(...sim.value.p))
const qMax = computed(() => Math.max(...sim.value.qi))
const cls = (j) => ({ picked: sim.value.rank[j] < k.value, missed: sim.value.rank[j] >= k.value && sim.value.p[j] > 0.04, hov: hov.value === j })

const ex = (i) => 34 + ((i - 1) / (S - 1)) * 480
const ey = (e) => 108 - Math.min(1, e) * 96
</script>

<style scoped>
.cap { font-size: 11px; color: var(--text-dim); margin: 4px 0; }
.strip { display: flex; gap: 1px; height: 64px; align-items: flex-end; border-bottom: 1px solid var(--border-strong); min-width: 430px; }
.c { flex: 1; height: 100%; display: flex; align-items: flex-end; cursor: crosshair; }
.c div { width: 100%; min-height: 2px; background: var(--border-strong); border-radius: 1px 1px 0 0; }
.c.picked div { background: var(--accent); }
.c.idx.picked div { background: var(--eye); }
.c.missed div { background: var(--danger); }
.c.hov { background: var(--accent-soft); }
.ok { stroke: var(--left); stroke-dasharray: 4 4; }
.lg { text-anchor: end; font-size: 9px; fill: var(--left); }
.yl { text-anchor: end; font-size: 9px; fill: var(--text-dim); }
.xl { text-anchor: middle; font-size: 9px; fill: var(--text-dim); }
.curve { fill: none; stroke: var(--accent); stroke-width: 1.5; }
.pt { fill: var(--accent); cursor: pointer; outline: none; }
.pt.now, .pt:focus-visible { fill: var(--eye); stroke: var(--text); stroke-width: 1.5; }
</style>
