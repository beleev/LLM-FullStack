<!--
  DoRA 实验台: 权重 = 幅度 m × 方向 V/‖V‖。
  只讲一件事: LoRA 用一个旋钮 (ΔV) 同时改长度和方向, 两者被绑在一起;
  DoRA 让低秩更新 ΔV 只管方向, 另设一个可训标量 m 只管长度 —— 沿径向拖 ΔV 时 DoRA 的结果纹丝不动。
  这里画的是权重矩阵里的一个权重向量 (2 维); 真实 DoRA 对每个向量各存一个 m (本仓库按输出行)。
-->
<template>
  <LabFrame
    title="DoRA — 把「转方向」和「改长度」拆成两个旋钮"
    sub="灰色箭头是冻结的预训练权重 w₀ (权重矩阵里的一个向量)。拖紫色圆点 = 低秩更新把它带到 w₀+ΔV。粉色是 LoRA 的结果 (就是 w₀+ΔV 本身), 绿色是 DoRA 的结果: 方向取自 w₀+ΔV, 长度由滑杆 m 单独决定。黄色 ★ 是全参微调想到达的目标。"
    module="llm_finetune/methods/dora.py"
    :challenge="{
      ask: '选「目标: 只转 35°」。只拖紫点, 让 LoRA 的误差 < 0.05 —— 紫点必须落在哪? 再看 DoRA: 紫点沿着从原点出发的射线来回拖, 绿色箭头动吗? 这说明 DoRA 对 ΔV 的梯度有什么性质?',
      answer: 'LoRA 要「只转不伸」, ΔV 必须精确落在那一个点上 (弦的另一端): 拖偏一点, 长度就跟着变 —— ΔM 和 ΔD 被同一个旋钮绑死, 这正是 DoRA 论文观察到的现象: LoRA 训练中幅度变化与方向变化强正相关, 而全参微调两者几乎独立甚至负相关。DoRA 里 W′ = m·V/‖V‖, 紫点沿射线移动只改变 ‖V‖, 被归一化消掉, 绿色箭头不动 —— 也就是说 loss 对 V 的梯度天然没有径向分量 (被投影到与 V 垂直的方向, 再乘 m/‖V‖), 低秩容量全部花在转方向上; 长度交给每个向量一个标量 m, 代价只是多 d 个参数 (相对 LoRA 的 2dr 多 1/(2r))。推理前同样可以 merge 回一个普通矩阵。',
    }"
  >
    <template #controls>
      <div class="row">
        <button v-for="(t, k) in TARGETS" :key="k" type="button" :class="{ active: tgt === k }" @click="tgt = k">{{ t.name }}</button>
        <button type="button" @click="reset">复位 (ΔV = 0, m = ‖w₀‖)</button>
      </div>
      <LabSlider v-model="m" label="DoRA 幅度 m" :min="0.5" :max="4.5" :step="0.05" :format="(v) => v.toFixed(2)" />
    </template>

    <svg ref="svgEl" viewBox="0 0 640 380" role="img" aria-label="二维权重向量的幅度与方向分解">
      <defs>
        <marker v-for="c in ['text-dim', 'right', 'left']" :id="'ah-' + c" :key="c" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
          <path d="M0,0 L10,5 L0,10 Z" :fill="`var(--${c})`" />
        </marker>
      </defs>
      <circle :cx="OX" :cy="OY" :r="N0 * SC" fill="none" stroke="var(--border-strong)" stroke-dasharray="3 4" />
      <circle :cx="OX" :cy="OY" :r="m * SC" fill="none" stroke="var(--left)" stroke-dasharray="2 5" opacity="0.7" />
      <text :x="OX + 4" :y="OY - m * SC - 4" class="t" fill="var(--left)">半径 m</text>
      <!-- 从原点过紫点的射线: DoRA 只看这条线的方向 -->
      <line :x1="OX" :y1="OY" :x2="px(ray)[0]" :y2="px(ray)[1]" stroke="var(--accent)" stroke-width="0.8" stroke-dasharray="2 3" />
      <line v-for="a in arrows" :key="a.id" :x1="OX" :y1="OY" :x2="px(a.v)[0]" :y2="px(a.v)[1]" :stroke="`var(--${a.c})`" :stroke-width="a.w" :marker-end="`url(#ah-${a.c})`" />
      <line :x1="px(W0)[0]" :y1="px(W0)[1]" :x2="px(v)[0]" :y2="px(v)[1]" stroke="var(--accent)" stroke-width="1.5" />
      <text v-for="a in arrows" :key="'l' + a.id" :x="px(a.v)[0] + 8" :y="px(a.v)[1] + a.dy" class="t" :fill="`var(--${a.c})`">{{ a.id }}</text>
      <text :x="px(target)[0]" :y="px(target)[1] + 6" text-anchor="middle" class="star">★</text>
      <!-- ★ 手放在 ΔV 上 -->
      <circle
        class="draggable" :cx="px(v)[0]" :cy="px(v)[1]" r="9" fill="var(--accent)" stroke="var(--bg-card)" stroke-width="2"
        tabindex="0" role="slider" aria-label="w₀+ΔV 的位置, 方向键移动" :aria-valuenow="v[0]" aria-valuemin="-1" aria-valuemax="7"
        @pointerdown="start($event, { svg: svgEl, onMove: ({ x, y }) => setV((x - OX) / SC, (OY - y) / SC) })"
        @keydown.right.prevent="setV(v[0] + 0.1, v[1])" @keydown.left.prevent="setV(v[0] - 0.1, v[1])"
        @keydown.up.prevent="setV(v[0], v[1] + 0.1)" @keydown.down.prevent="setV(v[0], v[1] - 0.1)"
      />
      <text :x="px(v)[0] + 12" :y="px(v)[1] - 10" class="t" fill="var(--accent)">w₀+ΔV</text>
    </svg>

    <template #stats>
      <table class="cmp">
        <thead><tr><th></th><th>LoRA</th><th>DoRA</th></tr></thead>
        <tbody>
          <tr><td>长度变化 ΔM</td><td class="mono">{{ sg(lora.dM) }}</td><td class="mono">{{ sg(dora.dM) }}</td></tr>
          <tr><td>方向变化 ΔD</td><td class="mono">{{ lora.dD.toFixed(1) }}°</td><td class="mono">{{ dora.dD.toFixed(1) }}°</td></tr>
        </tbody>
      </table>
      <div class="kv"><span>LoRA 距目标</span><b :class="lora.err < 0.05 ? 'good' : 'bad'">{{ lora.err.toFixed(3) }}</b></div>
      <div class="kv"><span>DoRA 距目标</span><b :class="dora.err < 0.05 ? 'good' : 'bad'">{{ dora.err.toFixed(3) }}</b></div>
      <p class="lab-note">
        LoRA 的两行数字由同一个紫点决定, 改一个另一个必然跟着动;
        DoRA 的 ΔD 只由紫点所在的射线决定, ΔM = m − ‖w₀‖ 只由滑杆决定。
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

const OX = 90, OY = 330, SC = 72
const W0 = [2.6, 1.0], N0 = Math.hypot(...W0)
const rot = ([x, y], deg) => { const a = deg * Math.PI / 180; return [x * Math.cos(a) - y * Math.sin(a), x * Math.sin(a) + y * Math.cos(a)] }
const TARGETS = [
  { name: '目标: 只转 35°', v: rot(W0, 35) },
  { name: '目标: 只放大 1.4×', v: W0.map((x) => x * 1.4) },
  { name: '目标: 转 25° 且缩到 0.7×', v: rot(W0, 25).map((x) => x * 0.7) },
]
const px = ([x, y]) => [OX + x * SC, OY - y * SC]

const v = ref([...W0])          // w₀ + ΔV
const m = ref(Math.round(N0 * 20) / 20)
const tgt = ref(0)
const svgEl = ref(null)
const { start } = useDrag()

const setV = (x, y) => { const p = [clamp(x, -1, 7.4), clamp(y, -0.5, 4.4)]; if (Math.hypot(...p) > 0.2) v.value = p.map((c) => Math.round(c * 50) / 50) }
const reset = () => { v.value = [...W0]; m.value = Math.round(N0 * 20) / 20 }

const target = computed(() => TARGETS[tgt.value].v)
const unit = computed(() => { const n = Math.hypot(...v.value); return v.value.map((c) => c / n) })
const doraW = computed(() => unit.value.map((c) => c * m.value))      // ★ W′ = m · V/‖V‖
const ray = computed(() => unit.value.map((c) => c * 7.6))

const measure = (w) => ({
  dM: Math.hypot(...w) - N0,
  dD: Math.acos(clamp((w[0] * W0[0] + w[1] * W0[1]) / (Math.hypot(...w) * N0), -1, 1)) * 180 / Math.PI,
  err: Math.hypot(w[0] - target.value[0], w[1] - target.value[1]),
})
const lora = computed(() => measure(v.value))                          // LoRA: W′ = w₀ + ΔV
const dora = computed(() => measure(doraW.value))
const arrows = computed(() => [
  { id: 'w₀', v: W0, c: 'text-dim', w: 2, dy: 14 },
  { id: 'LoRA', v: v.value, c: 'right', w: 2, dy: 16 },
  { id: 'DoRA', v: doraW.value, c: 'left', w: 3, dy: -8 },
])
const sg = (x) => (x >= 0 ? '+' : '') + x.toFixed(2)
</script>

<style scoped>
.t { font-size: 11px; fill: var(--text-muted); font-family: "SF Mono", Menlo, monospace; }
.star { font-size: 18px; fill: var(--warn); }
.cmp { width: 100%; border-collapse: collapse; font-size: 12px; }
.cmp th, .cmp td { border-bottom: 1px dashed var(--border); padding: 4px 6px; text-align: right; color: var(--text); }
.cmp th:first-child, .cmp td:first-child { text-align: left; color: var(--text-muted); }
.cmp th { color: var(--text-muted); font-weight: 500; }
</style>
