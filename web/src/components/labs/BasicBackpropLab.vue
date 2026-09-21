<!--
  反向传播实验台 (对应 llm_basic/model.py 的 linear_backward / cross_entropy_forward_backward)。
  一条最短的链: x, W → z = W·x → p = softmax(z) → L = -log p[y]。
  点节点看 "上游梯度 × 本地导数 = 本节点梯度", 链式法则只有这一句话。
-->
<template>
  <LabFrame
    title="反向传播 — 点一个节点, 看梯度怎么传到它"
    sub="前向从左到右算出 loss (灰色数字), 反向从右到左把 ∂L/∂· 传回来 (粉色数字)。点任意节点: 右侧会把它的梯度拆成 '上游梯度 × 本地导数'。拖 x、换目标类、走几步 SGD, 看每个数怎么跟着变。"
    module="llm_basic/model.py"
    run="python llm_basic/gradcheck.py"
    :challenge="{
      ask: '把目标类 y 换成模型当前概率最低的那一类, dz (logits 的梯度) 里哪一项绝对值最大? 为什么 softmax+CE 的梯度永远是 p − onehot 这么干净?',
      answer: 'dz = p − onehot(y): 目标类那一项是 p_y − 1, p_y 越小它越接近 −1, 绝对值最大 —— 错得越离谱推得越狠。单看 CE 的本地导数是 −1/p_y (p_y 小时会爆炸), 单看 softmax 的雅可比是 diag(p) − ppᵀ (很麻烦); 两者一乘, 1/p_y 正好被约掉。这就是 llm_basic 把 softmax 和 CE 合并成一个 forward_backward 函数的原因: 又简单又数值稳定。',
    }"
  >
    <template #controls>
      <LabSlider v-model="x1" label="输入 x₁" :min="-2" :max="2" :step="0.1" />
      <LabSlider v-model="x2" label="输入 x₂" :min="-2" :max="2" :step="0.1" />
      <div class="row">
        <span class="hint">目标类 y:</span>
        <button v-for="c in 3" :key="c" type="button" :class="{ active: y === c - 1 }" @click="y = c - 1">类 {{ c - 1 }}</button>
        <button type="button" @click="sgd">走一步 SGD (lr = 0.5)</button>
        <button type="button" @click="reset">重置 W</button>
        <span class="hint mono">已走 {{ steps }} 步</span>
      </div>
    </template>

    <svg viewBox="0 0 640 250" role="group" aria-label="计算图">
      <defs>
        <marker id="bp-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" class="arr" /></marker>
      </defs>
      <line v-for="e in edges" :key="e.id" :x1="e.x1" :y1="e.y1" :x2="e.x2" :y2="e.y2" class="edge" :class="{ back: e.back }" marker-end="url(#bp-arr)" />
      <g
        v-for="n in nodes" :key="n.id" class="node" :class="{ sel: sel === n.id }"
        tabindex="0" role="button" :aria-label="`节点 ${n.label}`" :aria-pressed="sel === n.id"
        @click="sel = n.id" @keydown.enter="sel = n.id"
      >
        <rect :x="n.x - 62" :y="n.y - 36" width="124" height="72" rx="8" />
        <text :x="n.x" :y="n.y - 18" class="nm">{{ n.label }}</text>
        <text :x="n.x" :y="n.y + 2" class="fw mono">{{ n.fw }}</text>
        <text :x="n.x" :y="n.y + 22" class="bw mono">{{ n.bw }}</text>
      </g>
      <text x="320" y="238" class="cap">灰 = 前向值 · 粉 = ∂L/∂该节点 · 粉色边 = 梯度传到所选节点走过的路</text>
    </svg>

    <template #stats>
      <div class="kv"><span>loss = −log p[y]</span><b :class="g.L < 0.3 ? 'good' : ''">{{ g.L.toFixed(3) }}</b></div>
      <div class="kv"><span>目标类概率 p[y]</span><b>{{ g.p[y].toFixed(3) }}</b></div>
      <div class="kv"><span>dW[0][0] 解析 / 数值</span><b :class="relErr < 1e-6 ? 'good' : 'bad'">{{ g.dW[0][0].toFixed(4) }} / {{ numeric.toFixed(4) }}</b></div>
      <div class="detail">
        <b>{{ info.title }}</b>
        <p><span class="k">上游梯度</span><span class="mono">{{ info.up }}</span></p>
        <p><span class="k">本地导数</span><span class="mono">{{ info.local }}</span></p>
        <p><span class="k">相乘得到</span><span class="mono res">{{ info.out }}</span></p>
      </div>
      <p class="lab-note">{{ info.note }}</p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { softmax } from '@/utils/labmath.js'

const W0 = [[0.6, -0.4], [-0.3, 0.8], [0.2, 0.1]]
const W = ref(W0.map((r) => [...r]))
const x1 = ref(1), x2 = ref(-0.5), y = ref(1), sel = ref('z'), steps = ref(0)

// 前向 + 反向一次算完, 和 llm_basic 的 forward/backward 成对函数是同一件事
const run = (Wm, x, yi) => {
  const z = Wm.map((r) => r[0] * x[0] + r[1] * x[1])
  const p = softmax(z)
  const L = -Math.log(p[yi])
  const dp = p.map((_, i) => (i === yi ? -1 / p[yi] : 0))
  const dz = p.map((pi, i) => pi - (i === yi ? 1 : 0))         // ★ softmax+CE 合并后的梯度: p − onehot
  const dW = dz.map((d) => [d * x[0], d * x[1]])               // 外积 dz ⊗ x
  const dx = [0, 1].map((j) => Wm.reduce((s, r, i) => s + r[j] * dz[i], 0)) // Wᵀ dz
  return { z, p, L, dp, dz, dW, dx }
}
const x = computed(() => [x1.value, x2.value])
const g = computed(() => run(W.value, x.value, y.value))

// 中心差分验一个元素 —— gradcheck 的最小版本
const numeric = computed(() => {
  const eps = 1e-5
  const at = (d) => run(W.value.map((r, i) => (i === 0 ? [r[0] + d, r[1]] : r)), x.value, y.value).L
  return (at(eps) - at(-eps)) / (2 * eps)
})
const relErr = computed(() => Math.abs(numeric.value - g.value.dW[0][0]) / Math.max(1e-12, Math.abs(numeric.value) + Math.abs(g.value.dW[0][0])))

const sgd = () => { const d = g.value.dW; W.value = W.value.map((r, i) => r.map((w, j) => w - 0.5 * d[i][j])); steps.value++ }
const reset = () => { W.value = W0.map((r) => [...r]); steps.value = 0 }

const v = (a) => '[' + a.map((t) => t.toFixed(2)).join(', ') + ']'
const nodes = computed(() => [
  { id: 'x', label: '输入 x', x: 80, y: 50, fw: v(x.value), bw: v(g.value.dx) },
  { id: 'W', label: '权重 W (3×2)', x: 80, y: 170, fw: '第0行 ' + v(W.value[0]), bw: '第0行 ' + v(g.value.dW[0]) },
  { id: 'z', label: 'z = W·x (logits)', x: 250, y: 110, fw: v(g.value.z), bw: v(g.value.dz) },
  { id: 'p', label: 'p = softmax(z)', x: 410, y: 110, fw: v(g.value.p), bw: v(g.value.dp) },
  { id: 'L', label: 'L = −log p[y]', x: 565, y: 110, fw: g.value.L.toFixed(3), bw: '1' },
])
// 梯度从 L 传到所选节点经过的边
const PATH = { L: [], p: ['pL'], z: ['pL', 'zp'], x: ['pL', 'zp', 'xz'], W: ['pL', 'zp', 'Wz'] }
const edges = computed(() => [
  { id: 'xz', x1: 142, y1: 62, x2: 188, y2: 96 }, { id: 'Wz', x1: 142, y1: 158, x2: 188, y2: 124 },
  { id: 'zp', x1: 312, y1: 110, x2: 346, y2: 110 }, { id: 'pL', x1: 472, y1: 110, x2: 501, y2: 110 },
].map((e) => ({ ...e, back: PATH[sel.value].includes(e.id) })))

const info = computed(() => {
  const G = g.value
  return {
    L: { title: 'L: 反向的起点', up: '—', local: '∂L/∂L = 1', out: '1', note: '反向传播永远从标量 loss 对自己的导数 1 出发。' },
    p: { title: 'p ← L', up: '1', local: `∂L/∂p[y] = −1/p[y] = ${(-1 / G.p[y.value]).toFixed(2)}`, out: v(G.dp), note: '只有目标类那一项有梯度; p[y] 越小, −1/p[y] 越大 —— 单独算这一步数值上很危险。' },
    z: { title: 'z ← p', up: v(G.dp), local: 'Jᵀ, J = diag(p) − ppᵀ', out: v(G.dz) + ' = p − onehot', note: 'softmax 的每个输出依赖所有输入, 所以本地导数是一个矩阵 (耦合项)。乘上上游后化简成 p − onehot, 1/p[y] 被约掉了。' },
    W: { title: 'W ← z', up: v(G.dz), local: `∂z_i/∂W_ij = x_j = ${v(x.value)}`, out: 'dz ⊗ x, 第0行 ' + v(G.dW[0]), note: 'dW = dz ⊗ x: 这就是 linear_backward 里的 x.T @ dout。把 x 拖到 0, dW 整个变 0 —— 输入为 0 的权重学不到东西。' },
    x: { title: 'x ← z', up: v(G.dz), local: '∂z/∂x = W', out: 'Wᵀ·dz = ' + v(G.dx), note: 'dx = Wᵀ dz: 这份梯度会继续往更前面的层传, 多层网络就是把这一步重复 n_layer 次。' },
  }[sel.value]
})
</script>

<style scoped>
.hint { font-size: 12px; color: var(--text-muted); }
.edge { stroke: var(--border-strong); stroke-width: 1.5; }
.edge.back { stroke: var(--right); stroke-width: 2.5; }
.arr { fill: var(--text-dim); }
.node { cursor: pointer; outline: none; }
.node rect { fill: var(--bg-elev); stroke: var(--border-strong); stroke-width: 1.2; }
.node.sel rect, .node:focus-visible rect { stroke: var(--accent); stroke-width: 2.5; fill: var(--accent-soft); }
.node text { text-anchor: middle; font-size: 11px; }
.nm { fill: var(--text); font-weight: 600; }
.fw { fill: var(--text-muted); }
.bw { fill: var(--right); }
.cap { text-anchor: middle; font-size: 11px; fill: var(--text-dim); }
.detail { border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 8px 10px; display: grid; gap: 4px; }
.detail b { color: var(--text); font-size: 13px; }
.detail p { display: grid; grid-template-columns: 64px 1fr; gap: 6px; }
.detail .k { color: var(--text-dim); }
.detail .res { color: var(--right); }
</style>
