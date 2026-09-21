<!--
  树形投机实验台 (对应 llm_infer/m19_tree_speculation/tree_spec.py)。
  一件事: 每个节点留 draft 的 top-w 候选长成树, tree attention mask (只看祖先 + 自己) 让 target 一次 forward 验完所有分支。
-->
<template>
  <LabFrame
    title="树形投机 — tree attention mask 一次验整棵树"
    sub="用 +/− 捏出一棵 draft 树 (BFS 编号, 与 tree_shape 一致)。左: 树, 绿色 = 本轮被 target 接受的路径; 右: tree mask。悬停 / 点击任一节点或 mask 的任一行, 看它能看见谁。"
    module="llm_infer/m19"
    run="python -m llm_infer.m19_tree_speculation.demo"
    :challenge="{
      ask: '默认树 [3,2,1] 有 16 个节点。同样验 16 个 token 的链 (K=15) 和它比, 谁每轮接受得多? 把 top-1 命中率拖到 0.9 再比一次。为什么同一层的兄弟节点 RoPE 位置相同?',
      answer: 'p1=0.5 时树 ≈1.60、链 K=15 ≈1.00: 链第一个猜错后面 14 个全废, 树在最容易错的第一层留了 3 个备胎 (命中率 0.5 → 0.78)。p1=0.9 时反过来, 链 ≈7.1 远超树的 2.90 (树最深只有 3 层) —— draft 足够准时深度比宽度值钱, 所以 EAGLE-2 按置信度动态长树。兄弟节点是同一个位置的不同候选, 最终只有一个会留下, 所以位置都是 n_ctx + depth; mask 保证它们互相看不见, 每个节点看到的恰好是从根到自己的那条链, 等价于每条路径各跑一次顺序 decode。',
    }"
  >
    <template #controls>
      <div v-for="(w, d) in widths" :key="d" class="row">
        <span class="wl">第 {{ d + 1 }} 层每节点孩子数</span>
        <button type="button" :disabled="w <= 1" :aria-label="`第 ${d + 1} 层减少`" @click="widths[d]--">−</button>
        <b class="mono wv">{{ w }}</b>
        <button type="button" :disabled="w >= 3" :aria-label="`第 ${d + 1} 层增加`" @click="widths[d]++">+</button>
        <button v-if="d === widths.length - 1 && d > 0" type="button" @click="widths.pop()">删掉这层</button>
      </div>
      <div class="row">
        <button type="button" :disabled="widths.length >= 3" @click="widths.push(1)">加一层</button>
        <button type="button" @click="seed++">换一组 (重新抽 target 的选择)</button>
      </div>
      <LabSlider v-model="p1" label="draft top-1 命中率 p1" :min="0.1" :max="0.95" :step="0.05" :format="(t) => t.toFixed(2)" />
    </template>

    <div class="pair">
      <svg :viewBox="`0 0 360 ${60 + widths.length * 56}`" class="tree" role="group" aria-label="draft token 树">
        <line v-for="i in tree.n - 1" :key="`e${i}`" :x1="pos[tree.parents[i]].x" :y1="pos[tree.parents[i]].y" :x2="pos[i].x" :y2="pos[i].y"
          :class="['edge', { ok: onPath(i) }]" />
        <g v-for="(p, i) in pos" :key="i" tabindex="0" role="button" :aria-label="`节点 ${i}`" class="node"
          :class="{ ok: onPath(i), anc: focus >= 0 && tree.anc[focus][i], me: focus === i }"
          @mouseenter="hover = i" @mouseleave="hover = -1" @focus="hover = i" @blur="hover = -1"
          @click="pin = pin === i ? -1 : i" @keydown.enter="pin = pin === i ? -1 : i">
          <circle :cx="p.x" :cy="p.y" :r="R" />
          <text v-if="tree.n <= 22" :x="p.x" :y="p.y + 3">{{ i }}</text>
        </g>
      </svg>
      <svg :viewBox="`0 0 ${(NCTX + tree.n) * C + 4} ${tree.n * C + 4}`" class="mask" role="group" aria-label="tree attention mask">
        <g v-for="i in tree.n" :key="i" @mouseenter="hover = i - 1" @mouseleave="hover = -1" @click="pin = pin === i - 1 ? -1 : i - 1">
          <rect v-for="j in NCTX" :key="`c${j}`" :x="(j - 1) * C + 2" :y="(i - 1) * C + 2" :width="C - 1" :height="C - 1" class="m ctx" :class="{ row: focus === i - 1 }" />
          <rect v-for="j in tree.n" :key="j" :x="(NCTX + j - 1) * C + 2" :y="(i - 1) * C + 2" :width="C - 1" :height="C - 1"
            class="m" :class="{ vis: tree.anc[i - 1][j - 1], row: focus === i - 1 }" />
        </g>
      </svg>
    </div>
    <p class="read mono">{{ readout }}</p>

    <template #stats>
      <div class="kv"><span>1 次 target 调用验证节点数</span><b>{{ tree.n }}</b></div>
      <div class="kv"><span>本轮接受 (+1 个 target token)</span><b>{{ path.length - 1 }} + 1</b></div>
      <div class="kv"><span>树: 期望接受 (公式 / 模拟)</span><b :class="eTree >= chain(tree.n - 1) ? 'good' : ''">{{ eTree.toFixed(2) }} / {{ emp.toFixed(2) }}</b></div>
      <div class="kv"><span>链, 同深度 K={{ widths.length }}</span><b>{{ chain(widths.length).toFixed(2) }}</b></div>
      <div class="kv"><span>链, 同验证量 K={{ tree.n - 1 }}</span><b :class="chain(tree.n - 1) > eTree ? 'good' : ''">{{ chain(tree.n - 1).toFixed(2) }}</b></div>
      <p class="lab-note">
        玩具模型 (每层独立): target 的 token 落在 draft 第 r 名的概率 = p1·0.4^(r−1) (总和截到 1)。留 w 个孩子的命中率
        H(w) = 前 w 名之和, 期望接受 = Σ_d Π_(i≤d) H(w_i)。同验证量的链要 {{ tree.n - 1 }} 次 draft 调用, 树只要 {{ widths.length }} 次。
        多验的 token 在 decode 带宽受限时几乎免费, 大 batch 算力受限时不免费。真实 demo: 树 [3,2,1] 2.58 token/target 调用, 链 K=3 1.86。
      </p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, reactive, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { mulberry32, range, sum } from '@/utils/labmath.js'

const NCTX = 3, ROUNDS = 20000, RHO = 0.4
const widths = reactive([3, 2, 1])
const p1 = ref(0.5), seed = ref(1), hover = ref(-1), pin = ref(-1)

// 与 tree_shape / ancestor_matrix 逐行对应: BFS 编号, 节点 0 = 根 (已确定的 out[-1])
const tree = computed(() => {
  const parents = [-1], depth = [0]
  let level = [0]
  widths.forEach((w, d) => {
    const nxt = []
    for (const p of level) for (let c = 0; c < w; c++) { nxt.push(parents.length); parents.push(p); depth.push(d + 1) }
    level = nxt
  })
  const n = parents.length
  const anc = range(n).map((i) => range(n).map((j) => i === j))
  for (let i = 1; i < n; i++) anc[parents[i]].forEach((v, j) => { if (v) anc[i][j] = true }) // ★ anc[i] |= anc[parent]: 父先于子, 一遍即闭包
  const kids = range(n).map((i) => range(n).filter((j) => parents[j] === i))
  return { parents, depth, n, anc, kids }
})
const focus = computed(() => { const f = hover.value >= 0 ? hover.value : pin.value; return f < tree.value.n ? f : -1 })

// 命中率: 第 r 名 = p1·ρ^(r−1), 总和截到 1
const hits = computed(() => { let left = 1; return range(3).map((r) => { const h = Math.min(p1.value * RHO ** r, left); left -= h; return h }) })
const H = (w) => sum(hits.value.slice(0, w))
const eTree = computed(() => { let prod = 1, e = 0; for (const w of widths) { prod *= H(w); e += prod } return e })
const chain = (k) => sum(range(k).map((d) => p1.value ** (d + 1)))

// 一轮: 从根往下, target 的 token 是第几名 → 有这个孩子就进去, 否则断 (同 accept_tree)
const walk = (rand) => {
  const path = [0]
  for (const w of widths) {
    const u = rand()
    let r = 0, cum = hits.value[0]
    while (r < 3 && u >= cum) { r++; cum += hits.value[r] ?? 0 }
    if (r >= w) break
    path.push(tree.value.kids[path[path.length - 1]][r])
  }
  return path
}
const path = computed(() => walk(mulberry32(seed.value * 101 + 7)))
const emp = computed(() => { const rand = mulberry32(seed.value * 977 + 3); let t = 0; for (let i = 0; i < ROUNDS; i++) t += walk(rand).length - 1; return t / ROUNDS })
const onPath = (i) => path.value.includes(i)

// 画树: 叶子均匀铺开, 内部节点取孩子中点
const R = computed(() => (tree.value.n > 22 ? 5 : 9)), C = computed(() => (tree.value.n > 22 ? 8 : 14))
const pos = computed(() => {
  const { n, kids, depth } = tree.value, xs = Array(n).fill(0)
  const leaves = range(n).filter((i) => !kids[i].length)
  leaves.forEach((i, k) => { xs[i] = 20 + (320 * (k + 0.5)) / leaves.length })
  for (let i = n - 1; i >= 0; i--) if (kids[i].length) xs[i] = sum(kids[i].map((k) => xs[k])) / kids[i].length
  return range(n).map((i) => ({ x: xs[i], y: 24 + depth[i] * 56 }))
})
const readout = computed(() => {
  const i = focus.value, t = tree.value
  if (i < 0) return `悬停一个节点: 看它的 mask 行。mask 前 ${NCTX} 列是上下文 (全可见), 后 ${t.n} 列是树节点。`
  const a = range(t.n).filter((j) => t.anc[i][j] && j !== i)
  return `节点 ${i}: 深度 ${t.depth[i]} → RoPE 位置 = n_ctx + depth = ${NCTX} + ${t.depth[i]} = ${NCTX + t.depth[i]}; 可见 = 上下文 + 祖先 [${a.join(', ')}] + 自己 (共 ${NCTX + a.length + 1} 列)`
})
</script>

<style scoped>
.wl { font-size: 12px; color: var(--text-muted); min-width: 130px; }
.wv { min-width: 16px; text-align: center; color: var(--accent); }
.pair { display: flex; flex-wrap: wrap; gap: 16px; align-items: flex-start; }
.tree { flex: 1 1 260px; min-width: 220px; }
.mask { flex: 0 1 250px; min-width: 180px; }
.edge { stroke: var(--border-strong); stroke-width: 1.2; }
.edge.ok { stroke: var(--left); stroke-width: 2.5; }
.node { cursor: pointer; outline: none; }
.node circle { fill: var(--bg-elev); stroke: var(--border-strong); stroke-width: 1.2; }
.node.ok circle { fill: color-mix(in srgb, var(--left) 30%, transparent); stroke: var(--left); }
.node.anc circle { stroke: var(--accent); stroke-width: 2.5; }
.node.me circle, .node:focus-visible circle { stroke: var(--warn); stroke-width: 3; }
.node text { font-size: 9px; fill: var(--text); text-anchor: middle; pointer-events: none; }
.m { fill: var(--bg-elev); stroke: var(--border); stroke-width: 0.5; }
.m.ctx { fill: color-mix(in srgb, var(--text-dim) 35%, transparent); }
.m.vis { fill: var(--accent); }
.m.row { stroke: var(--warn); stroke-width: 1.5; }
.read { font-size: 11px; color: var(--text-muted); margin-top: 8px; line-height: 1.6; }
</style>
