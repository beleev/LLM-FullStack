<!--
  残差流实验台: 把 "Pre-LN 把 norm 挪进分支" 这件事画出来, 并让它的后果 (各层梯度尺度) 可测。
  左边是一个 block 的真实接线, 右边是 L 层摞起来之后反传回来的梯度。切换 Pre/Post 时 norm 方块会真的搬家。
-->
<template>
  <LabFrame
    title="残差流 — norm 挪一个位置, 深层网络就能开训"
    sub="主干那条粗带子是残差流 x [B,T,D], 从头贯到尾。attention 和 FFN 都不在主干上, 它们从旁边接出去、算完再加回来。norm 放在哪, 决定了主干还是不是一条干净的加法通路 —— 也就决定了梯度能不能原样回到第 0 层。"
    module="llm_models/layers/core/blocks.py"
    run="python -m llm_models.run_models.language_models.llama.train_llama"
    :challenge="{
      ask: '把层数拖到 48, 两种接法第 0 层的梯度尺度各是多少? 为什么 Post-LN 那个年代的论文都在讲 warmup?',
      answer: 'Pre-LN 恒为 1.00 —— 主干是纯加法, 反传时恒等项把梯度原样送到底, 与层数无关。Post-LN 一层里有两个子层, 就要过两次 norm, 每次乘 1/√(1+v) —— v=1 时是 0.707。48 层连乘 94 次, 第 0 层只剩 7e-15。顶层梯度正常、底层几乎为零, 同一个学习率没法同时喂饱两头。warmup 就是拿前几千步把学习率压住, 等各层尺度自己长匀。Pre-LN 直接把这个问题消掉了, 所以现在没人再为它调 warmup。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: pre }" @click="pre = true">Pre-LN (norm 在分支上)</button>
        <button type="button" :class="{ active: !pre }" @click="pre = false">Post-LN (norm 在主干上)</button>
        <button type="button" @click="flow = !flow">{{ flow ? '看前向' : '看反传' }}</button>
      </div>
      <LabSlider v-model="L" label="堆多少层" :min="2" :max="48" />
      <LabSlider v-model="v" label="分支输出方差 v" :min="0.1" :max="3" :step="0.1"
                 :format="(x) => x.toFixed(1)" />
    </template>

    <svg :viewBox="`0 0 ${W} ${H}`" role="img" :aria-label="`${pre ? 'Pre-LN' : 'Post-LN'} block 的残差流接线图`">
      <defs>
        <marker id="ar" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 z" fill="var(--text-dim)" />
        </marker>
      </defs>

      <!-- 主干: 一条从头贯到尾的带子, 宽度代表 D -->
      <rect :x="MX - TRUNK / 2" y="30" :width="TRUNK" :height="H - 70" class="trunk" />
      <text :x="MX" y="20" class="lbl mid">残差流 x [B, T, D]</text>

      <!-- 每个子层: 主干 → (norm) → 零件 → 回到 + -->
      <template v-for="sl in sublayers" :key="sl.id">
        <path :d="sl.outPath" class="wire" marker-end="url(#ar)" />
        <path v-if="sl.midPath" :d="sl.midPath" class="wire" marker-end="url(#ar)" />
        <path :d="sl.inPath" class="wire" marker-end="url(#ar)" />
        <g class="plus"><circle :cx="MX" :cy="sl.addY" r="11" /><text :x="MX" :y="sl.addY + 4" class="mid">+</text></g>
      </template>

      <!-- 零件方块: norm 在 Pre-LN 下落在分支上, 在 Post-LN 下压在主干上 -->
      <g
        v-for="n in boxes" :key="n.id"
        :class="['box', n.cls, { on: hover === n.id }]" tabindex="0" role="button" :aria-label="n.title"
        @mouseenter="hover = n.id" @mouseleave="hover = null"
        @focus="hover = n.id" @blur="hover = null"
      >
        <rect :x="n.x" :y="n.y" :width="n.w" :height="n.h" rx="6" />
        <text :x="n.x + n.w / 2" :y="n.y + n.h / 2 + 4" class="box-t">{{ n.label }}</text>
      </g>

      <text :x="MX" :y="H - 10" :class="['verdict mid', pre ? 'good' : 'bad']">
        {{ pre ? '主干从头到尾只有加法 → 反传有一条恒等通路' : '主干被 norm 截断 2L 次 → 梯度每次都要被缩放' }}
      </text>
    </svg>

    <!-- L 层摞起来: 每层一根柱, 高度 = 那一层拿到的梯度尺度 -->
    <div class="depth">
      <div class="depth-head mono">{{ flow ? '前向: 主干方差逐层怎么走' : '反传: 第 l 层拿到的梯度尺度 (顶层 = 1)' }}</div>
      <div class="bars" :style="{ gridTemplateColumns: `repeat(${L}, 1fr)` }">
        <div
          v-for="(g, l) in series" :key="l" class="bar-slot"
          :title="`第 ${l} 层: ${fmt(g)}`"
        >
          <div class="bar" :style="{ height: `${barH(g)}%`, background: barColor(g) }" />
        </div>
      </div>
      <div class="axis mono"><span>第 0 层 (最靠近输入)</span><span>第 {{ L - 1 }} 层 (最靠近输出)</span></div>
    </div>

    <template #stats>
      <div class="kv"><span>第 0 层梯度尺度</span><b :class="pre ? 'good' : 'bad'">{{ fmt(series[0]) }}</b></div>
      <div class="kv"><span>顶层 / 底层之比</span><b :class="pre ? 'good' : 'bad'">{{ fmt(1 / series[0]) }}</b></div>
      <div class="kv"><span>每层缩放因子</span><b>{{ pre ? '1.000 (恒等)' : perLayer.toFixed(3) }}</b></div>
      <div v-if="hover" class="kv-note">
        <b>{{ nodeBy(hover).title }}</b>
        <p>{{ nodeBy(hover).desc }}</p>
      </div>
      <p v-else class="lab-note">把鼠标放到任意一个方块上, 看它在这条流水线里干什么。</p>
      <p class="lab-note">
        这里的梯度用一个简化模型: 每过一次主干上的 norm, 梯度乘 1/√(1+v)，v 是分支输出相对主干的方差。
        真实网络里每层的 v 不一样, 但"Post-LN 要连乘 L 个因子、Pre-LN 有一条恒等通路"这一点是一样的。
      </p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { range } from '@/utils/labmath.js'

const pre = ref(true)
const flow = ref(false)
const L = ref(24)
const v = ref(1)
const hover = ref(null)

const W = 580, H = 340, MX = 190, TRUNK = 46
const BW = 132, BH = 34, BX = MX + 96          // 分支上的零件统一靠右摆

// 两个子层的纵向布局: 从主干哪里出去、加号在哪
const LAYOUT = [
  { id: 's1', op: 'attn', outY: 58, addY: 148 },
  { id: 's2', op: 'ffn', outY: 196, addY: 286 },
]
const META = {
  attn: { pre: 'GQA', post: 'GQA', cls: 'attn', title: 'attention',
    desc: '一个 block 里唯一让不同位置互相说话的地方。进去 [B,T,D], 出来还是 [B,T,D]。' },
  ffn: { pre: 'SwiGLU', post: 'SwiGLU', cls: 'ffn', title: 'FFN',
    desc: '逐位置独立的两层 MLP, 中间宽到 d_ff。一个 block 的参数大头在这里, LLaMA 里约占三分之二。' },
}
const NORM_DESC = {
  pre: { title: 'norm 在分支上 (Pre-LN)', desc: '只归一化送进子层的那一份副本。主干上的 x 一个字没动, 直接往下走。' },
  post: { title: 'norm 在主干上 (Post-LN)', desc: '先把子层输出加回主干, 再归一化整条主干 —— 主干从此不是纯加法, 反传要过这一道。' },
}

// 方块: Pre-LN 下 norm 在分支 (零件上方); Post-LN 下 norm 压在主干、在加号下方
const boxes = computed(() =>
  LAYOUT.flatMap((sl) => {
    const m = META[sl.op]
    const opBox = { id: sl.op, label: m[pre.value ? 'pre' : 'post'], cls: m.cls, title: m.title, desc: m.desc,
      x: BX, y: sl.outY + (pre.value ? 46 : 4), w: BW, h: BH }
    const nd = NORM_DESC[pre.value ? 'pre' : 'post']
    const normBox = pre.value
      ? { id: sl.id + 'n', label: 'RMSNorm', cls: 'norm', ...nd, x: BX, y: sl.outY - 6, w: BW, h: BH }
      : { id: sl.id + 'n', label: 'LayerNorm', cls: 'norm', ...nd, x: MX - BW / 2, y: sl.addY + 18, w: BW, h: BH }
    return [normBox, opBox]
  }))

// 走线: 出主干 → 竖到零件 → 进左边; 零件底 → 竖下来 → 回到加号
const sublayers = computed(() =>
  LAYOUT.map((sl) => {
    const opY = sl.outY + (pre.value ? 46 : 4)
    const enterY = pre.value ? sl.outY + BH / 2 - 6 : opY + BH / 2   // Pre-LN 先进 norm, Post-LN 直接进零件
    const x0 = MX + TRUNK / 2, xb = BX - 14, cx = BX + BW / 2
    return {
      id: sl.id,
      outPath: `M${x0},${sl.outY} H${xb} V${enterY} H${BX - 2}`,
      // Pre-LN: norm 的输出再喂给零件, 这一段要画出来
      midPath: pre.value ? `M${cx},${sl.outY - 6 + BH} V${opY - 2}` : null,
      inPath: `M${cx},${opY + BH} V${sl.addY} H${MX + 13}`,
      addY: sl.addY,
    }
  }))
const nodeBy = (id) => boxes.value.find((n) => n.id === id) || {}

// ── 梯度尺度: Pre-LN 主干是恒等; Post-LN 每层过两次 norm ──────────────
const perLayer = computed(() => 1 / Math.sqrt(1 + v.value))
const series = computed(() =>
  range(L.value).map((l) => {
    const above = L.value - 1 - l                 // 它上面还有多少层要穿过
    if (flow.value) return pre.value ? 1 + v.value * (l + 1) : 1   // 前向: Pre-LN 主干方差逐层累加
    return pre.value ? 1 : perLayer.value ** (2 * above)           // 每层两个子层 = 两次 norm
  }))

const fmt = (x) =>
  x === 0 ? '0' : x >= 1000 || x < 0.001 ? x.toExponential(1) : x.toFixed(x < 1 ? 3 : 2)
// 对数标高: 线性画的话 1e-8 和 0 看不出区别
const barH = (g) => {
  const vals = series.value.filter((x) => x > 0)
  const lo = Math.min(...vals), hi = Math.max(...vals)
  if (hi / lo < 1.01) return 100                    // 各层一样高 → 满格, 一眼看出"没有失衡"
  return Math.max(2, (Math.log(g) - Math.log(lo)) / (Math.log(hi) - Math.log(lo)) * 100)
}
const barColor = (g) => (flow.value ? 'var(--eye)' : g > 0.2 ? 'var(--left)' : g > 1e-3 ? 'var(--warn)' : 'var(--danger)')
</script>

<style scoped>
.trunk { fill: var(--accent-soft); stroke: var(--accent); stroke-width: 1; }
.wire { stroke: var(--text-dim); stroke-width: 1.5; fill: none; }
.box rect { fill: var(--bg-elev); stroke: var(--border-strong); stroke-width: 1.5; cursor: pointer; }
.box.norm rect { stroke: var(--warn); }
.box.attn rect { stroke: var(--accent); }
.box.ffn rect { stroke: var(--left); }
.box.on rect { fill: var(--bg-card); stroke-width: 3; }
.box:focus-visible rect { stroke: var(--warn); stroke-width: 3; }
.box-t { fill: var(--text); font-size: 12px; text-anchor: middle; font-family: "SF Mono", Menlo, monospace; pointer-events: none; }
.plus circle { fill: var(--bg); stroke: var(--accent); stroke-width: 2; }
.plus text { fill: var(--accent); font-size: 15px; text-anchor: middle; }
.lbl { fill: var(--text-dim); font-size: 11px; font-family: "SF Mono", Menlo, monospace; }
.mid { text-anchor: middle; }
.verdict { font-size: 11.5px; }
.verdict.good { fill: var(--left); }
.verdict.bad { fill: var(--danger); }

.depth { margin-top: 14px; }
.depth-head { font-size: 11px; color: var(--text-dim); margin-bottom: 6px; }
.bars { display: grid; gap: 1px; height: 84px; align-items: end; background: var(--code-bg); padding: 6px; border-radius: var(--radius-sm); }
.bar-slot { height: 100%; display: flex; align-items: flex-end; }
.bar { width: 100%; border-radius: 1px 1px 0 0; }
.axis { display: flex; justify-content: space-between; font-size: 10px; color: var(--text-dim); margin-top: 4px; }
.kv-note { border-left: 2px solid var(--accent); padding-left: 10px; }
.kv-note b { font-size: 13px; color: var(--text); }
.kv-note p { font-size: 12px; color: var(--text-muted); line-height: 1.6; margin-top: 3px; }
</style>
