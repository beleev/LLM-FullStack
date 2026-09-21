<!--
  LoRA 低秩补丁实验台 (对应 llm_finetune/methods/lora.py)。
  只讲一件事: 微调需要的 ΔW 本身就接近低秩, 所以用 r 很小的 B·A 去拟合它, 误差小、参数省;
  训练完 merge 回 W, 推理零开销。
  目标 ΔW* 由已知奇异值谱构造 (U·diag(s)·Vᵀ), 秩 r 的最优近似就是截断 SVD (Eckart–Young), 误差有闭式解。
-->
<template>
  <LabFrame
    title="LoRA — 用 B·A 拟合一个低秩的 ΔW"
    sub="左边是冻结的 W₀, 右边是「全参微调本来想学到的」ΔW*。拖秩 r: B (d×r) 和 A (r×d) 的乘积就是秩 r 下对 ΔW* 的最优近似。悬停 ΔW 的任意一格, 看它由 B 的哪一行和 A 的哪一列相乘得到。"
    module="llm_finetune/methods/lora.py"
    run="python -m llm_finetune.run_finetune.lora.train_lora"
    :challenge="{
      ask: '把「谱衰减」拖到 0.95 (ΔW* 几乎满秩) 再拖到 0.4 (ΔW* 很低秩), 同样 r = 2 时误差差多少? 这说明 LoRA 成立的前提是什么?',
      answer: 'LoRA 能用 r = 8 替代 d = 4096 的全量更新, 不是因为低秩分解有魔法。而是因为「微调带来的权重变化本身内在维度就很低」—— 这是 Aghajanyan 2020 / LoRA 论文的经验发现。谱衰减快时, 前几个奇异值已经占了几乎全部能量, r = 2 误差就只剩几个百分点。谱很平时, 误差 = sqrt(被丢掉的 s² 之和 / 全部 s² 之和), 丢多少亏多少。任务离预训练分布越远 (例如学一门新语言), ΔW 越不低秩, 这时就该加大 r 或改用全参。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: !merged }" @click="merged = false">训练态: W₀ 与 B·A 分开</button>
        <button type="button" :class="{ active: merged }" @click="merged = true">merge: W′ = W₀ + (α/r)·B·A</button>
        <button type="button" @click="seed++">换一组矩阵</button>
      </div>
      <LabSlider v-model="r" label="秩 r" :min="1" :max="D" />
      <LabSlider v-model="decay" label="ΔW* 谱衰减" :min="0.3" :max="0.95" :step="0.05" :format="(v) => v.toFixed(2)" />
      <LabSlider v-model="alpha" label="α (缩放分子)" :min="1" :max="32" />
      <LabSlider v-model="logd" label="真实层宽 d (算参数用)" :min="9" :max="13" :format="(v) => String(2 ** v)" />
    </template>

    <div class="mats" @mouseleave="hov = null">
      <template v-for="(m, k) in shown" :key="m.name">
        <span v-if="m.op" class="op mono">{{ m.op }}</span>
        <figure class="mat">
          <div class="cells hm" :style="{ gridTemplateColumns: `repeat(${m.data[0].length}, 12px)` }">
            <template v-for="(row, i) in m.data" :key="i">
              <span
                v-for="(v, j) in row" :key="j" class="px"
                :class="{ hl: isHl(m.name, i, j) }"
                :style="{ background: tint(v, m.max) }"
                @mouseenter="m.name === 'ΔW' && (hov = { i, j })"
              />
            </template>
          </div>
          <figcaption class="mono">{{ m.name }} <i>{{ m.shape }}</i><br><em>{{ m.note }}</em></figcaption>
        </figure>
      </template>
    </div>
    <div class="spec">
      <span class="mono cap">ΔW* 的奇异值谱 (亮的 = 被 r 保留)</span>
      <div class="bars">
        <i v-for="(s, k) in sv" :key="k" :class="{ keep: k < r }" :style="{ height: s * 100 + '%' }" :title="`s${k} = ${s.toFixed(3)}`" />
      </div>
    </div>

    <template #stats>
      <div class="kv"><span>可训参数 2·d·r</span><b>{{ fmtNum(2 * dReal * r) }}</b></div>
      <div class="kv"><span>占全量 d² 的</span><b class="good">{{ (200 * r / dReal).toFixed(2) }}%</b></div>
      <div class="kv"><span>重构相对误差</span><b :class="err < 0.1 ? 'good' : err > 0.3 ? 'bad' : ''">{{ (err * 100).toFixed(1) }}%</b></div>
      <div class="kv"><span>缩放 α/r</span><b>{{ (alpha / r).toFixed(2) }}</b></div>
      <div class="kv"><span>推理额外 matmul</span><b :class="merged ? 'good' : ''">{{ merged ? 0 : 2 }}</b></div>
      <p class="lab-note">
        <template v-if="merged">
          merge 后就是一个普通 Linear: 对同一个 x, 两种算法的输出差 max|Δy| = {{ mergeDiff.toExponential(1) }} (浮点舍入)。
        </template>
        <template v-else>
          前向 y = W₀x + (α/r)·B(Ax): 多两次小 matmul。α/r 让你换 r 时不必重调学习率 —— r 翻倍, 每个秩分量的贡献自动减半。
        </template>
      </p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { fmtNum, mulberry32, randn, range } from '@/utils/labmath.js'

const D = 12
const r = ref(2), decay = ref(0.55), alpha = ref(16), logd = ref(12), seed = ref(1)
const merged = ref(false)
const hov = ref(null)
const dReal = computed(() => 2 ** logd.value)

// 随机正交基: 高斯向量做 Gram-Schmidt。cols[k] 是第 k 个单位向量
const orthoBasis = (rand) => {
  const cols = []
  for (let k = 0; k < D; k++) {
    let v = range(D).map(() => randn(rand))
    for (const u of cols) { const p = dot(u, v); v = v.map((x, i) => x - p * u[i]) }
    const n = Math.hypot(...v)
    cols.push(v.map((x) => x / n))
  }
  return cols
}
const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0)

const base = computed(() => {
  const rand = mulberry32(seed.value * 7919)
  return { U: orthoBasis(rand), V: orthoBasis(rand), W0: range(D).map(() => range(D).map(() => randn(rand) * 0.5)) }
})
const sv = computed(() => range(D).map((k) => decay.value ** k))

// 取前 n 个奇异分量求和: Σ s_k · u_k v_kᵀ
const lowRank = (n) => range(D).map((i) => range(D).map((j) =>
  range(n).reduce((s, k) => s + sv.value[k] * base.value.U[k][i] * base.value.V[k][j], 0)))

const target = computed(() => lowRank(D))
const delta = computed(() => lowRank(r.value))            // ★ 秩 r 的最优近似 = 截断 SVD
// 写成 LoRA 的参数形式: ΔW = (α/r)·B·A, 取 A = V_rᵀ, B = U_r·S_r·(r/α)
const A = computed(() => range(r.value).map((k) => base.value.V[k]))
const B = computed(() => range(D).map((i) => range(r.value).map((k) => base.value.U[k][i] * sv.value[k] * r.value / alpha.value)))
// 闭式误差: sqrt(丢掉的 s² / 全部 s²), 与逐元素相减算出来的 Frobenius 误差一致
const err = computed(() => Math.sqrt(sv.value.slice(r.value).reduce((s, x) => s + x * x, 0) / sv.value.reduce((s, x) => s + x * x, 0)))

const Wm = computed(() => base.value.W0.map((row, i) => row.map((w, j) => w + delta.value[i][j])))
const mergeDiff = computed(() => {
  const x = range(D).map((i) => Math.sin(i + 1)), s = alpha.value / r.value
  const Ax = A.value.map((a) => dot(a, x))
  const y1 = base.value.W0.map((w, i) => dot(w, x) + s * dot(B.value[i], Ax))
  const y2 = Wm.value.map((w) => dot(w, x))
  return Math.max(...y1.map((v, i) => Math.abs(v - y2[i])))
})

const mk = (name, data, note, op = '') => ({
  name, data, note, op, shape: `${data.length}×${data[0].length}`, max: Math.max(...data.flat().map(Math.abs), 1e-9),
})
const shown = computed(() => (merged.value
  ? [mk('W′', Wm.value, '一个普通矩阵, 补丁消失'), mk('ΔW*', target.value, '想学到的更新', '　')]
  : [
    mk('W₀', base.value.W0, '冻结, 不更新'), mk('B', B.value, '可训, 初始化为 0', '+'), mk('A', A.value, '可训, 随机初始化', '×'),
    mk('ΔW', delta.value.map((row) => row.slice()), '(α/r)·B·A', '='), mk('ΔW*', target.value, '想学到的更新', '≈'),
  ]))

// 悬停 ΔW[i][j] → 高亮 B 的第 i 行、A 的第 j 列
const isHl = (name, i, j) => {
  const h = hov.value
  if (!h) return false
  return (name === 'ΔW' && i === h.i && j === h.j) || (name === 'B' && i === h.i) || (name === 'A' && j === h.j)
}
const tint = (v, max) => `color-mix(in srgb, var(${v >= 0 ? '--accent' : '--right'}) ${Math.round(Math.abs(v) / max * 100)}%, transparent)`
</script>

<style scoped>
.mats { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
.mat { margin: 0; }
.hm { gap: 1px; width: max-content; }
.px { width: 12px; height: 12px; border-radius: 1px; outline: 1px solid color-mix(in srgb, var(--border) 60%, transparent); outline-offset: -1px; }
.px.hl { outline: 2px solid var(--warn); outline-offset: -1px; }
.op { font-size: 18px; color: var(--text-muted); }
figcaption { font-size: 11px; color: var(--text); margin-top: 4px; line-height: 1.4; }
figcaption i { color: var(--text-dim); font-style: normal; }
figcaption em { color: var(--text-dim); font-style: normal; font-size: 10px; }
.spec { margin-top: 14px; }
.cap { font-size: 11px; color: var(--text-dim); }
.bars { display: flex; align-items: flex-end; gap: 3px; height: 48px; margin-top: 4px; max-width: 260px; }
.bars i { flex: 1; background: var(--border-strong); border-radius: 1px; min-height: 1px; }
.bars i.keep { background: var(--accent); }
</style>
