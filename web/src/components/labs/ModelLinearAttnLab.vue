<!--
  线性注意力 / Gated DeltaNet 实验台 (对应 llm_models/layers/sparse/linear_attention.py:GatedDeltaNet)。
  只讲一件事: softmax 注意力把每个 token 的 K/V 原件都留着 (越存越多), 线性注意力只有一块固定大小的状态矩阵 S —— 省内存, 但容量有限, 写法决定记性。
-->
<template>
  <LabFrame
    title="固定大小的状态 vs 越长越大的 KV"
    sub="逐 token 播放: 每来一个 token, 把 (k, v) 写进 4×4 的状态 S。点上排任意旧 token, 用它的 key 去读 S, 看读回来的 value 和当初写进去的差多少。注意 t6 故意复用了 t1 的 key (同一个 “变量” 被重新赋值)。"
    module="llm_models/layers/sparse/linear_attention.py"
    run="python -m llm_models.run_models.language_models.qwen3_next.infer_qwen3_next"
    :challenge="{
      ask: '播放到最后, 先用 “纯累加” 点 t1, 再切到 delta rule 点 t1。哪个读回的是 t6 写入的新值? 然后想: key 只有 4 维, 写入第 5 个不同的 key 之后, 为什么所有旧 token 的误差都开始上升?',
      answer: '纯累加 S += k·vᵀ 只会叠加: t1 和 t6 的 key 相同, 读回来是 v1 + v6 的混合, 旧值永远擦不掉。delta rule 先用 k 读出 S 里这个方向当前存的值 kᵀS, 减掉它, 再写新值: S ← α(S − β·k·kᵀS) + β·k·vᵀ, β=1 时等于精确覆写, 读回的就是 v6。容量问题两者都有: 4 维空间最多只有 4 个互相正交的 key, 第 5 个 key 必然和旧 key 有夹角, 写它就会污染旧槽位。状态大小固定的代价是 “有损”。α<1 让旧内容整体衰减, 给新内容腾地方。所以 Qwen3-Next 每 4 层留 1 层全注意力兜底精确召回。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: mode === 'delta' }" @click="mode = 'delta'">delta rule (先擦后写)</button>
        <button type="button" :class="{ active: mode === 'add' }" @click="mode = 'add'">纯累加 S += k·vᵀ</button>
        <button type="button" @click="seed++">换一组</button>
      </div>
      <LabSlider v-model="alpha" label="衰减门 α" :min="0.6" :max="1" :step="0.01" :format="(t) => t.toFixed(2)" />
      <LabSlider v-model="beta" label="写入强度 β" :min="0.1" :max="1" :step="0.05" :format="(t) => t.toFixed(2)" />
      <StepPlayer :stepper="stepper" :label="`t = ${t}`" />
    </template>

    <div class="cells toks" :style="{ gridTemplateColumns: `repeat(${T}, minmax(24px, 34px))` }" role="group" aria-label="点一个旧 token 作为查询">
      <button
        v-for="j in T" :key="j" type="button" class="cell tok"
        :class="{ dim: j - 1 > t, on: j - 1 === t, active: j - 1 === qj, same: keyOf[j - 1] !== j - 1 || j - 1 === DUP_OF }" :disabled="j - 1 > t"
        @click="qj = j - 1"
      >t{{ j - 1 }}</button>
    </div>

    <div class="wrap">
      <div>
        <p class="cap">线性注意力的全部记忆: S (4×4 = 16 个数, 永不增长)</p>
        <div class="cells" style="grid-template-columns: repeat(4, 34px)">
          <span v-for="(x, i) in now.S.flat()" :key="i" class="cell big mono" :style="{ background: heat(Math.abs(x) / 1.5, x >= 0 ? 'var(--accent)' : 'var(--right)') }">{{ x.toFixed(1) }}</span>
        </div>
      </div>
      <div>
        <p class="cap">softmax 注意力的 KV cache: {{ t + 1 }} 行 × 8 = {{ (t + 1) * 8 }} 个数</p>
        <div class="kvlist">
          <div v-for="j in t + 1" :key="j" class="kvrow" :class="{ hit: j - 1 === qj }"><i v-for="c in 4" :key="'k' + c" class="k" /><i v-for="c in 4" :key="'v' + c" class="v" /></div>
        </div>
      </div>
      <div class="errs">
        <p class="cap">用各旧 token 的 key 读 S 的相对误差</p>
        <div class="ebars">
          <div v-for="(e, j) in now.errs" :key="j" class="eb" :class="{ hit: j === qj }">
            <div class="col"><div :class="e > 0.5 ? 'bad' : e > 0.15 ? 'mid' : 'ok'" :style="{ height: Math.min(1, e) * 100 + '%' }" /></div>
            <span class="mono">t{{ j }}</span>
          </div>
        </div>
      </div>
    </div>

    <template #stats>
      <div class="kv"><span>状态大小 (恒定)</span><b class="good">16</b></div>
      <div class="kv"><span>KV cache 大小 (随 t 增长)</span><b :class="(t + 1) * 8 > 16 ? 'bad' : ''">{{ (t + 1) * 8 }}</b></div>
      <div class="kv"><span>查询 t{{ qSafe }} 的读回误差</span><b :class="now.errs[qSafe] < 0.15 ? 'good' : 'bad'">{{ (now.errs[qSafe] * 100).toFixed(0) }}%</b></div>
      <div class="kv"><span>所有旧 token 平均误差</span><b>{{ (mean(now.errs) * 100).toFixed(0) }}%</b></div>
      <p class="lab-note mono">应读回 {{ vec(now.target[qSafe]) }}<br>实际读回 {{ vec(now.read[qSafe]) }}</p>
      <p class="lab-note">KV cache 那一列对应误差恒为 0 —— 原件都在, 代价是它会一直长下去。</p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import StepPlayer from '@/components/lab/StepPlayer.vue'
import { useStepper } from '@/composables/useStepper.js'
import { mulberry32, randn, range, heat, sum } from '@/utils/labmath.js'

const T = 10, Dm = 4, DUP = 6, DUP_OF = 1 // t6 复用 t1 的 key
const mode = ref('delta'), alpha = ref(1), beta = ref(1), seed = ref(1), qj = ref(1)
const keyOf = range(T).map((j) => (j === DUP ? DUP_OF : j))
const mean = (a) => sum(a) / a.length
const vec = (a) => '[' + a.map((x) => x.toFixed(2)).join(', ') + ']'

const data = computed(() => {
  const r = mulberry32(seed.value * 97)
  const unit = () => { const v = range(Dm).map(() => randn(r)); const n = Math.hypot(...v); return v.map((x) => x / n) } // key 做 L2 归一化, 与 Python 一致
  const ks = range(T).map(unit)
  ks[DUP] = ks[DUP_OF]
  return { ks, vs: range(T).map(() => range(Dm).map(() => randn(r))) }
})

const frames = computed(() => {
  const { ks, vs } = data.value, a = alpha.value, b = beta.value
  let S = range(Dm).map(() => Array(Dm).fill(0))
  return range(T).map((t) => {
    const k = ks[t], v = vs[t]
    const kS = range(Dm).map((e) => sum(k.map((kd, d) => kd * S[d][e])))   // kᵀS: k 方向当前存的值
    // ★ delta: S ← α(S − β k (kᵀS)) + β k vᵀ;  纯累加: S ← αS + β k vᵀ
    S = S.map((row, d) => row.map((x, e) => a * (x - (mode.value === 'delta' ? b * k[d] * kS[e] : 0)) + b * k[d] * v[e]))
    const read = range(t + 1).map((j) => range(Dm).map((e) => sum(ks[j].map((kd, d) => kd * S[d][e]))))
    // 目标: 这个 key 最近一次写入的 value (t1 在 t6 之后应读回 v6)
    const target = range(t + 1).map((j) => vs[Math.max(...range(t + 1).filter((i) => keyOf[i] === keyOf[j]))])
    const errs = read.map((o, j) => Math.hypot(...o.map((x, e) => x - target[j][e])) / Math.hypot(...target[j]))
    return { S, read, target, errs }
  })
})

const stepper = useStepper(() => T, { interval: 700 })
stepper.step.value = T - 1
const t = computed(() => stepper.step.value)
const now = computed(() => frames.value[t.value])
const qSafe = computed(() => Math.min(qj.value, t.value))
watch(t, (v) => { if (qj.value > v) qj.value = v })
</script>

<style scoped>
.toks { margin-bottom: 14px; }
.tok { padding: 0; min-height: 26px; height: 26px; cursor: pointer; }
.tok.same { border-style: dashed; border-width: 2px; border-color: var(--warn); }
.wrap { display: flex; flex-wrap: wrap; gap: 20px; align-items: flex-start; }
.cap { font-size: 11px; color: var(--text-dim); margin-bottom: 6px; }
.cell.big { width: 34px; height: 30px; color: var(--text); }
.kvlist { display: grid; gap: 2px; }
.kvrow { display: flex; width: max-content; gap: 1px; padding: 1px; border: 1px solid transparent; border-radius: 2px; }
.kvrow.hit { border-color: var(--accent); }
.kvrow i { width: 9px; height: 7px; border-radius: 1px; }
.kvrow .k { background: color-mix(in srgb, var(--eye) 60%, transparent); }
.kvrow .v { background: color-mix(in srgb, var(--left) 60%, transparent); }
.ebars { display: flex; gap: 4px; }
.eb { display: grid; justify-items: center; gap: 2px; font-size: 9px; color: var(--text-muted); border: 1px solid transparent; border-radius: 3px; padding: 1px; }
.eb.hit { border-color: var(--accent); }
.col { width: 14px; height: 70px; display: flex; align-items: flex-end; border-bottom: 1px solid var(--border-strong); }
.col div { width: 100%; min-height: 1px; border-radius: 2px 2px 0 0; }
.col .ok { background: var(--left); } .col .mid { background: var(--warn); } .col .bad { background: var(--danger); }
</style>
