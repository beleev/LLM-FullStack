<!--
  张量并行切法 (对应 llm_train/m03_tensor_parallel)。
  只讲一件事: Y = GeLU(X·A)·B, A 列切 + B 行切时, 中间不需要任何通信, 只在最后 all-reduce 一次。
  两张卡上的每个小矩阵都是真算出来的, Δ 是和不切分的 dense 结果逐元素比出来的。
-->
<template>
  <LabFrame
    title="张量并行 — 通信到底该插在哪?"
    sub="两张卡算 Y = GeLU(X·A)·B。先选 A、B 各自怎么切, 再点三个「同步点」开关通信。目标: 让两张卡都拿到和 dense 完全一样的 Y (Δ = 0), 而且通信最少。"
    module="llm_train/m03"
    run="python -m llm_train.m03_tensor_parallel.demo"
    :challenge="{
      ask: 'A 行切时, 为什么不能像列切那样先各算各的 GeLU、最后再求和? 关掉同步点 ① 看看 Δ。',
      answer: '行切后每张卡算出的是 X·A 的一个「部分和」, 而 GeLU 不是线性的: GeLU(a+b) ≠ GeLU(a)+GeLU(b), 所以必须在 GeLU 之前先 all-reduce。列切则不同: 每张卡拿到的是 X·A 的一段完整的列, 逐元素的 GeLU 在本地做完全正确。接着 B 行切正好吃下这一段, 得到部分和, 最后只需一次 all-reduce。Megatron 的「列切→行切」就是把通信推迟到唯一躲不掉的那个求和点。',
    }"
  >
    <template #controls>
      <div class="row">
        <span class="lbl">A [d×4d]:</span>
        <button type="button" :class="{ active: splitA === 'col' }" @click="splitA = 'col'">按列切 (切输出维)</button>
        <button type="button" :class="{ active: splitA === 'row' }" @click="splitA = 'row'">按行切 (切输入维)</button>
      </div>
      <div class="row">
        <span class="lbl">B [4d×d]:</span>
        <button type="button" :class="{ active: splitB === 'col' }" @click="splitB = 'col'">按列切</button>
        <button type="button" :class="{ active: splitB === 'row' }" @click="splitB = 'row'">按行切</button>
        <button type="button" @click="seed++">换一组数</button>
        <button type="button" @click="sync = [false, false, false]">清空同步点</button>
      </div>
    </template>

    <div class="flow">
      <template v-for="(s, i) in run.stages" :key="i">
        <button
          v-if="s.slot !== undefined" type="button" class="sync" :class="{ active: sync[s.slot], waste: s.waste }"
          :aria-pressed="sync[s.slot]" @click="sync[s.slot] = !sync[s.slot]"
        >
          同步点 {{ '①②③'[s.slot] }} · {{ sync[s.slot] ? s.op : '点击插入通信' }}
        </button>
        <div v-else class="stage" :class="{ bad: s.invalid }">
          <span class="name mono">{{ s.name }}</span>
          <div v-if="!s.invalid" class="ranks">
            <div v-for="(m, r) in s.t" :key="r" class="rank">
              <span class="rk mono">卡{{ r }}</span>
              <div class="mat" :style="{ gridTemplateColumns: `repeat(${m[0].length}, 12px)` }">
                <i v-for="(v, j) in m.flat()" :key="j" :style="{ background: heat(Math.abs(v) / s.max, v >= 0 ? 'var(--accent)' : 'var(--right)') }" :title="v.toFixed(3)" />
              </div>
            </div>
          </div>
          <span class="kind">{{ s.invalid || kindText[s.kind] }}</span>
        </div>
      </template>
    </div>

    <template #stats>
      <div class="kv"><span>与 dense 的 max|Δ|</span><b :class="run.ok ? 'good' : 'bad'">{{ run.delta === null ? '—' : run.delta < 1e-12 ? '0' : run.delta.toFixed(3) }}</b></div>
      <div class="kv"><span>结论</span><b :class="run.ok ? 'good' : 'bad'" class="verdict">{{ run.verdict }}</b></div>
      <div class="kv"><span>通信次数</span><b :class="{ good: run.ok && run.calls === 1 }">{{ run.calls }}</b></div>
      <div class="kv"><span>每卡发送量 (单位 |X|)</span><b :class="{ good: run.ok && run.vol <= 1 }">{{ run.vol.toFixed(2) }}×</b></div>
      <p class="lab-note">
        ★ 完整 = 每卡都有整个张量; 一段列 = 每卡只有几列, 拼起来才完整 (all-gather); 部分和 = 形状完整但数值只是一部分, 加起来才对 (all-reduce)。
        all-reduce 每卡发 2(N−1)/N·size, all-gather 发 (N−1)/N·size; 中间激活 H 比 X 宽 4 倍 (图里画 2 倍), 所以在 H 上通信最贵。
      </p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import { heat, mulberry32, randn, range } from '@/utils/labmath.js'

const N = 2, T = 2, D = 4, HID = 8
const splitA = ref('col'), splitB = ref('row'), seed = ref(1)
const sync = ref([false, false, true])
const kindText = { full: '完整 (每卡一份)', shard: '一段列 (拼起来才完整)', partial: '部分和 (加起来才对)' }

const matmul = (a, b) => a.map((row) => b[0].map((_, j) => row.reduce((s, v, k) => s + v * b[k][j], 0)))
const erf = (x) => { // Abramowitz-Stegun 7.1.26, 误差 < 1.5e-7, 足够画图; dense 和 TP 用同一个函数所以 Δ 仍可精确到 0
  const t = 1 / (1 + 0.3275911 * Math.abs(x))
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x)
  return x >= 0 ? y : -y
}
const gelu = (m) => m.map((row) => row.map((v) => 0.5 * v * (1 + erf(v / Math.SQRT2))))
const cols = (m, lo, hi) => m.map((row) => row.slice(lo, hi))
const part = (w, r) => [(w / N) * r, (w / N) * (r + 1)]

// 线性层的两种切法: 每张卡只用自己手里的东西去算, 不偷看别的卡
const linear = (st, Wm, split) => {
  const wIn = Wm.length, wOut = Wm[0].length
  if (split === 'col') {
    if (st.kind === 'shard') return { invalid: '形状对不上: 列切的权重要吃完整输入, 但每卡只有一段列 → 先 all-gather' }
    return { kind: 'shard', t: range(N).map((r) => matmul(st.t[r], cols(Wm, ...part(wOut, r)))) }
  }
  return { // 行切: 卡 r 只有 W 的第 r 段行, 所以只吃输入的第 r 段列
    kind: 'partial',
    t: range(N).map((r) => matmul(st.kind === 'shard' ? st.t[r] : cols(st.t[r], ...part(wIn, r)), Wm.slice(...part(wIn, r)))),
  }
}

const run = computed(() => {
  const rand = mulberry32(seed.value * 97)
  const mk = (a, b, s) => range(a).map(() => range(b).map(() => randn(rand) * s))
  const X = mk(T, D, 1), A = mk(D, HID, 0.7), B = mk(HID, D, 0.7)
  const dense = matmul(gelu(matmul(X, A)), B)

  const stages = []
  let st = { kind: 'full', t: range(N).map(() => X) }, calls = 0, vol = 0
  let dead = false // 形状已经对不上之后, 后面的阶段不再重复报错
  const push = (name) => !dead && (dead = !!st.invalid, stages.push({ name, ...st, max: st.t ? Math.max(1e-9, ...st.t.flat(2).map(Math.abs)) : 1 }))
  const doSync = (slot) => {
    const size = st.t ? st.t[0].length * (st.kind === 'shard' ? st.t[0][0].length * N : st.t[0][0].length) : 0
    const op = st.kind === 'shard' ? 'all-gather (拼)' : st.kind === 'partial' ? 'all-reduce (加)' : '多余: 已经是完整的';
    stages.push({ slot, op, waste: sync.value[slot] && st.kind === 'full' })
    if (!sync.value[slot] || st.invalid) return
    calls++
    if (st.kind === 'shard') { vol += ((N - 1) / N) * size; const full = st.t[0].map((_, i) => st.t.flatMap((m) => m[i])); st = { kind: 'full', t: range(N).map(() => full) } }
    else { // ★ 对已经完整的张量再同步 (按 mean) 数值不变, 但通信照样花钱 → 标成"多余"
      vol += (2 * (N - 1) / N) * size
      const total = st.t[0].map((row, i) => row.map((_, j) => st.t.reduce((s, m) => s + m[i][j], 0) / (st.kind === 'full' ? N : 1)))
      st = { kind: 'full', t: range(N).map(() => total) }
    }
  }
  const step = (name, fn) => { if (!st.invalid) st = fn(st); push(name) }

  push('X')
  step(`X·A (${splitA.value === 'col' ? '列切' : '行切'})`, (s) => linear(s, A, splitA.value))
  doSync(0)
  step('GeLU', (s) => ({ ...s, t: s.t.map(gelu) }))
  doSync(1)
  step(`·B (${splitB.value === 'col' ? '列切' : '行切'})`, (s) => linear(s, B, splitB.value))
  doSync(2)
  push('Y')

  const done = !st.invalid && st.kind === 'full'
  const delta = done ? Math.max(...st.t[0].flatMap((row, i) => row.map((v, j) => Math.abs(v - dense[i][j])))) : null
  const ok = done && delta < 1e-12
  const verdict = st.invalid ? '形状对不上' : !done ? (st.kind === 'shard' ? '还差一次 all-gather' : '还差一次 all-reduce') : ok ? (calls === 1 ? '正确, 且通信最少' : '正确, 但通信偏多') : '数值错了'
  return { stages, calls, vol: vol / (T * D), delta, ok, verdict }
})
</script>

<style scoped>
.lbl { font-size: 12px; color: var(--text-muted); min-width: 64px; }
.flow { display: flex; flex-direction: column; gap: 6px; }
.stage { display: grid; grid-template-columns: 92px 1fr; gap: 4px 12px; align-items: center; padding: 8px 10px; border: 1px solid var(--border); border-radius: var(--radius-sm); background: var(--bg-elev); }
.stage.bad { border-color: var(--danger); }
.stage.bad .kind { color: var(--danger); }
.name { font-size: 12px; color: var(--text); }
.ranks { display: flex; flex-wrap: wrap; gap: 14px; }
.rank { display: flex; align-items: center; gap: 6px; }
.rk { font-size: 10px; color: var(--text-dim); }
.mat { display: grid; gap: 2px; }
.mat i { width: 12px; height: 12px; border-radius: 2px; border: 1px solid var(--border); }
.kind { grid-column: 2; font-size: 11px; color: var(--text-dim); }
.sync { align-self: flex-start; margin-left: 24px; font-size: 12px; border-style: dashed; }
.sync.active { border-style: solid; }
.sync.waste { border-color: var(--warn); color: var(--warn); }
.verdict { font-size: 13px !important; }
</style>
