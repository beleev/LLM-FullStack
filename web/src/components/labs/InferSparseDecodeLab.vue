<!-- 稀疏 decode 实验台 (llm_infer/m22): needle_context + quest_upper_bound / mean_score + select_blocks + evaluate 的 JS 移植。 -->
<template>
  <LabFrame
    title="稀疏注意力 decode — 只读 top-k 个 KV block"
    sub="1024 个 token 的 KV 切成 64 个 block (每块 16)。格子越亮 = 这个 block 真实分到的注意力越多 (decode 时并不知道)。用廉价摘要给每块打分, 只读分数最高的 k 块。带框 = 被选中, 「针」= 埋了与 q 对齐的 key。点一个 block 看它的分数。"
    module="llm_infer/m22"
    run="python -m llm_infer.m22_sparse_attention.demo"
    :challenge="{
      ask: '把「针的强度」拖到 0, 再比较 Quest 打分和随机选块的误差。稀疏注意力还有用吗?',
      answer: '几乎没用。强度为 0 时所有 key 都是噪声, 注意力均匀弥散在 1024 个 token 上 —— 没有「少数重要 token」, 读 1/8 的 KV 就只能拿到约 1/8 的注意力质量, 怎么打分都和随机差不多。稀疏 decode 的收益完全来自注意力高度集中这个经验事实 (训练过的 LLM 成立; 随机权重的 TinyLM 不成立, Python demo 也如实展示了这一点)。',
    }"
  >
    <template #controls>
      <div class="row">
        <button v-for="m in SCORERS" :key="m.id" type="button" :class="{ active: scorer === m.id }" @click="scorer = m.id">{{ m.name }}</button>
        <button type="button" @click="seed++">换一组</button>
      </div>
      <LabSlider v-model="k" label="读取 block 数 k" :min="2" :max="64" />
      <LabSlider v-model="strength" label="针的强度 (logit 增量)" :min="0" :max="12" :step="0.5" />
    </template>

    <div class="cells grid">
      <button
        v-for="b in NB" :key="b" type="button" class="cell blk" :class="{ picked: res.picked.has(b - 1), cur: sel === b - 1 }"
        :style="{ background: heat(Math.sqrt(ctx.mass[b - 1] / ctx.maxMass)) }"
        :aria-label="`block ${b - 1}`" :aria-pressed="sel === b - 1" @click="sel = b - 1"
      >{{ ctx.needles.includes(b - 1) ? '针' : b === 1 ? 'S' : b === NB ? '近' : '' }}</button>
    </div>
    <p class="lab-note info">
      block {{ sel }}: 真实注意力质量 {{ (ctx.mass[sel] * 100).toFixed(1) }}% · 当前打分 {{ res.scores[sel].toFixed(2) }}
      · Quest 上界 {{ ctx.quest[sel].toFixed(1) }} ≥ 块内真实 max q·k = {{ ctx.trueMax[sel].toFixed(1) }}
      {{ res.picked.has(sel) ? '· 已选中' : '· 未选中' }}{{ sel === 0 || sel === NB - 1 ? ' (强制保留)' : '' }}
    </p>

    <template #stats>
      <div class="kv"><span>读取的 KV</span><b>{{ (k / NB * 100).toFixed(1) }}%</b></div>
      <div class="kv"><span>覆盖的注意力质量 (recall)</span><b :class="res.recall > 0.9 ? 'good' : res.recall < 0.5 ? 'bad' : ''">{{ (res.recall * 100).toFixed(1) }}%</b></div>
      <div class="kv"><span>输出相对 L2 误差</span><b :class="res.err < 0.05 ? 'good' : res.err > 0.5 ? 'bad' : ''">{{ res.err.toFixed(4) }}</b></div>
      <div class="kv"><span>抓到的针</span><b :class="res.caught === ctx.needles.length ? 'good' : 'bad'">{{ res.caught }} / {{ ctx.needles.length }}</b></div>
      <p class="lab-note">
        S = 第 0 块 (attention sink), 近 = 最后一块 (最近 token): 永远选中, 算在 k 内。
        Quest 摘要 = 每块逐维 kmin / kmax; 上界保证不漏针, 但偏松 (这里平均是真实 max 的 {{ ctx.loose.toFixed(1) }}×)。
        收益的前提是注意力集中 —— 训练过的 LLM 才有; 强度 = 0 时打分几乎不比随机强。
      </p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { mulberry32, randn, range, softmax, sum, heat } from '@/utils/labmath.js'

const T = 1024, BS = 16, NB = T / BS, DIM = 32, N_NEEDLE = 4, PER = 2
const SCORERS = [{ id: 'quest', name: 'Quest 上界' }, { id: 'mean', name: '均值 key' }, { id: 'random', name: '随机选块' }]

const scorer = ref('quest')
const k = ref(8)
const strength = ref(10)
const seed = ref(1)
const sel = ref(0)

// 只跟 seed 走的部分: q, 噪声 K/V, 针的位置, 随机分数
const base = computed(() => {
  const rand = mulberry32(seed.value * 15485863)
  const vec = (n) => Float64Array.from({ length: n }, () => randn(rand))
  const q = vec(DIM), K = vec(T * DIM), V = vec(T * DIM)
  const pool = range(NB - 2).map((i) => i + 1) // 针不放在首 / 末块
  const needles = range(N_NEEDLE).map(() => pool.splice(Math.floor(rand() * pool.length), 1)[0]).sort((a, b) => a - b)
  const pos = needles.flatMap((b) => { const p = range(BS); return range(PER).map(() => b * BS + p.splice(Math.floor(rand() * p.length), 1)[0]) })
  return { q, K, V, needles, pos, rnd: range(NB).map(() => rand()) }
})

// 加上针之后的上下文 + 所有"真值" (真实 attention, 完整输出) + 两种摘要打分
const ctx = computed(() => {
  const { q, V, needles, pos, rnd } = base.value
  const K = base.value.K.slice(), qq = sum(Array.from(q, (x) => x * x))
  for (const p of pos) for (let i = 0; i < DIM; i++) K[p * DIM + i] += q[i] * (strength.value * Math.sqrt(DIM) / qq) // 使 q·k/√d 增加 strength
  const qk = range(T).map((t) => { let s = 0; for (let i = 0; i < DIM; i++) s += q[i] * K[t * DIM + i]; return s })
  const logits = qk.map((x) => x / Math.sqrt(DIM)), probs = softmax(logits)
  const out = (idx, p) => range(DIM).map((i) => idx.reduce((a, t, j) => a + p[j] * V[t * DIM + i], 0))
  const blocks = range(NB).map((b) => range(BS).map((j) => b * BS + j))
  const mass = blocks.map((ts) => sum(ts.map((t) => probs[t])))
  const trueMax = blocks.map((ts) => Math.max(...ts.map((t) => qk[t])))
  // ★ Quest: 逐维 max(q_i·kmin_i, q_i·kmax_i) 求和 —— q_i 正取 kmax, 负取 kmin, 所以 ≥ 块内任何 q·k
  const quest = blocks.map((ts) => sum(range(DIM).map((i) => {
    const col = ts.map((t) => K[t * DIM + i])
    return Math.max(q[i] * Math.min(...col), q[i] * Math.max(...col))
  })))
  const mean = blocks.map((ts) => sum(ts.map((t) => qk[t])) / BS) // q·mean(K_block)
  const loose = sum(quest.map((u, b) => u / Math.abs(trueMax[b]))) / NB
  return { V, logits, mass, maxMass: Math.max(...mass), trueMax, quest, mean, random: rnd, needles, full: out(range(T), probs), out, loose }
})

const res = computed(() => {
  const c = ctx.value, scores = c[scorer.value]
  const forced = [0, NB - 1] // select_blocks: 首块 (sink) + 末块 (最近) 强制保留, 算在 k 内
  const rest = range(NB).filter((b) => !forced.includes(b)).sort((a, b) => scores[b] - scores[a] || a - b)
  const picked = new Set([...forced, ...rest.slice(0, Math.max(k.value - 2, 0))])
  const idx = range(T).filter((t) => picked.has(Math.floor(t / BS)))
  const sparse = c.out(idx, softmax(idx.map((t) => c.logits[t]))) // 只在选中的 token 上重新 softmax
  const norm = (v) => Math.sqrt(sum(v.map((x) => x * x)))
  return {
    scores, picked, err: norm(sparse.map((x, i) => x - c.full[i])) / norm(c.full),
    recall: sum([...picked].map((b) => c.mass[b])), caught: c.needles.filter((b) => picked.has(b)).length,
  }
})
</script>

<style scoped>
.grid { grid-template-columns: repeat(16, minmax(0, 1fr)); max-width: 560px; }
.blk { min-width: 0; min-height: 0; height: 30px; padding: 0; font-size: 10px; color: var(--text); transition: none; }
.blk.picked { border: 2px solid var(--left); }
.blk.cur { outline: 2px solid var(--accent); outline-offset: 1px; }
.info { margin-top: 10px; min-height: 3.4em; }
</style>
