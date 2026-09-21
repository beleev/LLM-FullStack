<!--
  LLaDA 迭代去遮实验台 (对应 llm_models/models/language_models/llada.py:LLaDA.sample)。
  只讲一件事: 每步 “全部预测 → 只留最有把握的 → 其余重新遮住”, 剩余 [MASK] 数按线性日程递减。
  玩具模型: 一个位置的置信度 = 自身难度 + 周围已定稿 token 数 (上下文越多越有把握), 采样/重遮逻辑与 Python 一致。
-->
<template>
  <LabFrame
    title="LLaDA — 先填有把握的, 没把握的遮回去再想"
    sub="12 个位置一开始全是 [MASK]。每一行是一步去噪之后的样子: 紫 = 本步新定稿, 绿 = 之前已定稿, 灰字 = 本步预测过但因为置信度低被重新遮住, 红 = 定稿错了 (再也改不回来)。最下面一行是自回归在 “同样的串行步数” 里能写出多少。"
    module="llm_models/models/language_models/llada.py"
    run="python -m llm_models.run_models.language_models.llada.infer_llada"
    :challenge="{
      ask: '步数设为 4, 先用 “低置信度重遮”, 再切 “随机重遮”, 多换几组。先猜: 哪个定稿错误更多? 再把步数拖到 12 和 1, 错误数怎么变?',
      answer: '随机重遮会把没把握的预测也定稿, 而定稿的 token 置信度记为 +∞、永不重遮, 错了就一直错, 还会误导邻居。低置信度重遮让模型先定 “显然” 的位置, 这些位置又成为别人的上下文, 难的位置留到信息最充分时再定 —— 生成顺序由置信度决定, 而不是从左到右。步数是质量/速度旋钮: steps = 12 时每步只定 1 个, 最稳但要 12 次整段前向 (双向注意力没有 KV cache, 每次都重算全部 12 个位置, 总计算是自回归的 12 倍); steps = 1 一次定完, 每个位置都在零上下文下瞎猜。LLaDA 赚的是串行步数可以少于 token 数, 赔的是每步更贵。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: remask === 'low' }" @click="remask = 'low'">低置信度重遮 (low_confidence)</button>
        <button type="button" :class="{ active: remask === 'random' }" @click="remask = 'random'">随机重遮 (random)</button>
        <button type="button" @click="seed++">换一组</button>
      </div>
      <LabSlider v-model="steps" label="去噪步数 steps" :min="1" :max="12" />
      <StepPlayer :stepper="stepper" :label="`第 ${stepper.step.value} / ${steps} 步`" />
    </template>

    <div class="cells grid" :style="{ gridTemplateColumns: `52px repeat(${T}, minmax(38px, 1fr))` }">
      <template v-for="(row, s) in frames" :key="s">
        <span class="rl mono" :class="{ dimrow: s > stepper.step.value }">{{ s === 0 ? '初始' : '步 ' + s }}</span>
        <span
          v-for="(c, i) in row" :key="i" class="cell w" :class="[c.cls, { dim: s > stepper.step.value }]"
          :title="c.conf != null ? `置信度 ${c.conf.toFixed(2)}` : ''"
        >{{ c.text }}</span>
      </template>
      <span class="rl mono ar">自回归</span>
      <span v-for="i in T" :key="'ar' + i" class="cell w" :class="i <= arDone ? 'ok' : ''">{{ i <= arDone ? WORDS[i - 1] : '·' }}</span>
    </div>

    <template #stats>
      <div class="kv"><span>定稿错误数</span><b :class="errors === 0 ? 'good' : 'bad'">{{ errors }} / {{ T }}</b></div>
      <div class="kv"><span>串行前向次数 (LLaDA / 自回归)</span><b :class="steps < T ? 'good' : ''">{{ steps }} / {{ T }}</b></div>
      <div class="kv"><span>token 前向总量 (LLaDA / 自回归)</span><b :class="steps > 1 ? 'bad' : ''">{{ steps * T }} / {{ T }}</b></div>
      <div class="kv"><span>本步结束应剩 [MASK]</span><b>{{ Math.round(T * (1 - stepper.step.value / steps)) }}</b></div>
      <p class="lab-note">日程: 第 s 步后剩 round(T·(1 − s/steps)) 个 [MASK]。自回归有 KV cache, 每步只算 1 个 token; LLaDA 每步重算全部 {{ T }} 个。</p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import StepPlayer from '@/components/lab/StepPlayer.vue'
import { useStepper } from '@/composables/useStepper.js'
import { mulberry32, clamp, range } from '@/utils/labmath.js'

const WORDS = ['今天', '天气', '很', '好', '我们', '一起', '去', '公园', '散步', '然后', '喝', '咖啡']
const WRONG = ['昨天', '心情', '不', '坏', '他们', '单独', '回', '公司', '加班', '但是', '吃', '火锅']
const T = WORDS.length
const remask = ref('low'), steps = ref(4), seed = ref(1)

const frames = computed(() => {
  const r = mulberry32(seed.value * 7919)
  const base = range(T).map(() => 0.45 + r() * 0.35)               // 每个位置的先天难度
  let x = Array(T).fill(null)                                     // null = [MASK]
  const out = [x.map(() => ({ text: '▢', cls: '' }))]
  for (let s = 1; s <= steps.value; s++) {
    const masked = range(T).filter((i) => x[i] === null)
    const pred = {}
    masked.forEach((i) => {
      const ctx = [-2, -1, 1, 2].filter((d) => x[i + d] != null).length // 周围已定稿的 token 数
      const pTrue = clamp(base[i] + 0.12 * ctx, 0.05, 0.98)
      const ok = r() < pTrue
      const conf = ok ? pTrue : pTrue * 0.55                      // 猜错时模型自报的概率通常也偏低, 但不总是最低
      pred[i] = { ok, conf, rank: remask.value === 'low' ? conf : r() }
    })
    // ★ 线性日程: 本步结束后应剩 n 个 [MASK]; 把名次最低的 n 个重新遮住
    const n = Math.round(T * (1 - s / steps.value))
    const again = new Set([...masked].sort((a, b) => pred[a].rank - pred[b].rank).slice(0, n))
    const row = range(T).map((i) => {
      if (x[i] !== null) return { text: x[i].text, cls: x[i].ok ? 'ok' : 'bad' }
      const p = pred[i], text = p.ok ? WORDS[i] : WRONG[i]
      return again.has(i) ? { text, cls: 'ghost', conf: p.conf } : { text, cls: p.ok ? 'on' : 'bad', conf: p.conf }
    })
    x = x.map((v, i) => (v !== null || again.has(i) ? v : { text: pred[i].ok ? WORDS[i] : WRONG[i], ok: pred[i].ok }))
    out.push(row)
  }
  return out
})
const errors = computed(() => frames.value[frames.value.length - 1].filter((c) => c.cls === 'bad').length)

const stepper = useStepper(() => steps.value + 1, { interval: 800 })
stepper.step.value = steps.value
watch(frames, () => { stepper.pause(); stepper.step.value = steps.value })
const arDone = computed(() => Math.min(T, stepper.step.value))
</script>

<style scoped>
.grid { gap: 3px; min-width: 540px; }
.cell.w { min-width: 38px; height: 24px; font-size: 11px; font-family: inherit; }
.cell.ghost { color: var(--text-dim); border-style: dashed; text-decoration: line-through; }
.rl { font-size: 10px; color: var(--text-dim); align-self: center; }
.rl.dimrow { opacity: 0.35; }
.rl.ar { color: var(--eye); margin-top: 8px; }
.rl.ar ~ .cell { margin-top: 8px; }
</style>
