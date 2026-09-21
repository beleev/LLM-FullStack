<!--
  token 级 vs 序列级重要性比率 (GRPO vs GSPO) 实验台。
  只讲一件事: 奖励是给整条序列的, GRPO 却给每个 token 各算一个 ρ_t 各自 clip —— 单个 token 的比率是单样本估计, 噪声很大;
  GSPO 用长度归一化的序列比率 s = exp(mean_t log ρ_t), 整条序列共用一个权重, 要 clip 就整条一起 clip。
-->
<template>
  <LabFrame
    title="GSPO — 比率该按 token 算, 还是按整条序列算"
    sub="每根柱子是一条回答里某个 token 的 log ρ_t = log π_new − log π_old。上下拖动柱顶。黄线之外的 token 在 GRPO 里被单独 clip (变灰 = 这个 token 没有梯度); 绿线是 GSPO 的序列级比率 log s = 所有柱子的平均。"
    module="llm_finetune/methods/grpo.py"
    :challenge="{
      ask: '把噪声拖到 0.3 以上: GRPO 下有几个 token 被 clip? 此时 GSPO 的 s 离 1 有多远? 再看「不做长度归一的 Πρ_t」—— 如果序列有 1000 个 token 它会怎样?',
      answer: '每个 ρ_t 都只基于「这个位置采到的这一个 token」, 是方差很大的单样本重要性权重。噪声 0.3 时常有三四个 token 越界被 clip, 剩下的照常更新。于是同一条序列、同一个 Â, 有的 token 学有的不学, 而且哪些被 clip 基本是随机的。MoE 模型里路由一变, token 级比率抖得更厉害, 这是 GSPO 论文报告的 GRPO 训练崩溃来源。序列级比率取平均后噪声按 1/√T 缩小, s 通常离 1 很近, 所以 GSPO 的 ε 取得极小 (论文里是 3e-4 ~ 4e-4 量级), 越界时整条序列一起放弃。不做长度归一的 Πρ_t = exp(Σ log ρ_t) 会随长度指数发散或归零, 长序列几乎必然被 clip —— 所以必须取几何平均。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: A > 0 }" @click="A = 1">Â &gt; 0</button>
        <button type="button" :class="{ active: A < 0 }" @click="A = -1">Â &lt; 0</button>
        <button type="button" @click="reseed">换一组</button>
      </div>
      <LabSlider v-model="noise" label="token 级比率噪声" :min="0" :max="0.6" :step="0.02" :format="(v) => v.toFixed(2)" />
      <LabSlider v-model="drift" label="整体漂移 (策略真的变了多少)" :min="-0.2" :max="0.2" :step="0.01" :format="(v) => v.toFixed(2)" />
      <LabSlider v-model="epsSeq" label="GSPO 的 ε (序列级)" :min="0.01" :max="0.2" :step="0.01" :format="(v) => v.toFixed(2)" />
    </template>

    <svg ref="svgEl" viewBox="0 0 640 300" role="img" aria-label="各 token 的对数重要性比率">
      <rect x="40" :y="sy(HI)" width="590" :height="sy(LO) - sy(HI)" fill="var(--warn)" opacity="0.08" />
      <line v-for="b in [HI, LO]" :key="b" x1="40" x2="630" :y1="sy(b)" :y2="sy(b)" stroke="var(--warn)" />
      <text x="632" :y="sy(HI) - 4" text-anchor="end" class="t" fill="var(--warn)">log(1+0.2): token 级上界</text>
      <text x="632" :y="sy(LO) + 13" text-anchor="end" class="t" fill="var(--warn)">log(1−0.2): token 级下界</text>
      <line x1="40" x2="630" :y1="sy(0)" :y2="sy(0)" stroke="var(--border-strong)" />
      <text x="4" :y="sy(0) + 4" class="t">ρ=1</text>
      <g v-for="(v, i) in logr" :key="i">
        <rect :x="bx(i)" :y="Math.min(sy(v), sy(0))" width="34" :height="Math.abs(sy(v) - sy(0))" :fill="clipped[i] ? 'var(--text-dim)' : 'var(--accent)'" :opacity="clipped[i] ? 0.45 : 0.85" />
        <!-- ★ 手放在单个 token 的比率上 -->
        <circle
          class="draggable" :cx="bx(i) + 17" :cy="sy(v)" r="8" :fill="clipped[i] ? 'var(--text-dim)' : 'var(--accent)'" stroke="var(--bg-card)" stroke-width="2"
          tabindex="0" role="slider" :aria-label="`token ${i} 的 log ρ`" :aria-valuenow="v" aria-valuemin="-0.9" aria-valuemax="0.9"
          @pointerdown="start($event, { svg: svgEl, onMove: ({ y }) => setTok(i, (150 - y) / SC) })"
          @keydown.up.prevent="setTok(i, v + 0.05)" @keydown.down.prevent="setTok(i, v - 0.05)"
        />
        <text :x="bx(i) + 17" y="292" text-anchor="middle" class="t">t{{ i }}</text>
      </g>
      <line x1="40" x2="630" :y1="sy(logS)" :y2="sy(logS)" stroke="var(--left)" stroke-width="2.5" />
      <text x="44" :y="sy(logS) - 5" class="t" fill="var(--left)">log s = 平均 = {{ logS.toFixed(3) }}</text>
    </svg>

    <template #stats>
      <div class="kv"><span>GRPO: 被 clip 的 token</span><b :class="nClip ? 'bad' : 'good'">{{ nClip }} / {{ N }}</b></div>
      <div class="kv"><span>GSPO: 序列比率 s</span><b>{{ Math.exp(logS).toFixed(3) }}</b></div>
      <div class="kv"><span>GSPO: 整条序列</span><b :class="seqClipped ? 'bad' : 'good'">{{ seqClipped ? '整条被 clip' : '整条保留' }}</b></div>
      <div class="kv"><span>不归一的 Πρ_t</span><b>{{ Math.exp(logS * N).toFixed(3) }}</b></div>
      <p class="lab-note">
        clip 只拦"对自己有利方向"的越界: Â &gt; 0 拦 ρ 过大, Â &lt; 0 拦 ρ 过小。
        噪声只改变柱子的散布, 不改变平均 —— 所以拖噪声时绿线几乎不动, 被 clip 的 token 数却一直在变。
      </p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { useDrag } from '@/composables/useDrag.js'
import { clamp, mulberry32, randn, range, sum } from '@/utils/labmath.js'

const N = 12, SC = 150, HI = Math.log(1.2), LO = Math.log(0.8)
const sy = (v) => 150 - clamp(v, -0.93, 0.93) * SC
const bx = (i) => 52 + i * 48

const A = ref(1), noise = ref(0.2), drift = ref(0.02), epsSeq = ref(0.05), seed = ref(1)
const svgEl = ref(null)
const { start } = useDrag()

// 单位噪声去均值: 这样"噪声"滑杆严格不改变平均值, 只改散布
const unit = computed(() => { const rand = mulberry32(seed.value * 1777); const z = range(N).map(() => randn(rand)); const m = sum(z) / N; return z.map((x) => x - m) })
const gen = () => unit.value.map((z) => clamp(drift.value + noise.value * z, -0.9, 0.9))
const logr = ref(gen())
watch([noise, drift, unit], () => { logr.value = gen() })
const reseed = () => { seed.value++ }
const setTok = (i, v) => { logr.value = logr.value.map((x, k) => (k === i ? clamp(Math.round(v * 100) / 100, -0.9, 0.9) : x)) }

// token 级: 每个 ρ_t 各自判断是否越界 (只拦有利方向)
const clipped = computed(() => logr.value.map((v) => (A.value > 0 ? v > HI : v < LO)))
const nClip = computed(() => clipped.value.filter(Boolean).length)
// ★ 序列级: s = (π_new(y)/π_old(y))^(1/|y|) = exp(mean_t log ρ_t)
const logS = computed(() => sum(logr.value) / N)
const seqClipped = computed(() => (A.value > 0 ? logS.value > Math.log(1 + epsSeq.value) : logS.value < Math.log(1 - epsSeq.value)))
</script>

<style scoped>
.t { font-size: 11px; fill: var(--text-muted); font-family: "SF Mono", Menlo, monospace; }
</style>
