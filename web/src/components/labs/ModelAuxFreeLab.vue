<!--
  Aux-loss-free 负载均衡实验台 (对应 llm_models/models/moe/deepseekV3.py:DeepSeekMoE.update_routing_bias)。
  只讲一件事: 给每个专家一个不参与梯度的偏置 b, 只用来 “选谁”, 每步按 b += γ·sign(平均负载 − 自己的负载) 调 ——
  负载被拉平, 而门控权重 (用不带 b 的分数算) 一点没被扭曲。对照组: 传统 aux loss 是直接改路由分数。
-->
<template>
  <LabFrame
    title="Aux-loss-free 均衡 — 偏置只管选人, 不管权重"
    sub="256 个 token、8 个专家、top-2 路由, 路由器天生偏爱前几个专家。按播放看每一步负载柱怎么被拉平; 点任意专家的柱子, 下方折线会叠加它的偏置轨迹。拖 γ 感受 “太小追不上、太大来回震”。"
    module="llm_models/models/moe/deepseekV3.py"
    run="python -m llm_models.run_models.moe.deepseek.train_deepseek"
    :challenge="{
      ask: '两种方法都能把 max/mean 压到 1.2 左右 (256 个 token 的抽样噪声下限)。先猜: “路由分数被改动量” 这一项, 两者分别是多少? 再把 γ 拖到 0.1 看会发生什么。',
      answer: 'Aux-loss-free 的偏置 b 只加在 top-k 的排序分数上, 门控权重仍取自原始 sigmoid 分数, 所以分数改动量恒为 0 —— 均衡和语言建模目标互不干扰。Aux loss 则是往总 loss 里加 α·E·Σf_e·P_e, 梯度直接把热门专家的路由分数往下压。负载是平了, 但 “这个 token 本该更信任哪个专家” 的信息也被改了。α 大了伤效果、小了不均衡。γ=0.1 时偏置每步跳 0.1, 而 sigmoid 分数之间的差距也就 0.1 量级, 于是每步都矫枉过正, 负载来回震荡。因为只用 sign, 步长与 batch 大小无关, 好调, 但必须足够小 (DeepSeek-V3 用 0.001)。',
    }"
  >
    <template #controls>
      <div class="row">
        <button v-for="m in MODES" :key="m.id" type="button" :class="{ active: mode === m.id }" @click="mode = m.id">{{ m.label }}</button>
        <button type="button" @click="seed++">换一批 token</button>
      </div>
      <LabSlider v-if="mode === 'bias'" v-model="gExp" label="偏置步长 γ" :min="-3" :max="-1" :step="0.1" :format="(t) => (10 ** t).toFixed(3)" />
      <LabSlider v-if="mode === 'aux'" v-model="eta" label="aux 权重×学习率 η" :min="0.05" :max="2" :step="0.05" :format="(t) => t.toFixed(2)" />
      <LabSlider v-model="skew" label="路由器偏心程度" :min="0" :max="2" :step="0.1" :format="(t) => t.toFixed(1)" />
      <StepPlayer :stepper="stepper" :label="`训练步 ${stepper.step.value}`" />
    </template>

    <div class="bars" role="group" aria-label="各专家负载, 点击选择">
      <div class="meanline" :style="{ bottom: 34 + (now.mean / loadMax) * 120 + 'px' }"><span class="mono">均值 {{ now.mean }}</span></div>
      <button v-for="(l, e) in now.load" :key="e" type="button" class="ex" :class="{ active: e === selE }" :aria-pressed="e === selE" @click="selE = e">
        <span class="mono n">{{ l }}</span>
        <div class="col" :class="l > now.mean * 1.2 ? 'hotc' : l < now.mean * 0.8 ? 'coldc' : 'okc'" :style="{ height: (l / loadMax) * 120 + 'px' }" />
        <span class="mono">e{{ e }}</span>
        <span class="mono b">{{ now.adj[e] >= 0 ? '+' : '' }}{{ now.adj[e].toFixed(2) }}</span>
      </button>
    </div>
    <p class="cap">柱 = 本步每个专家接到的 token 数 · 底下的数 = {{ mode === 'aux' ? 'aux loss 累计把该专家 logit 改了多少' : '该专家的路由偏置 b' }}</p>

    <svg viewBox="0 0 520 120" role="img" aria-label="不均衡度随训练步变化">
      <line x1="30" x2="515" :y1="iy(1)" :y2="iy(1)" class="ideal" /><text x="26" :y="iy(1) + 3" class="yl">1.0</text>
      <text x="26" :y="iy(imbMax) + 8" class="yl">{{ imbMax.toFixed(1) }}</text>
      <polyline :points="imbLine" class="imb" />
      <polyline :points="adjLine" class="adj" />
      <line :x1="sx(stepper.step.value)" :x2="sx(stepper.step.value)" y1="4" y2="112" class="cursor" />
      <text x="515" y="12" class="lg">紫 = 不均衡度 max/mean · 橙 = e{{ selE }} 的偏置轨迹</text>
    </svg>

    <template #stats>
      <div class="kv"><span>不均衡度 max/mean</span><b :class="now.imb < 1.3 ? 'good' : 'bad'">{{ now.imb.toFixed(2) }}</b></div>
      <div class="kv"><span>首次 &lt; 1.3 的步数</span><b>{{ firstOk < 0 ? '未达到' : firstOk }}</b></div>
      <div class="kv"><span>路由分数被改动量</span><b :class="now.distort < 1e-9 ? 'good' : 'bad'">{{ now.distort.toFixed(3) }}</b></div>
      <div class="kv"><span>最后 20 步负载 CV (std/mean)</span><b :class="cvTail < 0.15 ? 'good' : 'bad'">{{ cvTail.toFixed(3) }}</b></div>
      <p class="lab-note">{{ MODES.find((m) => m.id === mode).note }}</p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import StepPlayer from '@/components/lab/StepPlayer.vue'
import { useStepper } from '@/composables/useStepper.js'
import { mulberry32, randn, range, sum } from '@/utils/labmath.js'

const N = 256, E = 8, K = 2, STEPS = 120
const PREF = [1.5, 1.0, 0.5, 0.1, -0.1, -0.4, -0.7, -1.0] // 路由器对各专家的先天偏好 (logit)
const MODES = [
  { id: 'bias', label: 'Aux-loss-free (偏置 b)', note: 'b 不进计算图、不收梯度, 每步训练后按负载用 sign 更新。选人用 s+b, 加权用 s。' },
  { id: 'aux', label: '传统 aux loss', note: 'L_aux = α·E·Σ f_e·P_e 的梯度直接压低热门专家的路由 logit: 分数本身被改了, 门控权重跟着变。' },
  { id: 'none', label: '不做均衡', note: '热门专家被选得多 → 学得好 → 更被偏爱, 真实训练里会滚雪球直到路由坍缩。' },
]
const mode = ref('bias'), gExp = ref(-2), eta = ref(0.5), skew = ref(1), seed = ref(1), selE = ref(0)
const sig = (x) => 1 / (1 + Math.exp(-x))

const frames = computed(() => {
  const gamma = 10 ** gExp.value
  let b = Array(E).fill(0), c = Array(E).fill(0)
  return range(STEPS + 1).map((step) => {
    const r = mulberry32(seed.value * 1000 + step)          // 每步一批新 token
    const load = Array(E).fill(0), gP = Array(E).fill(0)
    let distort = 0
    for (let n = 0; n < N; n++) {
      const raw = PREF.map((p) => randn(r) + skew.value * p)
      const s = raw.map((l, e) => sig(l + c[e]))
      // ★ 选人用 s + b, 但权重只用 s —— b 不碰门控
      const pick = range(E).sort((x, y) => s[y] + b[y] - (s[x] + b[x])).slice(0, K)
      pick.forEach((e) => load[e]++)
      s.forEach((v, e) => { gP[e] += v * (1 - v) / N; distort += Math.abs(v - sig(raw[e])) / (N * E) })
    }
    const mean = (N * K) / E
    const out = { load: [...load], adj: mode.value === 'aux' ? [...c] : [...b], mean, imb: Math.max(...load) / mean, distort }
    if (mode.value === 'bias') b = b.map((v, e) => v + gamma * Math.sign(mean - load[e]))
    if (mode.value === 'aux') {
      const g = c.map((_, e) => E * (load[e] / (N * K)) * gP[e])   // ∂(E·Σ f·P)/∂c_e, f 视为常数
      const gm = sum(g) / E                                        // 去掉公共分量: 整体平移不改变谁被选, 真实路由 (softmax) 对它也不敏感
      c = c.map((v, e) => v - eta.value * (g[e] - gm))
    }
    return out
  })
})

const stepper = useStepper(() => STEPS + 1, { interval: 90 })
stepper.step.value = STEPS
watch([mode, seed], () => { stepper.pause(); stepper.step.value = STEPS })
const now = computed(() => frames.value[stepper.step.value])
const loadMax = computed(() => Math.max(...frames.value.map((f) => Math.max(...f.load))))
const firstOk = computed(() => frames.value.findIndex((f) => f.imb < 1.3))
// 与 train_deepseek 打印的同一个指标: 最后 20 步负载的变异系数 std/mean 取平均
const cv = (load) => { const m = sum(load) / E; return Math.sqrt(sum(load.map((l) => (l - m) ** 2)) / E) / m }
const cvTail = computed(() => sum(frames.value.slice(-20).map((f) => cv(f.load))) / 20)

const imbMax = computed(() => Math.max(1.5, ...frames.value.map((f) => f.imb)))
const sx = (i) => 30 + (i / STEPS) * 485
const iy = (v) => 110 - ((v - 0.9) / (imbMax.value - 0.9)) * 86
const imbLine = computed(() => frames.value.map((f, i) => `${sx(i).toFixed(1)},${iy(f.imb).toFixed(1)}`).join(' '))
const adjLine = computed(() => {
  const a = frames.value.map((f) => f.adj[selE.value]), m = Math.max(0.01, ...a.map(Math.abs))
  return a.map((v, i) => `${sx(i).toFixed(1)},${(60 - (v / m) * 48).toFixed(1)}`).join(' ')
})
</script>

<style scoped>
.bars { position: relative; display: flex; gap: 6px; align-items: flex-end; height: 190px; padding-top: 10px; }
.ex { flex: 1; min-width: 30px; max-width: 60px; padding: 2px; min-height: 0; display: grid; justify-items: center; gap: 2px; font-size: 10px; background: transparent; color: var(--text-muted); border-color: transparent; }
.ex.active { border-color: var(--accent); background: var(--accent-soft); color: var(--text); }
.col { width: 70%; border-radius: 3px 3px 0 0; min-height: 1px; }
.okc { background: var(--left); } .hotc { background: var(--danger); } .coldc { background: var(--warn); }
.b { color: var(--eye); }
.meanline { position: absolute; left: 0; right: 0; border-top: 1px dashed var(--text-dim); font-size: 10px; color: var(--text-dim); pointer-events: none; }
.meanline span { position: absolute; right: 0; top: -14px; }
.cap { font-size: 11px; color: var(--text-dim); margin: 6px 0 10px; }
.ideal { stroke: var(--left); stroke-dasharray: 4 4; }
.yl { text-anchor: end; font-size: 9px; fill: var(--text-dim); }
.lg { text-anchor: end; font-size: 9px; fill: var(--text-dim); }
.imb { fill: none; stroke: var(--accent); stroke-width: 2; }
.adj { fill: none; stroke: var(--eye); stroke-width: 1.2; }
.cursor { stroke: var(--text-muted); stroke-dasharray: 3 3; }
</style>
