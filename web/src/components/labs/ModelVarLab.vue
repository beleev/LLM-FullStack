<!--
  VAR next-scale 预测实验台 (对应 llm_models/models/generative/var.py:VARModel.sample_tokens 与 layers/diffusion/vq.py 的多尺度残差量化)。
  只讲一件事: 图像的自回归单位不必是 “下一个像素/patch”, 可以是 “下一个分辨率”: 1×1 → 2×2 → 4×4 → 8×8,
  每一级量化的是 “上一级还没解释掉的残差”, 同一级内所有 token 并行生成。
-->
<template>
  <LabFrame
    title="VAR — 下一个尺度, 而不是下一个像素"
    sub="目标是一张 8×8 灰度图。按播放或点下面的尺度缩略图: 每一步, VAR 一次前向就并行吐出整张 s×s 的 token map, 叠加到累计重建上; 右边是光栅顺序自回归用同样次数的前向能画出多少。拖码本大小看量化粗细的影响。"
    module="llm_models/models/generative/var.py"
    run="python -m llm_models.run_models.generative.var.infer_var"
    :challenge="{
      ask: '走到第 2 步 (2×2)。先猜: VAR 这时的重建和光栅自回归这时的画面, 哪个更 “像” 目标? 全部走完各需要多少次前向? VAR 的 token 总数比光栅多还是少?',
      answer: '2 次前向后, VAR 已经有了整张图的大致明暗布局 (先构图再细化), 光栅自回归只画了左上角 2 个像素 —— 它要到第 64 步才知道右下角长什么样, 而且一维展开破坏了二维邻接关系。VAR 总共 4 次前向 (每级内部并行), 光栅要 64 次。代价: VAR 的 token 总数是 1+4+16+64 = 85 > 64, 序列更长; 且每一级只量化 “残差” (目标减去之前所有级上采样之和), 所以粗尺度的错误可以被细尺度修正。训练时用块状因果 mask: 同一尺度内互相可见, 只能看更粗的尺度。',
    }"
  >
    <template #controls>
      <LabSlider v-model="K" label="码本大小 K (量化级数)" :min="2" :max="16" />
      <div class="row">
        <button v-for="(s, i) in SCALES" :key="s" type="button" :class="{ active: step === i }" @click="stepper.step.value = i">{{ s }}×{{ s }}</button>
        <button type="button" @click="seed++">换一张图</button>
      </div>
      <StepPlayer :stepper="stepper" :label="`第 ${step + 1} 次前向`" />
    </template>

    <div class="panels">
      <figure><div class="img" :style="gridOf(8)"><i v-for="(v, i) in sim.target.flat()" :key="i" :style="{ background: shade(v) }" /></div><figcaption>目标图</figcaption></figure>
      <figure><div class="img" :style="gridOf(SCALES[step])"><i v-for="(v, i) in now.tokens.flat()" :key="i" class="tk mono" :style="{ background: shade(v.val) }">{{ SCALES[step] <= 4 ? v.idx : '' }}</i></div><figcaption>本步并行生成的 token map ({{ SCALES[step] }}×{{ SCALES[step] }}, 量化残差)</figcaption></figure>
      <figure><div class="img" :style="gridOf(8)"><i v-for="(v, i) in now.recon.flat()" :key="i" :style="{ background: shade(v) }" /></div><figcaption>VAR 累计重建 (前 {{ step + 1 }} 级之和)</figcaption></figure>
      <figure><div class="img" :style="gridOf(8)"><i v-for="(v, i) in sim.raster.flat()" :key="i" :style="i <= step ? { background: shade(v) } : {}" :class="{ todo: i > step }" /></div><figcaption>光栅自回归: {{ step + 1 }} 次前向 = {{ step + 1 }} 个像素</figcaption></figure>
    </div>

    <template #stats>
      <div class="kv"><span>VAR 重建 MSE</span><b :class="now.mse < 0.01 ? 'good' : ''">{{ now.mse.toFixed(4) }}</b></div>
      <div class="kv"><span>本步并行生成 token 数</span><b>{{ SCALES[step] ** 2 }}</b></div>
      <div class="kv"><span>生成整张图的前向次数 (VAR / 光栅)</span><b class="good">4 / 64</b></div>
      <div class="kv"><span>token 总数 (VAR / 光栅)</span><b>85 / 64</b></div>
      <p class="lab-note">每一级的输入不是原图, 而是 “目标 − 已有重建” 的残差: 下采样到 s×s → 查最近的码字 → 上采样回 8×8 加到重建上。</p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import StepPlayer from '@/components/lab/StepPlayer.vue'
import { useStepper } from '@/composables/useStepper.js'
import { mulberry32, clamp, range, heat } from '@/utils/labmath.js'

const H = 8, SCALES = [1, 2, 4, 8]
const K = ref(8), seed = ref(1)

const sim = computed(() => {
  const r = mulberry32(seed.value * 131)
  const blobs = range(3).map(() => ({ x: r() * 7, y: r() * 7, a: (r() - 0.35) * 1.8, w: 2 + r() * 8 }))
  const target = range(H).map((y) => range(H).map((x) => clamp(-0.3 + blobs.reduce((s, b) => s + b.a * Math.exp(-((x - b.x) ** 2 + (y - b.y) ** 2) / b.w), 0), -1, 1)))
  const levels = range(K.value).map((i) => -1 + (2 * i) / (K.value - 1))   // 标量码本
  const quant = (v) => { const idx = levels.reduce((b, l, i) => (Math.abs(l - v) < Math.abs(levels[b] - v) ? i : b), 0); return { idx, val: levels[idx] } }

  let recon = range(H).map(() => Array(H).fill(0))
  const stages = SCALES.map((s) => {
    const blk = H / s
    // ★ 残差 → 面积平均下采样到 s×s → 量化 → 最近邻上采样 → 累加
    const tokens = range(s).map((ty) => range(s).map((tx) => {
      let acc = 0
      for (let y = ty * blk; y < (ty + 1) * blk; y++) for (let x = tx * blk; x < (tx + 1) * blk; x++) acc += target[y][x] - recon[y][x]
      return quant(acc / (blk * blk))
    }))
    recon = recon.map((row, y) => row.map((v, x) => v + tokens[Math.floor(y / blk)][Math.floor(x / blk)].val))
    const mse = recon.flat().reduce((a, v, i) => a + (v - target.flat()[i]) ** 2, 0) / (H * H)
    return { tokens, recon, mse }
  })
  return { target, stages, raster: target.map((row) => row.map((v) => quant(v).val)) }
})

const stepper = useStepper(() => SCALES.length, { interval: 1100 })
const step = computed(() => stepper.step.value)
const now = computed(() => sim.value.stages[step.value])
const shade = (v) => heat((v + 1) / 2, 'var(--text)')
const gridOf = (n) => ({ gridTemplateColumns: `repeat(${n}, 1fr)` })
</script>

<style scoped>
.panels { display: grid; grid-template-columns: repeat(auto-fit, minmax(128px, 1fr)); gap: 14px; }
figure { margin: 0; display: grid; gap: 6px; justify-items: center; }
figcaption { font-size: 11px; color: var(--text-dim); text-align: center; line-height: 1.4; }
.img { display: grid; width: 128px; height: 128px; gap: 1px; border: 1px solid var(--border-strong); background: var(--bg); }
.img i { display: grid; place-items: center; font-style: normal; font-size: 10px; color: var(--accent); }
.img i.todo { background: repeating-linear-gradient(45deg, transparent 0 3px, var(--border) 3px 4px); }
</style>
