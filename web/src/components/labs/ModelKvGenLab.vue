<!--
  KV-cache 生成实验台 (对应 llm_models/utils/generation.py:GenerationMixin.generate)。
  只讲一件事: 自回归每步只新增 1 个 token, 旧 token 的 K/V 不会变 —— 存下来就不用重算。
  以及它的边界: 上下文满了窗口左移, 绝对位置全变, cache 作废。
-->
<template>
  <LabFrame
    title="KV cache 生成 — 每一步到底重算了多少"
    sub="上排: 当前这一步喂进模型的 token (橙 = 本步重新算 K/V, 绿 = 直接读缓存)。下排: 每一步的计算量柱状图, 点任意一根柱子跳到那一步。关掉 cache、或者把上下文上限 max_len 拖到比总长度短, 看柱子怎么长高。"
    module="llm_models/utils/generation.py"
    run="python -m llm_models.run_models.language_models.llama.infer_llama"
    :challenge="{
      ask: 'P=6, 生成 16 个 token。先猜开/关 cache 的总计算量差几倍? 再把 max_len 拖到 12: 为什么有 cache 也救不回来?',
      answer: '无 cache 每步重算整个前缀: 6+7+…+21 = 216 次 token 前向; 有 cache 只在第一步 prefill 6 个, 之后每步 1 个: 6+15 = 21, 约 10 倍, 且差距随长度平方增长。max_len=12 时, 长度一超过 12 窗口就要左移: 每个 token 的绝对位置都变了 (RoPE 角度/位置 embedding 全错位), 旧 K/V 不能复用, generate() 只能丢掉 cache 重新 prefill —— 退化回无 cache。这就是生产系统要么不让超长、要么用 SWA 滚动缓存的原因。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: useCache }" @click="useCache = true">use_cache = True</button>
        <button type="button" :class="{ active: !useCache }" @click="useCache = false">use_cache = False (每步重算前缀)</button>
      </div>
      <LabSlider v-model="P" label="prompt 长度 P" :min="2" :max="10" />
      <LabSlider v-model="N" label="生成 token 数" :min="4" :max="20" />
      <LabSlider v-model="maxLen" label="上下文上限 max_len" :min="8" :max="32" />
      <StepPlayer :stepper="stepper" :label="`第 ${stepper.step.value + 1} 步`" />
    </template>

    <div class="cells" :style="{ gridTemplateColumns: `repeat(${P + N}, minmax(18px, 26px))` }">
      <span
        v-for="(c, i) in cellsNow" :key="i" class="cell" :class="c.cls"
        :title="c.tip"
      >{{ i < P ? 'p' + i : 'g' + (i - P) }}</span>
    </div>
    <p class="legend">
      <span class="cell hot">·</span> 本步重算 K/V <span class="cell ok">·</span> 读缓存 <span class="cell on">·</span> 本步新采样出的 token
      <span class="cell dim">·</span> 尚未生成 / 已滑出窗口
    </p>

    <div class="bars" role="group" aria-label="每步计算量">
      <button
        v-for="(f, s) in sim.frames" :key="s" type="button" class="bar" :class="{ now: s === stepper.step.value, prefill: f.computed > 1 }"
        :style="{ height: 8 + (f.computed / sim.maxC) * 92 + 'px' }" :aria-label="`第 ${s + 1} 步, 重算 ${f.computed} 个 token`"
        @click="stepper.step.value = s"
      ><i>{{ f.computed }}</i></button>
    </div>
    <p class="axis-cap">每步喂进模型的 token 数 (= 每层 K/V 投影次数) · 点柱子跳转</p>

    <template #stats>
      <div class="kv"><span>本步重算 token 数</span><b :class="now.computed === 1 ? 'good' : 'bad'">{{ now.computed }}</b></div>
      <div class="kv"><span>缓存里的 K/V 条目</span><b>{{ now.cached }}</b></div>
      <div class="kv"><span>累计 token 前向 (全程)</span><b :class="sim.total <= P + N ? 'good' : 'bad'">{{ sim.total }}</b></div>
      <div class="kv"><span>相对 "全程有效 cache" 的倍数</span><b :class="ratio < 1.5 ? 'good' : 'bad'">{{ ratio.toFixed(1) }}×</b></div>
      <p class="lab-note">
        {{ now.why }}
      </p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import StepPlayer from '@/components/lab/StepPlayer.vue'
import { useStepper } from '@/composables/useStepper.js'
import { range, sum } from '@/utils/labmath.js'

const useCache = ref(true), P = ref(6), N = ref(16), maxLen = ref(32)

// 逐步复刻 GenerationMixin.generate 的三个分支
const sim = computed(() => {
  const frames = []
  let pos = 0 // cache.pos: 缓存里已有多少条
  for (let s = 0; s < N.value; s++) {
    const len = P.value + s                      // 本步开始时序列总长
    const win = Math.min(len, maxLen.value)      // 模型实际能看到的后缀
    let computed_, why
    if (!useCache.value) {
      computed_ = win; pos = 0
      why = `无 cache: 把最近 ${win} 个 token 全部重新前向, 只为了拿最后一个位置的 logits。`
    } else if (pos === 0 || pos + 1 > maxLen.value) {
      // ★ prefill, 或窗口已满: 左移后绝对位置全变, 旧 cache 作废
      computed_ = win; pos = win
      why = s === 0 ? `prefill: 一次性把 ${win} 个 prompt token 的 K/V 算好写入 cache。`
        : `上下文已满 (max_len=${maxLen.value}): 窗口左移使所有绝对位置改变, 旧 cache 作废, 重新 prefill ${win} 个。`
    } else {
      computed_ = 1; pos += 1
      why = `decode: 只喂 1 个新 token, 它的 Q 去查缓存里的 ${pos} 条 K/V。`
    }
    frames.push({ len, win, computed: computed_, cached: useCache.value ? pos : 0, why })
  }
  const total = sum(frames.map((f) => f.computed))
  return { frames, total, maxC: Math.max(...frames.map((f) => f.computed)) }
})
const ratio = computed(() => sim.value.total / (P.value + N.value - 1))

const stepper = useStepper(() => sim.value.frames.length, { interval: 450 })
watch(sim, () => { stepper.pause(); stepper.step.value = Math.min(stepper.step.value, N.value - 1) })
const now = computed(() => sim.value.frames[stepper.step.value] || sim.value.frames[0])

const cellsNow = computed(() => range(P.value + N.value).map((i) => {
  const f = now.value, lo = f.len - f.win
  if (i === f.len) return { cls: 'on', tip: '本步采样出的新 token' }
  if (i > f.len || i < lo) return { cls: 'dim', tip: i > f.len ? '尚未生成' : '已滑出窗口' }
  const fresh = i >= f.len - f.computed
  return { cls: fresh ? 'hot' : 'ok', tip: fresh ? '本步重算 K/V' : '读缓存' }
}))
</script>

<style scoped>
.legend { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 10px; margin: 10px 0 16px; font-size: 12px; color: var(--text-muted); }
.legend .cell { min-width: 16px; width: 16px; height: 16px; }
.bars { display: flex; align-items: flex-end; gap: 3px; height: 120px; padding-top: 18px; border-bottom: 1px solid var(--border-strong); }
.bar { flex: 1; min-width: 12px; max-width: 30px; padding: 0; min-height: 0; border-radius: 3px 3px 0 0; border: 1px solid var(--left); background: color-mix(in srgb, var(--left) 30%, transparent); position: relative; cursor: pointer; }
.bar.prefill { border-color: var(--warn); background: color-mix(in srgb, var(--warn) 30%, transparent); }
.bar.now { outline: 2px solid var(--accent); outline-offset: 1px; }
.bar i { position: absolute; top: -15px; left: 0; right: 0; text-align: center; font-style: normal; font-size: 9px; color: var(--text-dim); font-family: "SF Mono", Menlo, monospace; }
.axis-cap { font-size: 11px; color: var(--text-dim); margin-top: 6px; }
</style>
