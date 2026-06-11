<template>
  <div class="lab card">
    <div class="lab-head">
      <h3>温度实验台 — 一根滑杆连接采样与蒸馏</h3>
      <p class="lab-sub">
        softmax(z/T)。T&lt;1 让分布更尖（采样更确定），T&gt;1 把分布压平（蒸馏的"暗知识"显形）。
      </p>
    </div>

    <div class="lab-controls">
      <div class="mode-row">
        <button @click="seed++">重新随机 logits</button>
      </div>
      <div class="ctl">
        <label>温度 T</label>
        <input type="range" min="0.1" max="5" step="0.1" v-model.number="T" />
        <span class="val mono">{{ T.toFixed(1) }}</span>
      </div>
    </div>

    <div class="lab-body">
      <div>
        <div class="chart">
          <div v-for="(p, idx) in probs" :key="idx" class="bar-slot">
            <span class="bar-val mono" :style="{ bottom: `calc(${barPct(idx)}% + 3px)` }">
              {{ p.toFixed(2) }}
            </span>
            <div class="bar" :style="barStyle(idx)"></div>
          </div>
        </div>
        <div class="x-labels">
          <span v-for="idx in 8" :key="idx" class="mono">tok{{ idx - 1 }}</span>
        </div>
      </div>

      <div class="lab-stats">
        <div class="stat">
          <span class="num mono">{{ entropy.toFixed(2) }}</span>
          <span class="cap">熵 H(p) (nats)</span>
        </div>
        <div class="stat">
          <span class="num mono">{{ top1.toFixed(2) }}</span>
          <span class="cap">top-1 概率</span>
        </div>
        <div class="stat">
          <span class="num mono">{{ effective.toFixed(1) }}</span>
          <span class="cap">有效候选数 exp(H)</span>
        </div>
        <p class="regime cold">T → 0: 退化为 argmax (greedy 解码)</p>
        <p class="regime base">T = 1: 模型原始分布</p>
        <p class="regime hot">
          T &gt; 1: 次优 token 的相对排序被放大 —— 蒸馏时 student 学到的不止正确答案
          (KL 项要乘 T² 补偿梯度)
        </p>
        <p class="note">
          同一公式两处复用: 解码采样 llm_infer/m10_sampling, 知识蒸馏 llm_finetune/run_finetune/distill。
        </p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'

const T = ref(1.0)
const seed = ref(1)

// 确定性伪随机: logits 只随 seed 变, 滑动 T 不会重抽
const mulberry32 = (a) => () => {
  a |= 0
  a = (a + 0x6d2b79f5) | 0
  let t = Math.imul(a ^ (a >>> 15), 1 | a)
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296
}

const logits = computed(() => {
  const prng = mulberry32(seed.value * 2731)
  const zs = Array.from({ length: 8 }, () => prng() * 6 - 1)
  return [...zs].sort((a, b) => b - a) // 降序排列, 视觉稳定: tok0 永远是 top-1
})

// 数值稳定 softmax: 先减最大值再取指数
const probs = computed(() => {
  const scaled = logits.value.map((z) => z / T.value)
  const m = Math.max(...scaled)
  const exps = scaled.map((s) => Math.exp(s - m))
  const sum = exps.reduce((a, b) => a + b, 0)
  return exps.map((e) => e / sum)
})

const entropy = computed(() =>
  probs.value.reduce((s, p) => (p > 0 ? s - p * Math.log(p) : s), 0)
)
const top1 = computed(() => probs.value[0])
const effective = computed(() => Math.exp(entropy.value))

const barPct = (idx) => (probs.value[idx] / probs.value[0]) * 100

const barStyle = (idx) => ({
  height: `${barPct(idx)}%`,
  background: idx === 0 ? 'var(--accent)' : 'var(--text-dim)',
  opacity: idx === 0 ? 1 : 0.4,
})
</script>

<style scoped>
.lab { margin-bottom: 16px; }
.lab-head h3 { margin-bottom: 4px; }
.lab-sub { color: var(--text-muted); font-size: 13px; line-height: 1.6; margin-bottom: 14px; }

.lab-controls { display: flex; flex-direction: column; gap: 8px; margin-bottom: 14px; }
.mode-row { display: flex; flex-wrap: wrap; gap: 8px; }
.mode-row button { font-size: 12px; }
.ctl { display: grid; grid-template-columns: 110px 1fr 56px; gap: 10px; align-items: center; }
.ctl label { font-size: 12px; color: var(--text-muted); }
.ctl .val { font-size: 12px; text-align: right; color: var(--accent); }

.lab-body { display: grid; grid-template-columns: minmax(220px, 320px) 1fr; gap: 18px; align-items: start; }

.chart { position: relative; display: flex; gap: 4px; height: 190px; padding: 18px 6px 0; background: var(--code-bg); border-radius: var(--radius-sm); }
.bar-slot { flex: 1; position: relative; }
.bar { position: absolute; bottom: 0; left: 14%; width: 72%; border-radius: 2px 2px 0 0; }
.bar-val { position: absolute; left: 0; right: 0; text-align: center; font-size: 10px; color: var(--text-muted); }
.x-labels { display: flex; gap: 4px; padding: 3px 6px 0; }
.x-labels span { flex: 1; text-align: center; font-size: 10px; color: var(--text-dim); }

.lab-stats { display: flex; flex-direction: column; gap: 10px; }
.stat { display: flex; align-items: baseline; gap: 10px; }
.stat .num { font-size: 20px; color: var(--text); }
.stat .cap { font-size: 12px; color: var(--text-muted); }
.regime { font-size: 12px; color: var(--text-dim); line-height: 1.6; border-left: 2px solid var(--border); padding-left: 10px; }
.regime.cold { border-left-color: var(--accent); }
.regime.base { border-left-color: var(--eye); }
.regime.hot { border-left-color: var(--warn); }
.note { font-size: 12px; color: var(--text-dim); line-height: 1.7; border-left: 2px solid var(--accent-soft); padding-left: 10px; }

@media (max-width: 960px) {
  .lab-body { grid-template-columns: 1fr; }
}
</style>
