<template>
  <div class="lab card">
    <div class="lab-head">
      <h3>GRPO 实验台 — 组内排名替代 critic</h3>
      <p class="lab-sub">
        同一个 prompt 采 G 条回复，奖励减组均值除组标准差就是优势 Â —— 不需要任何 value 网络。
        对应 llm_finetune/run_finetune/grpo/train_grpo.py。
      </p>
    </div>

    <div class="lab-controls">
      <div class="mode-row">
        <button @click="seed++">重新采样</button>
      </div>
      <div class="ctl">
        <label>组大小 G</label>
        <input type="range" min="4" max="16" step="1" v-model.number="G" />
        <span class="val mono">{{ G }}</span>
      </div>
      <div class="ctl">
        <label>β (KL 强度)</label>
        <input type="range" min="0" max="0.10" step="0.01" v-model.number="beta" />
        <span class="val mono">{{ beta.toFixed(2) }}</span>
      </div>
    </div>

    <div class="lab-body">
      <div class="charts">
        <div>
          <p class="chart-title">奖励 r</p>
          <div class="chart">
            <div class="mean-line" :style="{ bottom: mean * 100 + '%' }">
              <span class="mean-tag mono">组均值 μ</span>
            </div>
            <div v-for="(r, idx) in rewards" :key="idx" class="bar-slot">
              <div class="bar r-bar" :style="{ height: r * 100 + '%' }"></div>
            </div>
          </div>
        </div>
        <div>
          <p class="chart-title">优势 Â = (r − μ) / σ</p>
          <div class="chart">
            <div class="zero-axis"></div>
            <div v-for="(a, idx) in advantages" :key="idx" class="bar-slot">
              <div class="bar" :class="a >= 0 ? 'a-pos' : 'a-neg'" :style="advStyle(a)"></div>
            </div>
          </div>
          <div class="x-labels">
            <span v-for="(a, idx) in advantages" :key="idx" class="mono">{{ a.toFixed(1) }}</span>
          </div>
        </div>
      </div>

      <div class="lab-stats">
        <div class="stat">
          <span class="num mono">{{ mean.toFixed(3) }}</span>
          <span class="cap">组均值 μ</span>
        </div>
        <div class="stat">
          <span class="num mono">{{ std.toFixed(3) }}</span>
          <span class="cap">组标准差 σ</span>
        </div>
        <div class="stat">
          <span class="num mono">{{ maxAdv.toFixed(2) }} / {{ minAdv.toFixed(2) }}</span>
          <span class="cap">max Â / min Â</span>
        </div>
        <p class="kl-row">β·KL 惩罚: 当前 β = {{ beta.toFixed(2) }} — KL 把策略锚在参考模型附近，β 越大越保守。</p>
        <table class="cmp">
          <thead>
            <tr><th></th><th>PPO</th><th>GRPO</th></tr>
          </thead>
          <tbody>
            <tr><td>需要 critic</td><td>是</td><td>否</td></tr>
            <tr><td>同时驻留模型</td><td>4 个 (policy/ref/RM/critic)</td><td>2-3 个</td></tr>
            <tr><td>baseline 来源</td><td>value 网络预测</td><td>组内均值</td></tr>
            <tr><td>奖励来源</td><td>reward model</td><td>规则验证 (RLVR) 或 RM</td></tr>
          </tbody>
        </table>
        <p class="note">
          正优势的回复 → 提高概率；负优势 → 压低。把"比组里平均好"变成监督信号，这就是 R1 的训练配方核心。
        </p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'

const G = ref(8)
const beta = ref(0.02)
const seed = ref(1)

// 确定性伪随机: 同一 seed 永远得到同一组奖励, "重新采样" 仅递增 seed
const mulberry32 = (a) => () => {
  a |= 0
  a = (a + 0x6d2b79f5) | 0
  let t = Math.imul(a ^ (a >>> 15), 1 | a)
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296
}

const clamp01 = (x) => Math.min(1, Math.max(0, x))

const rewards = computed(() => {
  const prng = mulberry32(seed.value * 977)
  return Array.from({ length: G.value }, () => clamp01(0.45 + (prng() - 0.5) * 0.7))
})

const mean = computed(() => rewards.value.reduce((s, r) => s + r, 0) / rewards.value.length)
const std = computed(() => {
  const m = mean.value
  return Math.sqrt(rewards.value.reduce((s, r) => s + (r - m) ** 2, 0) / rewards.value.length)
})
const advantages = computed(() => rewards.value.map((r) => (r - mean.value) / (std.value + 1e-4)))
const maxAdv = computed(() => Math.max(...advantages.value))
const minAdv = computed(() => Math.min(...advantages.value))
const maxAbs = computed(() => Math.max(...advantages.value.map((a) => Math.abs(a)), 1e-6))

const advStyle = (a) => {
  const h = `${(Math.abs(a) / maxAbs.value) * 47}%`
  return a >= 0 ? { bottom: '50%', height: h } : { top: '50%', height: h }
}
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

.charts { display: flex; flex-direction: column; gap: 14px; }
.chart-title { font-size: 12px; color: var(--text-muted); margin-bottom: 4px; }
.chart { position: relative; display: flex; gap: 3px; height: 110px; padding: 0 4px; background: var(--code-bg); border-radius: var(--radius-sm); }
.bar-slot { flex: 1; position: relative; }
.bar { position: absolute; left: 18%; width: 64%; border-radius: 1px; }
.r-bar { bottom: 0; background: var(--eye); }
.a-pos { background: var(--left); }
.a-neg { background: var(--danger); }
.mean-line { position: absolute; left: 4px; right: 4px; border-top: 1px dashed var(--warn); z-index: 1; }
.mean-tag { position: absolute; right: 0; bottom: 1px; font-size: 9px; color: var(--warn); }
.zero-axis { position: absolute; left: 4px; right: 4px; top: 50%; border-top: 1px solid var(--border); }
.x-labels { display: flex; gap: 3px; padding: 2px 4px 0; }
.x-labels span { flex: 1; text-align: center; font-size: 8px; color: var(--text-dim); overflow: hidden; white-space: nowrap; }

.lab-stats { display: flex; flex-direction: column; gap: 10px; }
.stat { display: flex; align-items: baseline; gap: 10px; }
.stat .num { font-size: 20px; color: var(--text); }
.stat .cap { font-size: 12px; color: var(--text-muted); }
.kl-row { font-size: 12px; color: var(--text-muted); line-height: 1.6; }
.cmp { width: 100%; border-collapse: collapse; font-size: 11px; }
.cmp th, .cmp td { border: 1px solid var(--border); padding: 4px 6px; text-align: left; color: var(--text-muted); }
.cmp th { color: var(--text); background: var(--bg-elev); }
.note { font-size: 12px; color: var(--text-dim); line-height: 1.7; border-left: 2px solid var(--accent-soft); padding-left: 10px; }

@media (max-width: 960px) {
  .lab-body { grid-template-columns: 1fr; }
}
</style>
