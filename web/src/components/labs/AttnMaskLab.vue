<template>
  <div class="lab card">
    <div class="lab-head">
      <h3>注意力掩码实验台 — 谁能看见谁</h3>
      <p class="lab-sub">
        同一套 QKV 投影, 换一张 mask 就是另一个模型: 全因果 = LLaMA, 带状 = Mistral SWA,
        带状+sink = StreamingLLM / GPT-OSS, top-k 稀疏 = DeepSeek DSA。
      </p>
    </div>

    <div class="lab-controls">
      <div class="mode-row" role="group" aria-label="选择注意力掩码模式">
        <button
          v-for="m in modes" :key="m.id"
          type="button"
          :class="{ active: mode === m.id }"
          :aria-pressed="mode === m.id"
          @click="mode = m.id"
        >{{ m.label }}</button>
      </div>
      <div class="ctl">
        <label for="mask-sequence-length">序列长度 T</label>
        <input id="mask-sequence-length" type="range" min="8" max="48" step="4" v-model.number="T" />
        <span class="val mono">{{ T }}</span>
      </div>
      <div class="ctl" v-if="mode === 'swa' || mode === 'sink'">
        <label for="mask-window-size">窗口 W</label>
        <input id="mask-window-size" type="range" min="2" max="16" step="1" v-model.number="W" />
        <span class="val mono">{{ W }}</span>
      </div>
      <div class="ctl" v-if="mode === 'sink'">
        <label for="mask-sink-count">sink 数 S</label>
        <input id="mask-sink-count" type="range" min="1" max="4" step="1" v-model.number="S" />
        <span class="val mono">{{ S }}</span>
      </div>
      <div class="ctl" v-if="mode === 'topk'">
        <label for="mask-top-k">稀疏候选 top-k</label>
        <input id="mask-top-k" type="range" min="2" max="16" step="1" v-model.number="K" />
        <span class="val mono">{{ K }}</span>
      </div>
    </div>

    <div class="lab-body">
      <div class="mask-visual">
        <svg
          :viewBox="`0 0 ${T * cell} ${T * cell}`"
          class="mask-grid"
          role="img"
          :aria-label="maskAriaLabel"
        >
          <template v-for="i in T" :key="i">
            <rect
              v-for="j in T" :key="j"
              :x="(j - 1) * cell" :y="(i - 1) * cell"
              :width="cell - 0.6" :height="cell - 0.6"
              :class="cellClass(i - 1, j - 1)"
            />
          </template>
        </svg>
        <ul class="mask-legend" aria-label="矩阵颜色图例">
          <li><span class="swatch visible" aria-hidden="true"></span>当前 query 可见</li>
          <li v-if="mode === 'sink'"><span class="swatch sink" aria-hidden="true"></span>永久保留的 sink</li>
          <li><span class="swatch evicted" aria-hidden="true"></span>被策略裁掉</li>
          <li><span class="swatch future" aria-hidden="true"></span>未来 token（因果屏蔽）</li>
        </ul>
      </div>

      <div class="lab-stats" aria-live="polite" aria-atomic="true">
        <div class="mode-summary">
          <span>当前机制</span>
          <strong>{{ currentModeLabel }}</strong>
          <p>{{ modeSummary }}</p>
        </div>
        <div class="stat">
          <span class="num mono">{{ visibleCount }}</span>
          <span class="cap">可见格子 (∝ 注意力 FLOPs)</span>
        </div>
        <div class="stat">
          <span class="num mono">{{ pctOfFull }}%</span>
          <span class="cap">相对全因果的计算量</span>
        </div>
        <div class="stat">
          <span class="num mono">{{ kvEntries }}</span>
          <span class="cap">推理 KV cache 条目上限</span>
        </div>
        <p class="note">{{ modeNote }}</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'

const modes = [
  { id: 'full', label: '全因果 (LLaMA)' },
  { id: 'swa', label: '滑动窗口 (Mistral)' },
  { id: 'sink', label: '窗口+sink (StreamingLLM)' },
  { id: 'topk', label: 'top-k 稀疏 (DSA)' },
]

const mode = ref('swa')
const T = ref(24)
const W = ref(6)
const S = ref(2)
const K = ref(6)
const cell = 12

// top-k 稀疏: 对角线 + 确定性伪随机挑 k-1 个历史位置 (模拟 indexer 选中的 token)
const hash = (i, j) => {
  let h = (i * 2654435761 + j * 40503) % 2147483647
  return (h ^ (h >> 7)) % 997
}

const isVisible = (i, j) => {
  if (j > i) return false                       // 因果性永远成立
  if (mode.value === 'full') return true
  if (mode.value === 'swa') return j > i - W.value
  if (mode.value === 'sink') return j > i - W.value || j < S.value
  // topk: 保留对角附近 2 个 + 按 hash 选出的 k-2 个
  if (i - j <= 1) return true
  const kept = []
  for (let p = 0; p <= i - 2; p++) kept.push([hash(i, p), p])
  kept.sort((a, b) => a[0] - b[0])
  return kept.slice(0, Math.max(0, K.value - 2)).some(([, p]) => p === j)
}

const cellClass = (i, j) => {
  if (j > i) return 'cell future'
  if (!isVisible(i, j)) return 'cell evicted'
  if (mode.value === 'sink' && j < S.value && j <= i - W.value) return 'cell sink'
  return 'cell visible'
}

const visibleCount = computed(() => {
  let n = 0
  for (let i = 0; i < T.value; i++)
    for (let j = 0; j <= i; j++) if (isVisible(i, j)) n++
  return n
})
const fullCount = computed(() => (T.value * (T.value + 1)) / 2)
const pctOfFull = computed(() => Math.round((visibleCount.value / fullCount.value) * 100))

const kvEntries = computed(() => {
  if (mode.value === 'full') return `${T.value} = O(T)`
  if (mode.value === 'swa') return `${Math.min(T.value, W.value)} = O(W)`
  if (mode.value === 'sink') return `${Math.min(T.value, W.value + S.value)} = O(S+W)`
  return `${T.value} = O(T)`
})

const modeNote = computed(() => ({
  full: '每个位置看全部历史。计算 O(T²), KV cache O(T) — 长上下文的双重瓶颈。',
  swa: '只看最近 W 个。信息跨层接力, L 层感受野 ≈ L·W。代码: llm_models/.../mistral.py',
  sink: '窗口滑动 + 永远保留开头 S 个"注意力下水道"。代码: llm_infer/m16_attention_sinks',
  topk: 'DSA: cache 全保留 (O(T) 显存), 但每步只对 indexer 选出的 k 个算注意力 — 省计算不省显存。',
}[mode.value]))

const currentModeLabel = computed(() => modes.find((item) => item.id === mode.value)?.label || mode.value)

const modeSummary = computed(() => ({
  full: `第 ${T.value} 个 token 可以回看全部 ${T.value} 个历史位置；上下文越长，计算与 KV 都持续增长。`,
  swa: `每个 query 只保留最近 ${Math.min(T.value, W.value)} 个位置；旧信息必须跨层逐步传递。`,
  sink: `最近 ${Math.min(T.value, W.value)} 个位置负责局部信息，开头 ${Math.min(T.value, S.value)} 个 sink 负责稳定全局锚点。`,
  topk: `全部 ${T.value} 个 KV 仍被保存，但每个 query 只选择最多 ${Math.min(T.value, K.value)} 个位置参与重计算。`,
}[mode.value]))

const maskAriaLabel = computed(() =>
  `${currentModeLabel.value} 的 ${T.value}×${T.value} 因果掩码。${visibleCount.value} 个历史关系可见，占全因果计算的 ${pctOfFull.value}%。${modeSummary.value}`
)
</script>

<style scoped>
.lab { margin-bottom: 16px; }
.lab-head h3 { margin-bottom: 4px; }
.lab-sub { color: var(--text-muted); font-size: 13px; line-height: 1.6; margin-bottom: 14px; }

.lab-controls { display: flex; flex-direction: column; gap: 8px; margin-bottom: 14px; }
.mode-row { display: flex; flex-wrap: wrap; gap: 8px; }
.mode-row button { min-height: 40px; font-size: 12px; }
.ctl { display: grid; grid-template-columns: 110px 1fr 56px; gap: 10px; align-items: center; }
.ctl label { font-size: 12px; color: var(--text-muted); }
.ctl .val { font-size: 12px; text-align: right; color: var(--accent); font-variant-numeric: tabular-nums; }

.lab-body { display: grid; grid-template-columns: minmax(220px, 320px) 1fr; gap: 18px; align-items: start; }
.mask-grid { width: 100%; border-radius: var(--radius-sm); background: var(--code-bg); padding: 4px; }
.mask-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 12px;
  margin-top: 8px;
  list-style: none;
  color: var(--text-muted);
  font-size: 10.5px;
}
.mask-legend li { display: inline-flex; align-items: center; gap: 5px; }
.swatch { width: 10px; height: 10px; border-radius: 2px; border: 1px solid var(--border-strong); }
.swatch.visible { background: var(--accent); }
.swatch.sink { background: var(--eye); }
.swatch.evicted { background: var(--border); }
.swatch.future { background: transparent; }

.cell { transition: fill 0.15s; }
.cell.visible { fill: var(--accent); }
.cell.sink { fill: var(--eye); }
.cell.evicted { fill: var(--border); }
.cell.future { fill: transparent; }

.lab-stats { display: flex; flex-direction: column; gap: 10px; }
.mode-summary {
  padding: 11px 13px;
  border: 1px solid color-mix(in srgb, var(--accent) 38%, var(--border));
  border-radius: var(--radius-sm);
  background: var(--accent-soft);
}
.mode-summary span { display: block; color: var(--accent); font-size: 9.5px; font-weight: 700; letter-spacing: 0.7px; }
.mode-summary strong { display: block; margin-top: 2px; font-size: 13px; }
.mode-summary p { margin-top: 4px; color: var(--text-muted); font-size: 11.5px; line-height: 1.65; text-wrap: pretty; }
.stat { display: flex; align-items: baseline; gap: 10px; }
.stat .num { font-size: 20px; color: var(--text); font-variant-numeric: tabular-nums; }
.stat .cap { font-size: 12px; color: var(--text-muted); }
.note { font-size: 12px; color: var(--text-muted); line-height: 1.7; border-left: 2px solid var(--accent-soft); padding-left: 10px; }

@media (max-width: 960px) {
  .lab-body { grid-template-columns: 1fr; }
}

@media (max-width: 640px) {
  .ctl { grid-template-columns: 1fr auto; gap: 2px 10px; }
  .ctl input[type="range"] { grid-column: 1 / -1; }
}
</style>
