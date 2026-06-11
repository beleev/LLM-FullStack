<template>
  <div class="lab card">
    <div class="lab-head">
      <h3>Ring Attention 步进器 — KV 块沿环传递</h3>
      <p class="lab-sub">
        序列切成 D 段常驻 D 张卡；每一步 KV 块向右邻居传一格，D-1 步后每张卡都见过完整序列，
        但任何时刻只持有 1/D 的 KV。对应 llm_train/m12_sequence_parallel/demo.py。
      </p>
    </div>

    <div class="lab-controls">
      <div class="mode-row">
        <button
          v-for="d in [2, 4, 8]" :key="d"
          :class="{ active: D === d }"
          @click="D = d"
        >D = {{ d }} 卡</button>
        <button :disabled="step === 0" @click="step = Math.max(0, step - 1)">◀ 上一步</button>
        <button :disabled="step === D - 1" @click="step = Math.min(D - 1, step + 1)">▶ 下一步</button>
      </div>
      <div class="ctl">
        <label>步骤 step</label>
        <input type="range" min="0" :max="D - 1" step="1" v-model.number="step" />
        <span class="val mono">{{ step }}</span>
      </div>
    </div>

    <div class="lab-body">
      <div class="ring">
        <template v-for="r in D" :key="r">
          <div class="dev">
            <div class="dev-head mono">卡 {{ r - 1 }}</div>
            <div class="chips">
              <span class="chip q mono">Q_{{ r - 1 }}</span>
              <span class="chip kv mono">KV_{{ srcOf(r - 1) }}</span>
            </div>
            <div class="strip">
              <i
                v-for="b in D" :key="b"
                :class="stripClass(r - 1, b - 1)"
                :title="stripTitle(r - 1, b - 1)"
              />
            </div>
          </div>
          <span v-if="r < D" class="arrow">→</span>
          <span v-else class="ring-back mono">⟲ 环回</span>
        </template>
      </div>

      <div class="lab-stats">
        <div class="stat">
          <span class="num mono">{{ donePairs }} / {{ D * D }}</span>
          <span class="cap">已完成块对 (含因果跳过)</span>
        </div>
        <div class="stat">
          <span class="num mono">{{ skipped }}</span>
          <span class="cap">因果跳过的块对</span>
        </div>
        <div class="stat">
          <span class="num mono">1/{{ D }}</span>
          <span class="cap">每卡常驻 KV: 1/{{ D }} 段 = T/{{ D }} · d · 2 floats</span>
        </div>
        <div class="stat">
          <span class="num mono">{{ step }} / {{ D - 1 }}</span>
          <span class="cap">通信轮数</span>
        </div>
        <div class="cover">
          <span class="cap">覆盖矩阵 (行 = Q 块/卡, 列 = KV 块)</span>
          <div class="cover-grid" :style="{ gridTemplateColumns: `repeat(${D}, 16px)` }">
            <template v-for="r in D" :key="r">
              <i v-for="c in D" :key="c" :class="coverClass(r - 1, c - 1)" />
            </template>
          </div>
        </div>
        <p class="note">
          增量合并用的 online softmax 与 FlashAttention 完全同一个技巧 ——
          单卡分块是 Flash，跨卡传块就是 Ring。通信只发生在相邻卡之间，可与计算重叠。
        </p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, ref, watch } from 'vue'

const D = ref(4)
const step = ref(0)

// 切换卡数时 step 不能越界
watch(D, (d) => { if (step.value > d - 1) step.value = d - 1 })

// 第 step 步, 卡 r 手里的 KV 块编号 (块沿环向右传 = 编号向后回退)
const srcOf = (r) => (r - step.value + D.value) % D.value
// 卡 r 在第几步遇到 KV 块 b
const seenAt = (r, b) => (r - b + D.value) % D.value

const donePairs = computed(() => D.value * (step.value + 1))
const skipped = computed(() => {
  let n = 0
  for (let r = 0; r < D.value; r++)
    for (let s = 0; s <= step.value; s++)
      if ((r - s + D.value) % D.value > r) n++
  return n
})

const stripClass = (r, b) => {
  if (seenAt(r, b) > step.value) return 'sq todo'
  return b <= r ? 'sq done' : 'sq skip'
}
const stripTitle = (r, b) => {
  if (seenAt(r, b) > step.value) return `块 ${b}: 第 ${seenAt(r, b)} 步才到达`
  return b <= r ? `块 ${b}: 已计算` : `块 ${b}: 因果跳过 (未来块)`
}
const coverClass = (r, c) => {
  if (seenAt(r, c) > step.value) return 'cv todo'
  return c <= r ? 'cv done' : 'cv skip'
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

.lab-body { display: grid; grid-template-columns: 1.5fr 1fr; gap: 18px; align-items: start; }

.ring { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; background: var(--code-bg); border-radius: var(--radius-sm); padding: 12px; }
.dev { border: 1px solid var(--border); border-radius: var(--radius-sm); background: var(--bg-elev); padding: 8px; display: flex; flex-direction: column; gap: 6px; min-width: 86px; }
.dev-head { font-size: 11px; color: var(--text-muted); }
.chips { display: flex; gap: 4px; flex-wrap: wrap; }
.chip { font-size: 10px; padding: 2px 6px; border-radius: 999px; border: 1px solid; }
.chip.q { color: var(--accent); border-color: var(--accent); background: var(--accent-soft); }
.chip.kv { color: var(--eye); border-color: var(--eye); background: color-mix(in srgb, var(--eye) 14%, transparent); }
.strip { display: flex; gap: 3px; }
.strip i { display: block; width: 12px; height: 12px; border-radius: 3px; }
.sq.done { background: var(--left); }
.sq.skip { background: var(--border); }
.sq.todo { background: var(--code-bg); border: 1px solid var(--border); }
.arrow { color: var(--text-dim); font-size: 16px; }
.ring-back { font-size: 11px; color: var(--text-dim); }

.lab-stats { display: flex; flex-direction: column; gap: 10px; }
.stat { display: flex; align-items: baseline; gap: 10px; }
.stat .num { font-size: 20px; color: var(--text); }
.stat .cap { font-size: 12px; color: var(--text-muted); }

.cover { display: flex; flex-direction: column; gap: 6px; }
.cover .cap { font-size: 12px; color: var(--text-muted); }
.cover-grid { display: grid; gap: 3px; }
.cover-grid i { display: block; width: 16px; height: 16px; border-radius: 3px; }
.cv.done { background: var(--accent); }
.cv.skip { background: var(--border); }
.cv.todo { background: var(--code-bg); border: 1px solid var(--border); }

.note { font-size: 12px; color: var(--text-dim); line-height: 1.7; border-left: 2px solid var(--accent-soft); padding-left: 10px; }

@media (max-width: 960px) {
  .lab-body { grid-template-columns: 1fr; }
}
</style>
