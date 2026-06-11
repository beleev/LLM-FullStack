<template>
  <div class="lab card">
    <div class="lab-head">
      <h3>MTP 实验台 — 一次前向，多步预测</h3>
      <p class="lab-sub">
        主干在位置 i 预测 t+1；每个 MTP 级联模块拼接一个真实 next-token 的 embedding，再多看一步。
        对应 llm_models/run_models/language_models/mtp。
      </p>
    </div>

    <div class="lab-controls">
      <div class="ctl">
        <label>MTP 深度 K</label>
        <input type="range" min="1" max="3" step="1" v-model.number="K" />
        <span class="val mono">{{ K }}</span>
      </div>
      <div class="ctl">
        <label>λ (MTP 损失权重)</label>
        <input type="range" min="0" max="1" step="0.1" v-model.number="lambda" />
        <span class="val mono">{{ lambda.toFixed(1) }}</span>
      </div>
      <div class="ctl">
        <label>聚焦位置 i</label>
        <input type="range" min="0" max="5" step="1" v-model.number="focus" />
        <span class="val mono">{{ focus }}</span>
      </div>
    </div>

    <div class="lab-body">
      <div class="strip-area">
        <p class="strip-title">输入序列 (聚焦位置 i = {{ focus }})</p>
        <div class="chip-row">
          <span v-for="t in 8" :key="t" class="chip mono" :class="{ focus: t - 1 === focus }">
            t{{ t - 1 }}
          </span>
        </div>
        <div v-for="row in rows" :key="row.label" class="pred-row">
          <div class="pred-head">
            <span class="row-label mono">{{ row.label }}</span>
            <span class="row-desc mono" :class="{ muted: row.oob }">{{ row.desc }}</span>
          </div>
          <div v-if="!row.oob" class="chip-row">
            <span v-for="t in 8" :key="t" class="chip mono" :class="chipClass(row, t - 1)">
              t{{ t - 1 }}
            </span>
          </div>
        </div>
      </div>

      <div class="lab-stats">
        <div class="stat">
          <span class="num mono">{{ K + 1 }}</span>
          <span class="cap">一次前向产出的预测数 (主 head + K 级 MTP)</span>
        </div>
        <div class="stat">
          <span class="num mono formula">L = L_main + {{ lambda.toFixed(1) }}·mean(L_mtp)</span>
          <span class="cap">损失公式</span>
        </div>
        <div class="stat">
          <span class="num mono">{{ K }}</span>
          <span class="cap">投机解码草稿长度 (llm_infer/m07_speculative_decoding)</span>
        </div>
        <p class="cost-row">
          参数开销: 每级 = 1 个拼接投影 + 1 个 Block (embedding 与 lm_head 共享,
          DeepSeek-V3 61 层主干上 ~1.5%)。
        </p>
        <p class="note">
          训练信号更密 + 表征被迫向前规划 + 推理免费拿草稿 (DeepSeek-V3: 接受率 85%+,
          解码 ~1.8×)。部署时 MTP 模块可整体丢弃。
        </p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'

const K = ref(1)
const lambda = ref(0.3)
const focus = ref(2)

const LAST = 7 // 序列共 8 个 token: t0..t7

// 每行: 主 head 直接预测 t(i+1); MTP-k 拼接真实 t(i+k) 的 embedding 后预测 t(i+1+k)
const rows = computed(() => {
  const out = []
  for (let k = 0; k <= K.value; k++) {
    const consume = focus.value + k
    const predict = focus.value + 1 + k
    const oob = predict > LAST
    out.push({
      label: k === 0 ? '主 head' : `MTP-${k}`,
      isMain: k === 0,
      consume: k === 0 || oob ? -1 : consume,
      predict: oob ? -1 : predict,
      oob,
      desc: oob
        ? `t${predict} 越界 → -100 屏蔽`
        : k === 0
          ? `位置 t${focus.value} → 预测 t${predict}`
          : `拼接 Emb(t${consume}) → 预测 t${predict}`,
    })
  }
  return out
})

const chipClass = (row, idx) => ({
  consume: idx === row.consume,
  predict: idx === row.predict && !row.isMain,
  'predict-main': idx === row.predict && row.isMain,
})
</script>

<style scoped>
.lab { margin-bottom: 16px; }
.lab-head h3 { margin-bottom: 4px; }
.lab-sub { color: var(--text-muted); font-size: 13px; line-height: 1.6; margin-bottom: 14px; }

.lab-controls { display: flex; flex-direction: column; gap: 8px; margin-bottom: 14px; }
.ctl { display: grid; grid-template-columns: 110px 1fr 56px; gap: 10px; align-items: center; }
.ctl label { font-size: 12px; color: var(--text-muted); }
.ctl .val { font-size: 12px; text-align: right; color: var(--accent); }

.lab-body { display: grid; grid-template-columns: minmax(220px, 320px) 1fr; gap: 18px; align-items: start; }

.strip-area { display: flex; flex-direction: column; gap: 10px; background: var(--code-bg); border-radius: var(--radius-sm); padding: 10px; }
.strip-title { font-size: 11px; color: var(--text-muted); }
.chip-row { display: flex; flex-wrap: wrap; gap: 4px; }
.chip { font-size: 11px; padding: 2px 6px; border: 1px solid var(--border); border-radius: var(--radius-sm); background: var(--bg-elev); color: var(--text-muted); }
.chip.focus { border-color: var(--accent); color: var(--accent); background: var(--accent-soft); }
.chip.consume { border-color: var(--eye); color: var(--eye); box-shadow: 0 0 0 1px var(--eye) inset; }
.chip.predict { border-color: var(--left); color: var(--left); box-shadow: 0 0 0 1px var(--left) inset; }
.chip.predict-main { border-color: var(--accent); color: var(--accent); background: var(--accent-soft); box-shadow: 0 0 0 1px var(--accent) inset; }

.pred-row { display: flex; flex-direction: column; gap: 5px; border-top: 1px dashed var(--border); padding-top: 8px; }
.pred-head { display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap; }
.row-label { font-size: 11px; color: var(--text); }
.row-desc { font-size: 11px; color: var(--text-muted); }
.row-desc.muted { color: var(--text-dim); }

.lab-stats { display: flex; flex-direction: column; gap: 10px; }
.stat { display: flex; align-items: baseline; gap: 10px; }
.stat .num { font-size: 20px; color: var(--text); }
.stat .num.formula { font-size: 13px; }
.stat .cap { font-size: 12px; color: var(--text-muted); }
.cost-row { font-size: 12px; color: var(--text-muted); line-height: 1.6; }
.note { font-size: 12px; color: var(--text-dim); line-height: 1.7; border-left: 2px solid var(--accent-soft); padding-left: 10px; }

@media (max-width: 960px) {
  .lab-body { grid-template-columns: 1fr; }
}
</style>
