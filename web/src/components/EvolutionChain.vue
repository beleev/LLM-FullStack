<template>
  <div class="evo-chain card">
    <div class="evo-head">
      <h3>{{ title || '演进逻辑链' }}</h3>
      <p v-if="subtitle" class="evo-sub">{{ subtitle }}</p>
    </div>
    <div class="evo-row">
      <template v-for="(step, i) in steps" :key="step.name">
        <div class="evo-step" :style="{ borderColor: step.color || 'var(--border)' }">
          <div class="evo-name" :style="{ color: step.color || 'var(--text)' }">
            {{ step.name }}
            <span v-if="step.year" class="evo-year">{{ step.year }}</span>
          </div>
          <div v-if="step.pain" class="evo-pain">
            <span class="lbl">上一代痛点</span>
            <span class="txt">{{ step.pain }}</span>
          </div>
          <div v-if="step.fix" class="evo-fix">
            <span class="lbl">本代解法</span>
            <span class="txt">{{ step.fix }}</span>
          </div>
        </div>
        <div v-if="i < steps.length - 1" class="evo-arrow" aria-hidden="true">→</div>
      </template>
    </div>
  </div>
</template>

<script setup>
defineProps({
  title:    { type: String, default: '' },
  subtitle: { type: String, default: '' },
  steps:    { type: Array,  required: true },
})
</script>

<style scoped>
.evo-chain { margin-bottom: 24px; }
.evo-head { margin-bottom: 14px; }
.evo-sub  { font-size: 12.5px; color: var(--text-muted); margin-top: 4px; }

.evo-row {
  display: flex;
  align-items: stretch;
  gap: 8px;
  overflow-x: auto;
  padding-bottom: 4px;
}
.evo-step {
  flex: 1 1 0;
  min-width: 180px;
  background: var(--bg-elev);
  border: 1px solid var(--border);
  border-top-width: 3px;
  border-radius: var(--radius-sm);
  padding: 12px 14px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.evo-name {
  font-size: 14px;
  font-weight: 600;
  font-family: "SF Mono", Menlo, monospace;
  display: flex;
  align-items: baseline;
  gap: 8px;
}
.evo-year {
  font-size: 11px;
  color: var(--text-dim);
  font-weight: 400;
}
.evo-pain, .evo-fix {
  font-size: 12px;
  line-height: 1.5;
}
.evo-pain .lbl, .evo-fix .lbl {
  display: block;
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.7px;
  margin-bottom: 2px;
}
.evo-pain .lbl { color: var(--danger); }
.evo-fix  .lbl { color: var(--left); }
.evo-pain .txt, .evo-fix .txt { color: var(--text-muted); }

.evo-arrow {
  align-self: center;
  color: var(--text-dim);
  font-size: 18px;
  padding: 0 4px;
  user-select: none;
}

@media (max-width: 960px) {
  .evo-row { flex-direction: column; }
  .evo-arrow { transform: rotate(90deg); }
}
</style>
