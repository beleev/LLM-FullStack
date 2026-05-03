<template>
  <div class="shape-tuple mono" :title="symbolicLabel">
    <span class="bracket">[</span>
    <template v-for="(val, i) in numeric" :key="i">
      <span class="comma" v-if="i > 0">, </span>
      <span class="dim">
        <span class="sym">{{ shape[i] }}</span>
        <span v-if="val !== shape[i]" class="eq">=</span>
        <span v-if="val !== shape[i]" class="num">{{ val }}</span>
      </span>
    </template>
    <span class="bracket">]</span>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { formatShape } from '@/data/inspector.js'

const props = defineProps({
  shape: { type: Array, required: true },  // 符号表达式数组
  ctx:   { type: Object, required: true },
})

const numeric = computed(() => formatShape(props.shape, props.ctx))
const symbolicLabel = computed(() => '[' + props.shape.join(', ') + ']')
</script>

<style scoped>
.shape-tuple {
  display: inline-flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 0;
  margin-top: 6px;
  font-size: 12px;
  color: var(--text-muted);
  padding: 4px 10px;
  background: var(--code-bg);
  border-radius: 5px;
  border: 1px solid var(--border);
  font-family: "SF Mono", Menlo, monospace;
  line-height: 1.5;
}
.bracket { color: var(--text-dim); }
.comma   { color: var(--text-dim); }
.dim     { display: inline-flex; align-items: baseline; gap: 2px; }
.sym     { color: var(--text); }
.eq      { color: var(--text-dim); opacity: 0.6; }
.num     { color: var(--accent); font-weight: 500; }
</style>
