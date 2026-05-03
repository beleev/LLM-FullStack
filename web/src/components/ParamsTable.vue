<template>
  <div class="params">
    <table>
      <thead>
        <tr>
          <th>权重</th>
          <th class="right">形状</th>
          <th class="right">参数量</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="(p, i) in rows" :key="i">
          <td>
            <code class="inline">{{ p.name }}</code>
            <div v-if="p.note" class="note">{{ p.note }}</div>
          </td>
          <td class="right">
            <ShapeTuple :shape="p.shape" :ctx="ctx" />
          </td>
          <td class="right mono">
            <span v-if="p.count !== null" :class="{ 'is-big': p.count >= 1e8 }">
              {{ p.countFmt }}
            </span>
            <span v-else class="muted">—</span>
          </td>
        </tr>
      </tbody>
      <tfoot>
        <tr>
          <td>合计</td>
          <td class="right muted" style="font-size: 11px;">{{ rows.filter(r => r.count !== null).length }} 个可数矩阵</td>
          <td class="right total mono">{{ totalFmt }}</td>
        </tr>
      </tfoot>
    </table>

    <div v-if="breakdown" class="breakdown">
      <span class="k">粗算:</span>
      <code class="inline">{{ breakdown }}</code>
    </div>
    <div v-if="cache" class="breakdown">
      <span class="k" style="color: var(--accent);">KV cache / token:</span>
      <code class="inline">{{ cache.expr }}</code>
      <span class="eq">=</span>
      <span class="mono accent-val">{{ cacheNum }} B/token · fp16</span>
      <div v-if="cache.note" class="note" style="margin-top: 4px;">{{ cache.note }}</div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import ShapeTuple from './ShapeTuple.vue'
import { evalExpr, computeParamCount, formatParams } from '@/data/inspector.js'

const props = defineProps({
  params:    { type: Array, required: true },
  ctx:       { type: Object, required: true },
  breakdown: { type: String, default: null },
  cache:     { type: Object, default: null }, // { expr, note }
})

const rows = computed(() => props.params.map(p => {
  const count = computeParamCount(p, props.ctx)
  return {
    ...p,
    count,
    countFmt: count !== null ? formatParams(count) : '—',
  }
}))

const totalCount = computed(() =>
  rows.value.reduce((sum, r) => r.count !== null ? sum + r.count : sum, 0)
)
const totalFmt = computed(() => formatParams(totalCount.value))

const cacheNum = computed(() => {
  if (!props.cache) return '—'
  const n = evalExpr(props.cache.expr, props.ctx)
  return typeof n === 'number' ? (n * 2).toLocaleString() : '—'
})
</script>

<style scoped>
.params table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
.params th {
  text-align: left;
  padding: 8px 10px;
  color: var(--text-muted);
  font-size: 11px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.6px;
  border-bottom: 1px solid var(--border);
}
.params td {
  padding: 10px 10px;
  border-bottom: 1px solid var(--border);
  vertical-align: top;
}
.params tfoot td {
  font-weight: 600;
  border-bottom: none;
  border-top: 1px solid var(--border-strong);
}
.params .right { text-align: right; }
.params .note { font-size: 11.5px; color: var(--text-muted); margin-top: 3px; line-height: 1.5; }
.params .muted { color: var(--text-dim); }
.params .total { color: var(--accent); font-size: 14px; }
.params .is-big { color: var(--warn); }

.breakdown {
  margin-top: 12px;
  padding: 10px 12px;
  background: var(--bg-elev);
  border-radius: 6px;
  font-size: 12px;
  color: var(--text-muted);
  border-left: 2px solid var(--accent);
}
.breakdown .k {
  text-transform: uppercase;
  letter-spacing: 0.5px;
  font-size: 10px;
  color: var(--text-dim);
  margin-right: 6px;
}
.breakdown .eq { color: var(--text-dim); margin: 0 6px; }
.breakdown .accent-val { color: var(--accent); }
</style>
