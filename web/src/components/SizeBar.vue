<!--
  张量有多大, 就画多长。[B,T,D] 和 [B,H,T,T] 写出来长得差不多,
  但后者随 T 平方增长 —— 拖 T 的时候这条会横着长出去, 那就是注意力矩阵吃显存的样子。
-->
<template>
  <span class="size" :title="`${fmtNum(n)} 个元素 / 样本 (全流程最大的 ${pct}%)`">
    <span class="fill" :style="{ width: pct + '%', background: color }" />
    <span class="n mono">{{ fmtNum(n) }}</span>
  </span>
</template>

<script setup>
import { computed } from 'vue'
import { numelOf } from '@/data/inspector.js'
import { fmtNum } from '@/utils/labmath.js'

const props = defineProps({
  shape: { type: Array, required: true },
  ctx: { type: Object, required: true },
  max: { type: Number, required: true },
})
const n = computed(() => numelOf(props.shape, props.ctx))
const pct = computed(() => Math.max(1.5, Math.round((n.value / props.max) * 100)))
// 占到全流程最大张量一半以上就变暖色 —— 通常就是那个 [B,H,T,T]
const color = computed(() => (pct.value > 50 ? 'var(--warn)' : 'var(--accent)'))
</script>

<style scoped>
.size { position: relative; display: inline-flex; align-items: center; min-width: 96px; height: 14px; flex: 1; max-width: 200px; background: var(--code-bg); border-radius: 3px; overflow: hidden; }
.fill { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 3px; opacity: 0.55; transition: none; }
.n { position: relative; font-size: 10px; color: var(--text-muted); padding-left: 6px; }
</style>
