<!--
  真源码片段: <SourceSnippet src="llm_infer/m07_speculative_decoding/demo.py:spec_decode_greedy" />
  内容在构建期直接读自 Python 文件, 不是手抄的, 所以永远和仓库一致。
-->
<template>
  <details class="source" :open="open" @toggle="onToggle">
    <summary>
      <span class="mono">{{ parsed.path }}</span>
      <span v-if="parsed.symbol" class="sym mono">· {{ parsed.symbol }}</span>
      <span v-if="data?.start" class="lines mono">L{{ data.start }}–{{ data.end }}</span>
    </summary>
    <p v-if="data?.error" class="err">{{ data.error }}</p>
    <CodeBlock v-else-if="data" :code="data.code" />
    <p v-else class="err">加载中…</p>
    <RepoLink :path="parsed.path" label="在 GitHub 上看完整文件" tiny />
  </details>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import CodeBlock from '@/components/CodeBlock.vue'
import RepoLink from '@/components/RepoLink.vue'
import { loadSource, parseRef } from '@/utils/sources.js'

const props = defineProps({
  src: { type: String, required: true },
  open: { type: Boolean, default: false },
})
const parsed = computed(() => parseRef(props.src))
const data = ref(null)
const fetchIt = async () => { data.value = await loadSource(props.src) }
const onToggle = (e) => { if (e.target.open && !data.value) fetchIt() }
watch(() => props.src, () => { data.value = null; if (props.open) fetchIt() }, { immediate: true })
</script>

<style scoped>
.source { border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 8px 12px; margin-top: 10px; background: var(--bg-elev); }
.source summary { cursor: pointer; font-size: 12px; color: var(--text-muted); display: flex; gap: 8px; align-items: baseline; flex-wrap: wrap; }
.source[open] summary { margin-bottom: 8px; }
.sym { color: var(--accent); }
.lines { margin-left: auto; color: var(--text-dim); font-size: 11px; }
.err { font-size: 12px; color: var(--text-dim); }
</style>
