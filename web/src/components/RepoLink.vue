<template>
  <a
    v-if="!isGlob"
    class="repo-ref"
    :class="[variant, { tiny }]"
    :href="href"
    target="_blank"
    rel="noopener"
    :title="`查看源码 · ${displayPath}`"
  >
    <span class="ref-icon" aria-hidden="true">{{ icon }}</span>
    <code class="ref-path">{{ displayLabel }}</code>
    <span v-if="lineDisplay" class="ref-line">L{{ lineDisplay }}</span>
    <span class="ref-arrow" aria-hidden="true">↗</span>
  </a>
  <code v-else class="repo-ref glob" :class="{ tiny }" :title="'通配路径 — 不直接跳转'">
    <span class="ref-icon" aria-hidden="true">∗</span>
    {{ displayLabel }}
  </code>
</template>

<script setup>
import { computed } from 'vue'
import { repoUrl, parseRef } from '@/utils/repo.js'

const props = defineProps({
  path:    { type: String, required: true },
  line:    { type: [Number, String], default: null },
  label:   { type: String, default: '' },
  variant: { type: String, default: 'chip' },
  tiny:    { type: Boolean, default: false },
})

const parsed = computed(() => parseRef(props.path))
const displayPath = computed(() => parsed.value.path)
const lineNum = computed(() => props.line ? Number(props.line) : parsed.value.line)
const lineDisplay = computed(() => lineNum.value || null)
const href = computed(() => repoUrl(displayPath.value, lineNum.value))
const displayLabel = computed(() => props.label || displayPath.value)
const isGlob = computed(() => /[{*]|\.\.|,/.test(displayPath.value))
const icon = computed(() => {
  if (displayPath.value.endsWith('/')) return '📁'
  if (displayPath.value.endsWith('.py')) return '🐍'
  return '📄'
})
</script>

<style scoped>
.repo-ref {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-family: "SF Mono", "JetBrains Mono", Menlo, monospace;
  font-size: 12px;
  color: var(--text);
  background: var(--code-bg);
  border: 1px solid var(--border);
  padding: 2px 8px;
  border-radius: 4px;
  text-decoration: none;
  transition: all 0.12s;
  line-height: 1.5;
  vertical-align: middle;
  max-width: 100%;
}
.repo-ref:hover {
  border-color: var(--accent);
  color: var(--accent);
  text-decoration: none;
}
.repo-ref:hover .ref-arrow { opacity: 1; transform: translate(1px, -1px); }
.repo-ref.pill {
  border-radius: 99px;
  padding: 2px 10px;
}
.repo-ref.plain {
  background: transparent;
  border: none;
  padding: 0;
}
.repo-ref.plain .ref-arrow { color: var(--accent); }
.repo-ref.tiny {
  font-size: 11px;
  padding: 1px 6px;
}

.ref-icon { font-size: 11px; opacity: 0.85; }
.ref-path { background: transparent; border: none; padding: 0; color: inherit; font: inherit; }
.ref-line {
  color: var(--text-dim);
  font-size: 10.5px;
  padding: 0 4px;
  border-left: 1px solid var(--border);
  margin-left: 2px;
}
.ref-arrow {
  font-size: 10px;
  color: var(--text-dim);
  opacity: 0.55;
  transition: all 0.12s;
}
.repo-ref.glob {
  cursor: default;
  color: var(--text-muted);
  border-style: dashed;
}
.repo-ref.glob:hover { color: var(--text-muted); border-color: var(--border); }
</style>
