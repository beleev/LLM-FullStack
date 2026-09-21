<!-- 带语法高亮和复制按钮的代码块。 -->
<template>
  <div class="codeblock">
    <button type="button" class="copy" @click="copy">{{ copied ? '已复制' : '复制' }}</button>
    <pre class="code" v-html="html" />
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'
import { highlightPython } from '@/utils/highlight.js'

const props = defineProps({
  code: { type: String, default: '' },
  lang: { type: String, default: 'python' },
})
// highlightPython 会先转义 HTML 再包 span, 所以 v-html 是安全的
const escape = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
const html = computed(() => (props.lang === 'python' ? highlightPython(props.code) : escape(props.code)))
const copied = ref(false)
const copy = async () => {
  try {
    await navigator.clipboard.writeText(props.code)
    copied.value = true
    setTimeout(() => (copied.value = false), 1200)
  } catch (_) { /* 无剪贴板权限时忽略 */ }
}
</script>

<style scoped>
.codeblock { position: relative; }
.copy {
  position: absolute; top: 6px; right: 6px; min-height: 0;
  padding: 2px 8px; font-size: 11px; opacity: 0; transition: opacity 0.15s;
}
.codeblock:hover .copy, .copy:focus-visible { opacity: 1; }
</style>
