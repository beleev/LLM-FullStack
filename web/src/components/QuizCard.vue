<!-- 章末自测: 选完立刻给对错和一句解释。全对记为"已掌握", 侧栏会打勾。 -->
<template>
  <section v-if="items.length" class="section quiz">
    <h2>章末自测 <span class="score mono">{{ correct }} / {{ items.length }}</span></h2>
    <p class="lead">合上代码答一遍。答错不要紧 —— 看解释, 再回实验台拖一拖验证。</p>
    <div v-for="(it, i) in items" :key="i" class="card q">
      <p class="q-text"><span class="mono q-idx">Q{{ i + 1 }}</span>{{ it.q }}</p>
      <div class="opts" role="radiogroup" :aria-label="`第 ${i + 1} 题`">
        <button
          v-for="(o, k) in it.options" :key="k" type="button" role="radio"
          :aria-checked="picked[i] === k"
          :class="['opt', stateOf(i, k)]" :disabled="picked[i] != null"
          @click="pick(i, k)"
        >{{ o }}</button>
      </div>
      <p v-if="picked[i] != null" class="why" :class="picked[i] === it.answer ? 'good' : 'bad'">
        {{ picked[i] === it.answer ? '✓ 正确。' : '✗ 再想想。' }} {{ it.why }}
      </p>
    </div>
    <button v-if="done" type="button" @click="retry">重做一遍</button>
  </section>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { quizBank } from '@/data/quiz/index.js'
import { useProgress } from '@/composables/useProgress.js'

const route = useRoute()
const progress = useProgress()
const items = computed(() => quizBank[route.name] || [])
const picked = ref([])
watch(items, () => (picked.value = []), { immediate: true })

const correct = computed(() => items.value.filter((it, i) => picked.value[i] === it.answer).length)
const done = computed(() => items.value.length > 0 && items.value.every((_, i) => picked.value[i] != null))
const pick = (i, k) => { picked.value[i] = k }
const retry = () => { picked.value = [] }
const stateOf = (i, k) => {
  if (picked.value[i] == null) return ''
  if (k === items.value[i].answer) return 'right'
  return picked.value[i] === k ? 'wrong' : ''
}
watch(done, (d) => { if (d) progress.setQuiz(route.name, correct.value, items.value.length) })
</script>

<style scoped>
.score { font-size: 12px; color: var(--accent); margin-left: auto; }
.q { margin-bottom: 12px; }
.q-text { font-size: 14px; margin-bottom: 10px; line-height: 1.6; }
.q-idx { color: var(--accent); margin-right: 8px; font-size: 12px; }
.opts { display: grid; gap: 6px; }
.opt { text-align: left; line-height: 1.5; padding: 8px 12px; }
.opt:disabled { opacity: 1; cursor: default; }
.opt.right { border-color: var(--left); background: color-mix(in srgb, var(--left) 14%, transparent); }
.opt.wrong { border-color: var(--danger); background: color-mix(in srgb, var(--danger) 12%, transparent); }
.why { margin-top: 10px; font-size: 13px; line-height: 1.6; padding-left: 10px; border-left: 2px solid var(--border); color: var(--text-muted); }
.why.good { border-left-color: var(--left); }
.why.bad { border-left-color: var(--danger); }
</style>
