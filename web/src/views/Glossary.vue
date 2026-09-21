<!-- 术语速查表: 搜索 + 按阶段过滤。每个词条一句话 + 一个关键数字 + 跳到对应章节。 -->
<template>
  <div>
    <h1 class="page-title">术语速查</h1>
    <p class="page-subtitle">读到陌生缩写先来这里。每条只给一句话和一个值得记住的数字, 想深入就点进对应章节。</p>

    <div class="bar">
      <input v-model="q" type="search" class="search" placeholder="搜索: KV cache / ZeRO / GRPO / 投机解码 …" aria-label="搜索术语" />
      <div class="btn-group">
        <button type="button" :class="{ active: !stage }" @click="stage = ''">全部</button>
        <button v-for="s in stages" :key="s.id" type="button" :class="{ active: stage === s.id }" @click="stage = s.id">
          {{ s.idx }} · {{ s.title }}
        </button>
      </div>
    </div>

    <p class="count mono">{{ shown.length }} / {{ glossary.length }} 条</p>
    <div class="grid grid-2">
      <article v-for="g in shown" :key="g.term" class="card term">
        <h3>{{ g.term }} <span v-if="g.aka" class="aka">{{ g.aka }}</span></h3>
        <p class="desc">{{ g.oneliner }}</p>
        <p v-if="g.number" class="num mono">{{ g.number }}</p>
        <router-link v-if="g.route" :to="{ name: g.route }" class="go">去这一章 →</router-link>
      </article>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'
import { glossary } from '@/data/glossary.js'
import { stages } from '@/data/models.js'

const q = ref('')
const stage = ref('')
const shown = computed(() => {
  const k = q.value.trim().toLowerCase()
  return glossary
    .filter((g) => !stage.value || g.stage === stage.value)
    .filter((g) => !k || `${g.term} ${g.aka || ''} ${g.oneliner}`.toLowerCase().includes(k))
    .sort((a, b) => a.term.localeCompare(b.term))
})
</script>

<style scoped>
.bar { display: flex; flex-direction: column; gap: 10px; margin-bottom: 12px; }
.search { width: 100%; max-width: 520px; min-height: 40px; padding: 8px 12px; font-size: 14px; color: var(--text); background: var(--bg-elev); border: 1px solid var(--border); border-radius: var(--radius-sm); }
.count { font-size: 11px; color: var(--text-dim); margin-bottom: 10px; }
.term h3 { margin-bottom: 6px; }
.aka { font-size: 11px; font-weight: 400; color: var(--text-dim); }
.num { margin-top: 8px; font-size: 12px; color: var(--accent); }
.go { display: inline-block; margin-top: 8px; font-size: 12px; }
</style>
