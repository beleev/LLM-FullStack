<template>
  <div class="app">
    <aside class="sidebar">
      <div class="brand">
        <div class="brand-row">
          <h1>LLM 全栈教程</h1>
          <button
            class="theme-toggle"
            type="button"
            @click="toggleTheme"
            :aria-label="theme === 'dark' ? '切换到浅色主题' : '切换到深色主题'"
            :title="theme === 'dark' ? '切到浅色 ☀︎' : '切到深色 ☾'"
          >
            <span v-if="theme === 'dark'" aria-hidden="true">☀︎</span>
            <span v-else aria-hidden="true">☾</span>
          </button>
        </div>
        <p>原理、架构、训练、微调、推理、Agent 六段闭环</p>
        <a class="repo-link" :href="repoUrl()" target="_blank" rel="noopener">
          <span class="gh-icon" aria-hidden="true">↗</span>
          GitHub · 仓库源码
        </a>
      </div>

      <nav>
        <!-- 序章 -->
        <div class="section-label">序章</div>
        <router-link :to="{ name: 'home' }" class="nav-link">
          <span class="idx">✦</span>
          <span>主线总览</span>
        </router-link>

        <!-- 六阶段 -->
        <template v-for="s in stages" :key="s.id">
          <div class="section-label">
            阶段 {{ s.idx }} · <span class="mono">{{ s.code }}</span>
            <span v-if="s.status === 'planned'" class="planned-tag">待补</span>
          </div>

          <!-- ready & 单页 -->
          <router-link
            v-if="s.status === 'ready' && s.route"
            :to="{ name: s.route }"
            class="nav-link"
          >
            <span class="idx">{{ s.idx }}</span>
            <span>{{ chapterTitle(s.route) }}</span>
          </router-link>

          <!-- ready & 阶段内子章 -->
          <router-link
            v-for="c in s.chapters || []"
            :key="c.route"
            :to="{ name: c.route }"
            class="nav-link sub"
          >
            <span class="idx">{{ s.idx }}.{{ subIdx(s, c.route) }}</span>
            <span>{{ chapterTitle(c.route) }}</span>
          </router-link>

          <!-- planned: 灰显, 不可点击 -->
          <div v-if="s.status === 'planned'" class="nav-link disabled">
            <span class="idx">·</span>
            <span>{{ s.title }}</span>
          </div>
        </template>

        <!-- 终章 -->
        <div class="section-label">终章</div>
        <router-link :to="{ name: 'compare' }" class="nav-link">
          <span class="idx">∎</span>
          <span>总览对照表</span>
        </router-link>

        <div class="section-label">关于</div>
        <div class="footer">
          <p class="mono">v0.4.0</p>
          <p>六个目录已接入 Web 教程, 每章都对照原始代码阅读。</p>
        </div>
      </nav>
    </aside>

    <main class="main">
      <router-view v-slot="{ Component, route }">
        <div v-if="route.meta.chapter" class="breadcrumb">
          {{ route.meta.chapter }} · {{ route.meta.title }}
        </div>
        <transition name="page" mode="out-in">
          <component :is="Component" :key="route.fullPath" />
        </transition>
      </router-view>
    </main>
  </div>
</template>

<script setup>
import { ref, onMounted, watch } from 'vue'
import { useRouter } from 'vue-router'
import { stages } from '@/data/models.js'
import { repoUrl } from '@/utils/repo.js'

const router = useRouter()
const allRoutes = router.getRoutes()

const chapterTitle = (name) => {
  const r = allRoutes.find(r => r.name === name)
  return r?.meta?.title || name
}

const subIdx = (stage, route) =>
  (stage.chapters || []).findIndex(c => c.route === route) + 1

// ── 主题切换 ──────────────────────────────────────────────────────
const THEME_KEY = 'llm-theme'
const initialTheme = (() => {
  try {
    const stored = localStorage.getItem(THEME_KEY)
    if (stored === 'light' || stored === 'dark') return stored
  } catch (_) { /* localStorage 不可用时降级 */ }
  if (typeof window !== 'undefined' && window.matchMedia) {
    return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark'
  }
  return 'dark'
})()
const theme = ref(initialTheme)
const applyTheme = (t) => { document.documentElement.dataset.theme = t }
applyTheme(theme.value)
onMounted(() => applyTheme(theme.value))
watch(theme, (t) => {
  applyTheme(t)
  try { localStorage.setItem(THEME_KEY, t) } catch (_) { /* ignore */ }
})
const toggleTheme = () => { theme.value = theme.value === 'dark' ? 'light' : 'dark' }
</script>

<style scoped>
.brand-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}
.theme-toggle {
  width: 28px;
  height: 28px;
  padding: 0;
  display: grid;
  place-items: center;
  font-size: 14px;
  line-height: 1;
  background: var(--bg-card);
  border: 1px solid var(--border);
  color: var(--text-muted);
  border-radius: var(--radius-sm);
}
.theme-toggle:hover {
  color: var(--accent);
  border-color: var(--accent);
}
.repo-link {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  margin-top: 10px;
  font-size: 11px;
  color: var(--text-muted);
  border-bottom: 1px dashed var(--border-strong);
  padding-bottom: 1px;
  width: max-content;
}
.repo-link:hover { color: var(--accent); border-color: var(--accent); text-decoration: none; }
.gh-icon { font-size: 10px; color: var(--accent); }

.footer {
  padding: 8px 24px 24px;
  font-size: 11px;
  color: var(--text-dim);
  line-height: 1.7;
}
.footer p { margin-top: 4px; }

.section-label .mono {
  font-size: 10px;
  color: var(--text-muted);
  text-transform: none;
  letter-spacing: 0.2px;
}
.planned-tag {
  margin-left: 6px;
  padding: 1px 6px;
  border-radius: 3px;
  background: var(--bg-card);
  border: 1px dashed var(--border);
  color: var(--text-dim);
  font-size: 9px;
  text-transform: uppercase;
  letter-spacing: 0.6px;
}

.nav-link.sub { padding-left: 38px; font-size: 12.5px; }
.nav-link.disabled {
  display: flex; align-items: center; gap: 12px;
  padding: 10px 24px;
  font-size: 13px;
  color: var(--text-dim);
  border-left: 2px solid transparent;
  cursor: not-allowed;
  opacity: 0.55;
}
.nav-link.disabled .idx { background: transparent; }

.page-enter-active, .page-leave-active {
  transition: opacity 0.18s, transform 0.18s;
}
.page-enter-from { opacity: 0; transform: translateY(4px); }
.page-leave-to   { opacity: 0; transform: translateY(-4px); }
</style>
