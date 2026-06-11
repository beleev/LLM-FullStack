<template>
  <div v-if="page">
    <h1 class="page-title">{{ page.title }}</h1>
    <p class="page-subtitle">{{ page.subtitle }}</p>

    <ChapterIntro
      :tldr="page.tldr"
      :question="page.question"
      :code="page.code"
      :prereq="prev"
      :next-step="next"
    />

    <section v-if="labList.length" class="section">
      <h2>0. 动手实验台</h2>
      <p class="lead">
        先动手, 再读字。拖动参数, 观察右侧数字与图形的联动 —— 每个实验台都对应一个可运行的 Python 模块。
      </p>
      <component :is="lab" v-for="(lab, i) in labList" :key="i" />
    </section>

    <section class="section">
      <h2>1. 本章抓手</h2>
      <p class="lead">
        先用三句话锁定概念边界, 再回到原始代码。读代码时只追关键变量,
        不把注意力分散到框架细节上。
      </p>
      <div class="grid grid-3">
        <div v-for="p in page.points" :key="p.title" class="card point-card">
          <h3>{{ p.title }}</h3>
          <p class="desc">{{ p.body }}</p>
        </div>
      </div>
    </section>

    <section class="section">
      <h2>2. 知识怎么接上前后文</h2>
      <p class="lead">
        每个子章都不是孤立概念。下面按“上一层抽象 → 本章机制 → 后续用途”读。
      </p>
      <div class="card link-card">
        <div v-for="l in page.links" :key="`${l.from}-${l.to}`" class="link-row">
          <component
            :is="topicRef(l.from).linked ? RepoLink : 'span'"
            v-bind="topicRef(l.from).linked ? { path: topicRef(l.from).path, label: l.from, tiny: true } : {}"
            class="mono endpoint"
          >{{ l.from }}</component>
          <span class="arrow">→</span>
          <component
            :is="topicRef(l.to).linked ? RepoLink : 'span'"
            v-bind="topicRef(l.to).linked ? { path: topicRef(l.to).path, label: l.to, tiny: true } : {}"
            class="mono endpoint"
          >{{ l.to }}</component>
          <span class="body">{{ l.body }}</span>
        </div>
      </div>
    </section>

    <section class="section">
      <h2>3. 原始代码对照</h2>
      <p class="lead">
        这一节只列读代码必须抓住的行级意图: 变量代表什么, 为什么这样写, 它验证了什么。
      </p>
      <div class="card" style="padding: 0; overflow-x: auto;">
        <table class="topic-table">
          <thead>
            <tr>
              <th>概念</th>
              <th>代码位置 / 表达式</th>
              <th>读这一行要理解什么</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="r in page.sourceRows" :key="r.concept">
              <td class="concept">{{ r.concept }}</td>
              <td class="mono small">
                <RepoLink
                  v-if="topicRef(r.code).linked"
                  :path="topicRef(r.code).path"
                  :label="r.code"
                  tiny
                />
                <span v-else>{{ r.code }}</span>
              </td>
              <td>{{ r.takeaway }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <section class="section">
      <h2>4. 最小代码骨架</h2>
      <p class="lead">
        下面是从原始代码里抽出的核心控制流。完整实现仍以对应 Python 文件为准。
      </p>
      <div class="card">
        <h3>{{ page.snippetTitle }} <span v-if="page.run" class="tag">可运行</span></h3>
        <pre class="code">{{ page.snippet }}</pre>
        <p v-if="page.run" class="hint">
          运行入口: <code class="inline">{{ page.run }}</code>
        </p>
      </div>
    </section>

    <ChapterNav :prev="prev" :next="next" />
  </div>

  <div v-else>
    <h1 class="page-title">章节未找到</h1>
    <p class="page-subtitle">当前路由没有对应的 topicPages 配置。</p>
    <ChapterNav :prev="{ name: 'home', label: '主线总览' }" />
  </div>
</template>

<script setup>
import { computed, defineAsyncComponent } from 'vue'
import { useRoute } from 'vue-router'
import ChapterIntro from '@/components/ChapterIntro.vue'
import ChapterNav from '@/components/ChapterNav.vue'
import RepoLink from '@/components/RepoLink.vue'
import { learningPath, stageBy, topicPages } from '@/data/models.js'
import { looksLikeRepoRef, normalizeRepoRef, splitRefs } from '@/utils/repo.js'

// 可交互实验台注册表: topicPages[x].widgets = ['BpeLab', ...] 即可挂载
const LABS = {
  AttnMaskLab: defineAsyncComponent(() => import('@/components/labs/AttnMaskLab.vue')),
  BpeLab: defineAsyncComponent(() => import('@/components/labs/BpeLab.vue')),
  MoeRouteLab: defineAsyncComponent(() => import('@/components/labs/MoeRouteLab.vue')),
  RingAttnLab: defineAsyncComponent(() => import('@/components/labs/RingAttnLab.vue')),
  GrpoLab: defineAsyncComponent(() => import('@/components/labs/GrpoLab.vue')),
  SoftmaxTempLab: defineAsyncComponent(() => import('@/components/labs/SoftmaxTempLab.vue')),
  RetrievalLab: defineAsyncComponent(() => import('@/components/labs/RetrievalLab.vue')),
  MtpLab: defineAsyncComponent(() => import('@/components/labs/MtpLab.vue')),
}

const route = useRoute()
const page = computed(() => topicPages[route.name])
const labList = computed(() =>
  (page.value?.widgets || []).map(name => LABS[name]).filter(Boolean)
)

const currentIndex = computed(() =>
  learningPath.findIndex(item => item.route === route.name)
)

const toNav = (item) => item ? { name: item.route, label: item.label } : null
const prev = computed(() => toNav(learningPath[currentIndex.value - 1]))
const next = computed(() => toNav(learningPath[currentIndex.value + 1]))

const stageCode = computed(() => stageBy[route.meta.stage]?.code || '')
const contextRefs = computed(() =>
  splitRefs(page.value?.code || '', 'auto').map(r => normalizeRepoRef(r.label || r.path).path)
)

const topicRef = (ref) => {
  const raw = String(ref || '').trim()
  if (!looksLikeRepoRef(raw)) return { linked: false, path: raw }

  const parsed = normalizeRepoRef(raw)
  if (parsed.path.startsWith('llm_')) return { linked: true, path: raw }

  const path = parsed.path
  const match = contextRefs.value.find(p => p.endsWith(path) || p.endsWith(`/${path}`))
  if (match) return { linked: true, path: match }

  if (stageCode.value === 'llm_basic/' && /^[\w.-]+\.(py|txt|bin|npz|md)(?::|$)/.test(raw)) {
    return { linked: true, path: `${stageCode.value}${raw}` }
  }

  if (stageCode.value && path.includes('/')) {
    return { linked: true, path: `${stageCode.value}${raw}` }
  }

  return { linked: false, path: raw }
}
</script>

<style scoped>
.point-card h3 {
  margin-bottom: 6px;
}

.link-card {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.link-row {
  display: grid;
  grid-template-columns: minmax(120px, 0.9fr) 24px minmax(120px, 0.9fr) 2fr;
  gap: 10px;
  align-items: baseline;
  padding: 10px 12px;
  background: var(--bg-elev);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  font-size: 13px;
}
.link-row .endpoint {
  color: var(--text);
  font-size: 12px;
}
.link-row .arrow {
  color: var(--accent);
  text-align: center;
}
.link-row .body {
  color: var(--text-muted);
  line-height: 1.55;
}

table.topic-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
table.topic-table th {
  text-align: left;
  padding: 12px 14px;
  background: var(--bg-elev);
  color: var(--text-muted);
  font-size: 11px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.7px;
  border-bottom: 1px solid var(--border-strong);
}
table.topic-table td {
  padding: 10px 14px;
  border-bottom: 1px solid var(--border);
  vertical-align: top;
}
table.topic-table .concept {
  color: var(--text);
  font-weight: 600;
  white-space: nowrap;
}
table.topic-table .small {
  color: var(--text-muted);
  font-size: 12px;
}
.hint {
  margin-top: 10px;
  font-size: 12px;
  color: var(--text-dim);
  line-height: 1.6;
}

@media (max-width: 960px) {
  .link-row {
    grid-template-columns: 1fr;
    gap: 4px;
  }
  .link-row .arrow {
    text-align: left;
  }
}
</style>
