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

    <section v-if="hasLabs" class="section">
      <h2>先动手</h2>
      <p class="lead">
        拖一下滑杆, 右边的数字会跟着动。看懂哪个数字被什么牵着走, 下面的字就好读了。
      </p>
      <LabMount />
    </section>

    <section class="section">
      <h2>三句话</h2>
      <p class="lead">
        这一章全部的内容就这三条。带「重点」的那条是走的时候必须带上的。
      </p>
      <div class="grid grid-3">
        <div v-for="p in page.points" :key="p.title" class="card point-card" :class="{ key: p.key }">
          <h3>{{ p.title }} <span v-if="p.key" class="key-tag">重点</span></h3>
          <p class="desc">{{ p.body }}</p>
        </div>
      </div>
    </section>

    <section class="section">
      <h2>它接在哪</h2>
      <p class="lead">
        没有哪个技术是凭空出现的。左边是它替换掉的东西, 右边是它后来被用在哪。
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
      <h2>去代码里找什么</h2>
      <p class="lead">
        打开源码之前先看这张表。只盯这几行, 其余的先放过。
      </p>
      <div class="card" style="padding: 0; overflow-x: auto;">
        <table class="topic-table">
          <thead>
            <tr>
              <th>概念</th>
              <th>代码位置 / 表达式</th>
              <th>这一行在说什么</th>
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
      <h2>骨架长这样</h2>
      <p class="lead">
        只留控制流, 细节都砍了。想看完整实现就展开下面的真源码 —— 那是直接从仓库文件里读出来的, 不会和代码脱节。
      </p>
      <div class="card">
        <h3>{{ page.snippetTitle }} <span v-if="page.run" class="tag">可运行</span></h3>
        <CodeBlock :code="page.snippet" />
        <!-- page.source: 直接从仓库 Python 文件取的真源码, 不会和代码漂移 -->
        <SourceSnippet v-for="s in sources" :key="s" :src="s" />
        <p v-if="page.run" class="hint">
          跑一下: <code class="inline">{{ page.run }}</code>
        </p>
      </div>
    </section>

    <QuizCard />

    <ChapterNav :prev="prev" :next="next" />
  </div>

  <div v-else>
    <h1 class="page-title">章节未找到</h1>
    <p class="page-subtitle">当前路由没有对应的 topicPages 配置。</p>
    <ChapterNav :prev="{ name: 'home', label: '主线总览' }" />
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import ChapterIntro from '@/components/ChapterIntro.vue'
import ChapterNav from '@/components/ChapterNav.vue'
import RepoLink from '@/components/RepoLink.vue'
import LabMount from '@/components/LabMount.vue'
import CodeBlock from '@/components/CodeBlock.vue'
import SourceSnippet from '@/components/SourceSnippet.vue'
import QuizCard from '@/components/QuizCard.vue'
import { labMap } from '@/data/labMap.js'
import { learningPath, stageBy, topicPages } from '@/data/models.js'
import { looksLikeRepoRef, normalizeRepoRef, splitRefs } from '@/utils/repo.js'

const route = useRoute()
const page = computed(() => topicPages[route.name])
const hasLabs = computed(() =>
  (page.value?.widgets?.length || 0) + (labMap[route.name]?.length || 0) > 0
)
const sources = computed(() => [].concat(page.value?.source || []))

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
/* key: true 的那条是这一章必须带走的 */
.point-card.key { border-color: var(--accent); }
.key-tag {
  font-size: 10px; font-weight: 500; padding: 1px 6px; border-radius: 3px;
  background: var(--accent-soft); color: var(--accent); letter-spacing: 0.3px;
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
