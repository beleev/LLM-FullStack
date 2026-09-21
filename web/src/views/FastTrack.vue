<!--
  速成路线: 75 章太多了, 这一页回答"我到底该读哪些、按什么顺序、要多久"。
  三个档位 (冲刺 / 主干 / 全部) 共用同一条 learningPath 顺序, 只是筛掉的多少不同。
-->
<template>
  <div>
    <h1 class="page-title">速成路线</h1>
    <p class="page-subtitle">
      六个阶段一共 {{ all.length }} 章, 全读完没必要。先选一个档位, 按顺序走 ——
      主干读完就能讲清"一个大模型怎么训出来、怎么上线、怎么变成会行动的系统", 扩展随时回来补。
    </p>

    <div class="btn-group levels">
      <button
        v-for="l in LEVELS" :key="l.id" type="button"
        :class="{ active: level === l.id }" @click="level = l.id"
      >{{ l.label }} · {{ countOf(l.id) }} 章, {{ l.hint }}</button>
    </div>

    <div class="card summary">
      <div class="sum-item"><span class="v mono">{{ list.length }}</span><span class="k">章</span></div>
      <div class="sum-item"><span class="v mono">{{ totalTime(list.map((c) => c.route)) }}</span><span class="k">粗估用时 (含动手)</span></div>
      <div class="sum-item"><span class="v mono">{{ labCount }}</span><span class="k">个实验台</span></div>
      <div class="sum-item"><span class="v mono" :class="{ good: readCount === list.length }">{{ readCount }}</span><span class="k">已读</span></div>
      <router-link v-if="nextUp" :to="{ name: nextUp.route }">
        <button type="button" class="active">{{ readCount ? '接着读' : '从第一章开始' }}: {{ nextUp.label }} →</button>
      </router-link>
    </div>
    <p class="assume">用时按每分钟 300 字粗估, 每个实验台另算 4 分钟 —— 只用来排计划, 不必当真。</p>

    <section v-for="g in groups" :key="g.stage" class="section">
      <h2>{{ g.title }} <span class="stage-code mono">{{ g.code }}</span></h2>
      <ol class="path">
        <li v-for="c in g.items" :key="c.route" :class="{ done: progress.isVisited(c.route) }">
          <router-link :to="{ name: c.route }" class="row">
            <span class="tier" :class="tierOf(c.route)" :title="TIER_META[tierOf(c.route)].desc">
              {{ isSprint(c.route) ? '★' : TIER_META[tierOf(c.route)].mark }}
            </span>
            <span class="label">{{ c.label }}</span>
            <span class="hint">{{ c.hint || '' }}</span>
            <span class="meta mono">
              <span v-if="labsOf(c.route)" class="lab-n" :title="`${labsOf(c.route)} 个实验台`">⚙{{ labsOf(c.route) }}</span>
              {{ minutesOf(c.route) }}′
            </span>
            <span class="state">{{ progress.isMastered(c.route) ? '✓' : progress.isVisited(c.route) ? '·' : '' }}</span>
          </router-link>
        </li>
      </ol>
    </section>

    <section class="section">
      <h2>这三档是怎么分的</h2>
      <div class="grid grid-3">
        <div v-for="l in LEVELS" :key="l.id" class="card">
          <h3>{{ l.label }}</h3>
          <p class="desc">{{ WHY[l.id] }}</p>
        </div>
      </div>
      <p class="assume">
        扩展章不是"次要内容", 是分支和深水区 —— 里面全是 2026 年真实系统正在用的东西。
        只是先走完主干, 再读它们才有落点。
      </p>
    </section>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'
import { learningPath, stages } from '@/data/models.js'
import { LEVELS, TIER_META, inLevel, isSprint, tierOf } from '@/data/tiers.js'
import { labsOf, minutesOf, totalTime } from '@/utils/readtime.js'
import { useProgress } from '@/composables/useProgress.js'

const WHY = {
  all: '六个阶段的全部章节。每一章都对应仓库里可以运行的代码, 读到哪跑到哪。',
  core: '课程主线。从 numpy 手写反向传播, 一路到 Agent 能调工具, 中间不跳步。',
  sprint: '只有一天的话读这 12 章。每个阶段挑出不读就接不上下一阶段的那几页。',
}

const progress = useProgress()
const level = ref('core')

const all = computed(() => learningPath.filter((p) => p.route !== 'home'))
const list = computed(() => all.value.filter((c) => inLevel(c.route, level.value)))

const countOf = (id) => all.value.filter((c) => inLevel(c.route, id)).length
const labCount = computed(() => list.value.reduce((s, c) => s + labsOf(c.route), 0))
const readCount = computed(() => list.value.filter((c) => progress.isVisited(c.route)).length)
const nextUp = computed(() => list.value.find((c) => !progress.isVisited(c.route)) || list.value[0])

// 按阶段分组, 组内保持 learningPath 的顺序
const groups = computed(() =>
  stages
    .map((s) => ({
      stage: s.id,
      title: `阶段 ${s.idx} · ${s.title}`,
      code: s.code,
      items: list.value.filter((c) => c.route === s.route || (s.chapters || []).some((ch) => ch.route === c.route))
        .map((c) => ({ ...c, hint: (s.chapters || []).find((ch) => ch.route === c.route)?.hint })),
    }))
    .filter((g) => g.items.length),
)
</script>

<style scoped>
.levels { margin-bottom: 16px; }
.summary { display: flex; flex-wrap: wrap; align-items: center; gap: 24px; }
.sum-item { display: flex; flex-direction: column; }
.sum-item .v { font-size: 20px; color: var(--text); }
.sum-item .v.good { color: var(--left); }
.sum-item .k { font-size: 11px; color: var(--text-dim); }
.assume { font-size: 12px; color: var(--text-dim); margin-top: 8px; line-height: 1.7; }
.stage-code { font-size: 11px; font-weight: 400; color: var(--text-dim); }

.path { list-style: none; counter-reset: step; }
.path .row {
  display: grid;
  grid-template-columns: 18px minmax(120px, 1.1fr) 2fr 62px 14px;
  gap: 10px;
  align-items: baseline;
  padding: 7px 10px;
  border-bottom: 1px solid var(--border);
  color: var(--text-muted);
}
.path .row:hover { background: var(--bg-card); text-decoration: none; }
.path li.done .label { color: var(--text-dim); }
.tier { font-size: 10px; text-align: center; }
.tier.core { color: var(--accent); }
.tier.ext { color: var(--text-dim); }
.label { color: var(--text); font-size: 13px; }
.hint { font-size: 12px; color: var(--text-dim); }
.meta { font-size: 11px; color: var(--text-dim); text-align: right; font-variant-numeric: tabular-nums; }
.lab-n { color: var(--accent); margin-right: 4px; }
.state { font-size: 11px; color: var(--left); }

@media (max-width: 900px) {
  .path .row { grid-template-columns: 18px 1fr 52px; }
  .hint, .state { display: none; }
}
</style>
