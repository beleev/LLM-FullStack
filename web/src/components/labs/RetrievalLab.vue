<template>
  <div class="lab card">
    <div class="lab-head">
      <h3>检索实验台 — 关键词计数 vs TF-IDF 余弦</h3>
      <p class="lab-sub">
        同一个查询，两种打分。idf 把高频词权重压低，罕见词主导排序。
        对应 llm_agent/m08_retrieval/demo.py。
      </p>
    </div>

    <div class="lab-controls">
      <div class="ctl">
        <label>查询 query</label>
        <input type="text" v-model="query" spellcheck="false" class="mono query-input" />
        <span class="val mono">{{ queryWords.length }} 词</span>
      </div>
      <div class="mode-row">
        <button :class="{ active: mode === 'keyword' }" @click="mode = 'keyword'">关键词计数</button>
        <button :class="{ active: mode === 'tfidf' }" @click="mode = 'tfidf'">TF-IDF 余弦</button>
      </div>
    </div>

    <div class="lab-body">
      <div v-for="(r, idx) in ranked" :key="r.id" class="doc" :class="{ top: idx < 3 }">
        <div class="doc-head">
          <span class="rank mono">#{{ idx + 1 }}</span>
          <span class="title mono">{{ r.id }}</span>
          <span class="score mono">{{ fmtScore(r.score) }}</span>
        </div>
        <div class="bar-track"><div class="bar" :style="{ width: barWidth(r.score) }"></div></div>
        <p class="body-text">
          <template v-for="(seg, si) in r.segments" :key="si">
            <mark v-if="seg.hit">{{ seg.text }}</mark><template v-else>{{ seg.text }}</template>
          </template>
        </p>
      </div>

      <div class="lab-stats">
        <div class="stat">
          <span class="num mono">{{ tiedCount }}</span>
          <span class="cap">并列文档数 (同分; 仅关键词模式有意义)</span>
        </div>
        <p class="note">
          关键词整数分大量并列、烂大街的词与关键词等权；TF-IDF 给出连续分级。
          把 embed() 换成神经网络向量就是工业级语义检索，工具接口不变。
        </p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'

// 与 llm_agent/m08_retrieval/demo.py 相同的固定语料
const CORPUS = [
  { id: 'paged_attention', body: 'paged attention fixes memory fragmentation in the kv cache with block tables, like virtual memory pages for the model.' },
  { id: 'kv_cache', body: 'the kv cache is the model memory of past keys and values, decode becomes incremental.' },
  { id: 'lora', body: 'lora finetunes a model with low rank adapters, under one percent of weights.' },
  { id: 'dpo', body: 'dpo aligns a model with preference pairs, no reward model needed.' },
  { id: 'ring_attention', body: 'ring attention passes kv blocks between devices so long context fits in device memory.' },
  { id: 'sampling', body: 'temperature and top k shape how a model samples the next token.' },
]
const N = CORPUS.length

const query = ref('model memory fragmentation')
const mode = ref('tfidf')

const words = (text) => text.toLowerCase().match(/[a-z0-9]+/g) || []
const queryWords = computed(() => [...new Set(words(query.value))])

// 预计算: 文档词频 (title+body) 与 idf = ln(N/(1+df)) + 1, 语料固定故只算一次
const docTf = CORPUS.map((d) => {
  const tf = new Map()
  for (const w of words(`${d.id} ${d.body}`)) tf.set(w, (tf.get(w) || 0) + 1)
  return tf
})
const idf = (() => {
  const df = new Map()
  for (const tf of docTf) for (const w of tf.keys()) df.set(w, (df.get(w) || 0) + 1)
  return new Map([...df].map(([w, c]) => [w, Math.log(N / (1 + c)) + 1]))
})()

const vectorize = (tf) => {
  const raw = new Map([...tf].map(([w, c]) => [w, c * (idf.get(w) ?? 1)]))
  const norm = Math.sqrt([...raw.values()].reduce((s, v) => s + v * v, 0)) || 1
  return new Map([...raw].map(([w, v]) => [w, v / norm]))
}
const docVecs = docTf.map(vectorize)

const queryVec = computed(() => {
  const tf = new Map()
  for (const w of words(query.value)) tf.set(w, (tf.get(w) || 0) + 1)
  return vectorize(tf)
})

const scoreOf = (i) => {
  if (mode.value === 'keyword') {
    return queryWords.value.filter((w) => docTf[i].has(w)).length
  }
  let dot = 0
  for (const [w, v] of queryVec.value) dot += v * (docVecs[i].get(w) || 0)
  return dot
}

// 命中词高亮: 把 body 切成 (普通段, 命中段) 序列, 不用 v-html
const segmentsOf = (body) => {
  const qset = new Set(queryWords.value)
  const segs = []
  let last = 0
  for (const m of body.matchAll(/[a-z0-9]+/g)) {
    if (!qset.has(m[0])) continue
    if (m.index > last) segs.push({ text: body.slice(last, m.index), hit: false })
    segs.push({ text: m[0], hit: true })
    last = m.index + m[0].length
  }
  if (last < body.length) segs.push({ text: body.slice(last), hit: false })
  return segs
}

const ranked = computed(() =>
  CORPUS.map((d, i) => ({ ...d, score: scoreOf(i), segments: segmentsOf(d.body) }))
    .sort((a, b) => b.score - a.score)
)
const maxScore = computed(() => Math.max(0, ...ranked.value.map((r) => r.score)))

const fmtScore = (s) => (mode.value === 'tfidf' ? s.toFixed(2) : String(s))
const barWidth = (s) => (maxScore.value > 0 ? `${((s / maxScore.value) * 100).toFixed(1)}%` : '0%')

const tiedCount = computed(() => {
  if (mode.value !== 'keyword') return '—'
  const freq = new Map()
  for (const r of ranked.value) freq.set(r.score, (freq.get(r.score) || 0) + 1)
  return [...freq.values()].filter((c) => c > 1).reduce((s, c) => s + c, 0)
})
</script>

<style scoped>
.lab { margin-bottom: 16px; }
.lab-head h3 { margin-bottom: 4px; }
.lab-sub { color: var(--text-muted); font-size: 13px; line-height: 1.6; margin-bottom: 14px; }

.lab-controls { display: flex; flex-direction: column; gap: 8px; margin-bottom: 14px; }
.mode-row { display: flex; flex-wrap: wrap; gap: 8px; }
.mode-row button { font-size: 12px; }
.ctl { display: grid; grid-template-columns: 110px 1fr 56px; gap: 10px; align-items: center; }
.ctl label { font-size: 12px; color: var(--text-muted); }
.ctl .val { font-size: 12px; text-align: right; color: var(--accent); }
.query-input {
  font-size: 12px; padding: 6px 10px; background: var(--code-bg); color: var(--text);
  border: 1px solid var(--border); border-radius: var(--radius-sm);
}
.query-input:focus { outline: none; border-color: var(--accent); }

.lab-body { display: flex; flex-direction: column; gap: 10px; }
.doc { border: 1px solid var(--border); border-radius: var(--radius-sm); background: var(--bg-elev); padding: 10px 12px; }
.doc.top { border-color: var(--accent); }
.doc-head { display: flex; align-items: baseline; gap: 10px; margin-bottom: 6px; }
.rank { font-size: 11px; color: var(--text-dim); }
.title { font-size: 12px; color: var(--text); }
.score { margin-left: auto; font-size: 12px; color: var(--accent); }
.bar-track { height: 6px; border-radius: 3px; background: var(--code-bg); overflow: hidden; margin-bottom: 6px; }
.bar { height: 6px; background: var(--accent); border-radius: 3px; transition: width 0.2s; }
.body-text { font-size: 12px; color: var(--text-muted); line-height: 1.6; }
.body-text mark { background: var(--accent-soft); color: var(--text); border-radius: 2px; padding: 0 1px; }

.lab-stats { display: flex; align-items: baseline; gap: 16px; flex-wrap: wrap; margin-top: 4px; }
.stat { display: flex; align-items: baseline; gap: 10px; }
.stat .num { font-size: 20px; color: var(--text); }
.stat .cap { font-size: 12px; color: var(--text-muted); }
.note { font-size: 12px; color: var(--text-dim); line-height: 1.7; border-left: 2px solid var(--accent-soft); padding-left: 10px; flex: 1; min-width: 240px; }
</style>
