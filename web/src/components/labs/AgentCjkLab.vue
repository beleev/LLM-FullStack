<!--
  中文检索: 字符 bigram 分词 + BM25 式 idf (对应 llm_agent/core/utils.py:tokenize 与 core/retrieval.py:TfidfIndex)。
  只讲一件事: 分词器只认 [a-z0-9]+ 时, 中文查询的得分恒为 0 —— 检索质量先死在分词上。
-->
<template>
  <LabFrame
    title="中文检索 — 为什么要字符 bigram"
    sub="语料和 m08 demo 完全相同 (6 篇英文 + 2 篇中文)。改查询、切换分词器和 idf 公式, 看 token 怎么切、排序怎么变。悬停任意一个 token, 看它的 idf 和命中了哪些文档。"
    module="llm_agent/m08"
    run="python -m llm_agent.m08_retrieval.demo"
    :challenge="{
      ask: '用「只认英文」分词器查「显存碎片怎么解决」, top-1 得分是多少? 换成 bigram 之后, 为什么「显存」这个 token 的 idf 比「碎片」低?',
      answer: '得分全是 0: 正则 [a-z0-9]+ 把整句中文丢光了, 查询向量是空的, 模型拿到 no matches 只能瞎编。bigram 不需要词典: 「显存碎片」→ 显存 / 存碎 / 碎片, 其中「存碎」是噪声, 但噪声 token 几乎不会在别的文档出现, 影响很小。「显存」在两篇中文文档里都出现 (df=2), 「碎片」只在一篇 (df=1) —— idf = ln(1 + (N−df+0.5)/(df+0.5)) 让越罕见的词权重越高, 几乎每篇都有的词趋近 0。',
    }"
  >
    <template #controls>
      <div class="ctl">
        <label for="cjk-q">查询 query</label>
        <input id="cjk-q" v-model="query" type="text" class="mono q" spellcheck="false" />
        <span class="val mono">{{ qTokens.length }} token</span>
      </div>
      <div class="row">
        <button v-for="q in QUERIES" :key="q" type="button" :class="{ active: query === q }" @click="query = q">{{ q }}</button>
      </div>
      <div class="row">
        <button type="button" :class="{ active: bigram }" @click="bigram = true">英文按词 + 中文 bigram</button>
        <button type="button" :class="{ active: !bigram }" @click="bigram = false">只认 [a-z0-9]+ (旧)</button>
        <button type="button" :class="{ active: bm25 }" @click="bm25 = !bm25">idf: {{ bm25 ? 'BM25 式 ln(1+(N−df+.5)/(df+.5))' : '旧式 ln(N/(1+df))+1' }}</button>
      </div>
    </template>

    <div class="tokens">
      <button
        v-for="(w, i) in qTokens" :key="i" type="button" class="tok mono" :class="{ oov: !(w in index.idf), on: hover === w }"
        @mouseenter="hover = w" @mouseleave="hover = ''" @focus="hover = w" @blur="hover = ''"
      >{{ w }}<em>{{ w in index.idf ? index.idf[w].toFixed(2) : '语料外' }}</em></button>
      <span v-if="!qTokens.length" class="empty">分词结果为空 —— 查询向量是零向量</span>
    </div>
    <div v-for="(d, i) in ranked" :key="d.title" class="doc" :class="{ top: i === 0 && d.score > 0, hit: hover && d.tf[hover] }">
      <span class="mono rank">#{{ i + 1 }}</span>
      <span class="mono title">{{ d.title }}</span>
      <span class="track"><span class="fill" :style="{ width: d.score * 100 + '%' }" /></span>
      <span class="mono score">{{ d.score.toFixed(2) }}</span>
    </div>

    <template #stats>
      <div class="kv"><span>top-1</span><b class="small" :class="ranked[0].score > 0 ? 'good' : 'bad'">{{ ranked[0].score > 0 ? ranked[0].title : 'no matches' }}</b></div>
      <div class="kv"><span>top-1 余弦</span><b :class="ranked[0].score > 0 ? '' : 'bad'">{{ ranked[0].score.toFixed(2) }}</b></div>
      <div class="kv"><span>得分 &gt; 0 的文档</span><b>{{ ranked.filter((d) => d.score > 0).length }} / {{ ranked.length }}</b></div>
      <div class="kv"><span>悬停 token 的 df</span><b>{{ hover ? (index.df[hover] || 0) : '—' }}</b></div>
      <p class="lab-note">这仍是 TF-IDF 余弦, 只是 idf 换成了 BM25 的那一项 (完整 BM25 还有词频饱和 k1 与长度归一 b)。把 embed() 换成神经向量就是稠密检索, 工具接口不变。</p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'

// 与 llm_agent/m08_retrieval/demo.py 的 DOCS 相同
const DOCS = {
  paged_attention: 'paged attention fixes memory fragmentation in the kv cache with block tables, like virtual memory pages for the model.',
  kv_cache: 'how does decode stay fast: the kv cache is the model memory of past keys and values.',
  faq: 'how does the model work, how does the team manage the model, and how does the budget get set.',
  lora: 'how does the team manage finetuning cost: lora trains low rank adapters for the model.',
  dpo: 'how does the team manage alignment: dpo tunes the model on preference pairs, no reward model.',
  sampling: 'how does the model pick the next token: temperature and top k control randomness.',
  分页注意力: '分页注意力用块表管理 KV 缓存, 解决显存碎片问题, 思路类似操作系统的虚拟内存分页。',
  低秩微调: 'LoRA 只训练低秩适配器, 可训练参数不到百分之一, 显存占用大幅下降。',
}
const QUERIES = ['显存碎片怎么解决', 'how does the model manage memory fragmentation', 'LoRA 显存占用']
const query = ref(QUERIES[0]), bigram = ref(true), bm25 = ref(true), hover = ref('')

// ★ utils.py:tokenize —— 英文按词; 中文连续段切成相邻两字的 bigram (单字段保留该字)
const tokenize = (text) => {
  const out = []
  for (const run of text.toLowerCase().match(/[a-z0-9]+|[一-鿿]+/g) || []) {
    if (/[a-z0-9]/.test(run[0])) out.push(run)
    else if (bigram.value) for (let i = 0; i < Math.max(1, run.length - 1); i++) out.push(run.slice(i, i + 2))
  }
  return out
}
const count = (tokens) => tokens.reduce((tf, w) => ((tf[w] = (tf[w] || 0) + 1), tf), {})

const index = computed(() => {
  const N = Object.keys(DOCS).length, df = {}
  const tfs = Object.entries(DOCS).map(([title, body]) => ({ title, tf: count(tokenize(`${title} ${body}`)) }))
  tfs.forEach(({ tf }) => Object.keys(tf).forEach((w) => (df[w] = (df[w] || 0) + 1)))
  const idf = Object.fromEntries(Object.entries(df).map(([w, c]) => [w, bm25.value ? Math.log(1 + (N - c + 0.5) / (c + 0.5)) : Math.log(N / (1 + c)) + 1]))
  return { tfs, df, idf }
})
// 归一化的稀疏向量; 权重 (1+ln tf)·idf: 同一个词重复 3 次不该有 3 倍话语权; 语料外的词直接丢
const vec = (tf, idf) => {
  const v = Object.fromEntries(Object.entries(tf).filter(([w]) => w in idf).map(([w, c]) => [w, (1 + Math.log(c)) * idf[w]]))
  const norm = Math.sqrt(Object.values(v).reduce((s, x) => s + x * x, 0)) || 1
  return Object.fromEntries(Object.entries(v).map(([w, x]) => [w, x / norm]))
}
const qTokens = computed(() => tokenize(query.value))
const ranked = computed(() => {
  const { tfs, idf } = index.value
  const q = vec(count(qTokens.value), idf)
  return tfs.map(({ title, tf }) => {
    const d = vec(tf, idf)
    return { title, tf, score: Object.keys(q).reduce((s, w) => s + q[w] * (d[w] || 0), 0) }
  }).sort((a, b) => b.score - a.score)
})
</script>

<style scoped>
.q { min-height: 32px; min-width: 0; background: var(--bg-elev); color: var(--text); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 0 8px; font-size: 12px; }
.tokens { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 12px; min-height: 44px; }
.tok { display: flex; flex-direction: column; align-items: center; min-height: 0; padding: 3px 8px; font-size: 12px; border: 1px solid var(--accent); border-radius: 4px; background: var(--accent-soft); color: var(--text); }
.tok em { font-style: normal; font-size: 9px; color: var(--text-dim); }
.tok.oov { border-color: var(--border); background: none; color: var(--text-dim); }
.tok.on { outline: 2px solid var(--warn); }
.empty { font-size: 12px; color: var(--danger); align-self: center; }
.doc { display: grid; grid-template-columns: 26px 120px 1fr 40px; gap: 8px; align-items: center; padding: 4px 6px; border-radius: 4px; font-size: 12px; color: var(--text-muted); }
.doc.top { color: var(--text); background: var(--accent-soft); }
.doc.hit { outline: 1px solid var(--warn); }
.rank { color: var(--text-dim); font-size: 11px; } .title { font-size: 11px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.track { height: 8px; border-radius: 4px; background: var(--border); overflow: hidden; } .fill { display: block; height: 100%; background: var(--accent); }
.score { text-align: right; }
.small { font-size: 12px !important; text-align: right; }
</style>
