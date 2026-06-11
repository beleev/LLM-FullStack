<template>
  <div class="lab card">
    <div class="lab-head">
      <h3>BPE 分词实验台 — 看着词表长出来</h3>
      <p class="lab-sub">
        从单字符出发，每一步把语料中最高频的相邻 token 对合并成一个新 token。
        对应可运行代码 llm_basic/bpe.py。
      </p>
    </div>

    <div class="lab-controls">
      <textarea v-model="text" rows="3" maxlength="2000" spellcheck="false" class="mono corpus"></textarea>
      <div class="ctl">
        <label>合并次数</label>
        <input type="range" min="0" max="40" step="1" v-model.number="numMerges" />
        <span class="val mono">{{ numMerges }}</span>
      </div>
    </div>

    <div class="lab-body">
      <div class="token-view">
        <span
          v-for="(t, idx) in tokens" :key="idx"
          class="token mono" :class="{ multi: isMulti(t) }"
        >{{ disp(t) }}</span>
      </div>

      <div class="lab-stats">
        <div class="stat">
          <span class="num mono">{{ tokens.length }} / {{ charCount }}</span>
          <span class="cap">token 数 / 字符数</span>
        </div>
        <div class="stat">
          <span class="num mono">{{ compression }}</span>
          <span class="cap">压缩率 (字符数 ÷ token 数)</span>
        </div>
        <div class="merge-log">
          <p class="log-title">合并记录</p>
          <p v-for="(m, idx) in merges" :key="idx" class="log-line mono">
            {{ idx + 1 }}. '{{ disp(m.a) }}'+'{{ disp(m.b) }}' → '{{ disp(m.a + m.b) }}' (出现 {{ m.count }} 次)
          </p>
          <p v-if="merges.length === 0" class="log-line mono">（拖动滑块开始合并）</p>
        </div>
        <p class="note">
          词表大小是「序列长度 ↔ embedding 参数量」的权衡旋钮；真实 GPT-2 在 50 万倍大的语料上做
          5 万次合并，思想与这里完全一致 (Python 版是 byte-level，可处理任意 UTF-8)。
        </p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'

const DEFAULT_TEXT =
  'First Citizen:\nBefore we proceed any further, hear me speak.\n' +
  'All: Speak, speak.\nFirst Citizen: You are all resolved rather to die than to famish?'
const MAX_CHARS = 2000

const text = ref(DEFAULT_TEXT)
const numMerges = ref(12)

const clipped = computed(() => text.value.slice(0, MAX_CHARS))
const charCount = computed(() => Array.from(clipped.value).length)

// char-level BPE: 每轮统计相邻 token 对频次, 合并最高频的一对 (平手取先出现者)
const bpe = computed(() => {
  let toks = Array.from(clipped.value)
  const applied = []
  for (let step = 0; step < numMerges.value; step++) {
    if (toks.length < 2) break
    const counts = new Map()
    for (let k = 0; k < toks.length - 1; k++) {
      const key = toks[k].length + ':' + toks[k] + toks[k + 1]
      const e = counts.get(key)
      if (e) e.count += 1
      else counts.set(key, { a: toks[k], b: toks[k + 1], count: 1 })
    }
    let best = null
    for (const e of counts.values()) if (!best || e.count > best.count) best = e
    if (!best) break
    const next = []
    for (let k = 0; k < toks.length; k++) {
      if (k < toks.length - 1 && toks[k] === best.a && toks[k + 1] === best.b) {
        next.push(best.a + best.b)
        k += 1
      } else {
        next.push(toks[k])
      }
    }
    toks = next
    applied.push({ a: best.a, b: best.b, count: best.count })
  }
  return { toks, applied }
})

const tokens = computed(() => bpe.value.toks)
const merges = computed(() => bpe.value.applied)
const compression = computed(() =>
  tokens.value.length > 0 ? (charCount.value / tokens.value.length).toFixed(2) : '—'
)

const isMulti = (t) => Array.from(t).length > 1
const disp = (t) => t.replace(/ /g, '␣').replace(/\n/g, '⏎')
</script>

<style scoped>
.lab { margin-bottom: 16px; }
.lab-head h3 { margin-bottom: 4px; }
.lab-sub { color: var(--text-muted); font-size: 13px; line-height: 1.6; margin-bottom: 14px; }

.lab-controls { display: flex; flex-direction: column; gap: 8px; margin-bottom: 14px; }
.corpus {
  width: 100%; resize: vertical; font-size: 12px; line-height: 1.6; padding: 8px 10px;
  background: var(--code-bg); color: var(--text);
  border: 1px solid var(--border); border-radius: var(--radius-sm);
}
.corpus:focus { outline: none; border-color: var(--accent); }
.ctl { display: grid; grid-template-columns: 110px 1fr 56px; gap: 10px; align-items: center; }
.ctl label { font-size: 12px; color: var(--text-muted); }
.ctl .val { font-size: 12px; text-align: right; color: var(--accent); }

.lab-body { display: grid; grid-template-columns: minmax(220px, 1fr) 1fr; gap: 18px; align-items: start; }
.token-view {
  display: flex; flex-wrap: wrap; gap: 3px; align-content: flex-start;
  background: var(--code-bg); border-radius: var(--radius-sm); padding: 10px;
  max-height: 300px; overflow-y: auto;
}
.token {
  font-size: 11px; line-height: 1.5; padding: 1px 4px; border-radius: 3px;
  background: var(--bg-elev); color: var(--text-muted); white-space: pre;
}
.token.multi { background: var(--accent); color: var(--code-bg); }

.lab-stats { display: flex; flex-direction: column; gap: 10px; }
.stat { display: flex; align-items: baseline; gap: 10px; }
.stat .num { font-size: 20px; color: var(--text); }
.stat .cap { font-size: 12px; color: var(--text-muted); }
.merge-log {
  max-height: 180px; overflow-y: auto; padding: 8px 10px;
  background: var(--bg-elev); border: 1px solid var(--border); border-radius: var(--radius-sm);
}
.log-title { font-size: 12px; color: var(--text-muted); margin-bottom: 6px; }
.log-line { font-size: 11px; line-height: 1.7; color: var(--text-dim); }
.note { font-size: 12px; color: var(--text-dim); line-height: 1.7; border-left: 2px solid var(--accent-soft); padding-left: 10px; }

@media (max-width: 960px) {
  .lab-body { grid-template-columns: 1fr; }
}
</style>
