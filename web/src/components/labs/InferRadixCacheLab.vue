<!-- Radix 前缀缓存实验台 (对应 llm_infer/m05_radix_cache): JS 里老老实实建一棵基数树, 逐条请求重放。 -->
<template>
  <LabFrame
    title="Radix cache — 公共前缀只算一次"
    sub="每一行是树上的一个节点, 行里的 token 串就是「父 → 它」这条边 (边压缩: 一条边可以存很多 token)。逐条放入请求: 绿色 = 这条请求命中的前缀 (KV 直接复用, 不用前向), 蓝色 = 新插入的部分, 橙色 = 刚被从中间劈开的边。把容量往小拖, 看 LRU 从叶子开始驱逐。点任意节点看它代表的完整前缀。"
    module="llm_infer/m05"
    run="python -m llm_infer.m05_radix_cache.demo"
    :challenge="{
      ask: '容量不够时, 为什么只能驱逐叶子? 如果直接扔掉「最久没用」的中间节点会怎样?',
      answer: 'KV 依赖它前面的全部 token。中间节点的 KV 是所有后代前缀的一部分。扔了这个中间节点, 后代的 KV 虽然还在显存里却再也匹配不上 (从根走不到), 等于一起作废。所以驱逐顺序只能是叶子 → 变成叶子的父节点 → …。另外注意右侧对比: radix 能命中任意长度 n, 而 block hash 只能命中 ⌊n/bs⌋·bs —— 共享 100 个 token、bs=16 时只能复用 96 个。',
    }"
  >
    <template #controls>
      <div class="row">
        <button v-for="(p, k) in PRESETS" :key="k" type="button" :class="{ active: preset === k }" @click="preset = k">{{ p.name }}</button>
        <button type="button" class="active" :disabled="stepper.step.value >= reqs.length - 1" @click="stepper.next()">下一条请求 ▶</button>
      </div>
      <LabSlider v-model="cap" label="缓存容量" :min="8" :max="64" unit=" tok" />
      <LabSlider v-model="bsExp" label="对比: block hash 的 block" :min="2" :max="4" :format="(v) => 2 ** v" unit=" tok" />
      <StepPlayer :stepper="stepper" :label="`请求 ${stepper.step.value + 1} / ${reqs.length}`" />
    </template>

    <p class="cap">请求 {{ stepper.step.value + 1 }}:</p>
    <div class="strip">
      <span v-for="(tk, i) in cur.ids" :key="i" class="cell tok" :class="i < cur.hit ? 'ok' : 'on'">{{ tk }}</span>
    </div>

    <p class="cap">radix tree ({{ tree.rows.length }} 个节点):</p>
    <div class="tree" role="tree">
      <div v-for="r in tree.rows" :key="r.id" class="node" :style="{ paddingLeft: r.depth * 22 + 'px' }">
        <button
          type="button" class="edge" role="treeitem"
          :class="{ path: selPath.has(r.id), sel: r.id === selId }" @click="selId = r.id === selId ? -1 : r.id"
        >
          <span class="twig">└</span>
          <span v-for="(tk, i) in r.tokens" :key="i" class="cell tok" :class="rowClass(r)">{{ tk }}</span>
          <span class="meta mono">{{ r.tokens.length }} tok · 上次用于 #{{ r.last + 1 }}{{ r.leaf ? ' · 叶' : '' }}</span>
        </button>
      </div>
      <p v-if="!tree.rows.length" class="lab-note">树被驱逐空了。</p>
    </div>
    <p v-if="selNode" class="lab-note sel-note">
      选中节点代表的完整前缀 ({{ selPrefix.length }} tok): <span class="mono">{{ selPrefix.join(' ') }}</span>
    </p>

    <template #stats>
      <div class="kv"><span>本条命中 / 长度</span><b :class="cur.hit ? 'good' : ''">{{ cur.hit }} / {{ cur.ids.length }}</b></div>
      <div class="kv"><span>同样前缀, block hash 只能命中</span><b :class="blockHit < cur.hit ? 'bad' : ''">{{ blockHit }}</b></div>
      <div class="kv"><span>累计命中率 (省掉的 prefill)</span><b class="good">{{ (tree.hitRate * 100).toFixed(1) }}%</b></div>
      <div class="kv"><span>树里 token / 容量</span><b :class="tree.size > cap ? 'bad' : ''">{{ tree.size }} / {{ cap }}</b></div>
      <div class="kv"><span>累计驱逐</span><b :class="tree.evicted ? 'bad' : ''">{{ tree.evicted }} tok</b></div>
      <p class="lab-note">
        {{ cur.msg }}<br />
        匹配只有三步: 按「子边首 token」找孩子 → 沿边求最长公共前缀 → 停在边中间就 <code class="inline">_split</code>。
        当前请求走过的节点被锁住 (ref&gt;0), 不会被驱逐。
      </p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import StepPlayer from '@/components/lab/StepPlayer.vue'
import { useStepper } from '@/composables/useStepper.js'

const SYS = ['你', '是', '严谨', '的', '助手', '。']
const SHOT = ['Q', '1+1', 'A', '2', ';', 'Q', '2+3', 'A', '5', ';']
const A1 = ['sys', 'A:你好'], A2 = [...A1, '→嗨!', 'A:讲个', '笑话'], A3 = [...A2, '→从前…', 'A:再', '来', '一个']
const B1 = ['sys', 'B:在吗'], B2 = [...B1, '→在', 'B:帮我', '改', 'bug'], B3 = [...B2, '→贴代码', 'B:如下', '…']
const PRESETS = {
  system: { name: '共享 system prompt', reqs: [
    [...SYS, '翻译', ':', 'hello'], [...SYS, '翻译', ':', 'world'], [...SYS, '总结', ':', '这篇', '文章'],
    [...SYS, '翻译', ':', 'hello', 'world'], [...SYS, '写', '一首', '诗'], [...SYS, '总结', ':', '这段', '代码'] ] },
  fewshot: { name: 'few-shot', reqs: [
    [...SHOT, 'Q', '7+8', 'A'], [...SHOT, 'Q', '3+9', 'A'], [...SHOT.slice(0, 5), 'Q', '4+4', 'A', '8', ';', 'Q', '9+1', 'A'],
    [...SHOT, 'Q', '7+8', 'A'], [...SHOT, 'Q', '6+6', 'A'], [...SHOT.slice(0, 5), 'Q', '4+4', 'A', '8', ';', 'Q', '5+5', 'A'] ] },
  chat: { name: '多轮对话 (两个用户交替)', reqs: [A1, B1, A2, B2, A3, B3] },
}

const preset = ref('system')
const cap = ref(64)
const bsExp = ref(2)
const selId = ref(-1)
const reqs = computed(() => PRESETS[preset.value].reqs)
const stepper = useStepper(() => reqs.value.length, { interval: 1100 })
watch(preset, () => { stepper.reset(); selId.value = -1 })

// 把第 0..upto 条请求依次放进一棵新树 —— 纯函数, 滑杆一动整棵树重建
const tree = computed(() => {
  let nid = 0
  const mk = (tokens, parent) => ({ id: nid++, tokens, parent, children: new Map(), last: 0 })
  const root = mk([], null)
  let size = 0, evicted = 0, totHit = 0, totLen = 0, cur = null
  for (let c = 0; c <= stepper.step.value; c++) {
    const ids = reqs.value[c], path = []
    let node = root, i = 0, split = -1
    while (i < ids.length && node.children.has(ids[i])) {   // 孩子按「子边首 token」索引
      let child = node.children.get(ids[i]), n = 0
      while (n < child.tokens.length && child.tokens[n] === ids[i + n]) n++
      if (n < child.tokens.length) {
        // ★ 分叉落在边中间: 劈成 上半 (公共部分) + 下半 (原来的后续), KV 槽位跟着 token 一起切
        const upper = mk(child.tokens.slice(0, n), node)
        upper.last = child.last
        node.children.set(ids[i], upper)
        child.tokens = child.tokens.slice(n); child.parent = upper
        upper.children.set(child.tokens[0], child)
        split = upper.id; child = upper
      }
      child.last = c; path.push(child.id); node = child; i += n
    }
    const hit = i
    let leaf = null
    if (i < ids.length) { leaf = mk(ids.slice(i), node); leaf.last = c; node.children.set(ids[i], leaf); size += leaf.tokens.length }
    // LRU 驱逐: 只看叶子, 当前请求路径上的节点锁住
    const locked = new Set([...path, leaf?.id]), gone = []
    while (size > cap.value) {
      const leaves = []
      const walk = (nd) => { nd.children.size ? nd.children.forEach(walk) : nd !== root && !locked.has(nd.id) && leaves.push(nd) }
      walk(root)
      if (!leaves.length) break
      const v = leaves.reduce((a, b) => (b.last < a.last ? b : a))
      v.parent.children.delete(v.tokens[0]); size -= v.tokens.length; evicted += v.tokens.length; gone.push(v.tokens.join(' '))
    }
    totHit += hit; totLen += ids.length
    const msg = [hit ? `沿树命中 ${hit} 个 token` : '没有公共前缀, 整条都要 prefill',
      split >= 0 ? '命中停在一条边的中间 → 劈开这条边' : '', gone.length ? `超容量, 驱逐叶子: 「${gone.join('」「')}」` : ''].filter(Boolean).join('; ')
    cur = { ids, hit, path: new Set(path), leaf: leaf?.id ?? -1, split, msg: msg + '。' }
  }
  const rows = [], byId = new Map()
  const flat = (nd, depth) => nd.children.forEach((ch) => {
    rows.push({ id: ch.id, tokens: ch.tokens, depth, last: ch.last, leaf: !ch.children.size }); byId.set(ch.id, ch); flat(ch, depth + 1)
  })
  flat(root, 0)
  return { rows, byId, size, evicted, hitRate: totHit / totLen, cur }
})

const cur = computed(() => tree.value.cur)
const blockHit = computed(() => Math.floor(cur.value.hit / 2 ** bsExp.value) * 2 ** bsExp.value)
const rowClass = (r) => (r.id === cur.value.split ? 'hot' : cur.value.path.has(r.id) ? 'ok' : r.id === cur.value.leaf ? 'on' : '')

const selNode = computed(() => tree.value.byId.get(selId.value))
const selChain = computed(() => { const out = []; for (let n = selNode.value; n && n.parent; n = n.parent) out.unshift(n); return out })
const selPath = computed(() => new Set(selChain.value.map((n) => n.id)))
const selPrefix = computed(() => selChain.value.flatMap((n) => n.tokens))
</script>

<style scoped>
.cap { font-size: 12px; color: var(--text-muted); margin: 10px 0 6px; }
.cap:first-child { margin-top: 0; }
.strip { display: flex; flex-wrap: wrap; gap: 3px; }
.tok { min-width: 0; padding: 0 6px; font-size: 11px; white-space: nowrap; }
.tree { display: flex; flex-direction: column; gap: 3px; }
.edge { display: flex; flex-wrap: wrap; align-items: center; gap: 3px; width: 100%; min-height: 30px; padding: 3px 6px; text-align: left; background: transparent; border: 1px solid transparent; }
.edge.path { border-color: var(--border-strong); background: var(--bg-elev); }
.edge.sel { border-color: var(--accent); }
.twig { color: var(--text-dim); font-size: 12px; }
.meta { margin-left: auto; font-size: 10px; color: var(--text-dim); }
.sel-note { margin-top: 10px; }
</style>
