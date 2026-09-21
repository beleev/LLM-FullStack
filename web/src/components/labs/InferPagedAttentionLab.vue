<!-- PagedAttention 实验台 (对应 llm_infer/m02_paged_attention): 事件列表 → computed 重放, 拖滑杆时结果确定。 -->
<template>
  <LabFrame
    title="PagedAttention — 把 KV 显存切成 block 按需分配"
    sub="上面是物理 block 池 (共 384 个 token 的 KV 容量), 颜色 = 属于哪个请求, 填充高度 = 这个 block 写了多少。下面是每个请求的 block table (逻辑页 → 物理块)。自己接请求、decode、结束请求, 看物理块如何不连续地被复用; 右侧同时算一笔账: 如果按 max_len 连续预留会浪费多少。"
    module="llm_infer/m02"
    run="python -m llm_infer.m02_paged_attention.demo"
    :challenge="{
      ask: '分页之后, attention 本身算得更快了吗? 把 block 大小从 4 拖到 32, 浪费率怎么变, 为什么真实系统不干脆用 1?',
      answer: '没有更快 —— 经页表间接寻址反而多一次查表。分页买到的是显存利用率: 浪费只剩每个请求最后一个 block 的尾巴 (平均 bs/2 个 token), 而连续预留要为每条请求押上整个 max_len。利用率高 → 同一张卡能塞更多并发 → batch 更大 → 吞吐更高。block 越小浪费越少, 但页表越长、kernel 访存越碎, 所以 vLLM 默认取 16。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" class="active" @click="push({ type: 'new', len: nextLen })">新请求 (prompt {{ nextLen }} token)</button>
        <button type="button" @click="reset(seed)">重置</button>
        <button type="button" @click="reset(seed + 1)">换一组</button>
        <span class="msg" :class="{ bad: st.bad }">{{ st.msg }}</span>
      </div>
      <LabSlider v-model="bsExp" label="block 大小" :min="2" :max="5" :format="(v) => 2 ** v" unit=" tok" />
      <LabSlider v-model="maxLen" label="连续预留的 max_len" :min="64" :max="256" :step="16" unit=" tok" />
      <LabSlider v-model="pos" label="查地址: 逻辑位置 pos" :min="0" :max="Math.max(0, (sel?.len || 1) - 1)" />
    </template>

    <div class="pool">
      <span
        v-for="(b, i) in st.blocks" :key="i" class="cell blk"
        :class="{ dim: focus !== null && b.owner !== null && b.owner !== focus, hit: i === addr.block }"
        :style="blockStyle(b)" :title="b.owner === null ? `物理块 ${i}: 空闲` : `物理块 ${i}: 请求 ${b.owner}, 已写 ${b.fill}/${bs}`"
      >{{ i }}</span>
    </div>

    <div class="reqs">
      <div
        v-for="r in st.live" :key="r.id" class="req" :class="{ sel: r.id === selId }"
        @mouseenter="hover = r.id" @mouseleave="hover = null"
      >
        <button type="button" class="name" :style="{ borderColor: color(r.id) }" @click="selId = r.id">
          请求 {{ r.id }} · {{ r.len }} tok
        </button>
        <span class="table mono">
          [<button
            v-for="(b, i) in r.table" :key="i" type="button" class="tb"
            :class="{ on: r.id === selId && i === addr.page }" :title="`逻辑页 ${i} → 物理块 ${b}`"
            @click="selId = r.id; pos = i * bs"
          >{{ b }}</button>]
        </span>
        <span class="ops">
          <button type="button" @click="push({ type: 'dec', id: r.id, n: 1 })">decode +1</button>
          <button type="button" @click="push({ type: 'dec', id: r.id, n: 8 })">+8</button>
          <button type="button" @click="push({ type: 'fin', id: r.id })">结束</button>
        </span>
      </div>
      <p v-if="!st.live.length" class="lab-note">池子空了。点「新请求」。</p>
    </div>

    <template #stats>
      <div class="kv"><span>分页: 已分配里的浪费</span><b class="good">{{ pct(st.pagedWaste) }}</b></div>
      <div class="kv"><span>连续预留 max_len: 浪费</span><b class="bad">{{ pct(st.contWaste) }}</b></div>
      <div class="kv">
        <span>这 {{ st.live.length }} 条请求, 连续预留装得下</span>
        <b :class="st.contFit < st.live.length ? 'bad' : ''">{{ st.contFit }} 条</b>
      </div>
      <div class="kv"><span>空闲物理块</span><b>{{ st.free }} / {{ st.blocks.length }}</b></div>
      <div v-if="sel" class="kv">
        <span>请求 {{ sel.id }} 的 pos {{ addr.pos }}</span>
        <b>块 {{ addr.block }} · 槽 {{ addr.slot }}</b>
      </div>
      <p class="lab-note">
        地址翻译只有一行: <code class="inline">block_table[pos // bs], pos % bs</code>。
        例: bs=8, pos=37, 页表 [9,4,6,1,3] → 37//8=4 → 物理块 3, 槽 37%8=5。<br />
        几条请求交替 decode, 各自新要的块就交错排开; 还回来的块接在空闲链表尾部 (与 m02 一致), 池子转过一圈后新请求拿到的全是零散块 —— 页表让"逻辑连续"不再需要"物理连续"。
      </p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { mulberry32, range, sum, clamp } from '@/utils/labmath.js'

const POOL = 384, CAP = 64 // 池容量 (token) 与单请求长度上限
const PALETTE = ['var(--accent)', 'var(--left)', 'var(--right)', 'var(--warn)', 'var(--danger)', 'var(--eye)']
const color = (id) => PALETTE[id % PALETTE.length]

const bsExp = ref(3)
const bs = computed(() => 2 ** bsExp.value)
const maxLen = ref(128)
const seed = ref(1)
const events = ref([])
const selId = ref(0)
const hover = ref(null)
const pos = ref(0)

// 第 i 个新请求的 prompt 长度只由 (seed, i) 决定, 重放时不会变
const lenOf = (i) => 5 + Math.floor(mulberry32(seed.value * 97 + i)() * 28)
const nextLen = computed(() => lenOf(events.value.filter((e) => e.type === 'new').length))
const push = (e) => { events.value = [...events.value, e]; if (e.type === 'new') selId.value = st.value.reqs.length - 1 }
const reset = (s) => {
  seed.value = s
  // 预置一段剧情: 接 4 条 → 交替 decode (新块交错) → 结束 1 号 → 再接 1 条
  const ev = range(4).map((i) => ({ type: 'new', len: lenOf(i) }))
  ev.push({ type: 'dec', id: 0, n: 8 }, { type: 'dec', id: 2, n: 8 }, { type: 'dec', id: 3, n: 8 }, { type: 'dec', id: 0, n: 8 })
  ev.push({ type: 'fin', id: 1 }, { type: 'new', len: lenOf(4) })
  events.value = ev; selId.value = 4; pos.value = 0
}
reset(1)

const st = computed(() => {
  const B = bs.value, nb = POOL / B, need = (len) => Math.ceil(len / B)
  const free = range(nb) // 空闲链表: 从头取, 还回来的接在尾部
  const reqs = []
  let msg = '', bad = false
  for (const e of events.value) {
    if (e.type === 'new') {
      const r = { id: reqs.length, len: e.len, table: [], live: false }
      reqs.push(r)
      bad = need(e.len) > free.length
      if (bad) { msg = `请求 ${r.id} 被拒: 需要 ${need(e.len)} 个 block, 只剩 ${free.length} 个`; continue }
      r.table = free.splice(0, need(e.len)); r.live = true
      msg = `请求 ${r.id} 拿到物理块 [${r.table.join(', ')}]`
      continue
    }
    const r = reqs[e.id]
    if (!r?.live) continue
    if (e.type === 'fin') { free.push(...r.table); msg = `请求 ${r.id} 结束, 归还 ${r.table.length} 个 block`; bad = false; r.table = []; r.live = false; continue }
    const len = Math.min(CAP, r.len + e.n), extra = need(len) - r.table.length
    bad = extra > free.length || len === r.len
    if (bad) { msg = len === r.len ? `请求 ${r.id} 已到长度上限 ${CAP}` : `请求 ${r.id} decode 失败: 没有空闲 block (真实系统此时会抢占)`; continue }
    // ★ 按需分配: 只有跨过 block 边界时才多要一个物理块
    r.table.push(...free.splice(0, extra)); r.len = len
    msg = extra ? `请求 ${r.id} 跨过 block 边界, 新增物理块 ${r.table.at(-1)}` : `请求 ${r.id} 写进最后一个 block 的空槽, 不需要新块`
  }
  const live = reqs.filter((r) => r.live)
  const blocks = range(nb).map(() => ({ owner: null, fill: 0 }))
  live.forEach((r) => r.table.forEach((b, i) => { blocks[b] = { owner: r.id, fill: Math.min(B, r.len - i * B) } }))
  const used = sum(live.map((r) => r.len)), alloc = sum(live.map((r) => r.table.length)) * B
  const reserved = live.length * maxLen.value
  return {
    reqs, live, blocks, msg, bad, free: free.length,
    pagedWaste: alloc ? (alloc - used) / alloc : 0,
    contWaste: reserved ? (reserved - sum(live.map((r) => Math.min(r.len, maxLen.value)))) / reserved : 0,
    contFit: Math.min(live.length, Math.floor(POOL / maxLen.value)), // 最好情况: 不算外部碎片
  }
})

const sel = computed(() => st.value.live.find((r) => r.id === selId.value) || st.value.live[0] || null)
const focus = computed(() => hover.value ?? sel.value?.id ?? null)
watch(sel, (r) => { if (r) pos.value = clamp(pos.value, 0, r.len - 1) })
const addr = computed(() => {
  const r = sel.value
  if (!r) return { pos: 0, page: -1, block: -1, slot: 0 }
  const p = clamp(pos.value, 0, r.len - 1), page = Math.floor(p / bs.value)
  return { pos: p, page, block: r.table[page], slot: p % bs.value }
})

const pct = (x) => (x * 100).toFixed(1) + '%'
const blockStyle = (b) => {
  if (b.owner === null) return {}
  const c = color(b.owner), f = Math.round((b.fill / bs.value) * 100)
  return {
    borderColor: c,
    background: `linear-gradient(to top, color-mix(in srgb, ${c} 55%, transparent) ${f}%, color-mix(in srgb, ${c} 12%, transparent) ${f}%)`,
  }
}
</script>

<style scoped>
.pool { display: flex; flex-wrap: wrap; gap: 3px; margin-bottom: 14px; }
.blk { width: 26px; height: 26px; color: var(--text); }
.blk.hit { outline: 2px solid var(--text); outline-offset: 1px; }
.reqs { display: flex; flex-direction: column; gap: 6px; }
.req { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; padding: 4px 6px; border: 1px solid transparent; border-radius: var(--radius-sm); }
.req.sel { border-color: var(--border-strong); background: var(--bg-elev); }
.req button { font-size: 11px; min-height: 26px; padding: 2px 8px; }
.name { border-left-width: 4px; min-width: 116px; text-align: left; }
.table { font-size: 12px; color: var(--text-muted); display: flex; flex-wrap: wrap; gap: 2px; align-items: center; }
.tb { min-width: 26px; padding: 2px 4px !important; font-family: inherit; }
.tb.on { border-color: var(--accent); color: var(--accent); }
.ops { display: flex; gap: 4px; margin-left: auto; }
.msg { font-size: 12px; color: var(--text-muted); }
.msg.bad { color: var(--danger); }
</style>
