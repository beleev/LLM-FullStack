<!-- 分层 KV offload 实验台 (对应 llm_infer/m20_kv_offload): TieredKVCache 的 JS 移植, 时间全部来自代价模型。 -->
<template>
  <LabFrame
    title="分层 KV offload — GPU 装不下的历史, 降级而不是丢弃"
    sub="多轮对话: 每个用户每轮的 prompt = 共享 system prompt + 自己的全部历史 + 新消息, 用户轮流发言。每根柱子是一个请求的 prompt, 按颜色分成: 在 GPU / CPU / 磁盘命中的 token 与必须重算的 token; 折线是它的 TTFT。拖各层容量, 点柱子看明细。注意: 所有时间来自代价模型 (8000 tok/s 重算、128 KiB/token), 不是实测。"
    module="llm_infer/m20"
    run="python -m llm_infer.m20_kv_offload.demo"
    :challenge="{
      ask: '把磁盘带宽拖到 0.5 GB/s, 再切到「命中就加载」。多了一层缓存, 平均 TTFT 为什么反而变差? 切回「取小」后又发生了什么?',
      answer: '0.5 GB/s 时搬一个 block (2 MiB) 要 4.2 ms, 而重算它只要 2 ms —— 命中了也不该搬。「命中就加载」会老老实实去搬, TTFT 比没有这一层还差。「取小」在每层比较 load = 固定延迟 + 字节/带宽 与 recompute, 慢就直接重算, 最坏也不比没有这层差。带宽够快时还有第二个门槛: 固定延迟要靠命中块数摊薄, 交叉点 n* = 延迟 / (每块重算 − 每块加载)。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: !alwaysLoad }" @click="alwaysLoad = false">加载 / 重算取小</button>
        <button type="button" :class="{ active: alwaysLoad }" @click="alwaysLoad = true">命中就加载</button>
        <button type="button" @click="seed++">换一组</button>
      </div>
      <LabSlider v-model="users" label="用户数 (各 6 轮)" :min="2" :max="12" />
      <LabSlider v-model="capGpu" label="GPU 层容量" :min="0" :max="256" :step="8" unit=" blk" />
      <LabSlider v-model="capCpu" label="CPU 层容量 (25 GB/s)" :min="0" :max="512" :step="16" unit=" blk" />
      <LabSlider v-model="capDisk" label="磁盘层容量" :min="0" :max="4096" :step="128" unit=" blk" />
      <LabSlider v-model="diskBw" label="磁盘带宽" :min="0.5" :max="7" :step="0.5" unit=" GB/s" />
      <LabSlider v-model="diskLat" label="磁盘每次加载固定延迟" :min="0" :max="20" unit=" ms" />
    </template>

    <svg :viewBox="`0 0 ${W} ${H + 16}`" role="img" aria-label="每个请求的命中构成与 TTFT">
      <g
        v-for="(r, i) in sim.reqs" :key="i" tabindex="0" role="button" :aria-label="`请求 ${i + 1}`"
        @mouseenter="pick = i" @click="pick = i" @keydown.enter="pick = i"
      >
        <rect :x="i * bw" y="0" :width="bw" :height="H" fill="transparent" />
        <rect
          v-for="s in r.segs" :key="s.k" :x="i * bw + 0.5" :y="H - (s.y0 + s.n) * ky"
          :width="Math.max(1, bw - 1)" :height="s.n * ky" :fill="COLORS[s.k]" :opacity="s.k === 3 ? 0.35 : 0.85"
        />
        <rect v-if="i === pickI" :x="i * bw" y="0" :width="bw" :height="H" class="pick" />
      </g>
      <polyline :points="sim.reqs.map((r, i) => `${(i + 0.5) * bw},${H - (r.ttft / sim.maxTtft) * (H - 4)}`).join(' ')" class="ttft" />
      <text x="0" :y="H + 13" class="ax">请求 1 (第 1 轮)</text>
      <text :x="W" :y="H + 13" class="ax" text-anchor="end">请求 {{ sim.reqs.length }} (第 6 轮) · 折线 = TTFT, 峰值 {{ sim.maxTtft.toFixed(0) }} ms</text>
    </svg>
    <p class="legend">
      <span v-for="(n, k) in NAMES" :key="k"><i :style="{ background: COLORS[k], opacity: k === 3 ? 0.35 : 0.85 }" />{{ n }}</span>
    </p>
    <p class="lab-note detail">
      请求 {{ pickI + 1 }} · 用户 {{ cur.user }} 第 {{ cur.turn }} 轮 · prompt {{ cur.len }} tok =
      GPU {{ cur.hit[0] }} + CPU {{ cur.hit[1] }} + 磁盘 {{ cur.hit[2] }} + 重算 {{ cur.miss }};
      TTFT <b>{{ cur.ttft.toFixed(1) }} ms</b> (全量重算 {{ (cur.len / 8).toFixed(1) }} ms){{ cur.skipped ? `; 有 ${cur.skipped} 个命中 token 因为加载更慢而选择重算` : '' }}
    </p>

    <template #stats>
      <div class="kv"><span>命中 GPU / CPU / 磁盘</span><b>{{ pct(sim.hit[0]) }} / {{ pct(sim.hit[1]) }} / {{ pct(sim.hit[2]) }}</b></div>
      <div class="kv"><span>miss (必须重算)</span><b :class="sim.miss > 0.5 ? 'bad' : 'good'">{{ pct(sim.miss) }}</b></div>
      <div class="kv">
        <span>平均 TTFT (无缓存 {{ sim.noCache.toFixed(1) }} ms)</span>
        <b :class="sim.ttft > sim.noCache * 0.9 ? 'bad' : 'good'">{{ sim.ttft.toFixed(1) }} ms</b>
      </div>
      <div class="kv"><span>磁盘交叉点 n*</span><b :class="nStar === null ? 'bad' : ''">{{ nStar === null ? '永不' : nStar.toFixed(2) + ' 块' }}</b></div>
      <p class="lab-note">
        每块 (16 tok = 2 MiB): 重算 2.00 ms · CPU 加载 0.084 ms · 磁盘加载 {{ diskPerBlock.toFixed(3) }} ms (+ 每次 {{ diskLat }} ms)。<br />
        n* = 延迟 ÷ (每块重算 − 每块加载): 命中块数少于它时重算更快; 每块加载 ≥ 重算时永远不该加载。
        查找沿 hash 链走, 第一个全层 miss 处就停 —— 后面的块即使还在也用不上。
      </p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { mulberry32, range, sum } from '@/utils/labmath.js'

const BS = 16, BLOCK_BYTES = BS * 131072, RECOMP_MS = (n) => n / 8        // 8000 tok/s
const W = 640, H = 170
const COLORS = ['var(--left)', 'var(--accent)', 'var(--warn)', 'var(--danger)']
const NAMES = ['GPU 命中', 'CPU 命中', '磁盘命中', 'miss → 重算']

const users = ref(8), capGpu = ref(64), capCpu = ref(192), capDisk = ref(4096)
const diskBw = ref(3), diskLat = ref(2), alwaysLoad = ref(false), seed = ref(0), pick = ref(0)

const diskPerBlock = computed(() => (BLOCK_BYTES / (diskBw.value * 1e9)) * 1e3)
const nStar = computed(() => (diskPerBlock.value >= 2 ? null : diskLat.value / (2 - diskPerBlock.value)))

const sim = computed(() => {
  const tiers = [
    { cap: capGpu.value, bw: Infinity, lat: 0 }, { cap: capCpu.value, bw: 25, lat: 0.2 },
    { cap: capDisk.value, bw: diskBw.value, lat: diskLat.value },
  ]
  const lru = tiers.map(() => new Map())                                  // Map 的插入序 = LRU 序, 队首最旧
  const insert = (lv, h) => {
    if (lv >= lru.length) return
    lru[lv].delete(h); lru[lv].set(h, 1)
    while (lru[lv].size > tiers[lv].cap) { const old = lru[lv].keys().next().value; lru[lv].delete(old); insert(lv + 1, old) } // 溢出 → 降级
  }
  // 链式 hash 的替身: 前 4 块是共享 system prompt, 之后的块由 (用户, 块号) 唯一确定 (历史只追加)
  const ids = (u, len) => range(Math.floor(len / BS)).map((i) => (i < 4 ? `S${i}` : `u${u}:${i}`))
  const store = (hs) => [...hs].reverse().forEach((h) => { lru.slice(1).forEach((m) => m.delete(h)); insert(0, h) })
  const loadMs = (t, nb) => (nb === 0 || t.bw === Infinity ? 0 : t.lat + (nb * BLOCK_BYTES) / (t.bw * 1e9) * 1e3)

  const rand = mulberry32(seed.value + 11), msg = () => 40 + Math.floor(rand() * 60)
  const hist = Array(users.value).fill(64), reqs = []
  for (let turn = 1; turn <= 6; turn++) for (let u = 0; u < users.value; u++) {
    const len = (hist[u] += msg()), hs = ids(u, len)
    const where = []
    for (const h of hs) { const lv = lru.findIndex((m) => m.has(h)); if (lv < 0) break; where.push(lv) } // 首个 miss 处停
    const nb = tiers.map((_, lv) => where.filter((w) => w === lv).length)
    const miss = len - where.length * BS
    let ttft = RECOMP_MS(miss), skipped = 0
    tiers.forEach((t, lv) => {
      const load = loadMs(t, nb[lv]), re = RECOMP_MS(nb[lv] * BS)
      const useLoad = alwaysLoad.value || load <= re                      // ★ 逐层判断: 加载 vs 重算
      ttft += useLoad ? load : re
      if (!useLoad) skipped += nb[lv] * BS
    })
    store(hs)
    store(ids(u, (hist[u] += msg())))                                     // decode 生成的回复 KV 也登记进缓存
    let y0 = 0
    const hit = nb.map((n) => n * BS)
    const segs = [...hit, miss].map((n, k) => { const s = { k, n, y0 }; y0 += n; return s })
    reqs.push({ user: u, turn, len, hit, miss, ttft, skipped, segs })
  }
  const total = sum(reqs.map((r) => r.len))
  return {
    reqs, hit: [0, 1, 2].map((k) => sum(reqs.map((r) => r.hit[k])) / total), miss: sum(reqs.map((r) => r.miss)) / total,
    ttft: sum(reqs.map((r) => r.ttft)) / reqs.length, noCache: sum(reqs.map((r) => RECOMP_MS(r.len))) / reqs.length,
    maxLen: Math.max(...reqs.map((r) => r.len)), maxTtft: Math.max(...reqs.map((r) => r.ttft), 1),
  }
})
const bw = computed(() => W / sim.value.reqs.length)
const ky = computed(() => H / sim.value.maxLen)
const pickI = computed(() => Math.min(pick.value, sim.value.reqs.length - 1)) // 用户数变少时夹回合法范围
const cur = computed(() => sim.value.reqs[pickI.value])
const pct = (x) => (x * 100).toFixed(1) + '%'
</script>

<style scoped>
svg { min-width: 520px; } /* 窄屏: 图保持可读, 由 .lab-viz 横向滚动 */
.pick { fill: none; stroke: var(--text); stroke-width: 1.5; pointer-events: none; }
.ttft { fill: none; stroke: var(--text); stroke-width: 1.5; pointer-events: none; }
.ax { font-size: 11px; fill: var(--text-dim); }
.legend { display: flex; flex-wrap: wrap; gap: 12px; font-size: 11px; color: var(--text-muted); margin-top: 6px; }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 4px; }
.detail { margin-top: 10px; }
.detail b { color: var(--text); }
</style>
