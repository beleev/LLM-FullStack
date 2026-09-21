<!--
  滑窗 / 全注意力交替实验台 (对应 llm_models/models/moe/gpt_oss.py:GPTOSSMini.is_swa; mask 见 utils/masks.py:build_sliding_window_mask)。
  只讲一件事: 单层滑窗只看 W 个 token, 但信息可以跨层接力; 隔一层插一层全注意力, 感受野立刻铺满, KV cache 却只多一半。
-->
<template>
  <LabFrame
    title="滑窗 × 全注意力交替 — 信息怎么一层层传过来"
    sub="最上面一行是最后一层的输出。点其中一个 token: 往下每一行高亮 “它间接用到了哪些位置”, 最底一行就是它在输入上的感受野。切换层排布、拖窗口 W 和层数, 对比感受野和 KV cache 总量。"
    module="llm_models/models/moe/gpt_oss.py"
    run="python -m llm_models.run_models.moe.gpt_oss.infer_gpt_oss"
    :challenge="{
      ask: 'T=40, W=8, 4 层 (和 infer_gpt_oss 的 mini 模型同设定)。先猜: 全滑窗时最后一个 token 能看到多少个输入? 换成 “滑窗/全 交替” 呢? 两者的 KV cache 差多少?',
      answer: '全滑窗: 每层往回多看 W−1 = 7 个, 4 层共 4×7+1 = 29 个, 最前面 11 个 token 完全看不到 —— 想覆盖 40 个 token 需要 ⌈39/7⌉ = 6 层, 而且越远的信息被转手的次数越多、越糊。交替排布里只要有一层全注意力, 感受野立刻是全部 40 个。KV: 全滑窗每层只留 W=8 条, 共 32; 交替是各层 [8, 40, 8, 40] = 96; 全注意力是 4×40 = 160 —— 交替比全注意力省 40%, 与 Python demo 打印的一致 (而且裁掉窗口外的旧 K/V 不改变输出: 它们本来就被 mask 成 −inf)。真实尺度下差距更大: GPT-OSS 的 W=128, 上下文 131K, 滑窗层的 KV 几乎可以忽略, 总 cache ≈ 全注意力模型的一半。',
    }"
  >
    <template #controls>
      <div class="row">
        <button v-for="p in PATTERNS" :key="p.id" type="button" :class="{ active: pattern === p.id }" @click="pattern = p.id">{{ p.label }}</button>
      </div>
      <LabSlider v-model="W" label="滑动窗口 W" :min="2" :max="8" />
      <LabSlider v-model="L" label="层数" :min="1" :max="8" />
    </template>

    <div class="cells grid" :style="{ gridTemplateColumns: `74px repeat(${T}, 14px)` }">
      <template v-for="(row, r) in rows" :key="r">
        <span class="rl mono" :class="row.kind">{{ row.label }}</span>
        <template v-for="(on, i) in row.reach" :key="i">
          <button v-if="r === 0" type="button" class="cell pick" :class="{ on: i === q }" :aria-label="`选择 token ${i}`" :aria-pressed="i === q" @click="q = i" />
          <span v-else class="cell" :class="{ [row.kind === 'full' ? 'hot' : 'ok']: on, dim: !on && i <= q, fut: i > q }" />
        </template>
      </template>
    </div>
    <p class="cap">绿 = 经滑窗层传来 · 橙 = 经全注意力层传来 · 最底一行 = 输入 token 上的感受野</p>

    <template #stats>
      <div class="kv"><span>token {{ q }} 的感受野</span><b :class="field === q + 1 ? 'good' : 'bad'">{{ field }} / {{ q + 1 }}</b></div>
      <div class="kv"><span>KV cache 条目 (所有层合计)</span><b>{{ kv }}</b></div>
      <div class="kv"><span>相对全注意力 {{ L }}×{{ T }}</span><b :class="kv < L * T ? 'good' : ''">{{ ((kv / (L * T)) * 100).toFixed(0) }}%</b></div>
      <div class="kv"><span>纯滑窗看全 {{ q + 1 }} 个需要的层数</span><b>{{ needLayers }}</b></div>
      <p class="lab-note">滑窗层的 KV 是滚动缓冲, 上限 W, 与上下文长度无关; 全注意力层的 KV 随长度线性增长。交替 = 用一半层的代价买回全部感受野。</p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { range, sum } from '@/utils/labmath.js'

const T = 40
const PATTERNS = [
  { id: 'swa', label: '全滑窗 (Mistral)' },
  { id: 'alt', label: '滑窗 / 全 交替 (GPT-OSS)' },
  { id: 'full', label: '全注意力 (LLaMA)' },
]
const pattern = ref('alt'), W = ref(8), L = ref(4), q = ref(T - 1)

// 第 l 层 (0 起) 的类型: 交替排布里偶数层滑窗、奇数层全注意力
const kinds = computed(() => range(L.value).map((l) => (pattern.value === 'full' || (pattern.value === 'alt' && l % 2 === 1) ? 'full' : 'swa')))

const rows = computed(() => {
  let reach = range(T).map((i) => i === q.value)
  const out = [{ label: '输出 (点选)', kind: 'pick', reach }]
  for (let l = L.value - 1; l >= 0; l--) {
    const kind = kinds.value[l], next = Array(T).fill(false)
    // ★ 位置 i 在这一层能看到: 滑窗 [i−W+1, i], 全注意力 [0, i]
    reach.forEach((on, i) => { if (on) for (let j = kind === 'full' ? 0 : Math.max(0, i - W.value + 1); j <= i; j++) next[j] = true })
    reach = next
    out.push({ label: `L${l} · ${kind === 'full' ? '全' : '滑窗'}${l === 0 ? ' → 输入' : ''}`, kind, reach })
  }
  return out
})
const field = computed(() => rows.value[rows.value.length - 1].reach.filter(Boolean).length)
const kv = computed(() => sum(kinds.value.map((k) => (k === 'full' ? T : Math.min(T, W.value)))))
const needLayers = computed(() => Math.max(1, Math.ceil(q.value / (W.value - 1))))
</script>

<style scoped>
.grid { gap: 2px; min-width: 720px; }
.cell { min-width: 0; width: 14px; height: 18px; }
.cell.fut { opacity: 0.12; }
.pick { padding: 0; min-height: 0; cursor: pointer; border-radius: 3px; }
.pick.on { background: var(--accent); border-color: var(--accent); }
.rl { font-size: 10px; color: var(--text-dim); align-self: center; white-space: nowrap; }
.rl.full { color: var(--warn); } .rl.swa { color: var(--left); }
.cap { font-size: 11px; color: var(--text-dim); margin-top: 8px; }
</style>
