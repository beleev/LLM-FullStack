<!--
  SFT 标签对齐实验台 (对应 llm_finetune/data/instruction_data.py)。
  只讲一件事: labels 已经错位一格 (labels = x[:, 1:]) 之后, prompt mask 的边界是 P-1 而不是 P。
  这是本仓库真实出现过的 off-by-one: 写成 labels[:, :P] 会把"第一个 response token"也 mask 掉。
-->
<template>
  <LabFrame
    title="SFT 标签对齐 — mask 边界差一格会丢掉什么"
    sub="上排是模型输入 idx = x[:-1], 下排是已经错位一格的 labels = x[1:]: 每一列 = '看到上面这个 token, 要预测下面这个 token'。点击下排任意格子移动 mask 边界, 看哪些位置真正进入 loss。"
    module="llm_finetune/data"
    run="python -m llm_finetune.run_finetune.sft.train_sft"
    :challenge="{
      ask: '把 response 长度拖到 1 (单 token 答案, 例如分类 / 选择题), 再点「错误写法 [:P]」。还剩几个 token 在教模型? loss 会是多少?',
      answer: '0 个。labels 已经左移一格, 位置 P-1 的输入是最后一个 prompt token (答:), 它的 label 恰好是第一个 response token。[:P] 把它也 mask 了 —— 单 token 答案时全部标签都是 -100, cross_entropy 对空集求平均得到 nan。答案较长时 bug 更隐蔽: loss 照常下降, 但模型从没学过「回答该怎么开头」, 而开头恰恰是决定整条回复走向的 token。正确写法是 labels[:, :P-1] = -100。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: k === P }" @click="k = P">错误写法 labels[:, :P] = -100</button>
        <button type="button" :class="{ active: k === P - 1 }" @click="k = P - 1">正确写法 labels[:, :P-1] = -100</button>
        <button type="button" :class="{ active: k === 0 }" @click="k = 0">不 mask (预训练式)</button>
      </div>
      <LabSlider v-model="P" label="prompt 长度 P" :min="3" :max="7" />
      <LabSlider v-model="R" label="response 长度" :min="1" :max="4" />
    </template>

    <div class="cells grid" :style="{ gridTemplateColumns: `64px repeat(${T}, 52px)` }" @mouseleave="hover = -1">
      <span class="rl">位置 t</span>
      <span v-for="t in T" :key="'p' + t" class="pos mono" :class="{ hl: hover === t - 1 }">{{ t - 1 }}</span>

      <span class="rl">输入 idx</span>
      <span
        v-for="(tok, t) in idx" :key="'i' + t"
        class="cell tok" :class="[t < P ? 'prompt' : 'resp', { hl: hover === t }]"
        @mouseenter="hover = t"
      >{{ tok }}</span>

      <span class="rl">预测 ↓</span>
      <span v-for="t in T" :key="'a' + t" class="arrow" :class="{ hl: hover === t - 1, off: t - 1 < k }">↓</span>

      <span class="rl">labels</span>
      <button
        v-for="(c, t) in cols" :key="'l' + t" type="button"
        class="cell tok lbl" :class="[c.cls, { hl: hover === t }]"
        :aria-label="`位置 ${t} 的 label ${c.tok}, 点击移动 mask 边界`"
        @mouseenter="hover = t" @focus="hover = t" @click="k = c.masked ? t : t + 1"
      >{{ c.masked ? '-100' : c.tok }}</button>

      <span class="rl">进 loss?</span>
      <span v-for="(c, t) in cols" :key="'s' + t" class="mark" :class="c.cls">{{ c.mark }}</span>
    </div>
    <p class="lab-note" style="margin-top: 12px;">
      {{ hover >= 0 ? hoverText : '悬停任意一列: 看这个位置的输入、目标, 以及它为什么该 / 不该进 loss。' }}
    </p>

    <template #stats>
      <div class="kv"><span>当前 mask</span><b>[:{{ kLabel }}]</b></div>
      <div class="kv"><span>进入 loss 的位置数</span><b>{{ stat.inLoss }}</b></div>
      <div class="kv"><span>response 被监督</span><b :class="stat.resp === R ? 'good' : 'bad'">{{ stat.resp }} / {{ R }}</b></div>
      <div class="kv"><span>prompt 混进 loss</span><b :class="stat.leak ? 'bad' : 'good'">{{ stat.leak }}</b></div>
      <div class="kv">
        <span>首个回答 token「{{ resp[0] }}」</span>
        <b :class="stat.first ? 'good' : 'bad'">{{ stat.first ? '学到了' : '被 mask' }}</b>
      </div>
      <p class="lab-note">
        <template v-if="stat.inLoss === 0">全部标签都是 -100: cross_entropy 对空集取平均 → <b>nan</b>。</template>
        <template v-else>
          ★ 边界是 P−1 = {{ P - 1 }}: 因为 labels[t] = x[t+1], 第一个 response token x[P] 落在 labels 的第 P−1 格。
        </template>
      </p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'

const PROMPT = ['<s>', '问:', '法国', '的', '首都', '?', '答:']
const RESP = { 1: ['巴黎'], 2: ['巴黎', '</s>'], 3: ['巴黎', '。', '</s>'], 4: ['巴黎', '市', '。', '</s>'] }

const P = ref(6)
const R = ref(3)
const k = ref(6)          // mask 写成 labels[:, :k] = -100; 默认就是仓库里那个 bug
const hover = ref(-1)

// prompt 永远以 <s> 开头、以「答:」结尾, 中间按 P 截取
const prompt = computed(() => ['<s>', ...PROMPT.slice(PROMPT.length - (P.value - 1))])
const resp = computed(() => RESP[R.value])
const x = computed(() => [...prompt.value, ...resp.value])   // 完整序列, 长 P+R
const T = computed(() => x.value.length - 1)
const idx = computed(() => x.value.slice(0, -1))             // 输入: 去掉最后一个
const labels = computed(() => x.value.slice(1))              // ★ 标签: 整体左移一格, labels[t] = x[t+1]

// P 变了以后, 让 mask 跟着保持"同一种写法" (P / P-1 / 0)
watch(P, (p, old) => { k.value = k.value === old ? p : k.value === old - 1 ? p - 1 : Math.min(k.value, T.value) })
watch(T, (t) => { if (k.value > t) k.value = t })

const cols = computed(() => labels.value.map((tok, t) => {
  const isResp = t + 1 >= P.value            // labels[t] = x[t+1] 是不是 response token
  const masked = t < k.value
  const cls = masked ? (isResp ? 'bad' : 'dim') : (isResp ? 'ok' : 'hot')
  const mark = masked ? (isResp ? '✗ 丢了' : '—') : (isResp ? '✓' : '⚠ 泄漏')
  return { tok, isResp, masked, cls, mark }
}))

const stat = computed(() => {
  const live = cols.value.filter((c) => !c.masked)
  return {
    inLoss: live.length,
    resp: live.filter((c) => c.isResp).length,
    leak: live.filter((c) => !c.isResp).length,
    first: !cols.value[P.value - 1].masked,
  }
})
const kLabel = computed(() => (k.value === P.value ? 'P' : k.value === P.value - 1 ? 'P-1' : String(k.value)))

const hoverText = computed(() => {
  const t = hover.value, c = cols.value[t]
  if (!c) return ''
  const head = `位置 ${t}: 输入「${idx.value[t]}」→ 目标「${c.tok}」。`
  if (c.isResp && c.masked) return head + '目标是 response token 却被 mask —— 这部分监督信号白白丢掉。'
  if (c.isResp) return head + '目标是 response token, 进入 loss, 正是 SFT 想教的东西。'
  if (c.masked) return head + '目标还是 prompt 的一部分, mask 掉是对的: 不教模型复述用户的问题。'
  return head + '目标是 prompt token 却进了 loss: 模型在学"怎么提问", 稀释了回答的梯度。'
})
</script>

<style scoped>
.grid { align-items: center; row-gap: 4px; }
.rl { font-size: 11px; color: var(--text-dim); }
.pos { font-size: 10px; color: var(--text-dim); text-align: center; }
.tok { height: 30px; font-size: 12px; min-width: 0; }
.tok.prompt { color: var(--text-muted); }
.tok.resp { border-color: var(--left); color: var(--text); }
.lbl { cursor: pointer; padding: 0; min-height: 0; }
.lbl.dim { opacity: 0.45; }
.arrow { text-align: center; color: var(--accent); font-size: 12px; }
.arrow.off { color: var(--text-dim); opacity: 0.4; }
.mark { text-align: center; font-size: 10px; color: var(--text-dim); white-space: nowrap; }
.mark.ok { color: var(--left); }
.mark.bad { color: var(--danger); }
.mark.hot { color: var(--warn); }
.hl { outline: 2px solid var(--accent); outline-offset: 1px; border-radius: 3px; }
</style>
