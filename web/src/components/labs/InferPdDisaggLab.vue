<!-- P/D 分离实验台 (llm_infer/m15): 同卡混跑的 ITL 尖刺 vs 分离后要付的 KV 传输。全部时间来自代价模型。 -->
<template>
  <LabFrame
    title="P/D 分离 — 拿 KV 传输换掉 decode 卡顿"
    sub="几个用户在 decode, 这时来了一条长 prompt。同卡: 这一步被 prefill 占满, 所有人卡住。分离: P 节点算 prefill, KV 过网络发给 D 节点, D 节点的节奏不受影响。代价模型 (非实测): decode 每步 25 ms, prefill 8000 tok/s。"
    module="llm_infer/m15"
    run="python -m llm_infer.m15_pd_disaggregation.demo"
    :challenge="{
      ask: '链路延迟设为 0, 把 prompt 长度从 1K 拖到 16K —— 盈亏带宽怎么变? 为什么?',
      answer: '不变。KV 字节数 ∝ prompt 长度, prefill 耗时也 ∝ prompt 长度, 两者一除长度就消掉了: 盈亏带宽 = 每 token KV 字节 × prefill 速度 = 128 KiB × 8000 tok/s ≈ 1.05 GB/s ≈ 8.4 Gbps (注意 ×8)。所以搬 KV 几乎总比让别人等 prefill 划算; 真正要小心的是单位 —— 网卡标 Gbps (bit), KV 按 byte 算, 少除一个 8 就会把传输时间低估 8 倍。另外传输推迟的是新请求的第 2 个 token, 不是 TTFT: 首 token 在 P 节点上就产出了。',
    }"
  >
    <template #controls>
      <div class="row">
        <button v-for="(m, i) in MODELS" :key="m.name" type="button" :class="{ active: model === i }" @click="model = i">{{ m.name }} · {{ m.kib }} KiB/token</button>
      </div>
      <LabSlider v-model="kTok" label="prompt 长度" :min="1" :max="16" unit="K tok" />
      <LabSlider v-model="bwIdx" label="链路带宽" :min="0" :max="LINKS.length - 1" :format="() => `${gbps} Gbps`" />
      <div class="row">
        <span class="unit mono">= {{ gbps }} × 10⁹ ÷ 8 = <b>{{ (gbps / 8).toFixed(2) }} GB/s</b></span>
        <button v-for="p in PRESETS" :key="p.name" type="button" :class="{ active: gbps === p.gbps }" @click="bwIdx = LINKS.indexOf(p.gbps)">{{ p.name }}</button>
      </div>
      <LabSlider v-model="latency" label="链路固定延迟" :min="0" :max="20" unit=" ms" />
      <div class="row">
        <button type="button" :class="{ active: wrongUnit }" :aria-pressed="wrongUnit" @click="wrongUnit = !wrongUnit">把 Gbps 当成 GB/s (常见错误)</button>
      </div>
    </template>

    <svg class="tl" :viewBox="`0 0 ${W} 190`" role="img" aria-label="同卡与分离两种部署的时间线">
      <g v-for="(lane, li) in lanes" :key="lane.name">
        <text x="0" :y="lane.y + 15" class="lbl">{{ lane.name }}</text>
        <rect
          v-for="(s, i) in lane.segs" :key="i" :x="X0 + s.t0 * sx" :y="lane.y" :width="Math.max(1, (s.t1 - s.t0) * sx - 1)" height="22" rx="2"
          :class="['seg', s.kind, { hov: hov === `${li}-${i}` }]" tabindex="0" role="img" :aria-label="`${s.label} ${(s.t1 - s.t0).toFixed(1)} ms`"
          @mouseenter="hov = `${li}-${i}`" @focus="hov = `${li}-${i}`"
        />
      </g>
      <line :x1="X0 + arrive * sx" :x2="X0 + arrive * sx" y1="0" y2="168" class="arr" />
      <text :x="X0 + arrive * sx + 4" y="182" class="lbl">长 prompt 到达</text>
    </svg>
    <p class="lab-note hovline">{{ hovText }}</p>
    <p class="legend"><span class="sw dec" />decode 步 <span class="sw pre" />prefill <span class="sw xfer" />KV 传输 <span class="sw neo" />新请求 decode</p>

    <template #stats>
      <div class="kv"><span>同卡: 最大 ITL</span><b class="bad">{{ fmtMs(coloItl) }}</b></div>
      <div class="kv"><span>分离: 最大 ITL</span><b class="good">{{ fmtMs(STEP) }}</b></div>
      <div class="kv"><span>KV {{ kvGB.toFixed(2) }} GB · 传输</span><b>{{ fmtMs(xfer) }}</b></div>
      <div v-if="wrongUnit" class="kv"><span>错算 (忘了 ÷8)</span><b class="bad">{{ fmtMs(xferWrong) }}</b></div>
      <div class="kv"><span>新请求第 2 个 token 推迟</span><b :class="xfer < prefillMs ? 'good' : 'bad'">+{{ fmtMs(xfer) }}</b></div>
      <div class="kv"><span>盈亏带宽</span><b>{{ isFinite(breakEven) ? breakEven.toFixed(1) + ' Gbps' : '∞' }}</b></div>
      <p class="lab-note">
        盈亏带宽: 传输耗时 = 它换掉的那次 prefill 卡顿 ({{ fmtMs(prefillMs) }})。当前链路是它的 {{ (gbps / breakEven).toFixed(1) }}×。
        TTFT 两种部署相同 —— 首 token 在 P 节点产出; 传输只推迟第 2 个 token。
        <template v-if="wrongUnit">忘了 ÷8 会把传输部分低估 8 倍: {{ fmtMs(xferWrong) }} vs 真实 {{ fmtMs(xfer) }}。</template>
      </p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { range } from '@/utils/labmath.js'

const W = 720, X0 = 104, STEP = 25, PREFILL_TPS = 8000, WARM = 3, USERS_AFTER = 5
const MODELS = [{ name: 'LLaMA-3-8B (GQA)', kib: 128 }, { name: 'LLaMA-2-7B (MHA)', kib: 512 }]
const LINKS = [10, 25, 50, 100, 200, 400, 800, 3600]
const PRESETS = [{ name: '10 GbE', gbps: 10 }, { name: '100 GbE', gbps: 100 }, { name: '400G IB', gbps: 400 }, { name: 'NVLink ≈3600', gbps: 3600 }]
const fmtMs = (ms) => (ms >= 1000 ? (ms / 1000).toFixed(2) + ' s' : ms.toFixed(ms < 10 ? 1 : 0) + ' ms')

const model = ref(0)
const kTok = ref(4)
const bwIdx = ref(3)
const latency = ref(1)
const wrongUnit = ref(false)
const hov = ref('')

const gbps = computed(() => LINKS[bwIdx.value])
const kvBytes = computed(() => kTok.value * 1024 * MODELS[model.value].kib * 1024)
const kvGB = computed(() => kvBytes.value / 1e9)
const prefillMs = computed(() => (kTok.value * 1024 / PREFILL_TPS) * 1e3)
// ★ 单位: 带宽标的是 bit/s, KV 是 byte → bytes/s = Gbps · 1e9 / 8 (与 m15 KVLink.transfer_ms 相同)
const xfer = computed(() => latency.value + (kvBytes.value / (gbps.value * 1e9 / 8)) * 1e3)
const xferWrong = computed(() => latency.value + (kvBytes.value / (gbps.value * 1e9)) * 1e3)
const coloItl = computed(() => prefillMs.value + STEP) // 上一个 token → (整段 prefill) → 下一个 decode 步结束
const breakEven = computed(() => (prefillMs.value > latency.value ? (kvBytes.value * 8) / ((prefillMs.value - latency.value) / 1e3) / 1e9 : Infinity))

const arrive = WARM * STEP
const lanes = computed(() => {
  const dec = (t0, n, kind = 'dec') => range(n).map((i) => ({ t0: t0 + i * STEP, t1: t0 + (i + 1) * STEP, kind, label: kind === 'neo' ? '新请求 decode 步' : 'decode 步' }))
  const pEnd = arrive + prefillMs.value, xEnd = pEnd + xfer.value
  const nD = Math.ceil((xEnd + USERS_AFTER * STEP) / STEP)
  return [
    { name: '同卡 GPU', y: 6, segs: [...dec(0, WARM), { t0: arrive, t1: pEnd, kind: 'pre', label: 'prefill (所有 decode 停住)' }, ...dec(pEnd, USERS_AFTER)] },
    { name: '分离 · D 节点', y: 50, segs: dec(0, nD) },
    { name: '分离 · P 节点', y: 80, segs: [{ t0: arrive, t1: pEnd, kind: 'pre', label: 'prefill → 首 token' }] },
    { name: '分离 · 链路', y: 110, segs: [{ t0: pEnd, t1: xEnd, kind: 'xfer', label: `KV 传输 ${kvGB.value.toFixed(2)} GB` }] },
    { name: '新请求 @ D', y: 140, segs: dec(xEnd, 3, 'neo') },
  ]
})
// 悬停文字从当前 lanes 现算: 拖滑杆后不会留下过期的数字
const hovText = computed(() => {
  const [li, i] = hov.value.split('-').map(Number), lane = lanes.value[li], s = lane?.segs[i]
  return s ? `${lane.name} · ${s.label}: ${(s.t1 - s.t0).toFixed(1)} ms` : '悬停任一段看它的时长。'
})
const total = computed(() => Math.max(...lanes.value.flatMap((l) => l.segs.map((s) => s.t1))))
const sx = computed(() => (W - X0) / total.value)
</script>

<style scoped>
.tl { min-width: 560px; } /* 窄屏: 图保持可读, 由 .lab-viz 横向滚动 */
.seg { stroke: var(--bg-card); outline: none; }
.seg.hov, .seg:focus-visible { stroke: var(--text); stroke-width: 1.5; }
.dec { fill: color-mix(in srgb, var(--left) 60%, transparent); background: var(--left); }
.pre { fill: var(--warn); background: var(--warn); }
.xfer { fill: var(--danger); background: var(--danger); }
.neo { fill: var(--accent); background: var(--accent); }
.arr { stroke: var(--text-dim); stroke-dasharray: 3 3; }
.lbl { font-size: 12px; fill: var(--text-muted); }
.unit { font-size: 12px; color: var(--text-muted); }
.unit b { color: var(--accent); }
.hovline { margin-top: 8px; min-height: 1.7em; }
.legend { font-size: 11px; color: var(--text-dim); margin-top: 6px; }
.sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin: 0 4px 0 10px; }
</style>
