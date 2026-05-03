<template>
  <div class="inspector">
    <!-- Shape 参数条 -->
    <div class="shape-bar card">
      <div class="shape-bar-head">
        <h3 style="margin: 0;">张量形状参数 <span class="tag">所有数值随滑条实时变化</span></h3>
        <button class="reset" @click="resetCtx">重置为 LLaMA-7B 默认</button>
      </div>
      <div class="shape-grid">
        <div v-for="k in sliderKeys" :key="k" class="shape-slider">
          <label>{{ sliderMeta[k].label }}</label>
          <input type="range"
                 :min="sliderMeta[k].min"
                 :max="sliderMeta[k].max"
                 :step="sliderMeta[k].step"
                 v-model.number="ctx[k]" />
          <span class="val mono">{{ ctx[k] }}</span>
        </div>
      </div>
      <div class="derived mono">
        <span>D_h = D/H = <b>{{ ctx.D_h }}</b></span>
        <span v-if="showFF">d_ff ({{ ctx._ff_mode }}) = <b>{{ ctx.d_ff }}</b></span>
      </div>
    </div>

    <!-- Tab 切换 -->
    <div class="inspector-tabs">
      <button v-for="t in tabs" :key="t.key"
              :class="{ active: tab === t.key }"
              @click="$emit('update:tab', t.key)">
        <span class="tab-k">{{ t.label }}</span>
        <span class="tab-v mono">{{ displayName(t.key) }}</span>
      </button>
    </div>

    <!-- Detail -->
    <div class="inspector-body card" v-if="activeSpec">
      <div class="body-head">
        <div>
          <h3 style="margin-bottom: 2px;">{{ activeSpec.title }}</h3>
          <p class="desc">{{ activeSpec.subtitle }}</p>
        </div>
      </div>

      <div class="body-grid">
        <div class="params-col">
          <div class="col-label">权重参数</div>
          <ParamsTable
            :params="activeSpec.params"
            :ctx="ctx"
            :breakdown="activeSpec.paramBreakdown"
            :cache="activeSpec.cachePerToken || null" />
        </div>
        <div class="flow-col">
          <div class="col-label">计算流 · 每一步的张量形状</div>
          <FlowDiagram :steps="activeSpec.flow" :ctx="ctx" />
          <div class="legend">
            <span class="legend-dot" style="background:#60a5fa;"></span> matmul
            <span class="legend-dot" style="background:#3dd68c;"></span> activation
            <span class="legend-dot" style="background:#9ca3af;"></span> reshape
            <span class="legend-dot" style="background:#c084fc;"></span> attention
            <span class="legend-dot" style="background:#f5a623;"></span> 条件/RoPE
            <span class="legend-dot" style="background:#ec4899;"></span> route
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, reactive, watch } from 'vue'
import FlowDiagram from './FlowDiagram.vue'
import ParamsTable from './ParamsTable.vue'
import { attnSpecs, ffnSpecs, normSpecs, posSpecs } from '@/data/inspector.js'

const props = defineProps({
  config: { type: Object, required: true }, // { attn, ffn, norm, pos }
  tab:    { type: String, default: 'attn' },
})
const emit = defineEmits(['update:tab'])

// --- shape context (所有 flow 共用) ---
// 所有字段都放在初始对象里, 确保 Vue 反应式追踪正常工作;
// D_h 与 d_ff 是派生量, 由 watch 同步更新, 不用 getter。
const defaultCtx = () => ({
  B: 2,
  T: 32,
  D: 4096,
  H: 32,
  D_h: 128,     // = D / H
  H_kv: 8,
  H_idx: 4,
  D_h_idx: 32,
  r_kv: 512,
  r_q: 1536,
  D_nope: 64,
  D_rope: 64,
  D_t: 16, D_h_ax: 56, D_w: 56,   // M-RoPE 三段 (合计 = D_h)
  E: 8,
  S: 2,
  K: 2,
  N: 16,        // Mamba 状态维
  max_len: 4096,
  c_dim: 4096,
  d_ff: 11008,  // SwiGLU 默认 (8/3 · 4096 对齐到 64)
  _ff_mode: 'SwiGLU 2/3 · 对齐到 64',
})

const ctx = reactive(defaultCtx())
const showFF = computed(() => true)

// 派生: D_h = floor(D / H)
watch(() => [ctx.D, ctx.H], () => {
  ctx.D_h = Math.max(1, Math.floor(ctx.D / ctx.H))
}, { immediate: true })

// 派生: d_ff 随 FFN variant 变化 (SwiGLU / MoE 用 8/3, 其它用 4·D)
watch(() => [ctx.D, props.config.ffn], () => {
  const f = props.config.ffn
  if (f === 'swiglu' || f === 'moe_mx' || f === 'moe_ds') {
    ctx.d_ff = Math.round(ctx.D * 8 / 3 / 64) * 64
    ctx._ff_mode = 'SwiGLU 2/3 · 对齐到 64'
  } else {
    ctx.d_ff = ctx.D * 4
    ctx._ff_mode = 'classic 4·D'
  }
}, { immediate: true })

// 派生: H_kv 必须 ≤ H 且能整除 H
watch(() => ctx.H, () => {
  if (ctx.H_kv > ctx.H) ctx.H_kv = ctx.H
  while (ctx.H % ctx.H_kv !== 0 && ctx.H_kv > 1) ctx.H_kv--
})

// 派生: M-RoPE 三段之和必须等于 D_h
watch(() => ctx.D_h, (D_h) => {
  // 按 ~1/4, ~3/8, ~3/8 分 (保证为偶数)
  const third = Math.max(2, Math.floor(D_h / 3 / 2) * 2)
  ctx.D_t = D_h - 2 * third
  ctx.D_h_ax = third
  ctx.D_w = third
}, { immediate: true })

function resetCtx() {
  Object.assign(ctx, defaultCtx())
}

// --- 参数滑条定义 ---
const sliderMeta = {
  B:      { label: 'B · batch',        min: 1,   max: 16,   step: 1 },
  T:      { label: 'T · seq_len',      min: 16,  max: 4096, step: 16 },
  D:      { label: 'D · d_model',      min: 512, max: 8192, step: 128 },
  H:      { label: 'H · n_heads',      min: 4,   max: 64,   step: 2 },
  H_kv:   { label: 'H_kv · kv_heads',  min: 1,   max: 64,   step: 1 },
  r_kv:   { label: 'r_kv · MLA latent',min: 64,  max: 1024, step: 32 },
  D_nope: { label: 'D_nope (MLA)',     min: 16,  max: 128,  step: 8 },
  D_rope: { label: 'D_rope (MLA)',     min: 16,  max: 128,  step: 8 },
  E:      { label: 'E · 路由专家',      min: 2,   max: 64,   step: 1 },
  K:      { label: 'K · top-k',        min: 1,   max: 8,    step: 1 },
  S:      { label: 'S · 共享专家',      min: 0,   max: 4,    step: 1 },
  N:      { label: 'N · SSM 状态',     min: 8,   max: 64,   step: 4 },
  c_dim:  { label: 'c_dim · 条件向量',  min: 128, max: 2048, step: 64 },
}
const sliderKeys = computed(() => {
  // 根据当前 tab 只展示相关滑条, 减轻视觉负担
  const base = ['B', 'T', 'D', 'H']
  if (props.tab === 'attn') {
    const a = props.config.attn
    if (a === 'gqa') return [...base, 'H_kv']
    if (a === 'mla' || a === 'dsa') return [...base, 'r_kv', 'D_nope', 'D_rope']
    if (a === 'ssm') return ['B', 'T', 'D', 'N']
    return base
  }
  if (props.tab === 'ffn') {
    const f = props.config.ffn
    if (f.startsWith('moe')) return [...base, 'E', 'K', ...(f === 'moe_ds' ? ['S'] : [])]
    return ['B', 'T', 'D']
  }
  if (props.tab === 'norm') {
    if (props.config.norm === 'ada_ln') return ['B', 'T', 'D', 'c_dim']
    return ['B', 'T', 'D']
  }
  if (props.tab === 'pos') return ['B', 'T', 'D', 'H']
  return base
})

// --- 当前选中的 spec ---
const tabs = [
  { key: 'attn', label: '注意力' },
  { key: 'ffn',  label: 'FFN' },
  { key: 'norm', label: '归一化' },
  { key: 'pos',  label: '位置编码' },
]

const displayName = (tab) => {
  const map = { attn: attnSpecs, ffn: ffnSpecs, norm: normSpecs, pos: posSpecs }
  const id = props.config[tab]
  return map[tab][id]?.title?.split(' (')[0] || id
}

const activeSpec = computed(() => {
  const map = { attn: attnSpecs, ffn: ffnSpecs, norm: normSpecs, pos: posSpecs }
  return map[props.tab][props.config[props.tab]]
})
</script>

<style scoped>
.inspector {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.shape-bar-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 10px;
}
.shape-bar-head .reset {
  font-size: 11px;
  padding: 4px 10px;
}

.shape-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 10px 18px;
}
.shape-slider {
  display: grid;
  grid-template-columns: 140px 1fr 60px;
  gap: 10px;
  align-items: center;
}
.shape-slider label { font-size: 12px; color: var(--text-muted); }
.shape-slider .val  {
  font-size: 12px; color: var(--text);
  text-align: right; font-family: 'SF Mono', Menlo, monospace;
}

.derived {
  margin-top: 10px;
  padding: 8px 12px;
  background: var(--bg-elev);
  border-radius: 5px;
  display: flex;
  gap: 24px;
  font-size: 11.5px;
  color: var(--text-muted);
  border-left: 2px solid var(--accent);
}
.derived b { color: var(--accent); font-weight: 600; }

.inspector-tabs {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
}
.inspector-tabs button {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 2px;
  padding: 8px 14px;
  min-width: 140px;
  text-align: left;
}
.inspector-tabs .tab-k { font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.6px; }
.inspector-tabs .tab-v { font-size: 12px; color: var(--text); }
.inspector-tabs button.active .tab-k { color: rgba(255,255,255,0.7); }
.inspector-tabs button.active .tab-v { color: #fff; }

.body-head { margin-bottom: 16px; }
.body-grid {
  display: grid;
  grid-template-columns: minmax(280px, 38%) 1fr;
  gap: 20px;
}
@media (max-width: 1000px) {
  .body-grid { grid-template-columns: 1fr; }
}
.col-label {
  font-size: 11px;
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: 0.7px;
  margin-bottom: 10px;
}

.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  margin-top: 14px;
  padding: 8px 10px;
  background: var(--bg-elev);
  border-radius: 5px;
  font-size: 11.5px;
  color: var(--text-muted);
}
.legend-dot {
  display: inline-block;
  width: 8px; height: 8px;
  border-radius: 2px;
  margin-right: 4px;
  vertical-align: middle;
}
</style>
