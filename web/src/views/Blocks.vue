<template>
  <div>
    <h1 class="page-title">Block 组装器</h1>
    <p class="page-subtitle">
      本库的核心设计洞察: <strong>模型差异集中在"构造参数"而非"新类"</strong>。
      从下面的零件菜单里各选一个, 看看你拼出了哪个模型。对应
      <RepoLink path="llm_models/layers/core/blocks.py:PreLNBlock" label="llm_models/layers/core/blocks.py::PreLNBlock(attn, ffn, norm_cls)" tiny />。
    </p>

    <ChapterIntro
      tldr="LLaMA、Mixtral、DeepSeek-V3 不是「新模型」, 是同一份 PreLNBlock 用不同的 attn/ffn/norm/pos 实例化出来的结果。"
      question="如果模型间的差异只是 4 个构造参数, 那「读 N 个模型源码」是不是变成了「读 N 张零件配置表」?"
      :goals="[
        '把不同模型抽象成 PreLNBlock 的 4 个零件配置',
        '理解 Pre-LN vs Post-LN 在训练稳定性上的差异',
        '能从 LLaMA / Mixtral / DeepSeek 中读出零件替换的位置',
      ]"
      :codes="[
        { path: 'llm_models/layers/core/blocks.py', label: 'blocks.py · PreLNBlock' },
      ]"
      :prereq="{ name: 'position', label: '位置编码 & RoPE' }"
      :next-step="{ name: 'moe', label: 'MoE 路由 — 把 ffn 槽位换成稀疏专家' }"
    />

    <div class="grid" style="grid-template-columns: 260px 1fr 300px; gap: 20px;">
      <!-- 零件选择 -->
      <div class="card">
        <h3>零件库 <span class="tag">4 个槽位</span></h3>
        <div v-for="slot in slots" :key="slot.key" class="slot-section">
          <div class="slot-title">{{ slot.label }}</div>
          <div class="slot-options">
            <button v-for="opt in slot.options" :key="opt.id"
                    :class="{ active: config[slot.key] === opt.id }"
                    @click="config[slot.key] = opt.id">
              {{ opt.name }}
            </button>
          </div>
        </div>
      </div>

      <!-- 中: Block 数据流可视化 -->
      <div class="card">
        <h3>数据流 · Pre-LN Block <span class="tag">{{ modelMatch.name }}</span></h3>
        <p class="desc" style="margin-bottom: 20px;">
          Pre-LN 把归一化从残差主路径移到子层分支内, 是训练稳定的关键 (Xiong et al., 2020)。
        </p>

        <svg viewBox="0 0 440 420" width="100%" :height="420">
          <defs>
            <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5"
                    markerWidth="6" markerHeight="6" orient="auto">
              <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--text-muted)" />
            </marker>
            <marker id="arrow-residual" viewBox="0 0 10 10" refX="9" refY="5"
                    markerWidth="6" markerHeight="6" orient="auto">
              <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--border-strong)" />
            </marker>
          </defs>

          <!-- ===== 输入 x ===== -->
          <rect x="180" y="10" width="80" height="30" rx="5"
                fill="var(--bg-elev)" stroke="var(--border-strong)" />
          <text x="220" y="29" text-anchor="middle" fill="var(--text)"
                font-size="13" font-family="SF Mono">x (in)</text>

          <!-- 分叉节点: x 同时进入 norm1 和 residual -->
          <circle cx="220" cy="52" r="3" fill="var(--text-muted)" />
          <line x1="220" y1="40" x2="220" y2="52"
                stroke="var(--text-muted)" stroke-width="1.5" />

          <!-- ===== 第一个子层: norm1 → attn1 ===== -->
          <!-- x 分叉 → norm1 (左下弯) -->
          <path d="M 220 52 C 220 70, 120 70, 120 80"
                fill="none" stroke="var(--text-muted)" stroke-width="1.5"
                marker-end="url(#arrow)" />

          <rect x="40" y="80" width="160" height="38" rx="5"
                :fill="partColor('norm')" fill-opacity="0.18"
                :stroke="partColor('norm')" stroke-width="1.5" />
          <text x="120" y="103" text-anchor="middle" fill="var(--text)"
                font-size="12" font-family="SF Mono">{{ partName('norm') }}</text>

          <!-- norm1 → attn1 (水平) -->
          <line x1="200" y1="99" x2="240" y2="99"
                stroke="var(--text-muted)" stroke-width="1.5" marker-end="url(#arrow)" />

          <rect x="240" y="80" width="160" height="38" rx="5"
                :fill="partColor('attn')" fill-opacity="0.18"
                :stroke="partColor('attn')" stroke-width="1.5" />
          <text x="320" y="98" text-anchor="middle" fill="var(--text)"
                font-size="12" font-family="SF Mono">{{ partName('attn') }}</text>
          <text x="320" y="112" text-anchor="middle" fill="var(--text-dim)" font-size="10">{{ partNote('attn') }}</text>

          <!-- RoPE 提示 (仅 rope / mrope 时显示) -->
          <text v-if="config.pos === 'rope' || config.pos === 'mrope'"
                x="320" y="135" text-anchor="middle"
                fill="var(--accent)" font-size="10" font-family="SF Mono">
            + {{ partName('pos') }} 注入 Q/K
          </text>

          <!-- attn1 → add1 (右下弯回中线) -->
          <path d="M 320 118 C 320 150, 232 150, 232 175"
                fill="none" stroke="var(--text-muted)" stroke-width="1.5"
                marker-end="url(#arrow)" />

          <!-- residual1: x 分叉右侧 → add1 左侧 (虚线, 右侧绕过) -->
          <path d="M 220 52 C 370 52, 370 175, 208 175"
                fill="none" stroke="var(--border-strong)" stroke-width="1.5"
                stroke-dasharray="4 3" marker-end="url(#arrow-residual)" />
          <text x="386" y="118" text-anchor="middle"
                fill="var(--text-dim)" font-size="10" font-family="SF Mono">residual</text>

          <!-- Add1 -->
          <circle cx="220" cy="175" r="13" fill="var(--bg-elev)" stroke="var(--border-strong)" stroke-width="1.5" />
          <text x="220" y="179" text-anchor="middle" fill="var(--text)" font-size="15">+</text>

          <!-- Add1 → 分叉节点 (下一子层) -->
          <line x1="220" y1="188" x2="220" y2="207"
                stroke="var(--text-muted)" stroke-width="1.5" />
          <circle cx="220" cy="207" r="3" fill="var(--text-muted)" />

          <!-- ===== 第二个子层: norm2 → ffn ===== -->
          <!-- 分叉 → norm2 -->
          <path d="M 220 207 C 220 225, 120 225, 120 235"
                fill="none" stroke="var(--text-muted)" stroke-width="1.5"
                marker-end="url(#arrow)" />

          <rect x="40" y="235" width="160" height="38" rx="5"
                :fill="partColor('norm')" fill-opacity="0.18"
                :stroke="partColor('norm')" stroke-width="1.5" />
          <text x="120" y="258" text-anchor="middle" fill="var(--text)"
                font-size="12" font-family="SF Mono">{{ partName('norm') }}</text>

          <line x1="200" y1="254" x2="240" y2="254"
                stroke="var(--text-muted)" stroke-width="1.5" marker-end="url(#arrow)" />

          <rect x="240" y="235" width="160" height="38" rx="5"
                :fill="partColor('ffn')" fill-opacity="0.18"
                :stroke="partColor('ffn')" stroke-width="1.5" />
          <text x="320" y="253" text-anchor="middle" fill="var(--text)"
                font-size="12" font-family="SF Mono">{{ partName('ffn') }}</text>
          <text x="320" y="267" text-anchor="middle" fill="var(--text-dim)" font-size="10">{{ partNote('ffn') }}</text>

          <!-- ffn → add2 -->
          <path d="M 320 273 C 320 305, 232 305, 232 330"
                fill="none" stroke="var(--text-muted)" stroke-width="1.5"
                marker-end="url(#arrow)" />

          <!-- residual2: add1-后 → add2 (虚线, 右侧) -->
          <path d="M 220 207 C 370 207, 370 330, 208 330"
                fill="none" stroke="var(--border-strong)" stroke-width="1.5"
                stroke-dasharray="4 3" marker-end="url(#arrow-residual)" />
          <text x="386" y="273" text-anchor="middle"
                fill="var(--text-dim)" font-size="10" font-family="SF Mono">residual</text>

          <!-- Add2 -->
          <circle cx="220" cy="330" r="13" fill="var(--bg-elev)" stroke="var(--border-strong)" stroke-width="1.5" />
          <text x="220" y="334" text-anchor="middle" fill="var(--text)" font-size="15">+</text>

          <!-- Add2 → x(out) -->
          <line x1="220" y1="343" x2="220" y2="368"
                stroke="var(--text-muted)" stroke-width="1.5" marker-end="url(#arrow)" />

          <!-- 输出 -->
          <rect x="180" y="368" width="80" height="28" rx="5"
                fill="var(--bg-elev)" stroke="var(--border)" />
          <text x="220" y="386" text-anchor="middle" fill="var(--text-muted)"
                font-size="12" font-family="SF Mono">x (out)</text>
        </svg>
      </div>

      <!-- 右: 匹配哪个模型? -->
      <div>
        <div class="card match-card" :style="{ borderColor: modelMatch.color }">
          <h3>
            <span :class="['pill', modelMatch.track ? tracks[modelMatch.track].cls : '']">{{ modelMatch.year || '?' }}</span>
            <span>匹配模型</span>
          </h3>
          <div class="match-name" :style="{ color: modelMatch.color }">{{ modelMatch.name }}</div>
          <p class="desc">{{ modelMatch.blurb }}</p>
          <pre class="code" style="margin-top: 12px;">{{ generatedCode }}</pre>
        </div>

        <div class="card" style="margin-top: 16px;">
          <h3>这一组合的总参数量影响</h3>
          <div class="stat" style="margin-top: 4px;">
            <div class="k">FFN 参数</div>
            <div class="v">{{ ffnParams }}</div>
            <div class="hint">{{ ffnNote }}</div>
          </div>
          <div class="stat" style="margin-top: 8px;">
            <div class="k">KV 每层 / token</div>
            <div class="v">{{ kvNote }}</div>
            <div class="hint">d_model=4096, n_heads=32 下</div>
          </div>
        </div>
      </div>
    </div>

    <!-- 快速切换到真实模型 -->
    <section class="section">
      <h2>一键载入预设</h2>
      <p class="lead">点击下面任一模型, 自动填充其零件配置:</p>
      <div class="btn-group">
        <button v-for="preset in presets" :key="preset.name"
                @click="loadPreset(preset)">
          {{ preset.name }} <span style="color: var(--text-dim); font-size: 10px; margin-left: 4px;">{{ preset.year }}</span>
        </button>
      </div>
    </section>

    <!-- ↓↓↓ 矩阵变换详情 ↓↓↓ -->
    <section class="section">
      <h2>内部矩阵变换</h2>
      <p class="lead">
        展开所选零件的"计算流 + 权重矩阵"。每一步都标注张量的符号形状与按当前参数计算出的数值;
        拖动形状滑条, 所有数字会同步刷新。源代码:
        <RepoLink path="llm_models/layers/core/{attention,feedforward,normalization,position_encoding}.py" label="llm_models/layers/core/{attention,feedforward,normalization,position_encoding}.py" tiny />。
      </p>
      <InspectorPanel :config="config" :tab="inspectorTab" @update:tab="inspectorTab = $event" />
    </section>

    <ChapterNav
      :prev="{ name: 'position', label: '位置编码 & RoPE', hint: '本章 pos 槽位的所有候选项的来历' }"
      :next="{ name: 'moe', label: 'MoE 路由', hint: '把 ffn 槽位拆成多专家 — Mixtral / DeepSeek 的两条路' }"
    />
  </div>
</template>

<script setup>
import { ref, reactive, computed } from 'vue'
import { tracks } from '@/data/models.js'
import InspectorPanel from '@/components/InspectorPanel.vue'
import ChapterIntro from '@/components/ChapterIntro.vue'
import ChapterNav from '@/components/ChapterNav.vue'
import RepoLink from '@/components/RepoLink.vue'

const inspectorTab = ref('attn')

const slots = [
  { key: 'attn', label: '注意力 (attn)', options: [
    { id: 'mha', name: 'MHA',  note: '原始, cache 最大' },
    { id: 'gqa', name: 'GQA',  note: 'KV 头 ÷ 4' },
    { id: 'mla', name: 'MLA',  note: 'latent 低秩' },
    { id: 'dsa', name: 'DSA',  note: 'MLA + 稀疏 top-k' },
    { id: 'ssm', name: 'SSM',  note: '线性 O(T), Mamba' },
  ]},
  { key: 'ffn', label: '前馈 (ffn)', options: [
    { id: 'relu',   name: 'ReLU',    note: '原始 Transformer' },
    { id: 'gelu',   name: 'GELU',    note: 'BERT / GPT 风' },
    { id: 'swiglu', name: 'SwiGLU',  note: '门控, LLaMA/Qwen' },
    { id: 'moe_mx', name: 'Mixtral-MoE', note: 'softmax top-2' },
    { id: 'moe_ds', name: 'DeepSeek-MoE', note: 'sigmoid + 共享' },
  ]},
  { key: 'norm', label: '归一化 (norm)', options: [
    { id: 'post_ln', name: 'Post-LN', note: '原始, 需 warmup' },
    { id: 'pre_ln',  name: 'Pre-LN LayerNorm', note: 'GPT-3 风' },
    { id: 'pre_rms', name: 'Pre-LN RMSNorm',   note: 'LLaMA/DeepSeek 风' },
    { id: 'ada_ln',  name: 'adaLN-Zero',       note: 'DiT 扩散' },
  ]},
  { key: 'pos', label: '位置编码 (pos)', options: [
    { id: 'sin',   name: 'Sinusoidal' },
    { id: 'learn', name: 'Learnable' },
    { id: 'rope',  name: 'RoPE' },
    { id: 'mrope', name: 'M-RoPE (三轴)' },
  ]},
]

const config = reactive({
  attn: 'gqa',
  ffn: 'swiglu',
  norm: 'pre_rms',
  pos: 'rope',
})

const partColorMap = {
  // attn
  mha: '#9ca3af', gqa: '#60a5fa', mla: '#34d399', dsa: '#ec4899', ssm: '#c084fc',
  // ffn
  relu: '#9ca3af', gelu: '#60a5fa', swiglu: '#34d399',
  moe_mx: '#f5a623', moe_ds: '#ec4899',
  // norm
  post_ln: '#9ca3af', pre_ln: '#60a5fa', pre_rms: '#34d399', ada_ln: '#ec4899',
  // pos
  sin: '#9ca3af', learn: '#9ca3af', rope: '#34d399', mrope: '#f5a623',
}
const partColor = (k) => partColorMap[config[k]] || '#fff'

const findOpt = (slot, id) => slots.find(s => s.key === slot).options.find(o => o.id === id)
const partName = (slot) => findOpt(slot, config[slot])?.name
const partNote = (slot) => findOpt(slot, config[slot])?.note || ''

// --- 模型匹配 ---
const presets = [
  { name: 'Transformer', year: 2017, config: { attn: 'mha', ffn: 'relu',   norm: 'post_ln', pos: 'sin'   } },
  { name: 'BERT',        year: 2018, config: { attn: 'mha', ffn: 'gelu',   norm: 'pre_ln',  pos: 'learn' } },
  { name: 'GPT-3',       year: 2020, config: { attn: 'mha', ffn: 'gelu',   norm: 'pre_ln',  pos: 'sin'   } },
  { name: 'LLaMA',       year: 2023, config: { attn: 'gqa', ffn: 'swiglu', norm: 'pre_rms', pos: 'rope'  } },
  { name: 'Mixtral',     year: 2024, config: { attn: 'gqa', ffn: 'moe_mx', norm: 'pre_rms', pos: 'rope'  } },
  { name: 'DeepSeek-V3', year: 2024, config: { attn: 'mla', ffn: 'moe_ds', norm: 'pre_rms', pos: 'rope'  } },
  { name: 'DeepSeek-V3.2', year: 2025, config: { attn: 'dsa', ffn: 'moe_ds', norm: 'pre_rms', pos: 'rope' } },
  { name: 'Mamba',       year: 2023, config: { attn: 'ssm', ffn: 'swiglu', norm: 'pre_rms', pos: 'sin'   } },
  { name: 'DiT',         year: 2023, config: { attn: 'mha', ffn: 'gelu',   norm: 'ada_ln',  pos: 'learn' } },
  { name: 'Qwen2-VL',    year: 2024, config: { attn: 'gqa', ffn: 'swiglu', norm: 'pre_rms', pos: 'mrope' } },
]
function loadPreset(p) { Object.assign(config, p.config) }

const modelMatch = computed(() => {
  const match = presets.find(p =>
    p.config.attn === config.attn &&
    p.config.ffn === config.ffn &&
    p.config.norm === config.norm &&
    p.config.pos === config.pos
  )
  if (match) {
    const meta = {
      'Transformer': { track: 'left', color: '#9ca3af',
        blurb: '原始 encoder-decoder + Post-LN, 全部零件最朴素' },
      'BERT': { track: 'left', color: '#60a5fa',
        blurb: '双向注意力 + MLM, 理解任务里程碑' },
      'GPT-3': { track: 'left', color: '#60a5fa',
        blurb: '把 Transformer 纯 decoder 堆到 175B, 生成式范式起点' },
      'LLaMA': { track: 'left', color: '#3dd68c',
        blurb: '现代开源 LLM 的事实模板: 四个零件全部换成最新版' },
      'Mixtral': { track: 'left', color: '#f5a623',
        blurb: 'LLaMA 骨架, FFN 换成 softmax top-k MoE' },
      'DeepSeek-V3': { track: 'left', color: '#ec4899',
        blurb: '把 KV cache 压到 latent + MoE 切更细, 671B/37B 激活' },
      'DeepSeek-V3.2': { track: 'left', color: '#ec4899',
        blurb: 'V3 + DSA, 把算力从 O(T²) 降到 O(T·k)' },
      'Mamba': { track: 'left', color: '#c084fc',
        blurb: '非注意力分支: SSM 线性 O(T), 另一条主线' },
      'DiT': { track: 'right', color: '#ec4899',
        blurb: '扩散 Transformer 骨架, adaLN-Zero 注入 timestep' },
      'Qwen2-VL': { track: 'eye', color: '#f5a623',
        blurb: 'LLaMA + M-RoPE, 视觉/文本共用 decoder' },
    }
    return { ...match, ...meta[match.name], color: meta[match.name]?.color || 'var(--accent)' }
  }
  return {
    name: '自定义组合',
    year: '—',
    track: null,
    color: 'var(--text-muted)',
    blurb: '这个组合不对应历史上的主流模型, 但也许是你下一篇论文的起点?',
  }
})

const generatedCode = computed(() => {
  const cls = {
    mha: 'MultiHeadAttention',
    gqa: 'GroupedQueryAttention',
    mla: 'MultiHeadLatentAttention',
    dsa: 'MultiHeadLatentSparseAttention',
    ssm: 'SelectiveSSM',
  }[config.attn]
  const ffn = {
    relu: 'FeedForward', gelu: 'GeLUFeedForward', swiglu: 'SwiGLUFeedForward',
    moe_mx: 'MixtralMoE', moe_ds: 'DeepSeekMoE',
  }[config.ffn]
  const norm = {
    post_ln: 'nn.LayerNorm', pre_ln: 'nn.LayerNorm',
    pre_rms: 'RMSNorm', ada_ln: 'AdaLNZeroBlock',
  }[config.norm]
  return `PreLNBlock(
    d_model=4096,
    attn=${cls}(d_model, n_heads),
    ffn=${ffn}(d_model, d_ff),
    norm_cls=${norm},
)`
})

// FFN 参数量粗估 (d_model=4096, d_ff=4*d_model; MoE 粗算每层总参数)
const ffnParams = computed(() => {
  const d = 4096, d_ff = 4 * d
  switch (config.ffn) {
    case 'relu':
    case 'gelu':   return (2 * d * d_ff).toLocaleString() + ' ≈ 134M'
    case 'swiglu': return (3 * d * Math.floor(d_ff * 2 / 3)).toLocaleString() + ' ≈ 100M'
    case 'moe_mx': return '8 × 100M ≈ 800M (激活 2×)'
    case 'moe_ds': return '64 × 12.5M + 2 × 12.5M ≈ 825M (激活 8×)'
  }
  return '—'
})
const ffnNote = computed(() => {
  if (config.ffn === 'swiglu') return 'd_ff 按 2/3 缩放以保持总参近似 GELU'
  if (config.ffn.startsWith('moe')) return '总参数大, 每 token 只激活 top-k 个专家'
  return '标准 FFN (d_ff = 4·d_model)'
})
const kvNote = computed(() => {
  const d = 4096, h = 32
  const kb = (x) => (x * 2 / 1024).toFixed(1) + ' KB'
  switch (config.attn) {
    case 'mha': return kb(2 * d)  // K+V 全维
    case 'gqa': return kb(2 * d / 4) + ' (kv_heads=8)'
    case 'mla': return kb(512 + 64) + ' (latent + rope)'
    case 'dsa': return kb(512 + 64) + ' + 稀疏'
    case 'ssm': return 'O(d_state) ≈ 0.1 KB'
  }
  return '—'
})
</script>

<style scoped>
.slot-section { margin-top: 14px; }
.slot-section:first-child { margin-top: 0; }
.slot-title {
  font-size: 11px;
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: 0.7px;
  margin-bottom: 6px;
}
.slot-options {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.slot-options button {
  text-align: left;
  padding: 8px 10px;
  font-size: 12px;
}

.match-card { transition: border-color 0.3s; }
.match-name {
  font-size: 22px;
  font-weight: 700;
  margin: 8px 0 8px;
  font-family: "SF Mono", Menlo, monospace;
}
</style>
