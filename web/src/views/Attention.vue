<template>
  <div>
    <h1 class="page-title">注意力的四代演进</h1>
    <p class="page-subtitle">
      KV cache 是长上下文 LLM 推理的最大显存杀手。Attention 的演进
      (MHA → MQA → GQA → MLA → DSA) 本质都是在回答同一个问题:
      <strong>如何在不损失效果的前提下, 把每层每个 token 要缓存的东西变小?</strong>
    </p>

    <ChapterIntro
      tldr="八年里 attention 没换公式 (softmax(QK^T)·V), 只换了 K/V 的存储方式 — 头数、低秩、稀疏依次出场。"
      question="自回归推理时, 每层每 token 要缓存的 K, V 能不能再小一点?"
      :goals="[
        '理解 MHA / GQA / MLA / DSA 的差异 = 怎么省 KV cache',
        '看清「头数 / KV 头数 / 压缩维 / 稀疏 top-k」之间的换算',
        '能对照源码读出: 每一种注意力的 forward 长什么样',
      ]"
      :codes="[{ path: 'llm_models/layers/core/attention.py' }]"
      :prereq="{ name: 'home', label: '总览时间轴' }"
      :next-step="{ name: 'position', label: '位置编码 — RoPE 是怎么注入到 Q/K 的' }"
    />

    <EvolutionChain
      title="演进逻辑链 · 每一代都解决上一代的瓶颈"
      subtitle="顺着箭头读, 你会看到「问题 → 解法 → 引发新问题 → 下一代解法」的清晰螺旋。"
      :steps="evoSteps"
    />

    <!-- 先建立实验任务，再让读者替换内部机制。 -->
    <section class="concept-workbench card" aria-labelledby="attention-workbench-title">
      <div class="workbench-heading">
        <span class="eyebrow">INTERACTIVE MENTAL MODEL</span>
        <div>
          <h2 id="attention-workbench-title">先固定任务，再替换「KV 怎么存」</h2>
          <p>
            把下面当成一次可控实验：上下文和模型规模不变，只替换 Attention 方案。
            这样柱图的差异就只来自内部机制，而不是参数口径变化。
          </p>
        </div>
      </div>

      <ol class="learning-steps" aria-label="交互阅读顺序">
        <li><span>1</span><strong>选上下文</strong><small>决定要记住多少 token</small></li>
        <li><span>2</span><strong>换 Attention</strong><small>观察 KV 存储机制变化</small></li>
        <li><span>3</span><strong>读结果</strong><small>把显存和计算差异说清楚</small></li>
      </ol>

      <fieldset class="context-presets">
        <legend>第一步 · 选择一个真实阅读场景</legend>
        <div class="preset-grid">
          <button
            v-for="preset in contextPresets"
            :key="preset.id"
            type="button"
            :class="{ active: activePresetId === preset.id }"
            :aria-pressed="activePresetId === preset.id"
            @click="applyContextPreset(preset)"
          >
            <span>{{ preset.label }}</span>
            <small>{{ preset.hint }}</small>
          </button>
        </div>
      </fieldset>

      <div class="preset-readout" aria-live="polite" aria-atomic="true">
        <span class="readout-kicker">当前实验</span>
        <strong>{{ activePresetLabel }}</strong>
        <span>
          {{ formatT(p.T) }} tokens × d_model {{ p.d_model }}。
          接下来切换 Attention，只观察存储方式带来的差异。
        </span>
      </div>
    </section>

    <!-- 顶部 variant 切换 + paper cite -->
    <div class="variant-tabs" role="group" aria-label="第二步 · 选择 Attention 方案">
      <button
        v-for="(v, index) in variants"
        :key="v.id"
        type="button"
        :class="{ active: current.id === v.id }"
        :aria-pressed="current.id === v.id"
        @click="current = v"
      >
        <span class="variant-index" aria-hidden="true">{{ index + 1 }}</span>
        <span class="variant-name">{{ v.name }}</span>
        <span class="yr">{{ v.year }}</span>
      </button>
    </div>

    <section
      class="concept-bridge"
      :style="{ '--variant-color': current.color }"
      aria-labelledby="current-concept-title"
    >
      <div class="bridge-heading">
        <span class="eyebrow">第二步 · 当前机制</span>
        <h2 id="current-concept-title">{{ current.fullName }}</h2>
        <p>{{ current.description }}</p>
      </div>
      <div class="causal-flow" aria-live="polite" aria-atomic="true">
        <article>
          <span class="causal-label">输入</span>
          <strong>{{ conceptStory.input }}</strong>
          <p>{{ conceptStory.inputNote }}</p>
        </article>
        <span class="causal-arrow" aria-hidden="true">→</span>
        <article class="mechanism">
          <span class="causal-label">内部机制</span>
          <strong>{{ conceptStory.mechanism }}</strong>
          <p>{{ conceptStory.mechanismNote }}</p>
        </article>
        <span class="causal-arrow" aria-hidden="true">→</span>
        <article class="outcome">
          <span class="causal-label">可观察结果</span>
          <strong>{{ conceptStory.outcome }}</strong>
          <p>{{ conceptStory.outcomeNote }}</p>
        </article>
      </div>
      <div class="concept-takeaway">
        <span>一句话带走</span>
        <strong>{{ conceptStory.takeaway }}</strong>
      </div>
    </section>

    <!-- 主布局: 左参数, 中公式/数据, 右可视化 -->
    <div class="grid grid-2" style="gap: 20px;">
      <!-- 左: 参数 + 公式 -->
      <div class="card">
        <h3>参数调节 <span class="tag">interactive</span></h3>
        <p class="desc" style="margin-bottom: 12px;">{{ current.description }}</p>
        <div class="form-row">
          <label for="attention-seq-length">序列长度 T</label>
          <input id="attention-seq-length" type="range" min="128" max="131072" step="128" v-model.number="p.T" />
          <span class="val">{{ formatT(p.T) }}</span>
        </div>
        <div class="form-row">
          <label for="attention-model-width">模型宽度 d_model</label>
          <input id="attention-model-width" type="range" min="512" max="8192" step="128" v-model.number="p.d_model" />
          <span class="val">{{ p.d_model }}</span>
        </div>
        <div class="form-row">
          <label for="attention-query-heads">查询头数 n_heads</label>
          <input id="attention-query-heads" type="range" min="4" max="64" step="2" v-model.number="p.n_heads" />
          <span class="val">{{ p.n_heads }}</span>
        </div>
        <div v-if="current.id === 'gqa'" class="form-row">
          <label for="attention-kv-heads">KV 头数 num_kv_heads</label>
          <input id="attention-kv-heads" type="range" :min="1" :max="p.n_heads" step="1" v-model.number="p.num_kv_heads" />
          <span class="val">{{ p.num_kv_heads }}</span>
        </div>
        <div v-if="current.id === 'mla' || current.id === 'dsa'" class="form-row">
          <label for="attention-kv-rank">KV 压缩维 kv_lora_rank</label>
          <input id="attention-kv-rank" type="range" min="64" max="1024" step="32" v-model.number="p.kv_lora_rank" />
          <span class="val">{{ p.kv_lora_rank }}</span>
        </div>
        <div v-if="current.id === 'mla' || current.id === 'dsa'" class="form-row">
          <label for="attention-rope-dim">位置维 qk_rope_head_dim</label>
          <input id="attention-rope-dim" type="range" min="16" max="128" step="8" v-model.number="p.qk_rope_head_dim" />
          <span class="val">{{ p.qk_rope_head_dim }}</span>
        </div>
        <div v-if="current.id === 'dsa'" class="form-row">
          <label for="attention-sparse-top-k">稀疏候选 sparse_top_k</label>
          <input id="attention-sparse-top-k" type="range" :min="32" :max="Math.min(2048, p.T)" step="32" v-model.number="p.sparse_top_k" />
          <span class="val">{{ p.sparse_top_k }}</span>
        </div>

        <div class="formula-box">
          {{ current.formula }}
        </div>

        <p class="parameter-feedback" aria-live="polite" aria-atomic="true">
          新增一个 token 时，本层需要写入
          <strong class="mono">{{ formatBytes(cacheBytesPerToken(current)) }}</strong> KV；
          当前整段上下文占
          <strong class="mono">{{ formatBytes(cacheBytes(current)) }}</strong>。
        </p>

        <div class="trade" style="margin-top: 16px;">
          <div class="tr-item">
            <span class="ok">✓</span>
            <span>{{ current.pros }}</span>
          </div>
          <div class="tr-item">
            <span class="no">✗</span>
            <span>{{ current.cons }}</span>
          </div>
        </div>

        <div style="margin-top: 12px; font-size: 11px; color: var(--text-dim);">
          <span class="mono">{{ current.paper }}</span>
          · 使用模型: <span class="mono">{{ current.usedIn.join(', ') }}</span>
        </div>
      </div>

      <!-- 右: KV cache 横向对比柱图 -->
      <div class="card">
        <h3>整段 KV cache 对比 <span class="tag">单层 · B=1 · fp16</span></h3>
        <p class="desc" style="margin-bottom: 16px;">
          以 <span class="mono">T={{ formatT(p.T) }}</span>, <span class="mono">d_model={{ p.d_model }}</span>,
          <span class="mono">n_heads={{ p.n_heads }}</span> 下，比较一层保存<strong>整段上下文</strong>所需的 KV cache。
          柱越短，长上下文推理的显存压力越小。
        </p>
        <div class="bars" aria-label="四种 Attention 的单层 KV cache 对比">
          <div v-for="v in variants" :key="v.id" class="bar-row" :class="{ current: v.id === current.id }">
            <div class="bar-label">
              <span class="bar-name">{{ v.name }}</span>
              <span class="bar-sub mono">{{ formatBytes(cacheBytes(v)) }}</span>
            </div>
            <div class="bar-track">
              <div class="bar-fill"
                   :style="{ width: barPct(v) + '%', background: v.color }"></div>
            </div>
            <div class="bar-rel mono">{{ relToMHA(v) }}</div>
          </div>
        </div>

        <div class="hero-stat">
          <div class="stat">
            <div class="k">新增 1 token / 层</div>
            <div class="v accent">{{ formatBytes(cacheBytesPerToken(current)) }}</div>
            <div class="hint">每步追加的 KV 存储</div>
          </div>
          <div class="stat">
            <div class="k">当前上下文 / 层</div>
            <div class="v">{{ formatBytes(cacheBytes(current) * 1) }}</div>
            <div class="hint">相对 MHA: {{ relToMHA(current) }}</div>
          </div>
          <div class="stat">
            <div class="k">128 层合计</div>
            <div class="v">{{ formatBytes(cacheBytes(current) * 128) }}</div>
            <div class="hint">未计 batch 与其他激活</div>
          </div>
          <div class="stat">
            <div class="k">整段关系数</div>
            <div class="v">{{ formatFlops(flopsPerQuery) }}</div>
            <div class="hint">观察 O(T²) / O(T·k) 趋势</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Attention 矩阵可视化 -->
    <section class="section">
      <h2>Attention 矩阵可视化</h2>
      <p class="lead">
        每行是一个 query, 每列是一个 key。颜色越亮, query 越"看"这个 key。因果 mask 使右上三角被屏蔽。
        <span v-if="current.id === 'dsa'">DSA 只在 top-k 位置有权重, 大部分格子是空的 (稀疏)。</span>
      </p>
      <div class="card">
        <div class="matrix-grid">
          <svg
            :viewBox="`0 0 ${matrixSize} ${matrixSize}`"
            width="100%"
            :height="matrixSize"
            style="max-width: 480px;"
            role="img"
            :aria-label="matrixAriaLabel"
          >
            <rect v-for="cell in matrixCells" :key="cell.i + ':' + cell.j"
                  :x="cell.j * cellSize" :y="cell.i * cellSize"
                  :width="cellSize - 0.5" :height="cellSize - 0.5"
                  :fill="cell.color" />
          </svg>
          <div class="legend-attn">
            <div class="legend-strip">
              <span v-for="i in 10" :key="i"
                    :style="{ background: colorFor(i / 10) }"></span>
            </div>
            <div class="legend-labels mono">
              <span>0</span><span>权重</span><span>1</span>
            </div>
            <p class="desc" style="margin-top: 16px; max-width: 280px;">
              N = <span class="mono">{{ N }}</span> 个 token 的因果注意力矩阵。
              <span v-if="current.id === 'dsa'">sparse_top_k = <span class="mono">{{ Math.min(p.sparse_top_k, N) }}</span>, 每行仅亮 top-k 列。</span>
            </p>
            <p class="matrix-reading">{{ matrixReading }}</p>
          </div>
        </div>
      </div>
    </section>

    <!-- mask 一族: 投影压缩之外的另一条降本路线 -->
    <section class="section">
      <h2>另外两条路线: 改 mask, 或干脆不存 KV</h2>
      <p class="lead">
        MHA→GQA→MLA 压缩的是 <b>KV 投影</b>; Mistral / StreamingLLM / DSA 改的是 <b>mask</b> — 谁能看见谁;
        而 <b>线性注意力</b> (Gated DeltaNet, <RepoLink path="llm_models/layers/sparse/linear_attention.py" label="linear_attention.py" tiny />)
        干脆用固定大小的状态矩阵替掉整个 KV cache — Qwen3-Next 用它替换了 75% 的层, 剩下 25% 全注意力兜底召回。
        三条路线正交, 可以叠加。详见章节「SWA · MTP · 混合线性」。
      </p>
      <AttnMaskLab />
    </section>

    <!-- 源码速览 -->
    <section class="section">
      <h2>核心代码</h2>
      <p class="lead">对应 <RepoLink path="llm_models/layers/core/attention.py" label="llm_models/layers/core/attention.py" tiny /> 里的 <span class="mono">{{ classFor(current.id) }}</span> — 当前选中的变体即此类。</p>
      <pre class="code" v-html="highlight(codeSnippet)"></pre>
    </section>

    <ChapterNav
      :prev="{ name: 'home', label: '时间轴总览', hint: '回到 2017–2025 全景图' }"
      :next="{ name: 'position', label: '位置编码 & RoPE', hint: '看完 attention 后, 再看 RoPE 是怎么注入 Q/K 的' }"
    />
  </div>
</template>

<script setup>
import { ref, reactive, computed, watch } from 'vue'
import { variants } from '@/data/attention.js'
import ChapterIntro from '@/components/ChapterIntro.vue'
import ChapterNav from '@/components/ChapterNav.vue'
import EvolutionChain from '@/components/EvolutionChain.vue'
import RepoLink from '@/components/RepoLink.vue'
import AttnMaskLab from '@/components/labs/AttnMaskLab.vue'

const evoSteps = [
  { name: 'MHA', year: 2017, color: '#9ca3af',
    pain: '(原点) 每个头一对独立 K/V', fix: '表达力满, 开局即上限' },
  { name: 'MQA / GQA', year: '2019 / 2023', color: '#60a5fa',
    pain: 'KV cache = T·d_model, 长上下文显存爆炸', fix: '让多个 Q 头共享同一对 K/V → cache ÷ groups' },
  { name: 'MLA', year: 2024, color: '#34d399',
    pain: 'GQA 已挤干 head 维度, 但 cache 还和 d_model 成正比', fix: 'KV 一次低秩压缩成 c_kv (≈ d/8) + 解耦 RoPE 头, cache 再降 ~8×' },
  { name: 'DSA', year: 2025, color: '#f472b6',
    pain: '128K 上下文下 cache 解决了, 但 O(T²) 算力爆炸', fix: 'Lightning Indexer 选 top-k 关键位置, attention 算力降到 O(T·k)' },
]

const contextPresets = [
  { id: 'chat', label: '日常对话 · 4K', hint: '约 10–20 页文本', T: 4096 },
  { id: 'document', label: '长文档 · 32K', hint: '整篇论文或代码库片段', T: 32768 },
  { id: 'long-context', label: '超长上下文 · 128K', hint: '显存与计算差异被放大', T: 131072 },
]

const current = ref(variants[0])

const p = reactive({
  T: 4096,
  d_model: 4096,
  n_heads: 32,
  num_kv_heads: 8,
  kv_lora_rank: 512,
  qk_rope_head_dim: 64,
  sparse_top_k: 512,
})

const applyContextPreset = (preset) => {
  p.T = preset.T
}

const activePresetId = computed(() =>
  contextPresets.find((preset) => preset.T === p.T)?.id || 'custom'
)
const activePresetLabel = computed(() =>
  contextPresets.find((preset) => preset.id === activePresetId.value)?.label || '自定义上下文'
)

// 保证 num_kv_heads ≤ n_heads 且可整除
watch(() => p.n_heads, () => {
  if (p.num_kv_heads > p.n_heads) p.num_kv_heads = p.n_heads
  // 简单兼容: 找到最近的约数
  while (p.n_heads % p.num_kv_heads !== 0 && p.num_kv_heads > 1) p.num_kv_heads--
})

const cacheBytes = (v) => v.cache({
  T: p.T,
  d_model: p.d_model,
  n_heads: p.n_heads,
  num_kv_heads: v.id === 'gqa' ? p.num_kv_heads : v.id === 'mha' ? p.n_heads : 1,
  kv_lora_rank: p.kv_lora_rank,
  qk_rope_head_dim: p.qk_rope_head_dim,
}) * 2  // fp16

const cacheBytesPerToken = (v) => v.cache({
  T: 1,
  d_model: p.d_model,
  n_heads: p.n_heads,
  num_kv_heads: v.id === 'gqa' ? p.num_kv_heads : v.id === 'mha' ? p.n_heads : 1,
  kv_lora_rank: p.kv_lora_rank,
  qk_rope_head_dim: p.qk_rope_head_dim,
}) * 2

const mhaBytes = computed(() => variants[0].cache({ T: p.T, d_model: p.d_model, n_heads: p.n_heads }) * 2)
const maxBytes = computed(() => Math.max(...variants.map(cacheBytes)))
const barPct = (v) => (cacheBytes(v) / maxBytes.value) * 100
const relToMHA = (v) => {
  const ratio = cacheBytes(v) / mhaBytes.value
  if (ratio < 0.001) return (ratio * 1000).toFixed(2) + '‰'
  return (ratio * 100).toFixed(1) + '%'
}

const flopsPerQuery = computed(() => {
  if (current.value.id === 'dsa') return p.T * Math.min(p.sparse_top_k, p.T)
  return p.T * p.T
})

const conceptStory = computed(() => {
  const shared = {
    input: `${formatT(p.T)} tokens × ${p.d_model} 维`,
    inputNote: `每层有 ${p.n_heads} 个 Q 头；实验保持模型宽度与上下文不变。`,
    outcome: `${formatBytes(cacheBytes(current.value))} / 层`,
    outcomeNote: `新增 1 token 写入 ${formatBytes(cacheBytesPerToken(current.value))} KV；相对 MHA 为 ${relToMHA(current.value)}。`,
  }

  const stories = {
    mha: {
      mechanism: `${p.n_heads} 个 Q 头各存一组 K/V`,
      mechanismNote: '没有共享或压缩，是后续方案比较时的显存基线。',
      takeaway: 'MHA 的表达路径最直接，但上下文每增长一倍，KV cache 也跟着增长一倍。',
    },
    gqa: {
      mechanism: `${p.n_heads} 个 Q 头共享 ${p.num_kv_heads} 组 K/V`,
      mechanismNote: `每 ${Math.max(1, p.n_heads / p.num_kv_heads).toFixed(0)} 个 Q 头复用一组 K/V，Q 的数量不变。`,
      takeaway: 'GQA 省的是“重复保存的 K/V 头”，不是缩短上下文，也没有改变 Attention 公式。',
    },
    mla: {
      mechanism: `KV 先压到 ${p.kv_lora_rank} 维 latent`,
      mechanismNote: `运行时只缓存 c_kv + ${p.qk_rope_head_dim} 维位置向量，需要时再升维还原 K/V。`,
      takeaway: 'MLA 把“存完整 K/V”改成“存可还原的压缩表示”，因此长上下文显存下降最明显。',
    },
    dsa: {
      mechanism: `MLA 压缩 + 每次只选 ${Math.min(p.sparse_top_k, p.T)} 个位置`,
      mechanismNote: 'KV 仍按 MLA 保存，但 Lightning Indexer 让 softmax 只处理重要历史位置。',
      takeaway: 'DSA 分两步解决瓶颈：MLA 省显存，稀疏 top-k 再省超长上下文的计算。',
    },
  }

  return { ...shared, ...stories[current.value.id] }
})

const formatT = (t) => t >= 1024 ? (t / 1024).toFixed(t % 1024 === 0 ? 0 : 1) + 'K' : String(t)
const formatBytes = (b) => {
  if (b < 1024) return b.toFixed(0) + ' B'
  if (b < 1024 * 1024) return (b / 1024).toFixed(1) + ' KB'
  if (b < 1024 * 1024 * 1024) return (b / 1024 / 1024).toFixed(2) + ' MB'
  return (b / 1024 / 1024 / 1024).toFixed(2) + ' GB'
}
const formatFlops = (f) => {
  if (f < 1e6) return f.toFixed(0)
  if (f < 1e9) return (f / 1e6).toFixed(1) + 'M'
  return (f / 1e9).toFixed(2) + 'G'
}

// --- Attention 矩阵可视化 ---
const N = 24  // 显示用的小矩阵
const matrixSize = 320
const cellSize = matrixSize / N

const matrixCells = computed(() => {
  const cells = []
  for (let i = 0; i < N; i++) {
    // 每行的分数: 近距离 + 一些随机热点, 因果 mask 使 j > i 为 0
    const scores = new Array(N).fill(0)
    for (let j = 0; j <= i; j++) {
      const dist = i - j
      // 模拟真实 attention: 相邻衰减 + 几个高分
      scores[j] = Math.exp(-dist * 0.12) + 0.25 * Math.sin(j * 0.7 + i * 0.3) + 0.2
      scores[j] = Math.max(0, scores[j])
    }
    // 对当前 variant 做处理
    if (current.value.id === 'dsa') {
      const k = Math.min(p.sparse_top_k >> 5, Math.min(8, i + 1))  // 缩放到小矩阵
      // 保留 top-k 最大的
      const top = [...scores.map((s, idx) => ({ s, idx }))].sort((a, b) => b.s - a.s).slice(0, k)
      const keep = new Set(top.map(x => x.idx))
      for (let j = 0; j < N; j++) if (!keep.has(j)) scores[j] = 0
    }
    // 归一化 (softmax 简化)
    const sum = scores.reduce((a, b) => a + b, 0) || 1
    for (let j = 0; j < N; j++) {
      const w = scores[j] / sum
      cells.push({ i, j, color: colorFor(w * 2.5) })  // 放大可视度
    }
  }
  return cells
})

const matrixReading = computed(() => current.value.id === 'dsa'
  ? '读图：右上角为空代表不能看未来；历史区域里只有少量亮点，代表 indexer 选出的 top-k 位置。'
  : '读图：右上角为空代表不能看未来；左下三角越亮，代表当前 token 越依赖那个历史位置。'
)

const matrixAriaLabel = computed(() =>
  `${current.value.name} 的 ${N}×${N} 因果注意力矩阵。${matrixReading.value}`
)

function colorFor(w) {
  w = Math.min(1, Math.max(0, w))
  // 暗蓝 → 亮青 → 黄
  const stops = [
    { t: 0,   c: [21, 27, 46] },
    { t: 0.3, c: [56, 69, 125] },
    { t: 0.6, c: [124, 107, 241] },
    { t: 0.9, c: [236, 72, 153] },
    { t: 1,   c: [251, 191, 36] },
  ]
  for (let i = 0; i < stops.length - 1; i++) {
    if (w <= stops[i + 1].t) {
      const t = (w - stops[i].t) / (stops[i + 1].t - stops[i].t)
      const c = stops[i].c.map((a, k) => Math.round(a + (stops[i + 1].c[k] - a) * t))
      return `rgb(${c[0]},${c[1]},${c[2]})`
    }
  }
  return `rgb(${stops.at(-1).c.join(',')})`
}

// --- 代码片段 ---
const classFor = (id) => ({
  mha: 'MultiHeadAttention', gqa: 'GroupedQueryAttention',
  mla: 'MultiHeadLatentAttention', dsa: 'MultiHeadLatentSparseAttention',
}[id])

const codeSnippet = computed(() => {
  const snippets = {
    mha: `class MultiHeadAttention(nn.Module):
    """原始 MHA (Vaswani et al., 2017)"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_head = d_model // num_heads
        self.heads = nn.ModuleList([
            SingleHeadSelfAttention(d_model, self.d_head)
            for _ in range(num_heads)
        ])
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, q, k=None, v=None, mask=None):
        # KV cache: num_heads · d_head = d_model (满)
        outs = [h(q, k, v, mask) for h in self.heads]
        return self.w_o(torch.cat(outs, dim=-1))`,
    gqa: `class GroupedQueryAttention(nn.Module):
    """GQA — Ainslie et al., 2023 (LLaMA-2 70B)"""
    def __init__(self, d_model, num_heads, num_kv_heads=None):
        super().__init__()
        self.num_groups = num_heads // num_kv_heads  # 每组 Q 共享 1 对 KV
        self.w_q = nn.Linear(d_model, num_heads    * head_dim)
        self.w_k = nn.Linear(d_model, num_kv_heads * head_dim)  # ← 更小
        self.w_v = nn.Linear(d_model, num_kv_heads * head_dim)

    def forward(self, q, k=None, v=None, mask=None, rope=None):
        Q = self.w_q(q).view(B, T, num_heads,    head_dim)
        K = self.w_k(k).view(B, T, num_kv_heads, head_dim)
        # 复制 num_groups 份, 对齐 Q
        K = K.repeat_interleave(self.num_groups, dim=1)
        # ... 标准 QK^T · V`,
    mla: `class MultiHeadLatentAttention(nn.Module):
    """MLA — DeepSeek-V2/V3, KV 低秩 + 解耦 RoPE"""
    def __init__(self, d_model, num_heads, kv_lora_rank=512,
                 qk_nope_head_dim=64, qk_rope_head_dim=32):
        # KV 一次压成低秩 c_kv + 共享 k_rope
        self.kv_down = nn.Linear(d_model, kv_lora_rank + qk_rope_head_dim)
        self.k_up = nn.Linear(kv_lora_rank, num_heads * qk_nope_head_dim)
        self.v_up = nn.Linear(kv_lora_rank, num_heads * v_head_dim)
        # Q 也可低秩 (DeepSeek-V3 省训练显存)

    def forward(self, x, rope=None):
        kv_mix = self.kv_down(x)                    # [B, T, lora + rope]
        c_kv, k_rope = torch.split(kv_mix, [lora, rope], dim=-1)
        # nope 段走 latent, rope 段独立旋转
        # 推理: 只缓存 c_kv + k_rope, 约 MHA 的 7%`,
    dsa: `class MultiHeadLatentSparseAttention(nn.Module):
    """DSA — DeepSeek-V3.2 (2025), MLA + Lightning Indexer"""
    def __init__(self, d_model, num_heads, kv_lora_rank=512,
                 indexer_heads=4, sparse_top_k=128):
        self.mla = MultiHeadLatentAttention(...)
        self.indexer = LightningIndexer(d_model, indexer_heads)
        self.sparse_top_k = sparse_top_k

    def forward(self, q, mask=None, rope=None):
        # 1) 廉价 indexer 打分
        idx_scores = self.indexer(q, mask=mask)       # [B, T, S]
        # 2) 先按 mask 屏蔽不可见, 再取 top-k (否则可能泄漏未来)
        sparse_mask = top_k_mask(idx_scores, mask, self.sparse_top_k)
        # 3) MLA 只在 top-k 位置算 softmax
        return self.mla(q, mask=sparse_mask, rope=rope)`
  }
  return snippets[current.value.id]
})

// 简易语法高亮
function highlight(s) {
  return s
    .replace(/#.*$/gm, m => `<span class="cm">${m}</span>`)
    .replace(/"""[\s\S]*?"""/g, m => `<span class="cm">${m}</span>`)
    .replace(/\b(class|def|return|for|in|None|if|else|super|self|import|from)\b/g, '<span class="kw">$1</span>')
    .replace(/\b(nn\.\w+|torch\.\w+|F\.\w+)\b/g, '<span class="fn">$1</span>')
    .replace(/\b\d+\b/g, m => `<span class="num">${m}</span>`)
}
</script>

<style scoped>
.concept-workbench {
  margin-bottom: 18px;
  border-color: color-mix(in srgb, var(--accent) 42%, var(--border));
  background:
    linear-gradient(135deg, color-mix(in srgb, var(--accent) 11%, transparent), transparent 48%),
    var(--bg-card);
  box-shadow: 0 14px 36px color-mix(in srgb, var(--accent) 10%, transparent);
}
.workbench-heading {
  display: grid;
  grid-template-columns: 150px minmax(0, 1fr);
  gap: 18px;
  align-items: start;
}
.eyebrow {
  color: var(--accent);
  font-family: "SF Mono", Menlo, monospace;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 1.1px;
}
.workbench-heading h2,
.bridge-heading h2 {
  margin: 0;
  font-size: 20px;
  line-height: 1.35;
  text-wrap: balance;
}
.workbench-heading p,
.bridge-heading p {
  margin-top: 6px;
  color: var(--text-muted);
  font-size: 13px;
  line-height: 1.7;
  text-wrap: pretty;
}
.learning-steps {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
  margin: 18px 0;
  list-style: none;
}
.learning-steps li {
  display: grid;
  grid-template-columns: 28px 1fr;
  gap: 0 10px;
  align-items: center;
  padding: 10px 12px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: color-mix(in srgb, var(--bg-elev) 82%, transparent);
}
.learning-steps li > span {
  grid-row: 1 / 3;
  width: 28px;
  height: 28px;
  display: grid;
  place-items: center;
  border-radius: 50%;
  background: var(--accent-soft);
  color: var(--accent);
  font-family: "SF Mono", Menlo, monospace;
  font-size: 11px;
  font-weight: 700;
}
.learning-steps strong { font-size: 12.5px; }
.learning-steps small { color: var(--text-muted); font-size: 10.5px; }
.context-presets { border: 0; }
.context-presets legend {
  margin-bottom: 8px;
  color: var(--text-muted);
  font-size: 12px;
  font-weight: 600;
}
.preset-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 8px;
}
.preset-grid button {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  min-height: 58px;
  padding: 9px 12px;
  text-align: left;
}
.preset-grid button span { font-size: 12.5px; font-weight: 600; }
.preset-grid button small { margin-top: 2px; color: var(--text-muted); font-size: 10.5px; }
.preset-grid button.active small { color: rgba(255, 255, 255, 0.76); }
.preset-readout {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 6px 10px;
  margin-top: 12px;
  padding: 10px 12px;
  border-left: 3px solid var(--accent);
  border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
  background: var(--code-bg);
  color: var(--text-muted);
  font-size: 12px;
}
.preset-readout strong { color: var(--text); }
.readout-kicker {
  color: var(--accent);
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.8px;
}

.variant-tabs {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 8px;
  margin-bottom: 12px;
  padding-bottom: 4px;
}
.variant-tabs button {
  display: grid;
  grid-template-columns: 24px 1fr auto;
  gap: 8px;
  align-items: center;
  min-height: 48px;
  padding: 8px 10px;
  text-align: left;
}
.variant-index {
  width: 24px;
  height: 24px;
  display: grid;
  place-items: center;
  border-radius: 50%;
  background: var(--bg-card);
  color: var(--text-muted);
  font-family: "SF Mono", Menlo, monospace;
  font-size: 10px;
}
.variant-name { font-weight: 650; }
.variant-tabs button .yr {
  font-size: 10px;
  color: var(--text-muted);
  font-family: "SF Mono", Menlo, monospace;
}
.variant-tabs button.active .yr { color: rgba(255,255,255,0.75); }
.variant-tabs button.active .variant-index { background: rgba(255, 255, 255, 0.18); color: #fff; }

.concept-bridge {
  --variant-color: var(--accent);
  margin-bottom: 20px;
  padding: 18px 20px;
  border: 1px solid color-mix(in srgb, var(--variant-color) 44%, var(--border));
  border-top: 3px solid var(--variant-color);
  border-radius: var(--radius);
  background: linear-gradient(
    135deg,
    color-mix(in srgb, var(--variant-color) 8%, var(--bg-card)),
    var(--bg-card) 55%
  );
}
.bridge-heading {
  display: grid;
  grid-template-columns: 150px minmax(0, 1fr);
  gap: 4px 18px;
}
.bridge-heading .eyebrow { grid-row: 1 / 3; color: var(--variant-color); }
.causal-flow {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr) auto minmax(0, 1fr);
  gap: 10px;
  align-items: stretch;
  margin-top: 16px;
}
.causal-flow article {
  padding: 12px 14px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--bg-elev);
}
.causal-flow article.mechanism { border-color: color-mix(in srgb, var(--variant-color) 52%, var(--border)); }
.causal-flow article.outcome { background: color-mix(in srgb, var(--variant-color) 7%, var(--bg-elev)); }
.causal-label {
  display: block;
  margin-bottom: 4px;
  color: var(--text-muted);
  font-family: "SF Mono", Menlo, monospace;
  font-size: 9.5px;
  font-weight: 700;
  letter-spacing: 0.8px;
}
.causal-flow strong { display: block; font-size: 13px; text-wrap: balance; }
.causal-flow p { margin-top: 5px; color: var(--text-muted); font-size: 11.5px; line-height: 1.6; text-wrap: pretty; }
.causal-arrow { align-self: center; color: var(--variant-color); font-size: 18px; }
.concept-takeaway {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr);
  gap: 12px;
  align-items: baseline;
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px dashed var(--border);
}
.concept-takeaway span { color: var(--variant-color); font-size: 10px; font-weight: 700; letter-spacing: 0.7px; }
.concept-takeaway strong { font-size: 12.5px; line-height: 1.7; text-wrap: pretty; }

.formula-box {
  margin-top: 16px;
  padding: 12px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--code-bg);
  color: var(--code-text);
  font-family: "SF Mono", Menlo, monospace;
  font-size: 12.5px;
  line-height: 1.6;
}
.parameter-feedback {
  margin-top: 10px;
  padding: 10px 12px;
  border-radius: var(--radius-sm);
  background: var(--accent-soft);
  color: var(--text-muted);
  font-size: 11.5px;
  line-height: 1.65;
}
.parameter-feedback strong { color: var(--text); font-variant-numeric: tabular-nums; }

.trade { font-size: 13px; color: var(--text-muted); }
.tr-item { display: flex; gap: 8px; align-items: flex-start; margin-top: 4px; }
.tr-item .ok { color: var(--left); }
.tr-item .no { color: var(--danger); }

.bars { display: flex; flex-direction: column; gap: 12px; }
.bar-row {
  display: grid;
  grid-template-columns: 110px 1fr 70px;
  gap: 12px;
  align-items: center;
  transition-property: margin, padding, background-color;
  transition-duration: 200ms;
  transition-timing-function: ease-out;
}
.bar-row.current { background: var(--bg-elev); margin: -6px -10px; padding: 6px 10px; border-radius: 6px; }
.bar-label { display: flex; flex-direction: column; gap: 2px; }
.bar-label .bar-name { font-size: 13px; font-weight: 500; }
.bar-label .bar-sub { font-size: 11px; color: var(--text-muted); }
.bar-track {
  height: 22px;
  background: var(--bg-elev);
  border-radius: 4px;
  overflow: hidden;
}
.bar-fill {
  height: 100%;
  transition: width 0.3s ease;
  border-radius: 4px;
}
.bar-rel { font-size: 12px; color: var(--text-muted); text-align: right; font-variant-numeric: tabular-nums; }

.hero-stat {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
  margin-top: 20px;
}

.matrix-grid {
  display: flex;
  gap: 24px;
  align-items: flex-start;
  flex-wrap: wrap;
}
.legend-attn { max-width: 320px; }
.legend-strip {
  display: flex;
  height: 14px;
  border-radius: 3px;
  overflow: hidden;
}
.legend-strip span { flex: 1; }
.legend-labels {
  display: flex;
  justify-content: space-between;
  font-size: 11px;
  color: var(--text-dim);
  margin-top: 4px;
}
.matrix-reading {
  margin-top: 12px;
  padding: 10px 12px;
  border-left: 3px solid var(--accent);
  background: var(--accent-soft);
  color: var(--text-muted);
  font-size: 11.5px;
  line-height: 1.65;
}

@media (max-width: 960px) {
  .workbench-heading,
  .bridge-heading { grid-template-columns: 1fr; }
  .bridge-heading .eyebrow { grid-row: auto; }
  .learning-steps,
  .preset-grid { grid-template-columns: 1fr; }
  .causal-flow {
    grid-template-columns: 1fr;
  }
  .causal-arrow { justify-self: center; transform: rotate(90deg); }
}

@media (max-width: 720px) {
  .variant-tabs { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .concept-workbench,
  .concept-bridge { padding: 16px; }
  .concept-takeaway { grid-template-columns: 1fr; gap: 4px; }
  .bar-row { grid-template-columns: 88px minmax(90px, 1fr) 58px; gap: 8px; }
  .hero-stat { grid-template-columns: 1fr; }
}
</style>
