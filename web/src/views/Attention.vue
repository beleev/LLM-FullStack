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

    <!-- 顶部 variant 切换 + paper cite -->
    <div class="variant-tabs">
      <button v-for="v in variants" :key="v.id"
              :class="{ active: current.id === v.id }"
              @click="current = v">
        <span>{{ v.name }}</span>
        <span class="yr">{{ v.year }}</span>
      </button>
    </div>

    <!-- 主布局: 左参数, 中公式/数据, 右可视化 -->
    <div class="grid grid-2" style="gap: 20px;">
      <!-- 左: 参数 + 公式 -->
      <div class="card">
        <h3>参数调节 <span class="tag">interactive</span></h3>
        <p class="desc" style="margin-bottom: 12px;">{{ current.description }}</p>
        <div class="form-row">
          <label>序列长度 T</label>
          <input type="range" min="128" max="131072" step="128" v-model.number="p.T" />
          <span class="val">{{ formatT(p.T) }}</span>
        </div>
        <div class="form-row">
          <label>d_model</label>
          <input type="range" min="512" max="8192" step="128" v-model.number="p.d_model" />
          <span class="val">{{ p.d_model }}</span>
        </div>
        <div class="form-row">
          <label>n_heads (Q)</label>
          <input type="range" min="4" max="64" step="2" v-model.number="p.n_heads" />
          <span class="val">{{ p.n_heads }}</span>
        </div>
        <div v-if="current.id === 'gqa'" class="form-row">
          <label>num_kv_heads</label>
          <input type="range" :min="1" :max="p.n_heads" step="1" v-model.number="p.num_kv_heads" />
          <span class="val">{{ p.num_kv_heads }}</span>
        </div>
        <div v-if="current.id === 'mla' || current.id === 'dsa'" class="form-row">
          <label>kv_lora_rank</label>
          <input type="range" min="64" max="1024" step="32" v-model.number="p.kv_lora_rank" />
          <span class="val">{{ p.kv_lora_rank }}</span>
        </div>
        <div v-if="current.id === 'mla' || current.id === 'dsa'" class="form-row">
          <label>qk_rope_head_dim</label>
          <input type="range" min="16" max="128" step="8" v-model.number="p.qk_rope_head_dim" />
          <span class="val">{{ p.qk_rope_head_dim }}</span>
        </div>
        <div v-if="current.id === 'dsa'" class="form-row">
          <label>sparse_top_k</label>
          <input type="range" :min="32" :max="Math.min(2048, p.T)" step="32" v-model.number="p.sparse_top_k" />
          <span class="val">{{ p.sparse_top_k }}</span>
        </div>

        <div style="margin-top: 16px; padding: 12px; background: var(--code-bg); border-radius: 6px; font-size: 12.5px; color: var(--code-text); font-family: 'SF Mono', Menlo, monospace;">
          {{ current.formula }}
        </div>

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
        <h3>KV cache / token 对比 <span class="tag">所有变体同参</span></h3>
        <p class="desc" style="margin-bottom: 16px;">
          以 <span class="mono">T={{ formatT(p.T) }}</span>, <span class="mono">d_model={{ p.d_model }}</span>,
          <span class="mono">n_heads={{ p.n_heads }}</span> 下, 每层 KV cache 的大小。
          fp16 (2 bytes/元素) 估算。
        </p>
        <div class="bars">
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
            <div class="k">当前 cache / token</div>
            <div class="v accent">{{ formatBytes(cacheBytes(current)) }}</div>
            <div class="hint">相对 MHA: {{ relToMHA(current) }}</div>
          </div>
          <div class="stat">
            <div class="k">总 cache (单层 × B=1)</div>
            <div class="v">{{ formatBytes(cacheBytes(current) * 1) }}</div>
            <div class="hint">128 层时 × 128</div>
          </div>
          <div class="stat">
            <div class="k">Attention 算力</div>
            <div class="v">{{ formatFlops(flopsPerQuery) }}</div>
            <div class="hint">每 query 一次 QK^T · V</div>
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
          <svg :viewBox="`0 0 ${matrixSize} ${matrixSize}`" width="100%" :height="matrixSize" style="max-width: 480px;">
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
              <span v-if="current.id === 'dsa'"><br>sparse_top_k = <span class="mono">{{ Math.min(p.sparse_top_k, N) }}</span>, 每行仅亮 top-k 列。</span>
            </p>
          </div>
        </div>
      </div>
    </section>

    <!-- 源码速览 -->
    <section class="section">
      <h2>核心代码</h2>
      <p class="lead">对应 <code class="inline">llm_models/layers/core/attention.py</code> 里的 <span class="mono">{{ classFor(current.id) }}</span> — 当前选中的变体即此类。</p>
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
.variant-tabs {
  display: flex;
  gap: 8px;
  margin-bottom: 20px;
  padding-bottom: 4px;
}
.variant-tabs button {
  display: flex;
  flex-direction: column;
  gap: 2px;
  align-items: center;
  padding: 10px 22px;
  min-width: 100px;
}
.variant-tabs button .yr {
  font-size: 10px;
  color: var(--text-dim);
  font-family: "SF Mono", Menlo, monospace;
}
.variant-tabs button.active .yr { color: rgba(255,255,255,0.75); }

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
  transition: all 0.2s;
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
.bar-rel { font-size: 12px; color: var(--text-muted); text-align: right; }

.hero-stat {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
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
</style>
