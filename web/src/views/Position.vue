<template>
  <div>
    <h1 class="page-title">位置编码 · 从 Sinusoidal 到 RoPE 再到 M-RoPE</h1>
    <p class="page-subtitle">
      无位置编码时, attention 是 "bag of tokens"——调换 token 顺序结果不变。本章用一个滑条演示:
      <strong>为什么 RoPE 能让 Q·K 内积只依赖 "相对距离"</strong>。
    </p>

    <ChapterIntro
      tldr="位置编码从「加在 embedding 上」走向「乘在 Q/K 上」, 再到「按轴拆分」, 越走越贴近 attention 真正用到的形式。"
      question="如果让 Q·K 内积只依赖 (m − n), 而不依赖绝对位置 m、n, 模型还需要学什么?"
      :goals="[
        '区分绝对位置编码 / 学得位置 / RoPE / M-RoPE 的用途',
        '理解 RoPE 怎么让 Q·K 内积只依赖相对距离',
        '看懂多模态场景为什么要按轴拆分位置编码',
      ]"
      :codes="[{ path: 'llm_models/layers/core/position_encoding.py' }]"
      :prereq="{ name: 'attention', label: '注意力的四代演进' }"
      :next-step="{ name: 'blocks', label: 'Block 组装器 — 把零件拼成完整模型' }"
    />

    <EvolutionChain
      title="演进逻辑链 · 位置信息怎么 &quot;挪&quot; 进 attention"
      :steps="evoSteps"
    />

    <!-- variant 切换 -->
    <div class="btn-group" style="margin-bottom: 20px;">
      <button v-for="v in variants" :key="v.id"
              :class="{ active: variant === v.id }"
              @click="variant = v.id">{{ v.label }}</button>
    </div>

    <div v-if="variant === 'sin'">
      <!-- Sinusoidal -->
      <div class="grid grid-2">
        <div class="card">
          <h3>Sinusoidal PE <span class="tag">加在 embedding 上</span></h3>
          <p class="desc" style="margin-bottom: 12px;">
            对每个 position 计算一个固定向量, 加到 token embedding 上。不同维度选不同频率,
            构成 "位置指纹"。绝对位置, 不可外推到训练长度之外。
          </p>
          <pre class="code" v-html="highlight(sinCode)"></pre>
        </div>
        <div class="card">
          <h3>频率谱 <span class="tag">d_model=64</span></h3>
          <p class="desc" style="margin-bottom: 8px;">每行是一个维度, 横轴是位置。低维波长短, 高维波长长。</p>
          <svg viewBox="0 0 480 320" width="100%" height="320">
            <path v-for="(line, idx) in sinLines" :key="idx"
                  :d="line" fill="none"
                  :stroke="sinColor(idx)" stroke-width="1" stroke-opacity="0.7" />
          </svg>
        </div>
      </div>
    </div>

    <div v-else-if="variant === 'rope'">
      <!-- RoPE -->
      <div class="grid grid-2">
        <div class="card">
          <h3>RoPE 旋转可视化 <span class="tag">拖动滑条体验</span></h3>
          <p class="desc" style="margin-bottom: 16px;">
            RoPE 把 Q/K 的相邻两维视为复数 <span class="mono">z = a + ib</span>,
            在位置 <span class="mono">m</span> 处乘以 <span class="mono">e^(imθ)</span>,
            即旋转角 <span class="mono">mθ</span>。下面 4 个点分别对应 4 个维度对,
            不同频率导致不同旋转速度。
          </p>

          <div class="form-row">
            <label>位置 m</label>
            <input type="range" min="0" max="64" step="1" v-model.number="m" />
            <span class="val">{{ m }}</span>
          </div>

          <svg viewBox="-160 -160 320 320" width="100%" height="320" style="max-width: 360px; margin: 10px auto; display: block;">
            <!-- 网格 -->
            <circle cx="0" cy="0" r="100" fill="none" stroke="var(--border)" stroke-dasharray="2 4" />
            <line x1="-140" y1="0" x2="140" y2="0" stroke="var(--border)" />
            <line x1="0" y1="-140" x2="0" y2="140" stroke="var(--border)" />
            <text x="140" y="-6" font-size="10" fill="var(--text-dim)" font-family="SF Mono, monospace">Re</text>
            <text x="6" y="-140" font-size="10" fill="var(--text-dim)" font-family="SF Mono, monospace">Im</text>
            <!-- 4 个维度对, 不同频率, 初始在 (1,0) -->
            <g v-for="(d, i) in 4" :key="i">
              <line x1="0" y1="0"
                    :x2="100 * Math.cos(m * freq(i))"
                    :y2="-100 * Math.sin(m * freq(i))"
                    :stroke="dimColor(i)" stroke-width="2" />
              <circle :cx="100 * Math.cos(m * freq(i))"
                      :cy="-100 * Math.sin(m * freq(i))"
                      r="5" :fill="dimColor(i)" />
              <text :x="110 * Math.cos(m * freq(i))"
                    :y="-110 * Math.sin(m * freq(i))"
                    font-size="10" fill="var(--text-muted)"
                    font-family="SF Mono, monospace">d{{ i * 2 }}</text>
            </g>
          </svg>

          <p class="desc" style="font-size: 12px;">
            低维 (d0) 频率高, 转得快, 对局部位置敏感; 高维 (d6) 频率低, 转得慢, 对远距离敏感。
          </p>
        </div>

        <div class="card">
          <h3>相对位置的关键证明 <span class="tag">核心直觉</span></h3>
          <p class="desc" style="margin-bottom: 12px;">把两个 token 分别旋转后做内积, 结果只依赖位置差 (m - n):</p>
          <pre class="code" style="font-size: 13px; line-height: 1.9;">Q_m = R(mθ) · q
K_n = R(nθ) · k

&lt;Q_m, K_n&gt; = &lt;R(mθ)·q, R(nθ)·k&gt;
           = q^T · R(mθ)^T · R(nθ) · k
           = q^T · R((n-m)θ) · k      ← 只依赖 (n-m)</pre>

          <p class="desc" style="margin-top: 12px;">换句话说: <strong>旋转后的 Q 和 K 做内积, 自动编码了相对距离</strong>, 不需要任何可学参数, 也天然支持长度外推。这就是为什么现代 LLM 几乎全面切换到 RoPE。</p>

          <div class="form-row" style="margin-top: 16px;">
            <label>Q 位置 m</label>
            <input type="range" min="0" max="32" step="1" v-model.number="qPos" />
            <span class="val">{{ qPos }}</span>
          </div>
          <div class="form-row">
            <label>K 位置 n</label>
            <input type="range" min="0" max="32" step="1" v-model.number="kPos" />
            <span class="val">{{ kPos }}</span>
          </div>

          <div class="stat" style="margin-top: 10px;">
            <div class="k">模拟相似度 (低维对)</div>
            <div class="v accent">{{ cosSim.toFixed(3) }}</div>
            <div class="hint">相对距离 |m - n| = {{ Math.abs(qPos - kPos) }}; 与绝对位置 m, n 无关</div>
          </div>
        </div>
      </div>
    </div>

    <div v-else>
      <!-- M-RoPE -->
      <div class="grid grid-2">
        <div class="card">
          <h3>M-RoPE: 三轴位置 <span class="tag">Qwen2-VL · 2024</span></h3>
          <p class="desc" style="margin-bottom: 16px;">
            视觉 patch 本质是 2D 网格 (时间 × 高 × 宽)。M-RoPE 把 head_dim 切成三段,
            每段用对应轴的位置独立 RoPE, 让同一个 attention 能同时处理:
          </p>
          <ul class="desc" style="padding-left: 16px; line-height: 1.9; list-style: disc;">
            <li><strong>文本 token</strong>: (t, t, t) 三轴相同 → 退化为 1D RoPE</li>
            <li><strong>视觉 patch</strong>: (t, h, w) 三轴互不相同 → 空间结构被保留</li>
          </ul>

          <div class="trade" style="margin-top: 16px;">
            <div class="tr-item">
              <span class="ok">✓</span>
              <span>视觉 + 文本共享同一个 decoder, 不需要额外 cross-attention</span>
            </div>
            <div class="tr-item">
              <span class="ok">✓</span>
              <span>动态分辨率友好, 不同尺寸 patch 网格都能用同一套编码</span>
            </div>
          </div>
        </div>

        <div class="card">
          <h3>位置索引示意</h3>
          <p class="desc" style="margin-bottom: 12px;">下图展示文本 "Hi" 后接一张 3×3 的 patch 网格, 再接文本 "there" 时, 每 token 的 (t, h, w) 位置索引:</p>
          <div class="mrope-grid mono">
            <!-- 文本 "Hi" -->
            <div v-for="(tok, i) in ['H', 'i']" :key="'t1-'+i" class="tok text">
              <div class="ch">{{ tok }}</div>
              <div class="ids">({{ i }}, {{ i }}, {{ i }})</div>
            </div>
            <!-- 3x3 视觉 grid -->
            <div v-for="r in 3" :key="'row-'+r" class="viz-row">
              <div v-for="c in 3" :key="'cell-'+r+c" class="tok viz">
                <div class="ch">■</div>
                <div class="ids">(2, {{ r - 1 }}, {{ c - 1 }})</div>
              </div>
            </div>
            <!-- 后续文本 "there" -->
            <div v-for="(tok, i) in ['t', 'h', 'e', 'r', 'e']" :key="'t2-'+i" class="tok text">
              <div class="ch">{{ tok }}</div>
              <div class="ids">({{ 3 + i }}, {{ 3 + i }}, {{ 3 + i }})</div>
            </div>
          </div>
          <p class="desc" style="font-size: 12px; margin-top: 10px;">
            视觉 9 个 patch 在时间轴上都占位 2 (下一文本 token 从 3 开始), h/w 轴按网格填充。
          </p>
        </div>
      </div>
    </div>

    <section class="section">
      <h2>代码路径</h2>
      <p class="lead">本页对应 <code class="inline">llm_models/layers/core/position_encoding.py</code> 的三个类 — 注意它们都在 <code class="inline">core/</code> 子包内, 因为任何 Transformer 都会复用其中之一。</p>
      <div class="grid grid-3">
        <div class="card"><h3 style="font-size: 13px;">SinPositionalEncoding</h3><p class="desc">原始 Transformer, 加在 embedding 上。绝对位置, 长度外推差。</p></div>
        <div class="card"><h3 style="font-size: 13px;">RotaryPositionalEncoding</h3><p class="desc">对 Q/K 做旋转, 编码相对位置。无可学参数, 长度可外推。</p></div>
        <div class="card"><h3 style="font-size: 13px;">MultimodalRotaryEmbedding</h3><p class="desc">head_dim 切 T/H/W 三段独立 RoPE; 文本 token 三轴相同时退化为标准 RoPE。</p></div>
      </div>
    </section>

    <ChapterNav
      :prev="{ name: 'attention', label: '注意力的四代演进', hint: 'KV cache 的瓶颈与解法' }"
      :next="{ name: 'blocks', label: 'Block 组装器', hint: '把 attention + ffn + norm + pos 拼起来, 数模型差异' }"
    />
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import ChapterIntro from '@/components/ChapterIntro.vue'
import ChapterNav from '@/components/ChapterNav.vue'
import EvolutionChain from '@/components/EvolutionChain.vue'

const variants = [
  { id: 'sin',   label: 'Sinusoidal (2017)' },
  { id: 'rope',  label: 'RoPE (2021)' },
  { id: 'mrope', label: 'M-RoPE (2024)' },
]
const variant = ref('rope')

const evoSteps = [
  { name: 'Sinusoidal', year: 2017, color: '#9ca3af',
    pain: '(原点) 没位置编码 = bag of tokens',
    fix: '正余弦不同频率 → 加在 embedding 上, 绝对位置' },
  { name: 'Learnable', year: 2018, color: '#60a5fa',
    pain: 'Sin 公式固定, 不一定贴合数据',
    fix: '每个位置一个可学向量 (BERT/ViT) — 但训不动训练长度之外' },
  { name: 'RoPE', year: 2021, color: '#34d399',
    pain: '加在 embedding 上 → V 也带位置, attention 模型不需要',
    fix: '把 Q/K 视作复数旋转 e^(imθ) → 内积自动只剩 (m−n), 无参数 + 可外推' },
  { name: 'M-RoPE', year: 2024, color: '#f5a623',
    pain: '视觉 patch 是 2D 网格, 文本 RoPE 一维不够用',
    fix: 'head_dim 切 (T, H, W) 三段独立旋转; 文本退化为 1D RoPE → 多模态共用 decoder' },
]

// --- Sinusoidal 曲线 ---
const sinLines = computed(() => {
  const lines = []
  const W = 480, H = 320, dModel = 16, positions = 80
  for (let i = 0; i < dModel; i++) {
    const freq = 1 / Math.pow(10000, (2 * Math.floor(i / 2)) / 64)
    const pts = []
    for (let p = 0; p < positions; p++) {
      const v = i % 2 === 0 ? Math.sin(p * freq) : Math.cos(p * freq)
      const x = (p / (positions - 1)) * W
      const y = H / 2 + (H / 2 - 16) * -v * 0.85 * (1 - i * 0.04)
      pts.push(`${p === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`)
    }
    lines.push(pts.join(' '))
  }
  return lines
})
const sinColor = (idx) => {
  const hues = [175, 195, 220, 250, 280, 310, 340, 10]
  return `hsl(${hues[idx % hues.length]}, 70%, 65%)`
}

// --- RoPE 旋转 ---
const m = ref(12)
const qPos = ref(5)
const kPos = ref(9)
// 4 个维度对的频率 (对数衰减)
const freq = (i) => 0.5 * Math.pow(0.3, i) // i=0 快, i=3 慢
const dimColor = (i) => ['#7c6bf1', '#3dd68c', '#f5a623', '#ec4899'][i]

// 模拟相似度: 假设 q = k = (1, 0) 向量, 旋转后内积 = cos(差角)
const cosSim = computed(() => {
  const diff = kPos.value - qPos.value
  return Math.cos(diff * freq(0))
})

// --- 代码 ---
const sinCode = `def forward(x):
    # pe[pos, 2i  ] = sin(pos / 10000^(2i/d))
    # pe[pos, 2i+1] = cos(pos / 10000^(2i/d))
    return x + pe[:, :x.size(1)]`

function highlight(s) {
  return s
    .replace(/#.*$/gm, m => `<span class="cm">${m}</span>`)
    .replace(/\b(def|return|for|in|None|if|else|self)\b/g, '<span class="kw">$1</span>')
    .replace(/\b\d+\b/g, m => `<span class="num">${m}</span>`)
}
</script>

<style scoped>
.mrope-grid {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
  padding: 14px;
  background: var(--bg-elev);
  border-radius: 6px;
}
.viz-row {
  display: inline-flex;
  gap: 6px;
  flex-direction: row;
}
.tok {
  min-width: 50px;
  padding: 6px 8px;
  border-radius: 5px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  text-align: center;
  font-size: 11px;
}
.tok.viz { border-color: color-mix(in srgb, var(--eye) 50%, var(--border)); }
.tok.text { border-color: color-mix(in srgb, var(--accent) 40%, var(--border)); }
.tok .ch { font-size: 14px; color: var(--text); margin-bottom: 3px; }
.tok .ids { font-size: 10px; color: var(--text-muted); }
</style>
