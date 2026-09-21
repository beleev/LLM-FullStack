<template>
  <div>
    <h1 class="page-title">位置编码 · 从 Sinusoidal 到 RoPE 再到 M-RoPE</h1>
    <p class="page-subtitle">
      不给位置编码，attention 就是一袋 token：把句子里的词打乱顺序，算出来一模一样。
      本章拖一根滑条就能看明白一件事：<strong>为什么 RoPE 一旋转，Q·K 内积就只剩下「隔了多远」，绝对位置自动消失</strong>。
    </p>

    <ChapterIntro
      tldr="位置信息的注入点一路往后挪: 先加在 embedding 上, 再乘到 Q/K 上, 最后按轴拆开 — 越挪越靠近 attention 真正用得上的地方。"
      question="如果 Q·K 内积只依赖 (m − n)、跟绝对位置 m 和 n 无关, 模型剩下要学的是什么?"
      :goals="[
        '分清绝对位置 / 可学位置 / RoPE / M-RoPE 各自解决什么',
        '自己推一遍 RoPE 为什么让内积只剩相对距离',
        '说清多模态为什么要把位置按三个轴拆开',
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
            每个位置算一个固定向量, 直接加到 token embedding 上。不同维度用不同频率,
            合起来就是这个位置的 "指纹"。它编的是绝对位置, 训练长度之外推不出去。
          </p>
          <pre class="code" v-html="highlight(sinCode)"></pre>
        </div>
        <div class="card">
          <h3>频率谱 <span class="tag">d_model=64</span></h3>
          <p class="desc" style="margin-bottom: 8px;">一条线是一个维度, 横轴是位置。低维波长短、抖得快, 高维波长长、几乎是条斜线。</p>
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
            RoPE 把 Q/K 的相邻两维当成一个复数 <span class="mono">z = a + ib</span>,
            在位置 <span class="mono">m</span> 处乘上 <span class="mono">e^(imθ)</span> —— 也就是转了 <span class="mono">mθ</span> 这么大的角。
            下面 4 个点是 4 个维度对, 频率不同, 转速就不同。拖滑条看它们分开。
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
            低维 (d0) 频率高、转得快, 分得清相邻 token 谁先谁后; 高维 (d6) 频率低、转得慢, 管的是远距离。
            像秒针和时针 —— 长度外推出事的, 正是 "时针" 那一端。
          </p>
        </div>

        <div class="card">
          <h3>相对位置的关键证明 <span class="tag">核心直觉</span></h3>
          <p class="desc" style="margin-bottom: 12px;">两个 token 各自旋转之后再做内积, 绝对位置会被约掉, 只剩位置差 (m - n):</p>
          <pre class="code" style="font-size: 13px; line-height: 1.9;">Q_m = R(mθ) · q
K_n = R(nθ) · k

&lt;Q_m, K_n&gt; = &lt;R(mθ)·q, R(nθ)·k&gt;
           = q^T · R(mθ)^T · R(nθ) · k
           = q^T · R((n-m)θ) · k      ← 只依赖 (n-m)</pre>

          <p class="desc" style="margin-top: 12px;">也就是说: <strong>转过的 Q 和 K 一做内积, 相对距离就自己冒出来了</strong>。没有一个可学参数, 位置往外推也不会遇到 "没训过的参数"。现代 LLM 几乎全换成了 RoPE, 原因就是这个。下面两根滑条随便拖: 只要 |m − n| 不变, 相似度就不动。</p>

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
            视觉 patch 天生是个网格 (时间 × 高 × 宽), 一根一维位置轴装不下。
            M-RoPE 把 head_dim 切成三段, 每段拿对应轴的位置各转各的, 于是同一个 attention 能同时喂两种输入:
          </p>
          <ul class="desc" style="padding-left: 16px; line-height: 1.9; list-style: disc;">
            <li><strong>文本 token</strong>: 三轴 id 都是 (t, t, t) → 严格退化成 1-D RoPE</li>
            <li><strong>视觉 patch</strong>: 三轴 id 各不相同 (t, h, w) → 二维结构保住了</li>
          </ul>

          <div class="trade" style="margin-top: 16px;">
            <div class="tr-item">
              <span class="ok">✓</span>
              <span>图和文共用一个 decoder, 不用再加一套 cross-attention</span>
            </div>
            <div class="tr-item">
              <span class="ok">✓</span>
              <span>动态分辨率随便来: patch 网格多大都是同一套编码</span>
            </div>
          </div>

          <p class="desc" style="margin-top: 14px; font-size: 12px;">
            "退化成 1-D RoPE" 这句以前是打折扣的 —— 本仓库修之前纯文本下还差 3.3e-3。
            现在三轴共用同一条频率轴 (按段分给 T/H/W), demo 实测数值差 0.0e+00, 并且写进了断言。
            但 H/W 分到的是较低频率, 4×4 小网格上未训练时的影响约 1e-5~1e-4, 不是零。
          </p>
        </div>

        <div class="card">
          <h3>位置索引示意</h3>
          <p class="desc" style="margin-bottom: 12px;">文本 "Hi" 后面接一张 3×3 的 patch 网格, 再接文本 "there"。看每个 token 拿到的 (t, h, w):</p>
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
            9 个视觉 patch 在时间轴上都算位置 2 (所以后面的文本从 3 接着数), h/w 轴则按它在网格里的行列填。
          </p>
        </div>
      </div>
    </div>

    <section class="section">
      <h2>代码路径</h2>
      <p class="lead">这三个类都在 <RepoLink path="llm_models/layers/core/position_encoding.py" label="llm_models/layers/core/position_encoding.py" tiny /> 里, 住在 <RepoLink path="llm_models/layers/core/" label="core/" tiny /> 子包下 —— 任何一个 Transformer 都得从中挑一个用。</p>
      <div class="grid grid-3">
        <div class="card"><h3 style="font-size: 13px;">SinPositionalEncoding</h3><p class="desc">原始 Transformer 那一版, 加在 embedding 上。绝对位置, 推不到训练长度之外。</p></div>
        <div class="card"><h3 style="font-size: 13px;">RotaryPositionalEncoding</h3><p class="desc">旋转 Q/K, 内积只剩相对距离。零可学参数, 长度能往外推 (配 YaRN 推得更远)。</p></div>
        <div class="card"><h3 style="font-size: 13px;">MultimodalRotaryEmbedding</h3><p class="desc">head_dim 切成 T/H/W 三段各转各的; 文本三轴 id 相同时, 结果和标准 RoPE 逐位相等。</p></div>
      </div>
    </section>

    <!-- 本章挂载的实验台 (data/labmap/*.js) 与章末自测 (data/quiz/*.js), 没配置时不渲染 -->

    <LabMount />

    <QuizCard />


    <ChapterNav
      :prev="{ name: 'attention', label: '注意力的四代演进', hint: 'KV cache 的瓶颈与解法' }"
      :next="{ name: 'blocks', label: 'Block 组装器', hint: '把 attention + ffn + norm + pos 拼起来, 数模型差异' }"
    />
  </div>
</template>

<script setup>
import LabMount from '@/components/LabMount.vue'
import QuizCard from '@/components/QuizCard.vue'
import { ref, computed } from 'vue'
import ChapterIntro from '@/components/ChapterIntro.vue'
import ChapterNav from '@/components/ChapterNav.vue'
import EvolutionChain from '@/components/EvolutionChain.vue'
import RepoLink from '@/components/RepoLink.vue'

const variants = [
  { id: 'sin',   label: 'Sinusoidal (2017)' },
  { id: 'rope',  label: 'RoPE (2021)' },
  { id: 'mrope', label: 'M-RoPE (2024)' },
]
const variant = ref('rope')

const evoSteps = [
  { name: 'Sinusoidal', year: 2017, color: '#9ca3af',
    pain: '(原点) 不给位置 = 一袋 token, 打乱顺序结果不变',
    fix: '按不同频率算一组正余弦, 加到 embedding 上 — 编的是绝对位置' },
  { name: 'Learnable', year: 2018, color: '#60a5fa',
    pain: 'Sin 是写死的公式, 不一定贴合数据',
    fix: '每个位置配一个可学向量 (BERT / ViT) — 但训练长度之外的位置从没被训过' },
  { name: 'RoPE', year: 2021, color: '#34d399',
    pain: '加在 embedding 上, V 也跟着带了位置 — attention 其实只需要 Q 和 K 带',
    fix: '把 Q/K 的相邻两维当复数, 在位置 m 处旋转 e^(imθ): 内积自动只剩 (m−n), 零参数, 能往外推' },
  { name: 'M-RoPE', year: 2024, color: '#f5a623',
    pain: '视觉 patch 是二维网格, 一根一维的位置轴装不下',
    fix: 'head_dim 切成 (T, H, W) 三段各转各的; 文本三轴 id 相同, 严格退化成 1-D RoPE → 图文共用一个 decoder' },
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
