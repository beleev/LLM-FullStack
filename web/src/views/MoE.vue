<template>
  <div>
    <h1 class="page-title">MoE 路由可视化</h1>
    <p class="page-subtitle">
      MoE (Mixture of Experts) 把一个巨大 FFN 切成 E 个小 FFN, 每个 token 只激活 top-k 个,
      用 "E/k 倍参数量、k 倍算力" 换取容量与算力的解耦。本页并排对比两种范式:
      <strong>Mixtral (softmax + 外部 aux)</strong> vs <strong>DeepSeek (sigmoid + 共享专家 + aux-free bias)</strong>。
    </p>

    <ChapterIntro
      tldr="MoE = 给 ffn 槽位换上「E 个小 FFN + 一个 router」。Mixtral 与 DeepSeek 不是版本号差异, 而是两套不同的路由哲学。"
      question="专家变多以后, 如何避免 router 把所有 token 都送到那几个明星专家? 让训练 loss 不被「负载均衡」污染?"
      :goals="[
        '看懂 MoE 是怎么把单 FFN 替换成 router + E 个小 FFN',
        '区分 Mixtral 的 softmax top-k 与 DeepSeek 的 sigmoid + shared 路由',
        '理解 aux-free 路由怎么避免「明星专家」吃掉所有 token',
      ]"
      :codes="[
        { path: 'llm_models/layers/sparse/moe.py', label: 'moe.py · MixtralMoE' },
        { path: 'llm_models/models/moe/deepseekV3.py', label: 'deepseekV3.py · DeepSeekMoE' },
      ]"
      :prereq="{ name: 'blocks', label: 'Block 组装器 — 看清 ffn 槽位' }"
      :next-step="{ name: 'diffusion', label: '扩散生成 — 走另一条主线' }"
    />

    <!-- 控制条 -->
    <div class="card" style="margin-bottom: 20px;">
      <div class="controls">
        <div class="form-row">
          <label>变体</label>
          <div class="btn-group" style="grid-column: span 2;">
            <button :class="{ active: variant === 'mixtral' }" @click="variant = 'mixtral'">Mixtral (softmax)</button>
            <button :class="{ active: variant === 'deepseek' }" @click="variant = 'deepseek'">DeepSeek (sigmoid)</button>
          </div>
        </div>
        <div class="form-row">
          <label>专家总数 E</label>
          <input type="range" min="4" max="32" step="2" v-model.number="numExperts" />
          <span class="val">{{ numExperts }}</span>
        </div>
        <div class="form-row">
          <label>top-k</label>
          <input type="range" :min="1" :max="Math.max(2, numExperts / 2)" step="1" v-model.number="topK" />
          <span class="val">{{ topK }}</span>
        </div>
        <div v-if="variant === 'deepseek'" class="form-row">
          <label>共享专家</label>
          <input type="range" min="0" max="4" step="1" v-model.number="numShared" />
          <span class="val">{{ numShared }}</span>
        </div>
        <div class="form-row">
          <label>Token 数</label>
          <input type="range" min="8" max="32" step="2" v-model.number="numTokens" />
          <span class="val">{{ numTokens }}</span>
        </div>
        <div class="form-row">
          <label>路由坍塌倾向</label>
          <input type="range" min="0" max="100" step="1" v-model.number="collapseBias" />
          <span class="val">{{ collapseBias }}%</span>
        </div>
      </div>

      <div style="display: flex; gap: 8px; margin-top: 12px;">
        <button @click="regenerate">🎲 重新采样 tokens</button>
        <button @click="animateFlow" :disabled="animating">▷ 播放路由动画</button>
      </div>
    </div>

    <!-- 主可视化: 左 tokens → 中 router → 右 experts -->
    <div class="card">
      <svg :viewBox="`0 0 ${W} ${H}`" width="100%" :height="H">
        <defs>
          <linearGradient id="token-grad" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0" stop-color="#7c6bf1" stop-opacity="0.8" />
            <stop offset="1" stop-color="#ec4899" stop-opacity="0.8" />
          </linearGradient>
        </defs>

        <!-- Tokens -->
        <g>
          <text x="50" y="24" fill="var(--text-muted)" font-size="12" font-family="SF Mono">tokens (batch)</text>
          <g v-for="(t, i) in tokens" :key="'t-'+i">
            <rect :x="20" :y="40 + i * tokenGap"
                  width="80" height="22" rx="4"
                  fill="var(--bg-elev)" stroke="var(--border)" />
            <text :x="60" :y="56 + i * tokenGap"
                  text-anchor="middle" fill="var(--text)" font-size="11" font-family="SF Mono">
              t{{ i }}
            </text>
          </g>
        </g>

        <!-- Router -->
        <g>
          <rect :x="routerX" :y="40" width="100" :height="H - 60" rx="6"
                fill="var(--bg-elev)" stroke="var(--accent)" stroke-opacity="0.6" />
          <text :x="routerX + 50" y="30" text-anchor="middle" fill="var(--accent)" font-size="12" font-weight="600">Router</text>
          <text :x="routerX + 50" :y="H - 10" text-anchor="middle"
                fill="var(--text-dim)" font-size="10" font-family="SF Mono">
            {{ variant === 'mixtral' ? 'softmax' : 'sigmoid + bias' }}
          </text>
        </g>

        <!-- 路由连线 -->
        <g>
          <path v-for="(edge, i) in edges" :key="'e-'+i"
                :d="edge.d"
                :stroke="edge.color"
                :stroke-opacity="edge.opacity"
                :stroke-width="edge.width"
                fill="none"
                :class="{ flowing: animating }" />
        </g>

        <!-- Experts -->
        <g>
          <g v-for="(e, i) in displayedExperts" :key="'exp-'+i">
            <rect :x="expertX" :y="40 + i * expertGap"
                  :width="140" :height="expertGap - 8"
                  rx="4"
                  :fill="e.shared ? 'rgba(245, 166, 35, 0.08)' : 'var(--bg-elev)'"
                  :stroke="e.shared ? 'var(--eye)' : (e.load > 0 ? 'var(--accent)' : 'var(--border)')"
                  :stroke-opacity="e.shared ? 0.8 : (0.4 + Math.min(1, e.load / maxLoad) * 0.6)" />
            <text :x="expertX + 10" :y="55 + i * expertGap"
                  fill="var(--text)" font-size="11" font-family="SF Mono">
              {{ e.shared ? 'Shared' : 'E' + i }}
            </text>
            <!-- load bar -->
            <rect :x="expertX + 60" :y="48 + i * expertGap"
                  :width="60" height="6" rx="2" fill="var(--border)" />
            <rect :x="expertX + 60" :y="48 + i * expertGap"
                  :width="60 * Math.min(1, e.load / maxLoad)" height="6" rx="2"
                  :fill="e.shared ? 'var(--eye)' : e.load > avgLoad * 1.8 ? 'var(--danger)' : 'var(--accent)'" />
            <text :x="expertX + 128" :y="54 + i * expertGap"
                  fill="var(--text-muted)" font-size="10" font-family="SF Mono"
                  text-anchor="end">
              {{ e.load }}
            </text>
          </g>
          <text :x="expertX + 70" y="24" text-anchor="middle" fill="var(--text-muted)" font-size="12" font-family="SF Mono">experts (load)</text>
        </g>
      </svg>

      <!-- 负载分析 -->
      <div class="load-analysis">
        <div class="stat">
          <div class="k">负载标准差</div>
          <div class="v" :style="{ color: loadStdColor }">{{ loadStd.toFixed(2) }}</div>
          <div class="hint">越低表示路由越均衡</div>
        </div>
        <div class="stat">
          <div class="k">最热门专家</div>
          <div class="v">E{{ hottestExpert.idx }}</div>
          <div class="hint">承载 {{ Math.round(hottestExpert.load / totalRoutedAssigns * 100) }}% 的路由</div>
        </div>
        <div class="stat">
          <div class="k">激活比例</div>
          <div class="v accent">{{ (topK / numExperts * 100).toFixed(1) }}%</div>
          <div class="hint">top-k / 总专家, 算力占比</div>
        </div>
      </div>
    </div>

    <!-- 对比表 -->
    <section class="section">
      <h2>Mixtral vs DeepSeek MoE 差异</h2>
      <div class="card">
        <table class="compare">
          <thead>
            <tr>
              <th></th>
              <th>Mixtral (2024)</th>
              <th>DeepSeek-V3 (2024)</th>
            </tr>
          </thead>
          <tbody>
            <tr><td class="row-label">路由打分</td>
              <td><code class="inline">softmax(logits)</code></td>
              <td><code class="inline">sigmoid(logits)</code></td>
            </tr>
            <tr><td class="row-label">专家归一化</td>
              <td>softmax 本身归一</td>
              <td>top-k 后再 renormalize</td>
            </tr>
            <tr><td class="row-label">共享专家</td>
              <td>❌ 无</td>
              <td>✅ 始终激活 (托底通用能力)</td>
            </tr>
            <tr><td class="row-label">负载均衡</td>
              <td>Switch-style 外部 aux loss</td>
              <td>aux-loss-free bias (不污染主 loss)</td>
            </tr>
            <tr><td class="row-label">专家粒度</td>
              <td>8 个大专家</td>
              <td>64–256 个细粒度专家</td>
            </tr>
            <tr><td class="row-label">代码位置</td>
              <td><RepoLink path="llm_models/layers/sparse/moe.py" label="layers/sparse/moe.py" tiny /></td>
              <td><RepoLink path="llm_models/models/moe/deepseekV3.py" label="models/moe/deepseekV3.py" tiny /></td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <ChapterNav
      :prev="{ name: 'blocks', label: 'Block 组装器', hint: 'MoE 只是 ffn 槽位的一种填法' }"
      :next="{ name: 'diffusion', label: '扩散生成', hint: '换一条主线: 用 attention 学「去噪」而不是「下一个 token」' }"
    />
  </div>
</template>

<script setup>
import { ref, reactive, computed, watch, onMounted } from 'vue'
import ChapterIntro from '@/components/ChapterIntro.vue'
import ChapterNav from '@/components/ChapterNav.vue'
import RepoLink from '@/components/RepoLink.vue'

const variant = ref('deepseek')
const numExperts = ref(8)
const topK = ref(2)
const numShared = ref(2)
const numTokens = ref(16)
const collapseBias = ref(30)  // 0–100: 越大越倾向路由坍塌

const animating = ref(false)

// 画布几何
const W = 900
const tokenGap = 28
const expertGap = 32
const tokenX = 100
const routerX = 280
const expertX = 520

const H = computed(() => Math.max(numTokens.value, numExperts.value + numShared.value) * Math.max(tokenGap, expertGap) + 80)

// --- Tokens 与路由计算 ---
const tokens = ref([])

function makeToken() {
  // 每个 token 一组对各专家的 "真实偏好"
  // collapseBias 越大, 让多数 token 都偏好前几个专家
  const E = numExperts.value
  const scores = []
  const collapse = collapseBias.value / 100
  for (let e = 0; e < E; e++) {
    // 基础随机 + 向前 2 个专家倾斜
    const base = Math.random()
    const bias = e < 2 ? collapse * 1.8 : 0
    scores.push(base + bias + (Math.random() - 0.5) * 0.3)
  }
  return { scores }
}

function regenerate() {
  tokens.value = Array.from({ length: numTokens.value }, makeToken)
}

// 监听 token 数量和参数变化重新生成
watch([numTokens, numExperts, collapseBias], regenerate, { immediate: true })
onMounted(regenerate)

// --- 路由决策 ---
const routingDecisions = computed(() => {
  return tokens.value.map((t, i) => {
    let probs
    if (variant.value === 'mixtral') {
      // softmax
      const max = Math.max(...t.scores)
      const exp = t.scores.map(s => Math.exp(s - max))
      const sum = exp.reduce((a, b) => a + b, 0)
      probs = exp.map(e => e / sum)
    } else {
      // sigmoid independent
      probs = t.scores.map(s => 1 / (1 + Math.exp(-s)))
    }
    // top-k
    const withIdx = probs.map((p, e) => ({ p, e }))
    const top = withIdx.sort((a, b) => b.p - a.p).slice(0, topK.value)
    // renormalize
    const totalP = top.reduce((a, x) => a + x.p, 0)
    const weights = top.map(x => ({ e: x.e, w: x.p / totalP }))
    return { token: i, weights }
  })
})

// 专家负载
const displayedExperts = computed(() => {
  const E = numExperts.value
  const shared = variant.value === 'deepseek' ? numShared.value : 0
  const result = []
  // routed first
  for (let e = 0; e < E; e++) {
    const load = routingDecisions.value.reduce((acc, r) =>
      acc + r.weights.filter(w => w.e === e).length, 0)
    result.push({ idx: e, load, shared: false })
  }
  // shared
  for (let s = 0; s < shared; s++) {
    result.push({ idx: 's' + s, load: numTokens.value, shared: true })
  }
  return result
})

const maxLoad = computed(() => Math.max(1, ...displayedExperts.value.map(e => e.load)))
const avgLoad = computed(() =>
  (numTokens.value * topK.value) / numExperts.value
)
const loadStd = computed(() => {
  const routed = displayedExperts.value.filter(e => !e.shared)
  const mean = routed.reduce((a, e) => a + e.load, 0) / routed.length
  const variance = routed.reduce((a, e) => a + Math.pow(e.load - mean, 2), 0) / routed.length
  return Math.sqrt(variance)
})
const loadStdColor = computed(() => {
  const v = loadStd.value
  if (v < 1) return 'var(--left)'
  if (v < 3) return 'var(--warn)'
  return 'var(--danger)'
})
const hottestExpert = computed(() => {
  const routed = displayedExperts.value.filter(e => !e.shared)
  return routed.reduce((max, e) => e.load > max.load ? e : max, { load: 0, idx: 0 })
})
const totalRoutedAssigns = computed(() => numTokens.value * topK.value)

// --- 边的绘制 ---
const edges = computed(() => {
  const result = []
  routingDecisions.value.forEach((r, ti) => {
    const tx = tokenX + 0  // right side of token
    const ty = 40 + ti * tokenGap + 11
    const rx1 = routerX
    // Token → router
    result.push({
      d: `M ${tx} ${ty} L ${rx1} ${ty}`,
      color: 'var(--border-strong)',
      opacity: 0.4,
      width: 1,
    })
    // Router → selected experts
    r.weights.forEach(({ e, w }) => {
      const ex = expertX
      const ey = 40 + e * expertGap + 11
      const mid = (routerX + 100 + ex) / 2
      result.push({
        d: `M ${routerX + 100} ${ty} C ${mid} ${ty}, ${mid} ${ey}, ${ex} ${ey}`,
        color: 'var(--accent)',
        opacity: 0.2 + w * 0.6,
        width: 1 + w * 2,
      })
    })
    // Shared expert connections (always on, dashed)
    if (variant.value === 'deepseek') {
      for (let s = 0; s < numShared.value; s++) {
        const ex = expertX
        const ey = 40 + (numExperts.value + s) * expertGap + 11
        const mid = (routerX + 100 + ex) / 2
        result.push({
          d: `M ${routerX + 100} ${ty} C ${mid} ${ty}, ${mid} ${ey}, ${ex} ${ey}`,
          color: 'var(--eye)',
          opacity: 0.15,
          width: 1,
        })
      }
    }
  })
  return result
})

async function animateFlow() {
  animating.value = true
  await new Promise(r => setTimeout(r, 1200))
  animating.value = false
}
</script>

<style scoped>
.controls {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px 24px;
}
@media (max-width: 960px) {
  .controls { grid-template-columns: 1fr; }
}

.load-analysis {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
  margin-top: 16px;
}

table.compare {
  width: 100%;
  border-collapse: collapse;
}
table.compare th, table.compare td {
  padding: 10px 12px;
  text-align: left;
  border-bottom: 1px solid var(--border);
  font-size: 13px;
}
table.compare th {
  color: var(--text-muted);
  font-size: 11px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.6px;
}
table.compare .row-label {
  color: var(--text-muted);
  font-size: 12px;
  width: 130px;
}

.flowing {
  stroke-dasharray: 4 4;
  animation: flow 1s linear infinite;
}
@keyframes flow {
  to { stroke-dashoffset: -16; }
}
</style>
