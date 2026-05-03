<template>
  <div>
    <h1 class="page-title">扩散生成 · DDPM 与 Flow Matching</h1>
    <p class="page-subtitle">
      扩散模型学 "如何把噪声一步步去掉"。拖动下面的 timestep 滑条, 看 x_0 从清晰到纯噪声的连续过程,
      并观察 <strong>DDPM (ε-pred, cosine schedule)</strong> 与
      <strong>Flow Matching (v-pred, linear path)</strong> 两种范式的形状差异。
    </p>

    <ChapterIntro
      tldr="扩散把生成问题变成「学习去噪函数」。DDPM 学预测噪声 ε, Flow Matching 学预测速度 v=ε−x₀ — 后者把弯路拉成直线。"
      question="如果训练目标只换了一个 (ε → v), 推理步数为什么能从 50 降到 28? 这跟「直线路径」的几何意义是什么?"
      :goals="[
        '理解扩散模型把生成问题变成「学习去噪」的完整逻辑',
        '区分 ε-prediction (DDPM) 与 v-prediction (Flow Matching) 的几何意义',
        '看懂 DiT + adaLN-Zero 怎么把扩散嫁接到 Transformer 骨架上',
      ]"
      :codes="[
        { path: 'llm_models/layers/diffusion/adaln.py', label: 'adaln.py · AdaLNZeroBlock' },
        { path: 'llm_models/training/diffusion.py', label: 'training/diffusion.py' },
      ]"
      :prereq="{ name: 'moe', label: 'MoE 路由 (左脑分支)' }"
      :next-step="{ name: 'train', label: '阶段 3 — 把训练循环扩到分布式' }"
    />

    <!-- 控制 -->
    <div class="card" style="margin-bottom: 20px;">
      <div class="controls-grid">
        <div>
          <div class="slot-title">调度器</div>
          <div class="btn-group">
            <button :class="{ active: scheduler === 'ddpm' }" @click="scheduler = 'ddpm'">DDPM (cosine)</button>
            <button :class="{ active: scheduler === 'fm' }" @click="scheduler = 'fm'">Flow Matching (linear)</button>
          </div>
        </div>
        <div>
          <div class="slot-title">预测目标</div>
          <div class="form-val mono">
            {{ scheduler === 'ddpm' ? 'target = ε (noise)' : 'target = v = ε − x₀' }}
          </div>
        </div>
      </div>
      <div class="form-row" style="margin-top: 14px;">
        <label>timestep t</label>
        <input type="range" min="0" max="1" step="0.01" v-model.number="t" />
        <span class="val">{{ t.toFixed(2) }}</span>
      </div>
      <div class="btn-group" style="margin-top: 10px;">
        <button @click="animate" :disabled="animating">{{ animating ? '播放中…' : '▷ 播放完整去噪' }}</button>
        <button @click="t = 1">置为纯噪声 (t=1)</button>
        <button @click="t = 0">置为原图 (t=0)</button>
      </div>
    </div>

    <!-- 主可视化: 三格 x_0 / x_t / 预测 -->
    <div class="grid grid-3">
      <div class="card panel">
        <h3>x₀ <span class="tag">clean</span></h3>
        <p class="desc">ground truth 原图 (教学用的合成环形图案)</p>
        <canvas ref="canvasX0" width="128" height="128" class="canvas" />
        <div class="stat" style="margin-top: 10px;">
          <div class="k">σ(x) · 信号强度</div>
          <div class="v mono">{{ sigmaX0.toFixed(3) }}</div>
        </div>
      </div>

      <div class="card panel">
        <h3>x_t <span class="tag">noised</span></h3>
        <p class="desc">{{ noiseFormula }}</p>
        <canvas ref="canvasXt" width="128" height="128" class="canvas" />
        <div class="stat" style="margin-top: 10px;">
          <div class="k">√ᾱ_t / √(1−ᾱ_t)</div>
          <div class="v mono">{{ alphaBar.toFixed(3) }} / {{ sqrt1ma.toFixed(3) }}</div>
        </div>
      </div>

      <div class="card panel">
        <h3>{{ scheduler === 'ddpm' ? 'ε̂' : 'v̂' }} <span class="tag">model pred</span></h3>
        <p class="desc">模型要学习的目标 (此处用真值 + 小噪声模拟)</p>
        <canvas ref="canvasPred" width="128" height="128" class="canvas" />
        <div class="stat" style="margin-top: 10px;">
          <div class="k">MSE(pred, target)</div>
          <div class="v mono">{{ mseLoss.toFixed(4) }}</div>
        </div>
      </div>
    </div>

    <!-- schedule 对比曲线 -->
    <section class="section">
      <h2>噪声调度曲线</h2>
      <p class="lead">
        DDPM 用 cosine β 让 <span class="mono">ᾱ_t</span> 的两端更平滑, 相比 linear 能避免 T→1 时过度噪声。
        Flow Matching 则是直线路径 <span class="mono">x_t = (1-t)·x₀ + t·ε</span>, 训练极简、推理步数更少。
      </p>
      <div class="card">
        <svg viewBox="0 0 600 240" width="100%" height="240">
          <!-- 坐标轴 -->
          <line x1="40" y1="20" x2="40" y2="200" stroke="var(--border)" />
          <line x1="40" y1="200" x2="580" y2="200" stroke="var(--border)" />
          <text x="30" y="24" text-anchor="end" fill="var(--text-dim)" font-size="10" font-family="SF Mono">1</text>
          <text x="30" y="204" text-anchor="end" fill="var(--text-dim)" font-size="10" font-family="SF Mono">0</text>
          <text x="40" y="220" text-anchor="start" fill="var(--text-dim)" font-size="10" font-family="SF Mono">t=0</text>
          <text x="580" y="220" text-anchor="end" fill="var(--text-dim)" font-size="10" font-family="SF Mono">t=1</text>

          <!-- DDPM cosine curve -->
          <path :d="ddpmCurve" fill="none" stroke="#7c6bf1" stroke-width="2" />
          <!-- FM linear curve -->
          <path :d="fmCurve" fill="none" stroke="#3dd68c" stroke-width="2" stroke-dasharray="4 4" />

          <!-- 当前 t 标记 -->
          <line :x1="40 + t * 540" :x2="40 + t * 540" y1="20" y2="200"
                stroke="var(--warn)" stroke-width="1.5" />
          <circle :cx="40 + t * 540"
                  :cy="200 - 180 * alphaBarCurve(t)"
                  r="5"
                  :fill="scheduler === 'ddpm' ? '#7c6bf1' : '#3dd68c'" />

          <!-- Legend -->
          <g transform="translate(420, 30)">
            <line x1="0" y1="0" x2="20" y2="0" stroke="#7c6bf1" stroke-width="2" />
            <text x="26" y="4" fill="var(--text-muted)" font-size="11">DDPM ᾱ_t (cosine)</text>
            <line x1="0" y1="18" x2="20" y2="18" stroke="#3dd68c" stroke-width="2" stroke-dasharray="4 4" />
            <text x="26" y="22" fill="var(--text-muted)" font-size="11">FM (1-t) 线性</text>
          </g>
        </svg>
      </div>
    </section>

    <!-- 关键公式 -->
    <section class="section">
      <h2>核心公式速查</h2>
      <div class="grid grid-2">
        <div class="card">
          <h3>DDPM (Ho et al., 2020)</h3>
          <pre class="code">x_t = √ᾱ_t · x₀ + √(1-ᾱ_t) · ε
ε ~ N(0, I)
target = ε                              # 预测噪声
loss = MSE(model(x_t, t), ε)

# 推理 (DDIM, 确定性少步):
x₀̂ = (x_t - √(1-ᾱ_t)·ε̂) / √ᾱ_t
x_{t-1} = √ᾱ_{t-1}·x₀̂ + √(1-ᾱ_{t-1})·ε̂</pre>
        </div>
        <div class="card">
          <h3>Flow Matching (Lipman et al., 2023)</h3>
          <pre class="code">x_t = (1-t)·x₀ + t·ε          # 线性路径
v_t = dx_t/dt = ε - x₀        # velocity
target = v                     # 预测速度
loss = MSE(model(x_t, t), v)

# 推理 (Euler ODE, 极少步):
v̂ = model(x_t, t)
x_{t-Δt} = x_t - Δt·v̂         # 直线反推</pre>
        </div>
      </div>
    </section>

    <div class="card" style="margin-top: 20px;">
      <h3>为什么 DiT/MM-DiT/Sora 都换成了 Flow Matching?</h3>
      <p class="desc">
        SD3 / FLUX / HunyuanVideo / Wan 2.2 等 2024 年后的 SOTA 生图/生视频全部从
        ε-pred 切到 v-pred + Rectified Flow, 原因有三:
      </p>
      <ul class="desc" style="padding-left: 20px; list-style: disc; line-height: 1.9; margin-top: 6px;">
        <li><strong>训练更稳</strong>: 线性路径的 velocity 目标量级始终接近, 不像 ε 在 t 接近 0 时方差剧烈变化</li>
        <li><strong>推理更快</strong>: 直线路径意味着欧拉法几步就能走完, SD3 推荐 28 步 (vs DDIM 50 步)</li>
        <li><strong>数学更简洁</strong>: 无 noise schedule 可调, 超参数更少, 复现更可靠</li>
      </ul>
    </div>

    <ChapterNav
      :prev="{ name: 'moe', label: 'MoE 路由', hint: '语言侧的稀疏化与生成侧的连续化是两种「减算力」哲学' }"
      :next="{ name: 'train', label: '阶段 3 · 规模化训练', hint: '从模型结构进入分布式训练主循环' }"
    />
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, onBeforeUnmount } from 'vue'
import ChapterIntro from '@/components/ChapterIntro.vue'
import ChapterNav from '@/components/ChapterNav.vue'

const t = ref(0.5)
const scheduler = ref('ddpm')
const animating = ref(false)

const canvasX0 = ref(null)
const canvasXt = ref(null)
const canvasPred = ref(null)

// --- ᾱ_t curves ---
const s = 0.008
function alphaBarCosine(tau) {
  const f = (x) => Math.cos(((x + s) / (1 + s)) * Math.PI / 2) ** 2
  return f(tau) / f(0)
}
function alphaBarCurve(tau) {
  return scheduler.value === 'ddpm' ? alphaBarCosine(tau) : (1 - tau)
}

const alphaBar = computed(() => alphaBarCurve(t.value))
const sqrtAB = computed(() => Math.sqrt(alphaBar.value))
const sqrt1ma = computed(() => Math.sqrt(Math.max(0, 1 - alphaBar.value)))

const ddpmCurve = computed(() => {
  const pts = []
  for (let i = 0; i <= 100; i++) {
    const tau = i / 100
    const v = alphaBarCosine(tau)
    pts.push(`${i === 0 ? 'M' : 'L'} ${40 + tau * 540} ${200 - v * 180}`)
  }
  return pts.join(' ')
})
const fmCurve = computed(() => {
  // 这里画 (1-t), 表示 FM 下 x_0 系数
  return `M 40 20 L 580 200`
})

const noiseFormula = computed(() => {
  if (scheduler.value === 'ddpm') return `x_t = √ᾱ_t·x₀ + √(1-ᾱ_t)·ε`
  return `x_t = (1-t)·x₀ + t·ε`
})

const sigmaX0 = computed(() => 0.412)  // 展示用的固定值

const mseLoss = computed(() => {
  // 模拟: t 大时噪声大, MSE 更高
  return 0.02 + t.value * 0.08 + (scheduler.value === 'ddpm' ? 0.01 : 0)
})

// --- Canvas 渲染 ---
// 画一个合成的环形/辐射图案作为 x_0, 然后根据调度器画 x_t 与 prediction

function drawPattern(ctx, w, h) {
  // 同心圆 + 辐射线 + 渐变
  const grad = ctx.createRadialGradient(w / 2, h / 2, 5, w / 2, h / 2, w * 0.6)
  grad.addColorStop(0, '#fef08a')
  grad.addColorStop(0.4, '#f472b6')
  grad.addColorStop(0.8, '#6366f1')
  grad.addColorStop(1, '#0a0e1a')
  ctx.fillStyle = grad
  ctx.fillRect(0, 0, w, h)

  ctx.strokeStyle = 'rgba(255, 255, 255, 0.35)'
  ctx.lineWidth = 1
  for (let i = 0; i < 12; i++) {
    const angle = i * Math.PI / 6
    ctx.beginPath()
    ctx.moveTo(w / 2, h / 2)
    ctx.lineTo(w / 2 + Math.cos(angle) * w * 0.45, h / 2 + Math.sin(angle) * h * 0.45)
    ctx.stroke()
  }
  for (let r = 10; r < w * 0.5; r += 14) {
    ctx.beginPath()
    ctx.arc(w / 2, h / 2, r, 0, Math.PI * 2)
    ctx.strokeStyle = `rgba(255, 255, 255, ${0.15 + (r / w) * 0.3})`
    ctx.stroke()
  }
}

// 伪随机噪声, 用固定种子保证 t 变化时 noise 一致, 看起来是"同一份噪声逐步加入"
function prng(seed) {
  let s = seed
  return () => {
    s = (s * 9301 + 49297) % 233280
    return s / 233280
  }
}

function drawNoisy(ctx, w, h, alphaBar) {
  // 先画清晰图像
  const temp = document.createElement('canvas')
  temp.width = w; temp.height = h
  const tctx = temp.getContext('2d')
  drawPattern(tctx, w, h)
  const clean = tctx.getImageData(0, 0, w, h)
  // 噪声混合
  const img = ctx.createImageData(w, h)
  const rng = prng(42)
  const sA = Math.sqrt(alphaBar)
  const sN = Math.sqrt(Math.max(0, 1 - alphaBar))
  for (let i = 0; i < clean.data.length; i += 4) {
    // Box-Muller for gaussian noise
    const u1 = rng(), u2 = rng()
    const n = Math.sqrt(-2 * Math.log(Math.max(u1, 1e-9))) * Math.cos(2 * Math.PI * u2)
    const noise = Math.min(255, Math.max(0, 128 + n * 80))
    for (let c = 0; c < 3; c++) {
      img.data[i + c] = clean.data[i + c] * sA + noise * sN
    }
    img.data[i + 3] = 255
  }
  ctx.putImageData(img, 0, 0)
}

function drawPred(ctx, w, h) {
  // 展示: 模型预测的 ε 或 v - 这里画一个噪声样, 加少量原图泄漏, 表示模型没完美学到
  const rng = prng(42)
  const img = ctx.createImageData(w, h)
  for (let i = 0; i < img.data.length; i += 4) {
    const u1 = rng(), u2 = rng()
    const n = Math.sqrt(-2 * Math.log(Math.max(u1, 1e-9))) * Math.cos(2 * Math.PI * u2)
    const v = Math.min(255, Math.max(0, 128 + n * 60))
    img.data[i] = v
    img.data[i + 1] = v * 0.9
    img.data[i + 2] = v * 1.1
    img.data[i + 3] = 255
  }
  ctx.putImageData(img, 0, 0)
}

function renderAll() {
  if (!canvasX0.value) return
  const w = 128, h = 128
  drawPattern(canvasX0.value.getContext('2d'), w, h)
  drawNoisy(canvasXt.value.getContext('2d'), w, h, alphaBar.value)
  drawPred(canvasPred.value.getContext('2d'), w, h)
}

watch([t, scheduler], renderAll)
onMounted(renderAll)

let animTimer = null
async function animate() {
  if (animating.value) return
  animating.value = true
  const steps = 40
  let i = 0
  animTimer = setInterval(() => {
    t.value = Math.max(0, 1 - i / steps)
    i++
    if (i > steps) {
      clearInterval(animTimer)
      animating.value = false
    }
  }, 60)
}
onBeforeUnmount(() => { if (animTimer) clearInterval(animTimer) })
</script>

<style scoped>
.controls-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
}
.form-val {
  font-size: 14px;
  color: var(--text);
  padding: 8px 12px;
  background: var(--bg-elev);
  border-radius: 5px;
  border: 1px solid var(--border);
  margin-top: 2px;
}

.panel .canvas {
  width: 100%;
  max-width: 220px;
  height: auto;
  aspect-ratio: 1;
  margin-top: 12px;
  display: block;
  border-radius: 6px;
  image-rendering: pixelated;
  border: 1px solid var(--border);
}

.slot-title {
  font-size: 11px;
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: 0.7px;
  margin-bottom: 6px;
}
</style>
