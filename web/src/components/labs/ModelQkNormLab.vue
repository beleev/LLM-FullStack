<!--
  QK-Norm 实验台 (对应 llm_models/layers/core/attention.py:GroupedQueryAttention 的 qk_norm 开关)。
  只讲一件事: logit = q·k/√d 随 ‖q‖‖k‖ 二次增长, softmax 会塌成 one-hot、梯度消失; 对 q/k 各做一次 RMSNorm 就把 logit 钉死在 ±g²√d 以内。
-->
<template>
  <LabFrame
    title="QK-Norm — 把注意力 logit 关进笼子"
    sub="训练中 q、k 的范数会慢慢长大。拖 '范数倍率' 模拟这个过程: 左边 8×8 因果注意力图越来越 '尖', 点任意一行看那个 query 的 logit 和概率。再打开 QK-Norm 拖同一根滑杆。"
    module="llm_models/layers/core/attention.py"
    run="python -m llm_models.run_models.foundation.attention.train_attention"
    :challenge="{
      ask: '关掉 QK-Norm, 把范数倍率从 1 拖到 4。先猜: 最大 logit 变成几倍? softmax 的 “梯度通过率” 会怎样?',
      answer: 'q、k 各放大 4 倍, 点积放大 16 倍 (二次增长)。logit 差距一拉大, softmax 就塌成 one-hot: 熵 → 0, 通过率 1 − Σp² → 0 —— softmax 的雅可比 diag(p) − ppᵀ 在 one-hot 处是全零矩阵, 这个 head 从此收不到梯度, 而 logit 本身还会在 bf16/fp16 里溢出。QK-Norm 把 q、k 归一到 RMS = 1 再乘可学增益 g: logit = g²·√d·cosθ, 上界 g²√d, 与范数倍率完全无关。模型仍可通过学 g 来调 “尖锐度”, 只是只剩这一个受控的旋钮。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: !qkNorm }" @click="qkNorm = false">无 QK-Norm</button>
        <button type="button" :class="{ active: qkNorm }" @click="qkNorm = true">QK-Norm (q、k 各过 RMSNorm)</button>
        <button type="button" @click="seed++">换一组 q / k</button>
      </div>
      <LabSlider v-model="scale" label="q、k 范数倍率" :min="0.5" :max="6" :step="0.1" unit="×" :format="(t) => t.toFixed(1)" />
      <LabSlider v-if="qkNorm" v-model="gain" label="可学增益 g" :min="0.5" :max="2" :step="0.05" :format="(t) => t.toFixed(2)" />
    </template>

    <div class="wrap">
      <div>
        <div class="cells" :style="{ gridTemplateColumns: `28px repeat(${T}, 26px)` }" role="group" aria-label="注意力矩阵, 点一行选择 query">
          <template v-for="(row, i) in probs" :key="i">
            <button type="button" class="rl mono" :class="{ active: i === qi }" :aria-pressed="i === qi" @click="qi = i">q{{ i }}</button>
            <span v-for="(p, j) in row" :key="j" class="cell" :class="{ rowsel: i === qi }" :style="j <= i ? { background: heat(p) } : { opacity: 0.15 }" :title="j <= i ? `p=${p.toFixed(3)}` : '因果 mask'" />
          </template>
        </div>
        <p class="cap">行 = query, 列 = key, 颜色 = 注意力概率</p>
      </div>
      <div class="bars">
        <div v-for="j in qi + 1" :key="j" class="b">
          <span class="mono lg" :class="{ hi: Math.abs(logits[qi][j - 1]) > 30 }">{{ logits[qi][j - 1].toFixed(1) }}</span>
          <div class="col"><div :style="{ height: probs[qi][j - 1] * 100 + '%' }" /></div>
          <span class="mono">k{{ j - 1 }}</span>
        </div>
        <p class="cap">q{{ qi }} 对各 key: 上 = logit, 柱 = softmax 概率</p>
      </div>
    </div>

    <template #stats>
      <div class="kv"><span>最大 |logit| (全矩阵)</span><b :class="maxLogit > 30 ? 'bad' : 'good'">{{ maxLogit.toFixed(1) }}</b></div>
      <div class="kv"><span>理论上界 g²·√d</span><b>{{ qkNorm ? (gain * gain * Math.sqrt(D)).toFixed(1) : '无上界 (∝ 倍率²)' }}</b></div>
      <div class="kv"><span>q{{ qi }} 的注意力熵 / 最大熵</span><b :class="H / Hmax < 0.15 && qi > 0 ? 'bad' : ''">{{ H.toFixed(2) }} / {{ Hmax.toFixed(2) }}</b></div>
      <div class="kv"><span>梯度通过率 1 − Σp²</span><b :class="pass < 0.05 && qi > 0 ? 'bad' : 'good'">{{ pass.toFixed(3) }}</b></div>
      <p class="lab-note">
        {{ qkNorm ? '开着 QK-Norm 时拖范数倍率, 图完全不动: 归一化把倍率整个除掉了, 只剩方向 (cosθ) 和增益 g 决定注意力。'
          : '倍率 ×2 → logit ×4。通过率接近 0 意味着 softmax 饱和: 反向时 dL/dlogit ≈ 0, 这个 head 的 q/k 投影学不动了。' }}
      </p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { mulberry32, randn, softmax, entropy, range, heat, sum } from '@/utils/labmath.js'

const T = 8, D = 16
const qkNorm = ref(false), scale = ref(1), gain = ref(1), seed = ref(3), qi = ref(5)

const vecs = computed(() => {
  const r = mulberry32(seed.value)
  const mk = () => range(T).map(() => range(D).map(() => randn(r)))
  return { q: mk(), k: mk() }
})
const rms = (v) => Math.sqrt(sum(v.map((t) => t * t)) / v.length)
const dot = (a, b) => sum(a.map((t, i) => t * b[i]))

const logits = computed(() => {
  // ★ 无 QK-Norm: 范数倍率原样进点积 (二次); 有: RMSNorm 先把它除掉, 再乘增益 g
  const prep = (v) => (qkNorm.value ? v.map((t) => (t / rms(v)) * gain.value) : v.map((t) => t * scale.value))
  const q = vecs.value.q.map(prep), k = vecs.value.k.map(prep)
  return q.map((qv, i) => k.map((kv, j) => (j <= i ? dot(qv, kv) / Math.sqrt(D) : -Infinity)))
})
const probs = computed(() => logits.value.map((row) => softmax(row)))
const maxLogit = computed(() => Math.max(...logits.value.flat().filter(Number.isFinite).map(Math.abs)))
const H = computed(() => entropy(probs.value[qi.value]))
const Hmax = computed(() => Math.log(qi.value + 1))
const pass = computed(() => 1 - sum(probs.value[qi.value].map((p) => p * p)))
</script>

<style scoped>
.wrap { display: flex; flex-wrap: wrap; gap: 20px; align-items: flex-start; }
.cells { gap: 2px; }
.cell { min-width: 0; width: 26px; height: 22px; }
.cell.rowsel { border-color: var(--accent); }
.rl { padding: 0; min-height: 22px; height: 22px; font-size: 10px; border-radius: 3px; }
.cap { font-size: 11px; color: var(--text-dim); margin-top: 6px; flex-basis: 100%; }
.bars { display: flex; flex-wrap: wrap; gap: 4px; flex: 1; min-width: 220px; }
.b { display: grid; justify-items: center; gap: 3px; font-size: 10px; color: var(--text-muted); width: 30px; }
.col { width: 20px; height: 110px; display: flex; align-items: flex-end; border-bottom: 1px solid var(--border-strong); }
.col div { width: 100%; background: var(--accent); border-radius: 2px 2px 0 0; min-height: 1px; }
.lg.hi { color: var(--danger); }
</style>
