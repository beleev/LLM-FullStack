<!--
  无 reference 的偏好优化实验台: DPO vs SimPO vs ORPO。
  只讲一件事: DPO 里 reference 顺带抵消了"长回答 sum log p 天然更低"这件事; 拿掉 reference 之后,
  SimPO 用长度归一化 + 目标间隔 γ 顶上, ORPO 用 SFT 项 + odds ratio 顶上。
  设定: policy 对 chosen / rejected 的每 token 平均 log p 由滑杆给, ref 对两者都是 −1.5/token。
-->
<template>
  <LabFrame
    title="拿掉 reference 之后 — SimPO 与 ORPO 靠什么不被长度骗"
    sub="两行格子是同一个 prompt 的 chosen / rejected 回答, 点第 n 个格子把长度设为 n。下面四根条是四种写法里送进 log σ(·) 的那个量 z (越右 = 模型认为自己已经排对)。把 rejected 点长, 看谁的 z 白白变大。"
    module="llm_finetune/methods/simpo.py · orpo.py"
    :challenge="{
      ask: '保持两条回答每 token 的平均 log p 都是 −1.5 (模型对两者毫无偏好), 把 rejected 从 8 点到 24 token。「裸 sum 差」的 z 变成多少? 它的梯度权重还剩多少? DPO 和 SimPO 为什么不上当?',
      answer: '裸 sum 差 = (−1.5×8) − (−1.5×24) = +24, σ(−24) ≈ 0: 模型什么偏好都没学, loss 却已经是 0 —— 只因为长序列的 log 概率之和天然更小。数据里 rejected 普遍更长时, 这个目标学到的只是「短的好」。DPO 不上当, 因为 ref 对同一条长回答也给出同样低的 sum log p, 相减后只剩 policy 相对 ref 的变化。SimPO 没有 ref, 改成按长度取平均 (每 token log p 之差), 长度直接约掉。代价是平均后数值范围很小, 所以 β 要取 2 左右。SimPO 还会再减一个目标间隔 γ, 逼模型把差距拉到 γ/β 以上才停手。ORPO 同样用每 token 平均概率算 odds, 再靠 SFT 项 (chosen 的 NLL) 锚住模型 —— 一个阶段、一个模型, 连单独的 SFT 都省了。',
    }"
  >
    <template #controls>
      <LabSlider v-model="lc" label="chosen 每 token log p" :min="-3" :max="-0.1" :step="0.05" :format="(v) => v.toFixed(2)" />
      <LabSlider v-model="lr" label="rejected 每 token log p" :min="-3" :max="-0.1" :step="0.05" :format="(v) => v.toFixed(2)" />
      <LabSlider v-model="gamma" label="SimPO 目标间隔 γ" :min="0" :max="2" :step="0.1" :format="(v) => v.toFixed(1)" />
    </template>

    <div class="seqs">
      <div v-for="s in seqs" :key="s.id" class="seq">
        <span class="rl" :style="{ color: s.color }">{{ s.name }} · {{ s.len }} tok</span>
        <div class="cells" :style="{ gridTemplateColumns: `repeat(${LMAX}, 18px)` }">
          <button
            v-for="n in LMAX" :key="n" type="button" class="cell tk" :class="{ on: n <= s.len }"
            :style="n <= s.len ? { background: heat(clamp(-s.lp / 3, 0.15, 1), s.color) } : null"
            :aria-label="`${s.name} 长度设为 ${n}`" @click="s.set(n)"
          />
        </div>
      </div>
    </div>

    <div class="bars" @mouseleave="hov = -1">
      <div v-for="(r, i) in rows" :key="r.id" class="bar-row" :class="{ hl: hov === i }" tabindex="0" @mouseenter="hov = i" @focus="hov = i">
        <span class="name">{{ r.name }}</span>
        <div class="track">
          <i class="zero" />
          <i class="fill" :style="barStyle(r.z)" :class="r.z >= 0 ? 'pos' : 'neg'" />
        </div>
        <span class="mono z">z = {{ sg(r.z) }}</span>
        <span class="mono w">权重 σ(−z) = {{ sigmoid(-r.z).toFixed(3) }}</span>
      </div>
    </div>
    <p class="lab-note" style="margin-top: 10px;">{{ hov >= 0 ? rows[hov].formula : '悬停某一行看它的公式。' }}</p>

    <template #stats>
      <p class="cap">rejected 每多写 1 个 token, z 白涨多少 (∂z/∂L_r):</p>
      <div v-for="r in rows" :key="r.id" class="kv">
        <span>{{ r.short }}</span><b :class="Math.abs(r.dz) < 1e-9 ? 'good' : 'bad'">{{ sg(r.dz) }}</b>
      </div>
      <div class="kv"><span>裸 sum 差还剩的梯度权重</span><b :class="sigmoid(-rows[0].z) < 0.05 ? 'bad' : ''">{{ sigmoid(-rows[0].z).toFixed(3) }}</b></div>
      <div class="kv"><span>ORPO 总 loss = NLL + λ·L_OR</span><b>{{ orpoLoss.toFixed(3) }}</b></div>
      <p class="lab-note">
        需要的前向: DPO = policy×2 + ref×2 (两个模型); SimPO / ORPO = policy×2 (一个模型)。
        DPO 这一行不为 0 只发生在 policy 已偏离 ref 时 —— 那是学到的偏好, 不是长度红利。
      </p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { clamp, heat } from '@/utils/labmath.js'

const LMAX = 24, REF = -1.5, B_DPO = 0.1, B_SIMPO = 2, LAMBDA = 0.1, ZMAX = 8
const lc = ref(-1.5), lr = ref(-1.5), gamma = ref(0.5), Lc = ref(8), Lr = ref(8)
const hov = ref(-1)

const seqs = computed(() => [
  { id: 'c', name: 'chosen', len: Lc.value, lp: lc.value, color: 'var(--left)', set: (n) => (Lc.value = n) },
  { id: 'r', name: 'rejected', len: Lr.value, lp: lr.value, color: 'var(--danger)', set: (n) => (Lr.value = n) },
])
const sigmoid = (z) => 1 / (1 + Math.exp(-z))
const softplus = (z) => Math.max(z, 0) + Math.log1p(Math.exp(-Math.abs(z)))
const logit = (avgLogp) => { const p = Math.exp(avgLogp); return Math.log(p / (1 - p)) }   // log odds, p = 每 token 平均概率

const rows = computed(() => [
  {
    id: 'naive', name: '裸 sum 差 (无 ref, 不归一)', short: '裸 sum 差',
    z: lc.value * Lc.value - lr.value * Lr.value, dz: -lr.value,
    formula: 'z = Σlog π(chosen) − Σlog π(rejected)。没有任何校正: 序列越长 Σlog p 越负, rejected 长 = 白送 margin。',
  },
  {
    id: 'dpo', name: `DPO (β = ${B_DPO})`, short: 'DPO',
    // ★ ref 对同一条序列也"越长越负", 相减后长度红利消失
    z: B_DPO * ((lc.value - REF) * Lc.value - (lr.value - REF) * Lr.value), dz: -B_DPO * (lr.value - REF),
    formula: 'z = β·[(Σlog π_c − Σlog ref_c) − (Σlog π_r − Σlog ref_r)]。ref 每 token 也是 −1.5, 长度项成对抵消。',
  },
  {
    id: 'simpo', name: `SimPO (β = ${B_SIMPO}, γ = ${gamma.value.toFixed(1)})`, short: 'SimPO',
    z: B_SIMPO * (lc.value - lr.value) - gamma.value, dz: 0,              // ★ 按长度平均, 再减目标间隔 γ
    formula: 'z = β·(Σlog π_c/|y_c| − Σlog π_r/|y_r|) − γ。长度归一化顶替 ref; γ 要求每 token 平均 log p 至少领先 γ/β。',
  },
  {
    id: 'orpo', name: 'ORPO (odds ratio 项)', short: 'ORPO',
    z: logit(lc.value) - logit(lr.value), dz: 0,
    formula: 'z = log[odds(chosen)/odds(rejected)], odds = p/(1−p), p = exp(每 token 平均 log p)。总 loss = chosen 的 NLL + λ·(−log σ(z)), SFT 项兼任锚点。',
  },
])
const orpoLoss = computed(() => -lc.value + LAMBDA * softplus(-rows.value[3].z))
const barStyle = (z) => { const w = clamp(Math.abs(z) / ZMAX, 0, 1) * 50; return z >= 0 ? { left: '50%', width: w + '%' } : { left: 50 - w + '%', width: w + '%' } }
const sg = (x) => (x >= 0 ? '+' : '') + x.toFixed(2)
</script>

<style scoped>
.seqs { display: flex; flex-direction: column; gap: 8px; margin-bottom: 16px; }
.rl { font-size: 11px; font-family: "SF Mono", Menlo, monospace; display: block; margin-bottom: 3px; }
.tk { min-width: 0; min-height: 0; width: 18px; height: 22px; padding: 0; cursor: pointer; }
.bars { display: flex; flex-direction: column; gap: 6px; min-width: 520px; }
.bar-row { display: grid; grid-template-columns: 170px 1fr 84px 150px; gap: 8px; align-items: center; font-size: 12px; color: var(--text-muted); padding: 2px 4px; border-radius: 4px; }
.bar-row.hl { background: var(--bg-elev); color: var(--text); }
.track { position: relative; height: 14px; background: var(--code-bg); border-radius: 2px; }
.zero { position: absolute; left: 50%; top: -2px; bottom: -2px; border-left: 1px solid var(--border-strong); }
.fill { position: absolute; top: 0; bottom: 0; border-radius: 2px; }
.fill.pos { background: var(--left); }
.fill.neg { background: var(--danger); }
.z { color: var(--text); }
.w { font-size: 11px; }
.cap { font-size: 12px; color: var(--text-muted); }
</style>
