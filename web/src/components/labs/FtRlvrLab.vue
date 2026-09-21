<!--
  prompt 相关的可验证奖励实验台。
  只讲一件事: 如果奖励函数不看 prompt (例如"输出落在词表后半区就给 1 分"), 最优策略就是无视 prompt 的常数输出 ——
  reward 涨到 1.0 也证明不了模型学会了"按题作答"。奖励必须是 (prompt, completion) 的函数, RL 才是在学条件生成。
-->
<template>
  <LabFrame
    title="可验证奖励要「看题」— 否则 RL 只学会一个常数"
    sub="每一行是一个 prompt, 右边 10 个格子是策略可能输出的答案 token, 点格子 = 决定策略在这道题上输出什么。绿框 = 当前奖励函数会给 1 分的输出。先用「常数策略」(无视 prompt, 所有题输出同一个 token) 试着拿满分。"
    module="llm_finetune/data/tasks.py"
    run="python -m llm_finetune.run_finetune.grpo.train_grpo"
    :challenge="{
      ask: '「区域奖励」下, 常数策略最高能拿多少分? 切到「看题奖励 (a+b) mod 10」, 常数策略最高还能拿多少? 绿框的形状从什么变成了什么?',
      answer: '区域奖励 (token ≥ 5 就给分) 下绿框是几条竖带: 每一行可行的输出都一样, 所以永远输出 7 的常数策略就是满分 100% —— 训练曲线很漂亮, 但策略根本没读 prompt, 它学到的是一个 unigram 偏好, 不是任务。看题奖励下每行只有一个绿框, 位置随 prompt 变: 常数策略最多蒙对 4 题里的 1 题 (25%), 要拿满分必须让输出依赖输入。这就是教学用 RL 任务最容易踩的坑: 验证「RL 管线能让 reward 上升」和验证「RL 学会了条件行为」是两回事。真实 RLVR (数学答案、单元测试) 天然是看题的; 自己造玩具任务时要主动保证这一点, 并且单独报告「常数基线」的得分。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" :class="{ active: reward === 'region' }" @click="reward = 'region'">区域奖励: token ≥ 5 (不看题)</button>
        <button type="button" :class="{ active: reward === 'sum' }" @click="reward = 'sum'">看题奖励: 输出 = (a+b) mod 10</button>
      </div>
      <div class="row">
        <button type="button" :class="{ active: constant }" @click="setConstant(true)">常数策略 (无视 prompt)</button>
        <button type="button" :class="{ active: !constant }" @click="setConstant(false)">条件策略 (每题各自输出)</button>
        <button type="button" @click="seed++">换一组题</button>
      </div>
    </template>

    <div class="cells grid" :style="{ gridTemplateColumns: `92px repeat(10, 30px) 52px` }">
      <span class="rl">prompt</span>
      <span v-for="d in 10" :key="'h' + d" class="hd mono">{{ d - 1 }}</span>
      <span class="hd">奖励</span>
      <template v-for="(p, i) in prompts" :key="i">
        <span class="rl mono">{{ p.a }} + {{ p.b }} = ?</span>
        <button
          v-for="d in 10" :key="d" type="button" class="cell ans"
          :class="{ win: rewardOf(p, d - 1) === 1, on: outs[i] === d - 1 }"
          :aria-label="`题 ${i + 1} 输出 ${d - 1}`" :aria-pressed="outs[i] === d - 1"
          @click="pick(i, d - 1)"
        >{{ outs[i] === d - 1 ? d - 1 : '' }}</button>
        <span class="mono rw" :class="rewardOf(p, outs[i]) ? 'good' : 'bad'">{{ rewardOf(p, outs[i]) }}</span>
      </template>
    </div>

    <template #stats>
      <div class="kv"><span>当前策略平均奖励</span><b :class="mean === 1 ? 'good' : mean < 0.5 ? 'bad' : ''">{{ (mean * 100).toFixed(0) }}%</b></div>
      <div class="kv"><span>常数策略的上限</span><b :class="constBest === 1 ? 'bad' : 'good'">{{ (constBest * 100).toFixed(0) }}%</b></div>
      <div class="kv"><span>输出依赖 prompt 吗</span><b>{{ depends ? '依赖' : '不依赖' }}</b></div>
      <div class="kv"><span>满分 ⇒ 读懂了题?</span><b :class="constBest === 1 ? 'bad' : 'good'">{{ constBest === 1 ? '推不出' : '是' }}</b></div>
      <p class="lab-note">
        常数策略上限 = 让所有题输出同一个 token 时能拿到的最高平均分。它等于 100% 的奖励函数,
        RL 会沿最短路径收敛到那个常数 —— 熵塌缩, 且与输入无关。
      </p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import { mulberry32, range, sum } from '@/utils/labmath.js'

const reward = ref('region'), constant = ref(true), seed = ref(1)

// 4 道题, 保证答案两两不同 (否则常数策略能蒙对不止一题, 对比不够干净)
const prompts = computed(() => {
  const rand = mulberry32(seed.value * 389), ps = []
  while (ps.length < 4) {
    const a = 1 + Math.floor(rand() * 9), b = 1 + Math.floor(rand() * 9)
    if (!ps.some((p) => (p.a + p.b) % 10 === (a + b) % 10)) ps.push({ a, b })
  }
  return ps
})
const outs = ref([7, 7, 7, 7])
watch(prompts, () => { outs.value = [7, 7, 7, 7]; constant.value = true })

// ★ 两种奖励函数的唯一区别: 签名里有没有 prompt
const rewardOf = (p, tok) => (reward.value === 'region' ? +(tok >= 5) : +(tok === (p.a + p.b) % 10))

const pick = (i, tok) => { outs.value = constant.value ? outs.value.map(() => tok) : outs.value.map((o, k) => (k === i ? tok : o)) }
const setConstant = (c) => { constant.value = c; if (c) outs.value = outs.value.map(() => outs.value[0]) }

const mean = computed(() => sum(prompts.value.map((p, i) => rewardOf(p, outs.value[i]))) / prompts.value.length)
const constBest = computed(() => Math.max(...range(10).map((t) => sum(prompts.value.map((p) => rewardOf(p, t))) / prompts.value.length)))
const depends = computed(() => new Set(outs.value).size > 1)
</script>

<style scoped>
.grid { align-items: center; gap: 4px; width: max-content; }
.rl { font-size: 12px; color: var(--text-muted); }
.hd { font-size: 10px; color: var(--text-dim); text-align: center; }
.ans { min-width: 0; min-height: 0; width: 30px; height: 30px; padding: 0; font-size: 13px; cursor: pointer; }
.ans.win { border: 2px solid var(--left); }
.ans.on { background: var(--accent); color: var(--bg-card); font-weight: 700; }
.rw { text-align: center; font-size: 14px; }
.rw.good { color: var(--left); }
.rw.bad { color: var(--danger); }
</style>
