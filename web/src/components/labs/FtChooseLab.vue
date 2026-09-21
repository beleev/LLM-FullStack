<!--
  微调方法选型计算器 (对应 llm_finetune/README.md 的 "各方法需要什么" 表)。
  只讲一件事: 11 种方法的差别不在 loss 好不好看, 在代价结构 ——
  要不要成对数据 / 要不要 verifier / 要不要在线采样 / 要不要常驻一份 ref 或 teacher / 落盘什么。
  所有字节数都是 llm_finetune 在 CPU 上实测的真实值 (基座 99,648 参数 = 389 KB), 不是估算。
-->
<template>
  <LabFrame
    title="选型计算器 — 11 种方法, 各自要你付什么"
    sub="左边勾你手上真有的东西, 拖显存预算。右边每一行是一种方法: 亮着 = 现在就能用, 暗掉 = 缺东西或超预算 (行尾写了缺什么)。点任意一行展开「什么时候它是对的选择」和运行命令。"
    module="llm_finetune/README.md"
    run="python -m llm_finetune.run_all"
    :challenge="{
      ask: '只勾「成对偏好」, 把显存预算从 1600 KB 往下拖。最先掉队的是哪一个? 到 1200 KB 时还剩谁? 拿掉的那份权重原本在干什么?',
      answer: 'DPO 最先掉队 —— 它要常驻 policy + ref 两份权重 (389 + 389 = 778 KB), 再加全参 Adam 状态 778 KB, 合计 1556 KB。SimPO / ORPO / RM 都只要一份权重, 1167 KB 就够。ref 干的事有两件: 一是当锚, 不让 policy 漂离 SFT 起点; 二是顺手抵消长度红利 (它对同一条长回答也给出同样低的 Σlog p, 相减就消了)。拿掉 ref 就得请人接班: SimPO 用长度归一化 + 目标间隔 γ, ORPO 用 NLL 项当锚。实测代价写在结果里 —— 留出集 EM: ORPO 0.543 > DPO 0.121 > SimPO 0.023, SimPO 什么锚都没有, 掉得最惨。',
    }"
  >
    <template #controls>
      <div class="row">
        <span class="lbl">我手上有:</span>
        <button
          v-for="h in HAVE" :key="h.id" type="button"
          :class="{ active: have[h.id] }" :aria-pressed="have[h.id]"
          @click="have[h.id] = !have[h.id]"
        >{{ have[h.id] ? '✓ ' : '' }}{{ h.label }}</button>
      </div>
      <div class="row">
        <span class="lbl">能在线采样 (每个 prompt 现采 G 条):</span>
        <button type="button" :class="{ active: online }" @click="online = true">能</button>
        <button type="button" :class="{ active: !online }" @click="online = false">不能 / 太贵</button>
      </div>
      <LabSlider
        v-model="budget" label="常驻显存预算" :min="200" :max="1800" :step="50"
        :format="(v) => fmtKB(v)"
      />
    </template>

    <table class="menu">
      <thead>
        <tr>
          <th>方法</th>
          <th class="num">前向/步</th>
          <th>常驻显存</th>
          <th>落盘</th>
          <th class="num">信号</th>
        </tr>
      </thead>
      <tbody>
        <template v-for="m in rows" :key="m.id">
          <tr :class="{ off: !m.ok, open: open === m.id }" @click="open = open === m.id ? '' : m.id">
            <td>
              <button type="button" class="name" :aria-expanded="open === m.id">{{ m.name }}</button>
              <span v-if="!m.ok" class="why">{{ m.why }}</span>
            </td>
            <td class="num mono fwd">{{ m.fwd }}</td>
            <td class="memcell">
              <div class="bar" :title="m.memText">
                <i v-for="(s, i) in m.segs" :key="i" :class="s.cls" :style="{ width: (s.kb / MAXKB) * 100 + '%' }" />
              </div>
              <span class="mono memv" :class="{ over: m.mem > budget }">{{ m.mem }} KB</span>
            </td>
            <td class="small">{{ m.save }}</td>
            <td class="num mono sig">
              {{ m.signal }}
              <i class="dens" :style="{ width: 4 + 30 * (Math.log2(m.signal) / Math.log2(112)) + 'px' }" />
            </td>
          </tr>
          <tr v-if="open === m.id" class="detail">
            <td colspan="5">
              <p>{{ m.when }}</p>
              <code class="inline">{{ m.run }}</code>
            </td>
          </tr>
        </template>
      </tbody>
    </table>
    <p class="legend">
      常驻显存条: <i class="sw base" />基座 <i class="sw ref" />ref / teacher <i class="sw ad" />adapter <i class="sw adam" />Adam 状态
    </p>

    <template #stats>
      <div class="kv"><span>现在能用</span><b :class="okCount ? 'good' : 'bad'">{{ okCount }} / 11</b></div>
      <div class="kv"><span>最省显存的可行方法</span><b>{{ cheapest }}</b></div>
      <div class="kv"><span>被显存卡掉</span><b :class="overCount ? 'bad' : 'good'">{{ overCount }}</b></div>
      <div class="kv"><span>信号最密的可行方法</span><b>{{ densest }}</b></div>
      <p class="lab-note">
        字节数按本仓库的玩具基座算: 99,648 个参数 = 389 KB (fp32), NF4 基座 59 KB, LoRA adapter 76 KB,
        全参 Adam 状态 778 KB。换成 7B 模型时每一栏同比放大, 相对关系不变。
        "信号/样本" = 一条样本给出多少个监督数字: 偏好对 1 个 bit, GRPO 一条回复 1 个标量,
        SFT 每个回复 token 1 个标签 (R=7), 蒸馏每个 token 一个 16 维分布 (7×16=112)。
      </p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, reactive, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'

const HAVE = [
  { id: 'demo', label: '示范回答 (x, y)' },
  { id: 'pref', label: '成对偏好 (x, y_w, y_l)' },
  { id: 'verify', label: '可编程验证器' },
  { id: 'teacher', label: '更强的 teacher' },
]
const have = reactive({ demo: true, pref: false, verify: false, teacher: false })
const online = ref(true)
const budget = ref(1250)
const open = ref('')

// 全部数值都在 300~1600 KB, 统一用 KB 才好横着比
const fmtKB = (kb) => Math.round(kb) + ' KB'

// 实测字节 (llm_finetune, CPU): 基座 99,648 参数 fp32 = 389 KB; NF4 基座 59 KB;
// LoRA adapter 19,456 参数 = 76 KB; DoRA 20,736 = 81 KB; student 26,256 = 103 KB。
// Adam 状态 = 可训参数 × 8 B: 全参 778 KB, LoRA 152 KB, DoRA 162 KB, student 205 KB。
const B = 389, NF4 = 59, LA = 76, DA = 81, STU = 103
const ADAM_FULL = 778, ADAM_LORA = 152, ADAM_DORA = 162, ADAM_STU = 205
const MAXKB = 1600

// need: 需要手上有哪几样; sample: 是否必须在线采样
const METHODS = [
  { id: 'sft', name: 'SFT (全参)', need: ['demo'], fwd: '1', signal: 7,
    segs: [{ kb: B, cls: 'base' }, { kb: ADAM_FULL, cls: 'adam' }], save: '全量权重 389KB',
    when: '数据是 (问, 答), 显存够, 想要上限最高的那一个。同一基座 300 步实测: 全参留出集 EM 0.809, 同配置 LoRA 只有 0.352 —— LoRA 省的是显存不是步数。',
    run: 'python -m llm_finetune.run_finetune.sft.train_sft' },
  { id: 'lora', name: 'LoRA', need: ['demo'], fwd: '1', signal: 7,
    segs: [{ kb: B, cls: 'base' }, { kb: LA, cls: 'ad' }, { kb: ADAM_LORA, cls: 'adam' }], save: 'adapter 76KB',
    when: '显存放不下全参的梯度和 Adam 状态, 或者一个基座要挂很多个任务。adapter 能独立分发、按请求热切换; 效果上限 ≤ 全参。',
    run: 'python -m llm_finetune.run_finetune.lora.train_lora' },
  { id: 'qlora', name: 'QLoRA', need: ['demo'], fwd: '1', signal: 7,
    segs: [{ kb: NF4, cls: 'base' }, { kb: LA, cls: 'ad' }, { kb: ADAM_LORA, cls: 'adam' }], save: 'adapter 76KB + NF4 基座 59KB',
    when: '连冻结的基座都放不下。整模型 389 KB → 59 KB (6.57×), 基座原任务留出集 EM 仍是 1.000; 代价是每次前向都要反量化一遍。',
    run: 'python -m llm_finetune.run_finetune.qlora.train_qlora' },
  { id: 'dora', name: 'DoRA', need: ['demo'], fwd: '1', signal: 7,
    segs: [{ kb: B, cls: 'base' }, { kb: DA, cls: 'ad' }, { kb: ADAM_DORA, cls: 'adam' }], save: 'adapter 81KB',
    when: '想在同样的 r 下离全参近一点, 且不在乎训练慢一些。3 个种子平均留出集 EM 0.382 → 0.522, 每层只多 d_out 个参数; 但每步要显式构造整个 W。',
    run: 'python -m llm_finetune.run_finetune.dora.train_dora' },
  { id: 'rm', name: 'Reward Model', need: ['pref'], fwd: '1 (2B 条)', signal: 1,
    segs: [{ kb: B, cls: 'base' }, { kb: ADAM_FULL, cls: 'adam' }], save: '全量权重 + value head',
    when: '后面要跑在线 RL, 而任务没法用程序判分。RM 把"A 比 B 好"变成随时可调用的标量分 (留出集偏好准确率 0.549 → 0.930); 它只学分差, 整体加一个常数 loss 不变。',
    run: 'python -m llm_finetune.run_finetune.rm.train_rm' },
  { id: 'dpo', name: 'DPO', need: ['pref'], fwd: '2 (2B 条)', signal: 1,
    segs: [{ kb: B, cls: 'base' }, { kb: B, cls: 'ref' }, { kb: ADAM_FULL, cls: 'adam' }], save: '全量权重 389KB',
    when: '有偏好对、有 SFT 起点、显存放得下第二份权重。跑的时候要同时盯 log π(chosen): 实测 200 步偏好准确率 0.965 → 0.996, 但 log π(chosen) −4.03 → −4.18, 贪心 EM 0.332 → 0.137。',
    run: 'python -m llm_finetune.run_finetune.dpo.train_dpo' },
  { id: 'simpo', name: 'SimPO', need: ['pref'], fwd: '1 (2B 条)', signal: 1,
    segs: [{ kb: B, cls: 'base' }, { kb: ADAM_FULL, cls: 'adam' }], save: '全量权重 389KB',
    when: '不想为 ref 再占一份显存。长度归一化顶替 ref, 每步 4.0 s vs DPO 5.4 s; 但它没有任何锚, 实测留出集 EM 掉到 0.023 —— 得靠小 lr / 少步数自己收着点。',
    run: 'python -m llm_finetune.run_finetune.simpo_orpo.train_simpo_orpo' },
  { id: 'orpo', name: 'ORPO', need: ['pref'], fwd: '1 (2B 条)', signal: 1,
    segs: [{ kb: B, cls: 'base' }, { kb: ADAM_FULL, cls: 'adam' }], save: '全量权重 389KB',
    when: '连 SFT 阶段都想省掉。loss 里的 NLL 项就是 SFT, 同时当锚: 从零训 300 步留出集 EM 0.973 (与纯 SFT 打平), 还把 rejected 压得更低 (−16.34 vs −14.85)。',
    run: 'python -m llm_finetune.run_finetune.simpo_orpo.train_simpo_orpo' },
  { id: 'grpo', name: 'GRPO 系 (DAPO · Dr.GRPO · GSPO)', need: ['verify'], sample: true, fwd: '采样 + 1 + μ', signal: 1,
    segs: [{ kb: B, cls: 'base' }, { kb: ADAM_FULL, cls: 'adam' }], save: '全量权重 389KB',
    when: '答案能被程序判对错, 而且采样代价付得起。β=0 时连 ref 都不用养。验收要看采样 pass@1 (0.186 → 0.287~0.326), 贪心 EM 基本不动 —— 四个变体的最终分数差异在噪声内, 区别在机制。',
    run: 'python -m llm_finetune.run_finetune.grpo.train_grpo' },
  { id: 'kd', name: '离线蒸馏 (forward KL)', need: ['teacher'], fwd: '2', signal: 112,
    segs: [{ kb: STU, cls: 'base' }, { kb: B, cls: 'ref' }, { kb: ADAM_STU, cls: 'adam' }], save: 'student 权重 103KB',
    when: '有一个更强的模型, 想把能力压进小模型, 且数据量有限。128 条固定数据上软标签明显更好 (forward KL 2.292 → 1.860); 数据无限时这个优势会消失。',
    run: 'python -m llm_finetune.run_finetune.distill.train_distill' },
  { id: 'opd', name: 'on-policy 蒸馏 (reverse KL)', need: ['teacher'], sample: true, fwd: '采样 + 2', signal: 112,
    segs: [{ kb: STU, cls: 'base' }, { kb: B, cls: 'ref' }, { kb: ADAM_STU, cls: 'adam' }], save: 'student 权重 103KB',
    when: '离线蒸馏出来的小模型一采样就串台。让学生自己写、老师逐 token 打分: 合格率 0.059 → 0.402; 代价是老师的少数派答法被彻底放弃 (log π −17 → −33)。先热身再上。',
    run: 'python -m llm_finetune.run_finetune.on_policy_distill.train_on_policy_distill' },
]

const LABEL = Object.fromEntries(HAVE.map((h) => [h.id, h.label.replace(/ \(.*/, '')]))
const SEG = { base: '基座', ref: 'ref/teacher', ad: 'adapter', adam: 'Adam 状态' }

const rows = computed(() => METHODS.map((m) => {
  const mem = m.segs.reduce((a, s) => a + s.kb, 0)
  const missing = m.need.filter((k) => !have[k])
  // ★ 可行性三连判: 数据够不够 → 能不能在线采样 → 显存装不装得下
  let why = ''
  if (missing.length) why = '缺 ' + missing.map((k) => LABEL[k]).join(' / ')
  else if (m.sample && !online.value) why = '必须在线采样'
  else if (mem > budget.value) why = '超预算 ' + fmtKB(mem - budget.value)
  return {
    ...m, mem, why, ok: !why,
    short: m.name.replace(/\s*\(.*/, ''),
    memText: m.segs.map((s) => SEG[s.cls] + ' ' + s.kb + ' KB').join(' + ') + ' = ' + mem + ' KB',
  }
}))

const okRows = computed(() => rows.value.filter((r) => r.ok))
const okCount = computed(() => okRows.value.length)
const overCount = computed(() => rows.value.filter((r) => r.why.startsWith('超预算')).length)
const pick = (list, cmp) => (list.length ? list.reduce(cmp).short : '—')
const cheapest = computed(() => pick(okRows.value, (a, b) => (b.mem < a.mem ? b : a)))
const densest = computed(() => pick(okRows.value, (a, b) => (b.signal > a.signal ? b : a)))
</script>

<style scoped>
.lbl { font-size: 12px; color: var(--text-muted); }
table.menu { width: 100%; border-collapse: collapse; font-size: 11.5px; min-width: 540px; table-layout: fixed; }
.menu th {
  text-align: left; padding: 5px 6px; color: var(--text-dim); font-weight: 500;
  font-size: 10.5px; letter-spacing: 0.4px; border-bottom: 1px solid var(--border-strong);
}
.menu td { padding: 8px 6px; border-bottom: 1px solid var(--border); vertical-align: top; line-height: 1.5; }
.menu .num { text-align: right; white-space: nowrap; }
.menu th:nth-child(1) { width: 27%; }
.menu th:nth-child(2) { width: 15%; }
.menu th:nth-child(3) { width: 21%; }
.menu th:nth-child(4) { width: 25%; }
.menu th:nth-child(5) { width: 12%; }
.fwd { font-size: 10.5px; white-space: normal; }
.menu tbody tr { cursor: pointer; }
.menu tbody tr:hover:not(.detail) { background: var(--bg-elev); }
.menu tr.off { opacity: 0.38; }
.menu tr.open { background: var(--bg-elev); }
.name {
  background: none; border: none; padding: 0; font: inherit; color: var(--text);
  font-weight: 600; text-align: left; cursor: pointer;
}
tr.open .name { color: var(--accent); }
.why { display: block; font-size: 10.5px; color: var(--warn); margin-top: 2px; }
.small { color: var(--text-muted); }

/* 常驻显存: 一条横条按 base / ref / adapter / Adam 拆开 */
.bar { display: flex; height: 9px; border-radius: 3px; overflow: hidden; background: var(--bg-elev); }
.bar i { display: block; height: 100%; }
.bar i.base, .sw.base { background: var(--accent); }
.bar i.ref, .sw.ref { background: var(--danger); }
.bar i.ad, .sw.ad { background: var(--left); }
.bar i.adam, .sw.adam { background: var(--border-strong); }
.memv { font-size: 10.5px; color: var(--text-muted); }
.memv.over { color: var(--danger); }
.legend { margin-top: 8px; font-size: 11px; color: var(--text-dim); }
.sw { display: inline-block; width: 10px; height: 8px; border-radius: 2px; margin: 0 3px 0 10px; vertical-align: baseline; }

.sig { line-height: 1.3; }
.dens { display: block; height: 6px; border-radius: 2px; background: var(--eye); margin-left: auto; margin-top: 2px; }

tr.detail td { background: var(--bg-elev); border-bottom: 1px solid var(--border-strong); cursor: default; }
tr.detail p { color: var(--text-muted); line-height: 1.7; margin-bottom: 6px; font-size: 12px; }
</style>
