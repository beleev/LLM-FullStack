<!--
  Skill 渐进式披露 (对应 llm_agent/core/skills.py 的 SkillRegistry / SkillTool)。
  只讲一件事: 常驻上下文的只有每个 skill 一行 description, 正文在被触发时才加载。
-->
<template>
  <LabFrame
    title="Skills — 渐进式披露省下多少上下文"
    sub="每个格子是一个已安装的 skill。常驻上下文的只有它的一行 name + description; 点一条用户请求, 看模型调用 skill 工具把哪一个的正文加载进来。也可以直接点格子手动加载 / 卸载。拖滑杆改变安装数量和手册长度。"
    module="llm_agent/m05"
    run="python -m llm_agent.m05_extensibility.demo"
    :challenge="{
      ask: '装 40 个 skill、每个正文 3000 token。全量常驻要多少 token? 渐进披露在「只触发 1 个」时是多少? 省下的那部分, 每一轮请求都在省还是只省一次?',
      answer: '全量 = 40 × (30 + 3000) ≈ 121K, 一个 200K 窗口的六成还没干活就没了; 渐进披露 = 40 × 30 + 3000 = 4.2K, 约 29 倍。而且 agent 每一轮都要重发整个上下文, 所以这个差价是每轮都付的。代价: 模型只凭一行 description 判断要不要加载 —— description 写得含糊, skill 就永远不会被触发。',
    }"
  >
    <template #controls>
      <LabSlider v-model="n" label="已安装 skill 数" :min="1" :max="40" />
      <LabSlider v-model="body" label="每个正文长度" :min="200" :max="5000" :step="100" unit=" tok" />
      <LabSlider v-model="desc" label="description 长度" :min="10" :max="100" :step="5" unit=" tok" />
      <div class="row">
        <button v-for="p in PROMPTS" :key="p.text" type="button" :class="{ active: prompt === p.text }" @click="ask(p)">{{ p.text }}</button>
        <button type="button" @click="loaded = []; prompt = ''">清空已加载</button>
      </div>
    </template>

    <div class="skills">
      <button
        v-for="s in skills" :key="s.name" type="button" class="skill" :class="{ loaded: loaded.includes(s.name) }"
        :aria-pressed="loaded.includes(s.name)" :title="s.desc" @click="toggle(s.name)"
      >
        <span class="mono nm">{{ s.name }}</span>
        <span class="ds">{{ loaded.includes(s.name) ? `正文已加载 +${body}` : s.desc }}</span>
      </button>
    </div>
    <p v-if="trace" class="mono trace">{{ trace }}</p>

    <div class="bars">
      <div v-for="b in bars" :key="b.label" class="bar-row">
        <span class="lbl">{{ b.label }}</span>
        <div class="track">
          <div class="seg cat" :style="{ width: b.cat / scale * 100 + '%' }" />
          <div class="seg bod" :style="{ width: b.bod / scale * 100 + '%' }" />
        </div>
        <span class="mono num">{{ fmtNum(b.cat + b.bod) }}</span>
      </div>
      <p class="legend"><i class="cat" /> 目录 (name + description) <i class="bod" /> 正文</p>
    </div>

    <template #stats>
      <div class="kv"><span>全量常驻</span><b class="bad">{{ fmtNum(allIn) }} tok</b></div>
      <div class="kv"><span>渐进披露</span><b class="good">{{ fmtNum(progressive) }} tok</b></div>
      <div class="kv"><span>节省倍数</span><b>{{ (allIn / progressive).toFixed(1) }}×</b></div>
      <div class="kv"><span>全量占 200K 窗口</span><b :class="allIn > 100000 ? 'bad' : ''">{{ (allIn / 2000).toFixed(1) }}%</b></div>
      <p class="lab-note">三层加载: ① 启动时只读 SKILL.md 的 frontmatter; ② 模型调用 <code class="inline">skill</code> 工具时正文才进上下文; ③ 正文引用的脚本/附件用到才读。skill 是用户自己安装的可信指令, 和 fetch 回来的网页不是一回事。</p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { fmtNum, range } from '@/utils/labmath.js'

const NAMED = [
  { name: 'pdf-report', desc: '生成 / 填写 PDF 报告', keys: ['pdf'] },
  { name: 'debug-agent', desc: '排查 agent 没有结果', keys: ['排查', '没有结果'] },
  { name: 'release-notes', desc: '从 git log 写发布说明', keys: ['发布说明', 'release'] },
  { name: 'sql-review', desc: '审查 SQL 迁移脚本', keys: ['sql', '迁移'] },
  { name: 'code-review', desc: '按团队清单做代码评审', keys: ['评审', 'review'] },
  { name: 'deploy', desc: '按流程发布到 staging', keys: ['发布到', '部署'] },
]
const PROMPTS = [
  { text: '排查为什么 agent 没有结果' },
  { text: '把这份数据导出成 PDF' },
  { text: '审一下这个 SQL 迁移, 再写发布说明' },
  { text: '今天天气怎么样' },
]

const n = ref(12), body = ref(3000), desc = ref(30)
const loaded = ref([]), prompt = ref(''), trace = ref('')

const skills = computed(() => range(n.value).map((i) => NAMED[i] || { name: `skill-${i + 1}`, desc: '别的团队手册', keys: [] }))
// 装的 skill 变少时, 已加载但被卸掉的要清理
watch(skills, (list) => { loaded.value = loaded.value.filter((x) => list.some((s) => s.name === x)) })

// 模型只看得到目录: 用 description 关键词命中模拟"模型判断相关" → 调 skill 工具
const ask = (p) => {
  prompt.value = p.text
  const hits = skills.value.filter((s) => s.keys.some((k) => p.text.toLowerCase().includes(k)))
  loaded.value = hits.map((s) => s.name)
  trace.value = hits.length
    ? hits.map((s) => `tool_use skill({"name":"${s.name}"}) → tool_result: 正文 ${body.value} tok`).join('\n')
    : '目录里没有相关 skill → 不加载任何正文, 只付目录的钱'
}
const toggle = (name) => {
  loaded.value = loaded.value.includes(name) ? loaded.value.filter((x) => x !== name) : [...loaded.value, name]
  trace.value = ''
}

// ★ 全量: N×(desc+body) 每轮都付; 渐进: N×desc 常驻 + 只为用到的正文付费
const catalog = computed(() => n.value * desc.value)
const allIn = computed(() => n.value * (desc.value + body.value))
const progressive = computed(() => catalog.value + loaded.value.length * body.value)
const scale = computed(() => Math.max(allIn.value, 1))
const bars = computed(() => [
  { label: '全量常驻', cat: catalog.value, bod: n.value * body.value },
  { label: '渐进披露', cat: catalog.value, bod: loaded.value.length * body.value },
])
</script>

<style scoped>
.skills { display: grid; grid-template-columns: repeat(auto-fill, minmax(118px, 1fr)); gap: 6px; }
.skill { display: flex; flex-direction: column; align-items: flex-start; gap: 2px; min-height: 0; padding: 6px 8px; text-align: left; border: 1px solid var(--border); border-radius: var(--radius-sm); background: var(--bg-elev); color: var(--text); }
.skill .nm { font-size: 11px; }
.skill .ds { font-size: 10px; color: var(--text-dim); line-height: 1.4; }
.skill.loaded { border-color: var(--left); background: color-mix(in srgb, var(--left) 18%, transparent); }
.skill.loaded .ds { color: var(--left); }
.trace { font-size: 11px; color: var(--text-muted); margin-top: 10px; white-space: pre-wrap; word-break: break-all; }
.bars { margin-top: 14px; display: flex; flex-direction: column; gap: 6px; }
.bar-row { display: grid; grid-template-columns: 64px 1fr 56px; gap: 8px; align-items: center; font-size: 12px; color: var(--text-muted); }
.track { height: 16px; display: flex; border: 1px solid var(--border); border-radius: 3px; overflow: hidden; }
.seg.cat, .legend i.cat { background: var(--accent); }
.seg.bod, .legend i.bod { background: var(--warn); }
.num { text-align: right; color: var(--text); }
.legend { font-size: 11px; color: var(--text-dim); }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin: 0 4px 0 10px; vertical-align: -1px; }
</style>
