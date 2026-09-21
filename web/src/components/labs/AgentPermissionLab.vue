<!--
  权限门追踪器 (对应 llm_agent/core/permissions.py 的 PermissionGate)。
  只讲一件事: 裁决顺序 deny → ask → allow → 模式兜底; 以及字符串黑名单为什么天生很弱。
-->
<template>
  <LabFrame
    title="权限门 — 一次工具调用是怎么被裁决的"
    sub="选模式、点规则开关、选或直接输入一条调用, 看它依次穿过 deny → ask → allow → 模式兜底。再把匹配方式切到「朴素 glob」, 用 rm -fr、双空格、大写去绕 deny 规则。"
    module="llm_agent/m03"
    run="python -m llm_agent.m03_permissions.demo"
    :challenge="{
      ask: '归一化匹配下, 「rm -r -f /」「RM  -RF /」都被拦住了。你能想出一条同样删光磁盘、却仍然穿过 deny 规则的命令吗? (试试 find / -delete)',
      answer: '能, 而且有无穷多条: rm --recursive --force /、find / -delete、python -c shutil.rmtree…… 归一化只堵住最廉价的绕过, 因为黑名单是在枚举「坏」, 而坏是无穷的。所以 deny 规则只是最后一道便宜的网; 真正的边界是默认拒绝的 allowlist (default 模式未命中就问人) + 命令解析 + OS 级沙箱。注意 find / -delete 在 default / auto 下落到「问人」, 无人可问时 fail closed = 拒绝。',
    }"
  >
    <template #controls>
      <div class="row">
        <button v-for="m in MODES" :key="m" type="button" class="mono" :class="{ active: mode === m }" @click="mode = m">{{ m }}</button>
      </div>
      <div class="row">
        <button v-for="(r, i) in rules" :key="i" type="button" class="rule mono" :class="[r.decision, { off: !r.on }]" :aria-pressed="r.on" @click="r.on = !r.on">
          {{ r.on ? '☑' : '☐' }} {{ r.decision }} · {{ r.tool }}{{ r.pattern ? ` "${r.pattern}"` : '' }}
        </button>
      </div>
      <div class="row">
        <button type="button" :class="{ active: normalized }" @click="normalized = true">归一化匹配 (现在的实现)</button>
        <button type="button" :class="{ active: !normalized }" @click="normalized = false">朴素 glob (旧实现)</button>
        <button type="button" :class="{ active: human }" @click="human = !human">有人审批: {{ human ? '会批准' : '无人可问' }}</button>
      </div>
      <div class="row">
        <button v-for="p in PRESETS" :key="p.arg" type="button" class="mono" :class="{ active: tool === p.tool && arg === p.arg }" @click="tool = p.tool; arg = p.arg">{{ p.arg }}</button>
      </div>
      <div class="row call">
        <select v-model="tool" class="mono" aria-label="工具"><option v-for="(t, n) in TOOLS" :key="n" :value="n">{{ n }}</option></select>
        <input v-model="arg" type="text" class="mono arg" spellcheck="false" aria-label="参数" />
        <span class="risk mono">risk={{ TOOLS[tool].risk }}</span>
      </div>
    </template>

    <div v-for="(part, pi) in result.parts" :key="pi" class="part">
      <p v-if="result.parts.length > 1" class="mono seg" :class="{ decisive: pi === result.idx }">第 {{ pi + 1 }} 段: {{ part.text }} → {{ part.verdict }}</p>
      <ol v-if="pi === result.idx" class="trace">
        <li v-for="s in part.stages" :key="s.name" :class="s.state">
          <span class="stage mono">{{ s.name }}</span>
          <span class="what">{{ s.detail }}</span>
        </li>
      </ol>
    </div>

    <template #stats>
      <div class="kv"><span>最终裁决</span><b :class="final.cls">{{ final.text }}</b></div>
      <div class="kv"><span>谁做的决定</span><b>{{ decisive.source }}</b></div>
      <div class="kv"><span>匹配时看到的文本</span><b class="small">{{ decisive.seen }}</b></div>
      <div class="kv"><span>deny 规则: 朴素 / 归一化</span><b :class="denyHit.naive === denyHit.norm ? '' : 'bad'">{{ denyHit.naive ? '命中' : '漏过' }} / {{ denyHit.norm ? '命中' : '漏过' }}</b></div>
      <p class="lab-note">{{ decisive.reason }}。复合命令 (&amp;&amp; || ; | &amp;) 会被拆开逐段评估, 任一段不过则整体不过 —— 否则 <code class="inline">echo hi &amp;&amp; rm -rf /</code> 能蹭到 allow "echo *"。</p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, reactive, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'

const MODES = ['plan', 'default', 'accept_edits', 'auto', 'dont_ask', 'bypass_permissions']
const TOOLS = {
  shell: { risk: 'high', readOnly: false },
  write_note: { risk: 'medium', readOnly: false },
  read_file: { risk: 'low', readOnly: true },
  search_docs: { risk: 'low', readOnly: true },
  mcp__weather__get_weather: { risk: 'high', readOnly: false },
}
const PRESETS = [
  { tool: 'shell', arg: 'rm -rf /' }, { tool: 'shell', arg: 'rm -fr /' }, { tool: 'shell', arg: 'RM  -r -f /' },
  { tool: 'shell', arg: 'find / -delete' }, { tool: 'shell', arg: 'echo hi && rm -rf /' }, { tool: 'shell', arg: 'echo $(cat ~/.ssh/id_rsa)' }, { tool: 'shell', arg: 'git push origin main' },
  { tool: 'write_note', arg: 'todo: fix login' }, { tool: 'read_file', arg: '.env' }, { tool: 'mcp__weather__get_weather', arg: 'Beijing' },
]
const rules = reactive([
  { tool: 'shell', pattern: '*rm -rf*', decision: 'deny', on: true },
  { tool: 'shell', pattern: 'git push*', decision: 'ask', on: true },
  { tool: 'shell', pattern: 'echo *', decision: 'allow', on: true },
  { tool: 'write_note', pattern: '', decision: 'allow', on: false },
  { tool: 'mcp__weather__*', pattern: '', decision: 'allow', on: false },
])
const mode = ref('default'), normalized = ref(true), human = ref(false)
const tool = ref('shell'), arg = ref('rm -fr /')

// fnmatch: 只需要 * 和 ?
const glob = (text, pat) => new RegExp('^' + pat.replace(/[.+^${}()|[\]\\]/g, '\\$&').replace(/\*/g, '[\\s\\S]*').replace(/\?/g, '.') + '$').test(text)
// ★ 与 permissions.py:normalize_command 逐行对应: 小写、压空白、相邻短 flag 合并并排序 (-r -f → -fr)
const FLAG = /^(\*?)-([a-z]+)(\*?)$/
const sortSet = (s) => [...new Set(s)].sort().join('')
const normalize = (text) => {
  const out = []
  for (const tok of text.toLowerCase().split(/\s+/).filter(Boolean)) {
    const m = tok.match(FLAG), prev = out.length ? out[out.length - 1].match(FLAG) : null
    if (m && prev && !prev[3] && !m[1]) out[out.length - 1] = `${prev[1]}-${sortSet(prev[2] + m[2])}${m[3]}`
    else if (m) out.push(`${m[1]}-${sortSet(m[2])}${m[3]}`)
    else out.push(tok)
  }
  return out.join(' ')
}
const matches = (rule, name, text, norm) => glob(name, rule.tool)
  && (!rule.pattern || (norm ? glob(normalize(text), normalize(rule.pattern)) : glob(text, rule.pattern)))

const DANGER = ['rm ', 'sudo', 'curl ', 'wget ', 'ssh ', 'chmod ', '>', '| sh', 'token', 'secret']
const SENSITIVE = ['.env', 'secret', 'id_rsa', '.ssh', 'credentials']
const ask = (reason) => ({ verdict: 'ASK', source: 'human', reason })
const fallback = (name, text) => {
  const t = TOOLS[name], m = mode.value
  if (m === 'plan') return t.readOnly ? { verdict: 'ALLOW', source: 'plan', reason: '只读工具, plan 模式放行' } : { verdict: 'DENY', source: 'plan', reason: 'plan 模式只读: 计划获批前任何写操作都拒绝' }
  if (m === 'dont_ask' || m === 'bypass_permissions') return { verdict: 'ALLOW', source: m, reason: '该模式对未命中规则的调用一律放行 (deny 规则仍生效)' }
  if (m === 'accept_edits') return t.risk === 'high' ? ask('accept_edits 只自动放行低/中风险, 高风险仍问人') : { verdict: 'ALLOW', source: m, reason: '低/中风险自动放行' }
  if (m === 'auto') {
    if (name === 'shell' && DANGER.some((x) => (normalize(text) + ' ').includes(x))) return { verdict: 'DENY', source: 'auto', reason: 'auto 分类器看到危险 shell 片段' }
    if (name === 'read_file' && SENSITIVE.some((x) => text.toLowerCase().includes(x))) return { verdict: 'DENY', source: 'auto', reason: 'auto 分类器: 敏感路径' }
    return t.risk === 'high' ? ask('auto 分类器对高风险工具拿不准, 回退问人') : { verdict: 'ALLOW', source: 'auto', reason: `${t.risk} 风险工具, auto 放行` }
  }
  return ask('default 模式: 没有规则命中就问人')
}

// 对应 PermissionGate._evaluate_one: 顺序就是优先级
const evaluateOne = (name, text) => {
  const stages = []
  let out = null
  for (const d of ['deny', 'ask', 'allow']) {
    const skip = out ? '已有结论, 不再检查'
      : d !== 'deny' && mode.value === 'plan' ? 'plan 模式下 ask/allow 规则不生效 (allow 也不能放行写操作)'
        : d === 'ask' && mode.value === 'bypass_permissions' ? 'bypass_permissions 跳过 ask 规则'
          : d === 'allow' && name === 'shell' && /\$\(|`/.test(text) ? '命令替换 $(…) / `…` 里能藏任何东西: 不享受 allow 规则' : ''
    const hit = skip ? null : rules.find((r) => r.on && r.decision === d && matches(r, name, text, normalized.value))
    if (hit) out = d === 'ask' ? ask(`命中 ask 规则 ${hit.tool} "${hit.pattern}"`) : { verdict: d.toUpperCase(), source: 'rule', reason: `命中 ${d} 规则 ${hit.tool} "${hit.pattern}"` }
    const n = rules.filter((r) => r.on && r.decision === d).length
    stages.push({ name: `${d} 规则`, state: hit ? 'hit' : skip ? 'skip' : 'miss', detail: hit ? `命中: ${hit.tool} "${hit.pattern}"` : skip || `${n} 条规则, 无命中 → 继续` })
  }
  const fb = out ? null : fallback(name, text)
  stages.push({ name: `模式兜底 (${mode.value})`, state: fb ? 'hit' : 'skip', detail: fb ? fb.reason : '已有结论, 不再检查' })
  return { ...(out || fb), stages, text, seen: normalized.value ? normalize(text) : text }
}

const result = computed(() => {
  const segs = tool.value === 'shell' ? arg.value.split(/&&|\|\||;|\||&|\n/).map((s) => s.trim()).filter(Boolean) : []
  const parts = (segs.length > 1 ? segs : [arg.value]).map((s) => evaluateOne(tool.value, s))
  const bad = parts.findIndex((p) => p.verdict !== 'ALLOW' && !(p.verdict === 'ASK' && human.value))
  return { parts, idx: bad < 0 ? 0 : bad }
})
const decisive = computed(() => result.value.parts[result.value.idx])
const final = computed(() => {
  const v = decisive.value.verdict
  if (v === 'ASK') return human.value ? { text: 'ASK → 人批准', cls: 'good' } : { text: 'ASK → 无人 = DENY', cls: 'bad' }
  return { text: v, cls: v === 'ALLOW' ? 'good' : 'bad' }
})
const denyHit = computed(() => {
  const hit = (norm) => result.value.parts.some((p) => rules.some((r) => r.on && r.decision === 'deny' && matches(r, tool.value, p.text, norm)))
  return { naive: hit(false), norm: hit(true) }
})
</script>

<style scoped>
.rule.off { opacity: 0.45; }
.rule.deny { border-color: var(--danger); }
.rule.ask { border-color: var(--warn); }
.rule.allow { border-color: var(--left); }
.call select { max-width: 150px; }
.call .arg { flex: 1; min-width: 160px; }
.risk { font-size: 12px; color: var(--accent); }
.call select, .arg { min-height: 32px; background: var(--bg-elev); color: var(--text); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 0 8px; font-size: 12px; max-width: 100%; min-width: 0; }
.seg { font-size: 12px; color: var(--text-dim); margin: 6px 0; word-break: break-all; }
.seg.decisive { color: var(--text); }
.trace { list-style: none; display: flex; flex-direction: column; gap: 6px; }
.trace li { display: grid; grid-template-columns: 150px 1fr; gap: 10px; align-items: center; border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 8px 10px; font-size: 12px; }
.trace li.miss { color: var(--text-muted); }
.trace li.skip { opacity: 0.4; }
.trace li.hit { border-color: var(--accent); background: var(--accent-soft); color: var(--text); }
.stage { font-size: 12px; }
.small { font-size: 12px !important; word-break: break-all; text-align: right; }
@media (max-width: 600px) { .trace li { grid-template-columns: 1fr; gap: 2px; } }
</style>
