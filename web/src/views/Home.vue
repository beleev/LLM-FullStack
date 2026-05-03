<template>
  <div>
    <h1 class="page-title">LLM 全栈 · 主线总览</h1>
    <p class="page-subtitle">
      把"训出一个能用、能部署、能行动的大模型系统"拆成 <strong>六个递进阶段</strong>。每个阶段对应仓库
      根目录下的一个子项目, 教程与代码逐行对照 — 你看到的每一段叙述都能在 Python
      文件里找到出处。
    </p>

    <ChapterIntro
      tldr="从 numpy 手写最小闭环 → 现代架构家族 → 规模化训练 → 任务微调 → 推理优化 → Agent 应用层, 这条主线把模型从公式带到能行动的系统。"
      question="如果一个新人要从零开始, 先看哪一段, 跳过哪一段?"
      :goals="[
        '看清 LLM 全栈的六个工程阶段, 各自解决什么问题',
        '知道每一阶段对应仓库里的哪个目录、入口文件',
        '能根据自己背景挑一条合适的阅读路径',
      ]"
      :codes="[
        { path: 'llm_basic/' },
        { path: 'llm_models/' },
        { path: 'llm_train/' },
        { path: 'llm_finetune/' },
        { path: 'llm_infer/' },
        { path: 'llm_agent/' },
      ]"
      :next-step="{ name: 'basic', label: '阶段 1 — 用 numpy 跑通整个 Transformer' }"
    />

    <!-- ── 新手 30 秒导览 ─────────────────────────────────────── -->
    <section class="section onboarding">
      <h2>新手 30 秒导览 <span class="lead-inline">第一次来? 先看这里</span></h2>
      <div class="grid grid-3 onboard-grid">
        <div class="card onboard-card">
          <div class="onboard-num">1</div>
          <h3>这是什么</h3>
          <p class="desc">
            一份<strong>边读代码、边看图</strong>的 LLM 全栈教程。每章对应仓库里一个 Python 文件, 鼠标点击即跳到 GitHub 源码。
          </p>
        </div>
        <div class="card onboard-card">
          <div class="onboard-num">2</div>
          <h3>需要先会什么</h3>
          <p class="desc">
            会一点 Python + 高中线性代数即可。<span class="muted-line">不需要 PyTorch — 阶段 1 用 numpy 把所有东西摊开。</span>
          </p>
        </div>
        <div class="card onboard-card">
          <div class="onboard-num">3</div>
          <h3>怎么读</h3>
          <p class="desc">
            <strong>顺序读</strong> 阶段 1→6, 每章先看 "本章一句话", 再点 "对应代码" 链接对照源文件。
          </p>
        </div>
      </div>
    </section>

    <!-- ── 怎么用这本教程 (图形化路径) ───────────────────── -->
    <section class="section">
      <h2>三种学习节奏 <span class="lead-inline">挑一条适合你的路径</span></h2>
      <div class="grid grid-3 path-grid">
        <div class="card path-card">
          <span class="path-tag green">🟢 入门</span>
          <h3>从零开始</h3>
          <ol class="path-steps">
            <li>读完 <router-link :to="{ name: 'basic' }">阶段 1</router-link> — 用 numpy 跑通最小闭环</li>
            <li>挑 <router-link :to="{ name: 'attention' }">注意力</router-link> 与 <router-link :to="{ name: 'blocks' }">Block 组装</router-link> 看现代零件</li>
            <li>看 <router-link :to="{ name: 'compare' }">总览对照表</router-link>, 决定下一站</li>
          </ol>
        </div>
        <div class="card path-card">
          <span class="path-tag orange">🟠 进阶</span>
          <h3>有 PyTorch 经验</h3>
          <ol class="path-steps">
            <li>跳过阶段 1, 直接看 <router-link :to="{ name: 'attention' }">阶段 2 · 注意力演进</router-link></li>
            <li>关注 <router-link :to="{ name: 'train' }">阶段 3 训练</router-link> 和 <router-link :to="{ name: 'finetune' }">阶段 4 微调</router-link></li>
            <li>再用 <router-link :to="{ name: 'infer' }">阶段 5 推理</router-link> 把模型部署起来</li>
          </ol>
        </div>
        <div class="card path-card">
          <span class="path-tag blue">🔵 工程</span>
          <h3>只关心落地</h3>
          <ol class="path-steps">
            <li>先看 <router-link :to="{ name: 'infer' }">阶段 5 · 推理优化</router-link></li>
            <li>再看 <router-link :to="{ name: 'agent' }">阶段 6 · Agent 应用层</router-link></li>
            <li>遇到原理盲点回到阶段 1/2 翻查</li>
          </ol>
        </div>
      </div>
    </section>

    <!-- ── 六阶段流程图 ───────────────────────────────────────────── -->
    <section class="section">
      <h2>六阶段 · 一张图 <span class="lead-inline">点击进入对应章节</span></h2>
      <p class="lead">
        前两阶段看清模型本身, 中间三阶段解决"怎么训、怎么改、怎么上线",
        最后一阶段把推理服务接成能调用工具、保留状态、隔离子任务的 Agent。
        每张卡都直接连到对应目录的原始代码。
      </p>

      <div class="stage-flow">
        <template v-for="(s, i) in stages" :key="s.id">
          <component
            :is="s.status === 'ready' ? 'router-link' : 'div'"
            :to="s.status === 'ready' ? toFor(s) : undefined"
            :class="['stage-card', s.status, `stage-${s.id}`]"
          >
            <div class="stage-head">
              <span class="stage-idx">{{ s.idx }}</span>
              <span class="stage-code mono">{{ s.code }}</span>
              <span :class="['stage-pill', s.status]">
                {{ s.status === 'ready' ? '就绪' : '待补' }}
              </span>
            </div>
            <h3 class="stage-title">{{ s.title }}</h3>
            <p class="stage-one">{{ s.oneliner }}</p>
            <div v-if="s.chapters" class="stage-files">
              <span v-for="c in s.chapters" :key="c.route" class="mono file-tag">{{ c.label }}</span>
            </div>
            <div v-else-if="s.files" class="stage-files">
              <RepoLink
                v-for="f in s.files"
                :key="f"
                :path="`${s.code}${f}`"
                :label="f"
                tiny
              />
            </div>
            <div v-else-if="s.note" class="stage-note">{{ s.note }}</div>
          </component>

          <div v-if="i < stages.length - 1" class="flow-arrow" aria-hidden="true">→</div>
        </template>
      </div>
    </section>

    <!-- ── 阶段 2 时间轴 (llm_models 17 模型) ─────────────────────── -->
    <section class="section">
      <h2>阶段 2 内部 · llm_models 演进时间轴</h2>
      <p class="lead">
        2017 → 2025, 三条主线 (左脑 / 眼耳 / 右脑) 各自的代表模型。
        <strong>悬停看零件配置</strong>; 点击节点跳到对应章节, 我们会用一份 PyTorch
        实现把零件还原。
      </p>

      <div class="legend">
        <span v-for="(t, k) in tracks" :key="k" class="legend-item">
          <span class="dot" :style="{ background: t.color }" />
          {{ t.label }}
        </span>
      </div>

      <div class="timeline-wrap card">
        <svg class="timeline-svg" :viewBox="`0 0 ${W} ${H}`" width="100%" :height="H">
          <g>
            <line v-for="y in years" :key="'yg-'+y"
                  :x1="xOfYear(y)" :x2="xOfYear(y)"
                  :y1="40" :y2="H - 20"
                  stroke="var(--border)" stroke-dasharray="2 4" />
            <text v-for="y in years" :key="'yt-'+y"
                  :x="xOfYear(y)" :y="H - 6"
                  text-anchor="middle" fill="var(--text-muted)"
                  font-size="11" font-family="SF Mono, Menlo, monospace">{{ y }}</text>
          </g>

          <g>
            <template v-for="(t, k) in tracks" :key="k">
              <line :x1="40" :x2="W - 20" :y1="trackY(k)" :y2="trackY(k)"
                    :stroke="t.color" stroke-opacity="0.25" stroke-width="2" />
              <text :x="16" :y="trackY(k) + 4" fill="var(--text-muted)" font-size="11">
                {{ k === 'left' ? '左脑' : k === 'eye' ? '眼耳' : '右脑' }}
              </text>
            </template>
          </g>

          <g v-for="m in timeline" :key="m.id"
             @click="goto(m.id)"
             @mouseenter="hover = m.id" @mouseleave="hover = null"
             class="node">
            <circle :cx="nodeX(m)" :cy="trackY(m.track)"
                    :r="hover === m.id ? 10 : 7"
                    :fill="tracks[m.track].color"
                    stroke="var(--bg-card)" stroke-width="2.5" />
            <text :x="labelX(m)"
                  :y="labelY(m)"
                  text-anchor="middle"
                  font-size="11"
                  :fill="hover === m.id ? 'var(--text)' : 'var(--text-muted)'"
                  :font-weight="hover === m.id ? 600 : 400">
              {{ m.name }}
            </text>
          </g>
        </svg>

        <div v-if="hover" class="hover-card">
          <div class="hover-head">
            <span :class="['pill', tracks[hovered.track].cls]">{{ hovered.kind }}</span>
            <strong>{{ hovered.name }}</strong>
            <span class="mono year">{{ hovered.year }}</span>
          </div>
          <p class="desc">{{ hovered.blurb }}</p>
          <div class="parts">
            <div v-for="(v, k) in hovered.parts" :key="k" class="part">
              <span class="k">{{ labels[k] }}</span>
              <span class="v mono">{{ v }}</span>
            </div>
          </div>
          <div class="hover-foot mono">
            <RepoLink :path="`llm_models/${hovered.file}`" :label="hovered.file" tiny />
          </div>
        </div>
      </div>
    </section>

    <section class="section">
      <h2>六个目录之间的依赖</h2>
      <p class="lead">
        这不是六块孤立知识: 后一阶段会复用前一阶段的抽象, 只是把约束换成更真实的工程约束。
      </p>
      <div class="card" style="padding: 0; overflow-x: auto;">
        <table class="path-table">
          <thead>
            <tr>
              <th>阶段</th>
              <th>接住上一章什么</th>
              <th>新增约束</th>
              <th>读代码时先看</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="r in pathRows" :key="r.stage">
              <td class="stage-name">{{ r.stage }}</td>
              <td>{{ r.from }}</td>
              <td>{{ r.constraint }}</td>
              <td><RepoLink :path="r.file" tiny /></td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <ChapterNav :next="{ name: 'basic', label: '阶段 1 · llm_basic', hint: '用 numpy 把 forward / backward / 采样 全部手写一遍' }" />
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { tracks, timeline, years, findModel, stages } from '@/data/models.js'
import ChapterIntro from '@/components/ChapterIntro.vue'
import ChapterNav from '@/components/ChapterNav.vue'
import RepoLink from '@/components/RepoLink.vue'

const router = useRouter()
const hover = ref(null)
const hovered = computed(() => hover.value ? findModel(hover.value) : null)

const W = 960
const H = 300
const paddingLeft = 70
const paddingRight = 40
const yearMin = Math.min(...years)
const yearMax = Math.max(...years)

const xOfYear = (y) => {
  if (yearMin === yearMax) return W / 2
  return paddingLeft + (W - paddingLeft - paddingRight) * (y - yearMin) / (yearMax - yearMin)
}
const trackY = (k) => ({ left: 80, eye: 160, right: 240 }[k])

const sameSlot = (m) =>
  timeline
    .filter(x => x.year === m.year && x.track === m.track)
    .sort((a, b) => a.id.localeCompare(b.id))

const slotOffset = (m, step) => {
  const same = sameSlot(m)
  if (same.length <= 1) return 0
  const idx = same.findIndex(x => x.id === m.id)
  return (idx - (same.length - 1) / 2) * step
}

const nodeX = (m) => xOfYear(m.year) + slotOffset(m, 30)
const labelX = (m) => xOfYear(m.year) + slotOffset(m, 84)
const labelY = (m) => trackY(m.track) - 28 + slotOffset(m, 18)

const labels = {
  attn: '注意力',
  ffn:  'FFN',
  norm: '归一化',
  pos:  '位置编码',
}

const pathRows = [
  { stage: 'llm_basic',    from: '从 token 到 loss 的完整闭环',     constraint: '不用 autograd, 反向必须手写并 gradcheck',  file: 'llm_basic/model.py' },
  { stage: 'llm_models',   from: '把单头、单层替换成现代组件',       constraint: '不同模型只是 attn / ffn / norm / pos 的组合', file: 'llm_models/layers/core/' },
  { stage: 'llm_train',    from: '同一个 loss/backward/update 主循环', constraint: 'batch、层、状态、精度、通信都要切分',     file: 'llm_train/full_loop/demo.py' },
  { stage: 'llm_finetune', from: '已有 base model 和标准 LM loss',   constraint: '数据少、算力少、目标更窄, 需要 SFT/LoRA/DPO', file: 'llm_finetune/methods/' },
  { stage: 'llm_infer',    from: '训练好的权重和自回归生成',         constraint: '延迟、吞吐、显存、合法输出同时受限',      file: 'llm_infer/full_engine/' },
  { stage: 'llm_agent',    from: '可服务化的模型生成能力',           constraint: '工具、权限、上下文、记忆、状态和子任务同时受控', file: 'llm_agent/full_loop/demo.py' },
]

// 节点 → 章节路由 (集中放在主入口, 避免散落到各章 view 中)
const goto = (id) => {
  const map = {
    transformer: 'blocks', bert: 'blocks',
    gpt3: 'attention', llama: 'attention', mamba: 'blocks',
    mixtral: 'moe', deepseek_v3: 'moe', deepseek_v32: 'attention',
    clip: 'compare', whisper: 'compare', qwen2_vl: 'position', omni: 'position',
    vae: 'diffusion', dit: 'diffusion', mmdit: 'diffusion',
    video_dit: 'diffusion', var: 'diffusion',
  }
  router.push({ name: map[id] || 'compare', query: { focus: id } })
}

// 阶段卡片跳转 — chapters 形式取首章
const toFor = (s) => {
  if (s.route) return { name: s.route }
  if (s.chapters?.length) return { name: s.chapters[0].route }
  return { name: 'home' }
}
</script>

<style scoped>
.lead-inline {
  font-size: 12px;
  color: var(--text-dim);
  font-weight: 400;
  letter-spacing: 0.2px;
  margin-left: 8px;
}

/* ── 新手导览 + 学习路径 ─────────────────────── */
.onboard-grid, .path-grid { gap: 14px; }
.onboard-card {
  position: relative;
  padding-top: 26px;
}
.onboard-num {
  position: absolute;
  top: -10px;
  left: 16px;
  width: 28px;
  height: 28px;
  border-radius: 50%;
  background: var(--accent);
  color: #fff;
  display: grid;
  place-items: center;
  font-weight: 600;
  font-size: 13px;
  font-family: "SF Mono", Menlo, monospace;
  box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}
.onboard-card h3 { margin-bottom: 8px; }
.muted-line { color: var(--text-dim); font-size: 12px; }

.path-card {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.path-tag {
  align-self: flex-start;
  font-size: 10px;
  padding: 2px 8px;
  border-radius: 99px;
  background: var(--bg-elev);
  border: 1px solid var(--border);
  font-family: "SF Mono", Menlo, monospace;
  letter-spacing: 0.4px;
}
.path-tag.green  { color: var(--left);  border-color: color-mix(in srgb, var(--left)  40%, var(--border)); }
.path-tag.orange { color: var(--eye);   border-color: color-mix(in srgb, var(--eye)   40%, var(--border)); }
.path-tag.blue   { color: var(--accent); border-color: color-mix(in srgb, var(--accent) 40%, var(--border)); }

.path-card h3 { margin: 4px 0 0; }
.path-steps {
  list-style: none;
  counter-reset: step;
  padding: 0;
  margin: 4px 0 0;
  font-size: 12.5px;
  color: var(--text-muted);
  line-height: 1.7;
}
.path-steps li {
  counter-increment: step;
  position: relative;
  padding-left: 22px;
  padding-bottom: 4px;
}
.path-steps li::before {
  content: counter(step);
  position: absolute;
  left: 0;
  top: 2px;
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: var(--bg-elev);
  border: 1px solid var(--border);
  color: var(--text-dim);
  font-family: "SF Mono", Menlo, monospace;
  font-size: 10px;
  display: grid;
  place-items: center;
}

/* ── 阶段流程卡 ───────────────────────────── */
.stage-flow {
  display: flex;
  align-items: stretch;
  gap: 6px;
  overflow-x: auto;
  padding-bottom: 4px;
}
.stage-card {
  flex: 1 1 0;
  min-width: 220px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-top-width: 3px;
  border-top-color: var(--accent);
  border-radius: var(--radius-sm);
  padding: 14px 16px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  text-decoration: none;
  color: inherit;
  transition: all 0.15s;
}
.stage-card.ready:hover {
  border-color: var(--accent);
  transform: translateY(-2px);
  text-decoration: none;
}
.stage-card.planned {
  border-style: dashed;
  border-top-style: dashed;
  border-top-color: var(--text-dim);
  background: linear-gradient(180deg, var(--bg-elev) 0%, var(--bg-card) 100%);
  cursor: not-allowed;
  opacity: 0.85;
}

.stage-card.stage-basic    { border-top-color: var(--left); }
.stage-card.stage-models   { border-top-color: var(--accent); }
.stage-card.stage-train    { border-top-color: var(--warn); }
.stage-card.stage-finetune { border-top-color: var(--eye); }
.stage-card.stage-infer    { border-top-color: var(--right); }

.stage-head {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 11px;
}
.stage-idx {
  width: 22px; height: 22px;
  border-radius: 50%;
  background: var(--bg-elev);
  border: 1px solid var(--border);
  display: grid; place-items: center;
  font-size: 11px;
  font-family: "SF Mono", Menlo, monospace;
  color: var(--text);
}
.stage-code { color: var(--text-muted); font-size: 11px; }
.stage-pill {
  margin-left: auto;
  font-size: 10px;
  padding: 1px 7px;
  border-radius: 99px;
  font-family: "SF Mono", Menlo, monospace;
  letter-spacing: 0.4px;
}
.stage-pill.ready   { background: color-mix(in srgb, var(--left) 18%, transparent); color: var(--left); }
.stage-pill.planned { background: var(--bg-elev); color: var(--text-dim); border: 1px dashed var(--border); }

.stage-title {
  font-size: 13.5px;
  font-weight: 600;
  margin-bottom: 0;
}
.stage-one {
  font-size: 12px;
  color: var(--text-muted);
  line-height: 1.55;
  margin: 0;
}
.stage-files {
  display: flex; flex-wrap: wrap; gap: 4px;
  margin-top: auto;
}
.file-tag {
  font-size: 10.5px;
  padding: 2px 6px;
  border-radius: 3px;
  background: var(--code-bg);
  border: 1px solid var(--border);
  color: var(--code-text);
}
.stage-note {
  font-size: 11.5px;
  color: var(--text-dim);
  font-style: italic;
  margin-top: auto;
}

.flow-arrow {
  align-self: center;
  color: var(--text-dim);
  font-size: 18px;
  user-select: none;
  flex-shrink: 0;
}

@media (max-width: 1100px) {
  .stage-flow { flex-direction: column; }
  .flow-arrow { transform: rotate(90deg); }
}

/* ── 时间轴 ───────────────────────────────── */
.legend {
  display: flex;
  gap: 24px;
  padding: 12px 0 16px;
  font-size: 13px;
  color: var(--text-muted);
}
.legend-item { display: inline-flex; align-items: center; gap: 8px; }
.dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
}

.timeline-wrap {
  position: relative;
  overflow-x: auto;
}
.timeline-svg {
  display: block;
  min-width: 960px;
}
.node { cursor: pointer; transition: all 0.15s; }

.hover-card {
  position: absolute;
  right: 20px;
  top: 20px;
  width: 320px;
  background: var(--bg-elev);
  border: 1px solid var(--border-strong);
  border-radius: var(--radius-sm);
  padding: 14px 16px;
  box-shadow: var(--shadow);
  font-size: 13px;
}
.hover-head {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}
.hover-head strong {
  font-size: 15px;
  color: var(--text);
  flex: 1;
}
.hover-head .year {
  color: var(--text-dim);
  font-size: 12px;
}
.hover-card .desc { color: var(--text-muted); margin-bottom: 12px; }
.hover-card .parts { display: grid; grid-template-columns: 1fr 1fr; gap: 6px 12px; }
.hover-card .part { font-size: 12px; }
.hover-card .part .k { color: var(--text-dim); display: block; font-size: 11px; }
.hover-card .part .v { color: var(--text); }
.hover-card .hover-foot {
  margin-top: 10px;
  padding-top: 8px;
  border-top: 1px dashed var(--border);
  font-size: 11px;
  color: var(--text-dim);
}

table.path-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
table.path-table th {
  text-align: left;
  padding: 12px 14px;
  background: var(--bg-elev);
  color: var(--text-muted);
  font-size: 11px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.7px;
  border-bottom: 1px solid var(--border-strong);
}
table.path-table td {
  padding: 10px 14px;
  border-bottom: 1px solid var(--border);
  vertical-align: top;
}
table.path-table .stage-name { color: var(--text); font-weight: 600; }
table.path-table .small { font-size: 12px; color: var(--text-muted); }
</style>
