<template>
  <div>
    <h1 class="page-title">总览对照表</h1>
    <p class="page-subtitle">
      一张表看完模型零件配置, 再把六个工程目录放回同一条主线:
      原理闭环、架构替换、规模化训练、任务适配、推理服务、Agent 应用层。
    </p>

    <ChapterIntro
      tldr="竖着读模型表 = 一个组件维度的演进史; 横着读六阶段表 = 一个 LLM 从公式到可行动系统的工程闭环。"
      question="遇到一个真实大模型系统, 你能快速判断问题发生在结构、训练、微调、推理还是 Agent harness 层吗?"
      :goals="[
        '快速对照 17 种主流模型的零件配置 (attn / ffn / norm / pos)',
        '看清「早期 → 现代」每个维度的演进轨迹',
        '把六阶段闭环放回同一张地图: 哪一层出问题该读哪个目录',
      ]"
      :codes="[
        { path: 'llm_basic/' },
        { path: 'llm_models/' },
        { path: 'llm_train/' },
        { path: 'llm_finetune/' },
        { path: 'llm_infer/' },
        { path: 'llm_agent/' },
      ]"
      :prereq="{ name: 'agent-full-loop', label: '阶段 6.6 · mini Agent harness' }"
    />

    <!-- 过滤器 -->
    <div class="btn-group" style="margin-bottom: 16px;">
      <button :class="{ active: trackFilter === null }" @click="trackFilter = null">全部</button>
      <button v-for="(t, k) in tracks" :key="k"
              :class="{ active: trackFilter === k }"
              @click="trackFilter = k">
        <span class="dot" :style="{ background: t.color }"></span>
        {{ t.label }}
      </button>
    </div>

    <div class="card" style="padding: 0; overflow-x: auto;">
      <table class="compare-table">
        <thead>
          <tr>
            <th>模型</th>
            <th>年份</th>
            <th>类型</th>
            <th>注意力</th>
            <th>FFN</th>
            <th>归一化</th>
            <th>位置编码</th>
            <th>源码</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="m in filtered" :key="m.id"
              :class="{ highlight: focusId === m.id }">
            <td>
              <span :class="['pill', tracks[m.track].cls]" style="font-size: 10px;">{{ trackShort(m.track) }}</span>
              <strong>{{ m.name }}</strong>
            </td>
            <td class="mono muted">{{ m.year }}</td>
            <td class="muted small">{{ m.kind }}</td>
            <td><code class="inline">{{ m.parts.attn }}</code></td>
            <td><code class="inline">{{ m.parts.ffn }}</code></td>
            <td><code class="inline">{{ m.parts.norm }}</code></td>
            <td><code class="inline">{{ m.parts.pos }}</code></td>
            <td>
              <RepoLink :path="modelFilePath(m.file)" tiny />
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <section class="section">
      <h2>设计取舍速查</h2>
      <p class="lead">每一行是一个维度, 从左到右是"早期 → 现代"的演进。</p>
      <div class="card">
        <table class="compare-table tradeoffs">
          <thead>
            <tr>
              <th>维度</th>
              <th>早期 (2017–2020)</th>
              <th>现代 (2023–2025)</th>
              <th>代码位置</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="row in tradeoffs" :key="row.topic">
              <td class="row-label">{{ row.topic }}</td>
              <td><code class="inline">{{ row.early }}</code></td>
              <td><code class="inline">{{ row.modern }}</code></td>
              <td>
                <span class="files-cell">
                  <RepoLink v-for="p in splitFiles(row.file)" :key="p" :path="prefixModels(p)" tiny />
                </span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <section class="section">
      <h2>六目录工程闭环</h2>
      <p class="lead">
        每个目录都只回答一个问题。按这个顺序读, 后一章就是把前一章的抽象放进更真实的约束里。
      </p>
      <div class="card" style="padding: 0; overflow-x: auto;">
        <table class="compare-table">
          <thead>
            <tr>
              <th>目录</th>
              <th>核心问题</th>
              <th>关键联系</th>
              <th>入口代码</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="row in projectRows" :key="row.code">
              <td class="row-label">
                <RepoLink :path="row.code" tiny variant="plain" />
              </td>
              <td>{{ row.question }}</td>
              <td>{{ row.link }}</td>
              <td>
                <span class="files-cell">
                  <RepoLink v-for="p in splitFiles(row.file)" :key="p" :path="prefixForCode(row.code, p)" tiny />
                </span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <section class="section">
      <h2>完整学习路径 · 按主线回顾</h2>
      <p class="lead">下面按六阶段主线给出快速跳转入口, 每一块都能落到本仓库原始代码。</p>
      <div class="grid grid-3">
        <router-link :to="{ name: 'basic' }" class="card chapter-card">
          <div class="chapter-idx">1</div>
          <h3>llm_basic · 最小闭环</h3>
          <p class="desc">numpy 手写 forward / backward / Adam / 采样</p>
        </router-link>
        <router-link :to="{ name: 'attention' }" class="card chapter-card">
          <div class="chapter-idx">2</div>
          <h3>llm_models · 架构家族</h3>
          <p class="desc">Attention / Position / Blocks / MoE / Diffusion 五章合集</p>
        </router-link>
        <router-link :to="{ name: 'train' }" class="card chapter-card">
          <div class="chapter-idx">3</div>
          <h3>llm_train · 规模化训练</h3>
          <p class="desc">DDP / TP / PP / ZeRO / AMP / checkpoint 组合闭环</p>
        </router-link>
        <router-link :to="{ name: 'finetune' }" class="card chapter-card">
          <div class="chapter-idx">4</div>
          <h3>llm_finetune · 任务适配</h3>
          <p class="desc">SFT / LoRA / DPO 三柱并排</p>
        </router-link>
        <router-link :to="{ name: 'infer' }" class="card chapter-card">
          <div class="chapter-idx">5</div>
          <h3>llm_infer · 推理优化</h3>
          <p class="desc">KV cache / paged attention / continuous batching / mini-vLLM</p>
        </router-link>
        <router-link :to="{ name: 'agent' }" class="card chapter-card">
          <div class="chapter-idx">6</div>
          <h3>llm_agent · Agent 应用层</h3>
          <p class="desc">Agent loop / tools / permissions / memory / subagents</p>
        </router-link>
      </div>
    </section>

    <ChapterNav
      :prev="{ name: 'agent-full-loop', label: '阶段 6.6 · mini Agent harness', hint: '从应用层闭环回到全局地图' }"
      :next="{ name: 'home', label: '返回主线总览', hint: '六阶段地图全景重看一遍' }"
    />
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useRoute } from 'vue-router'
import { tracks, timeline } from '@/data/models.js'
import ChapterIntro from '@/components/ChapterIntro.vue'
import ChapterNav from '@/components/ChapterNav.vue'
import RepoLink from '@/components/RepoLink.vue'

// 把 "a.py + b.py" 形式的多文件 / 含通配符的文件描述拆成可链接片段
const splitFiles = (s) => {
  if (!s) return []
  return String(s)
    .split(/(?:\s\+\s|\s·\s|,\s*)/)
    .map(x => x.trim())
    .filter(Boolean)
}

// 模型表里的 m.file 是相对 llm_models/ 的, 这里补回前缀
const modelFilePath = (f) => f.startsWith('llm_') ? f : `llm_models/${f}`

// tradeoffs 表的 file 是相对 llm_models/ 的
const prefixModels = (p) => p.startsWith('llm_') ? p : `llm_models/${p}`

// projectRows 的 file 是相对该 stage code 的目录的 (例如 row.code = 'llm_basic/')
const prefixForCode = (code, p) => {
  if (p.startsWith('llm_')) return p
  return `${code}${p}`
}

const route = useRoute()
const focusId = ref(route.query.focus || null)
const trackFilter = ref(null)

const filtered = computed(() => {
  const sorted = [...timeline].sort((a, b) => a.year - b.year || a.track.localeCompare(b.track))
  if (!trackFilter.value) return sorted
  return sorted.filter(m => m.track === trackFilter.value)
})

const trackShort = (k) => ({ left: '语言', eye: '多模', right: '生成' }[k])

const tradeoffs = [
  { topic: 'Normalization',    early: 'Post-LN (需 warmup)', modern: 'Pre-LN + RMSNorm',
    file: 'layers/core/{blocks,normalization}.py' },
  { topic: 'FFN 激活',         early: 'ReLU',               modern: 'GELU → SwiGLU (门控)',
    file: 'layers/core/feedforward.py' },
  { topic: '位置编码',          early: 'Sinusoidal / Learnable', modern: 'RoPE → M-RoPE',
    file: 'layers/core/position_encoding.py' },
  { topic: 'KV cache',         early: 'MHA (最大)',         modern: 'MQA → GQA → MLA → MLA+DSA',
    file: 'layers/core/attention.py' },
  { topic: 'FFN 结构',         early: '单 FFN',             modern: 'Mixtral MoE → DeepSeekMoE (sigmoid + shared + aux-free)',
    file: 'layers/sparse/moe.py + models/moe/deepseekV3.py' },
  { topic: '序列建模',          early: 'Attention O(T²)',   modern: 'Mamba SSM O(T) / DSA O(T·k)',
    file: 'layers/sparse/ssm.py + layers/core/attention.py' },
  { topic: '多模态融合',        early: 'Cross-attention (Flamingo)', modern: 'Prefix-token (VL) / 双脑 (Omni)',
    file: 'models/multimodal/{qwen2_vl,qwen2_5_omni}.py' },
  { topic: '多模态对齐',        early: 'VSE++ triplet',      modern: 'CLIP 对称对比 + 可学温度',
    file: 'models/multimodal/clip.py' },
  { topic: '扩散骨架',         early: 'UNet (SD 1.5)',      modern: 'DiT + adaLN-Zero (SD3/FLUX/Sora)',
    file: 'models/generative/dit.py + layers/diffusion/adaln.py' },
  { topic: '扩散目标',         early: 'ε-prediction (DDPM)', modern: 'Rectified Flow v-pred (SD3)',
    file: 'training/diffusion.py' },
  { topic: '图像生成范式',      early: '扩散连续去噪',       modern: '自回归 next-token (VAR/LlamaGen)',
    file: 'models/generative/var.py' },
]

const projectRows = [
  {
    code: 'llm_basic/',
    question: 'Transformer 的 forward/backward/update 最小闭环是什么?',
    link: '给后续所有 PyTorch/autograd 代码提供手写基线。',
    file: 'model.py · train.py · optim.py',
  },
  {
    code: 'llm_models/',
    question: '主流模型的差别到底落在哪些零件上?',
    link: '把 basic 的单层单头替换成 attention/ffn/norm/pos 的现代组合。',
    file: 'layers/core/ · models/',
  },
  {
    code: 'llm_train/',
    question: '同一个训练循环如何扩到多卡并保持等价?',
    link: '把 loss/grad/update 拆成 batch、矩阵、层、状态、精度和通信。',
    file: 'm01..m10/demo.py · full_loop/demo.py',
  },
  {
    code: 'llm_finetune/',
    question: '如何用少量数据和参数把 base model 拨到任务上?',
    link: 'SFT 改数据目标, LoRA 改可训练参数, DPO 改偏好优化流程。',
    file: 'methods/{sft,lora,dpo}.py',
  },
  {
    code: 'llm_infer/',
    question: '训练好的模型如何低延迟、高吞吐、可控地产出 token?',
    link: '把自回归生成接上 KV cache、分页、调度、前缀复用和采样约束。',
    file: 'm01..m15/demo.py · full_engine/engine.py',
  },
  {
    code: 'llm_agent/',
    question: '推理服务如何变成能调用工具、保留状态、隔离子任务的系统?',
    link: '把模型输出接上工具、权限、上下文、记忆、Hook、持久化和子智能体。',
    file: 'm01..m07/demo.py · full_loop/demo.py',
  },
]
</script>

<style scoped>
.files-cell {
  display: inline-flex;
  flex-wrap: wrap;
  gap: 4px;
  align-items: center;
}

table.compare-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
table.compare-table thead {
  background: var(--bg-elev);
}
table.compare-table th {
  text-align: left;
  padding: 12px 14px;
  color: var(--text-muted);
  font-size: 11px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.7px;
  border-bottom: 1px solid var(--border-strong);
}
table.compare-table td {
  padding: 10px 14px;
  border-bottom: 1px solid var(--border);
  vertical-align: middle;
}
table.compare-table tbody tr:hover {
  background: var(--bg-elev);
}
table.compare-table tbody tr.highlight {
  background: color-mix(in srgb, var(--accent) 10%, transparent);
}
table.compare-table tbody tr.highlight td:first-child {
  border-left: 2px solid var(--accent);
}
table.compare-table strong {
  margin-left: 8px;
  color: var(--text);
}
table.compare-table .muted { color: var(--text-muted); }
table.compare-table .small { font-size: 12px; }
table.compare-table .row-label { color: var(--text-muted); }

.dot {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  margin-right: 6px;
}

.chapter-card {
  display: block;
  text-decoration: none;
  color: inherit;
  position: relative;
  transition: all 0.15s;
}
.chapter-card:hover {
  border-color: var(--accent);
  transform: translateY(-2px);
  text-decoration: none;
}
.chapter-idx {
  position: absolute;
  top: 14px; right: 16px;
  width: 26px; height: 26px;
  border-radius: 50%;
  background: var(--accent-soft);
  color: var(--accent);
  display: grid;
  place-items: center;
  font-size: 13px;
  font-weight: 600;
  font-family: "SF Mono", Menlo, monospace;
}
.chapter-card h3 { padding-right: 40px; margin-bottom: 4px; }
.chapter-card.disabled {
  border-style: dashed;
  cursor: not-allowed;
  opacity: 0.55;
}
.chapter-card.disabled:hover {
  border-color: var(--border);
  transform: none;
}
.chapter-card.disabled .chapter-idx {
  background: var(--bg-elev);
  color: var(--text-dim);
}
</style>
