<template>
  <div>
    <h1 class="page-title">常见模型结构 · 把主流模型拆成四类零件</h1>
    <p class="page-subtitle">
      阶段 2 从 <RepoLink path="llm_basic/" label="llm_basic/" tiny /> 进入
      <RepoLink path="llm_models/" label="llm_models/" tiny />：不再手写每个梯度，
      而是把 attention、FFN、norm、position 这些零件组合成 Transformer、MoE、多模态和扩散模型。
    </p>

    <ChapterIntro
      tldr="主流模型的差别大多落在 4 个槽位: attention 怎么省 KV cache, FFN 是否门控/稀疏, norm 放哪里, position 怎么注入。"
      question="读一个新模型时, 能不能先判断它只是替换了哪些零件, 而不是把整套架构重新读一遍?"
      :goals="[
        '用同一张零件表理解 Transformer / LLaMA / Mixtral / DeepSeek',
        '把多模态和扩散模型放回 token 化 + Transformer block 的主线',
        '知道阶段 2 每个小标题应该对照哪一组源码',
      ]"
      :codes="[
        { path: 'llm_models/layers/core/' },
        { path: 'llm_models/models/' },
        { path: 'llm_models/layers/sparse/' },
        { path: 'llm_models/models/generative/' },
      ]"
      :prereq="{ name: 'basic-optim-sample', label: '阶段 1.4 · Adam 与采样' }"
      :next-step="{ name: 'attention', label: '阶段 2.1 · 注意力演进' }"
    />

    <section class="section">
      <h2>阶段 2 的阅读顺序</h2>
      <p class="lead">
        先看注意力，因为它决定长上下文推理成本；再看位置编码和 Block 组装；最后进入 MoE 与扩散生成两条分支。
      </p>
      <div class="grid grid-3">
        <router-link
          v-for="(c, i) in modelChapters"
          :key="c.route"
          :to="{ name: c.route }"
          class="card chapter-card"
        >
          <div class="chapter-idx">2.{{ i + 1 }}</div>
          <h3>{{ c.label }}</h3>
          <p class="desc">{{ c.hint }}</p>
        </router-link>
      </div>
    </section>

    <section class="section">
      <h2>四个源码入口</h2>
      <p class="lead">读模型代码时先从这些入口找槽位，再进入具体模型实现。</p>
      <div class="card" style="padding: 0; overflow-x: auto;">
        <table class="models-table">
          <thead>
            <tr>
              <th>槽位</th>
              <th>解决的问题</th>
              <th>源码</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="r in sourceRows" :key="r.slot">
              <td class="slot">{{ r.slot }}</td>
              <td>{{ r.problem }}</td>
              <td><RepoLink :path="r.file" tiny /></td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <section class="section">
      <h2>模型族谱</h2>
      <p class="lead">阶段 2 的 20 个模型共享同一套元数据，源码列已全部跳转到 GitHub。</p>
      <div class="card" style="padding: 0; overflow-x: auto;">
        <table class="models-table">
          <thead>
            <tr>
              <th>模型</th>
              <th>年份</th>
              <th>类型</th>
              <th>源码</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="m in sortedTimeline" :key="m.id">
              <td class="slot">{{ m.name }}</td>
              <td class="mono muted">{{ m.year }}</td>
              <td>{{ m.kind }}</td>
              <td><RepoLink :path="`llm_models/${m.file}`" tiny /></td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <ChapterNav
      :prev="{ name: 'basic-optim-sample', label: '阶段 1.4 · Adam 与采样', hint: '从最小闭环进入现代模型结构' }"
      :next="{ name: 'attention', label: '阶段 2.1 · 注意力演进', hint: '先拆 KV cache 成本最高的 attention 槽位' }"
    />
  </div>
</template>

<script setup>
import { computed } from 'vue'
import ChapterIntro from '@/components/ChapterIntro.vue'
import ChapterNav from '@/components/ChapterNav.vue'
import RepoLink from '@/components/RepoLink.vue'
import { modelChapters, timeline } from '@/data/models.js'

const sourceRows = [
  { slot: 'Attention', problem: 'MHA / GQA / MLA / DSA 如何改变 KV cache 和长上下文成本', file: 'llm_models/layers/core/attention.py' },
  { slot: 'Position', problem: 'Sin / Learnable / RoPE / M-RoPE 如何把位置信息注入模型', file: 'llm_models/layers/core/position_encoding.py' },
  { slot: 'Block', problem: 'attn、ffn、norm、pos 如何拼成可复用 Transformer block', file: 'llm_models/layers/core/blocks.py' },
  { slot: 'Sparse / Generative', problem: 'MoE、SSM、DiT、MM-DiT、VAR 如何扩展主干结构', file: 'llm_models/layers/sparse/' },
]

const sortedTimeline = computed(() =>
  [...timeline].sort((a, b) => a.year - b.year || a.track.localeCompare(b.track))
)
</script>

<style scoped>
.chapter-card {
  display: block;
  position: relative;
  color: inherit;
  text-decoration: none;
  transition: all 0.15s;
}
.chapter-card:hover {
  border-color: var(--accent);
  transform: translateY(-2px);
  text-decoration: none;
}
.chapter-idx {
  position: absolute;
  top: 14px;
  right: 16px;
  color: var(--accent);
  font-family: "SF Mono", Menlo, monospace;
  font-size: 12px;
}
.chapter-card h3 { padding-right: 42px; margin-bottom: 8px; }

table.models-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
table.models-table th {
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
table.models-table td {
  padding: 10px 14px;
  border-bottom: 1px solid var(--border);
}
table.models-table .slot {
  color: var(--text);
  font-weight: 600;
}
.muted { color: var(--text-muted); }
</style>
