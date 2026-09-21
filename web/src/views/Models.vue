<template>
  <div>
    <h1 class="page-title">常见模型结构 · 把主流模型拆成四类零件</h1>
    <p class="page-subtitle">
      阶段 1 在 <RepoLink path="llm_basic/" label="llm_basic/" tiny /> 里手写了每一个梯度，原理那一关已经过了。
      阶段 2 进 <RepoLink path="llm_models/" label="llm_models/" tiny /> 换成 PyTorch，
      把 attention、FFN、norm、position 四种零件装进同一个 Block，堆出 Transformer、MoE、多模态和扩散模型。
      <strong>换一个零件只需要改前向，反向交给 autograd</strong> —— 这才有可能在一个仓库里放下二十多种架构。
    </p>

    <ChapterIntro
      tldr="主流模型的差别几乎都落在 4 个槽位: attention 怎么省 KV cache, FFN 门不门控、稀不稀疏, norm 放在子层前还是后, 位置信息从哪注入。"
      question="拿到一个没读过的模型, 能不能先认出它只换了哪几个零件, 而不是把整套架构从头读一遍?"
      :goals="[
        '用同一张零件表读懂 Transformer / LLaMA / Mixtral / DeepSeek',
        '把多模态和扩散模型放回 “先 token 化, 再过 Transformer block” 这条主线',
        '知道每个小标题该对着哪一组源码看',
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

    <ModelArchitectureLab />

    <section class="section">
      <h2>阶段 2 的阅读顺序</h2>
      <p class="lead">
        先看注意力，因为它直接决定长上下文的推理成本：一个 token 在全部层上的 KV，
        LLaMA-2-7B (MHA) 要 512 KiB，LLaMA-3-8B 换成 GQA 后 128 KiB，DeepSeek-V3 换成 MLA 后 68.6 KiB。
        再看位置编码和 Block 组装，最后分两条分支：MoE 把 FFN 拆稀疏，扩散换一套生成方式。
        每一章的形状都一样 —— 上一代哪里疼，这一代怎么止疼，又引出了什么新疼。
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
      <p class="lead">读一个模型时先来这四个文件认槽位，认完再去看它自己的实现文件，一般只剩几十行。</p>
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
      <p class="lead">
        时间线上这 {{ timeline.length }} 个模型共用同一套元数据，源码列直接跳到 GitHub。
        按年份读一遍，你会发现新模型几乎从不从零开始，都是在上一个的某个槽位上改。
      </p>
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

    <!-- 本章挂载的实验台 (data/labmap/*.js) 与章末自测 (data/quiz/*.js), 没配置时不渲染 -->

    <LabMount />

    <QuizCard />


    <ChapterNav
      :prev="{ name: 'basic-optim-sample', label: '阶段 1.4 · Adam 与采样', hint: '从最小闭环进入现代模型结构' }"
      :next="{ name: 'attention', label: '阶段 2.1 · 注意力演进', hint: '先拆 KV cache 成本最高的 attention 槽位' }"
    />
  </div>
</template>

<script setup>
import LabMount from '@/components/LabMount.vue'
import QuizCard from '@/components/QuizCard.vue'
import { computed } from 'vue'
import ChapterIntro from '@/components/ChapterIntro.vue'
import ChapterNav from '@/components/ChapterNav.vue'
import ModelArchitectureLab from '@/components/labs/ModelArchitectureLab.vue'
import RepoLink from '@/components/RepoLink.vue'
import { modelChapters, timeline } from '@/data/models.js'

const sourceRows = [
  { slot: 'Attention', problem: '每个 token 要缓存多少东西: MHA → GQA → MLA → DSA 一路在砍这个数', file: 'llm_models/layers/core/attention.py' },
  { slot: 'Position', problem: '位置信息从哪进模型: Sin / Learnable 加在 embedding 上, RoPE / M-RoPE 乘在 Q/K 上', file: 'llm_models/layers/core/position_encoding.py' },
  { slot: 'Block', problem: '把 attn、ffn、norm、pos 装进同一个壳: 所有模型共用的那段控制流', file: 'llm_models/layers/core/blocks.py' },
  { slot: 'Sparse / Generative', problem: '主干之外的扩展: MoE 把 FFN 拆稀疏, SSM 换掉注意力, DiT / MM-DiT / VAR 换掉生成方式', file: 'llm_models/layers/sparse/' },
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
  transition: border-color 150ms ease-out, transform 150ms ease-out, box-shadow 150ms ease-out;
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
