<template>
  <div>
    <h1 class="page-title">推理与部署优化 · 从能生成到高吞吐服务</h1>
    <p class="page-subtitle">
      <RepoLink path="llm_infer/" label="llm_infer/" tiny /> 把训练好的自回归模型搬进服务环境:
      还是一个接一个地生成 token, 但目标从"loss 下降"换成了"首 token 快、吞吐高、显存稳、输出可控"。
    </p>

    <ChapterIntro
      tldr="decode 每出 1 个 token, 都要把整份权重和全部 KV 读一遍 —— 卡住的是带宽, 不是算力。所以这 22 个模块几乎都在省 KV 的搬运和存储: cache 省重算, 分页省碎片, 前缀复用省重复 prefill, 连续批省空转, 量化和 GQA/MLA 省字节。"
      question="为什么训练时最贵的是反向, 推理时最贵的却常常是 KV cache、调度和内存带宽?"
      :goals="[
        '说清 KV cache / 分页 / 前缀复用各自省掉的到底是什么',
        '看懂连续批和分块 prefill 怎么把 GPU 喂饱, 以及代价落在谁头上',
        '把 mini-vLLM 的主循环逐行对上真实的 vLLM / SGLang',
      ]"
      :codes="[
        { path: 'llm_infer/core/' },
        { path: 'llm_infer/full_engine/' },
        { path: 'llm_infer/m01_kv_cache/' },
        { path: 'llm_infer/m03_continuous_batching/' },
      ]"
      :prereq="prevChapter"
      :next-step="{ name: 'infer-kv-memory', label: '下一章 · KV 与缓存内存' }"
    />

    <section class="section">
      <h2>1. 推理优化的递进关系</h2>
      <p class="lead">
        先把逐步重算改成增量 decode, 再把 KV 显存做成可分配资源, 然后让请求每步重新组成 batch。
        后面的前缀复用、投机解码、量化、结构化输出都挂在这条服务主循环上, 顺序拆开读就不会乱。
      </p>
      <EvolutionChain
        title="从朴素 generate 到 mini-vLLM"
        subtitle="每一步优化都保留同一个模型语义: logits 不变, 只是执行路径更省。"
        :steps="inferChain"
      />
    </section>

    <section class="section">
      <h2>2. 第一性瓶颈 · 每步重算 vs KV cache</h2>
      <p class="lead">
        没有 cache 时, 第 t 步要把 prompt 加上已生成的 t 个 token 整段再 prefill 一遍。
        有 cache 后 prefill 只跑一次, decode 只给新 token 追加 K/V ——
        m01 的 demo 里累计过一遍模型的 token 数从 688 掉到 37 (18.6×), 输出 ids 逐个不变。
      </p>
      <div class="grid grid-2" style="gap: 16px;">
        <div class="card">
          <h3>朴素路径 <span class="tag">重复 prefill</span></h3>
          <pre class="code">{{ noCacheCode }}</pre>
          <p class="hint">
            每一步的成本随已生成长度往上爬。demo 会把这条路径和 cache 路径的输出 ids 对齐,
            确认优化没有改掉生成结果。
          </p>
        </div>
        <div class="card">
          <h3>增量路径 <span class="tag">KV cache</span></h3>
          <pre class="code">{{ cacheCode }}</pre>
          <p class="hint">
            首步 prefill 存下每层 K/V, 之后每次 <code class="inline">decode_step</code> 只喂一个新 token。
            省掉的是旧 token 的 K/V 投影和 MLP 重算 —— 新 query 仍要和全部 t 个 key 做点积。
            对应 <RepoLink path="llm_infer/m01_kv_cache/demo.py" label="llm_infer/m01_kv_cache/demo.py" tiny />。
          </p>
        </div>
      </div>
    </section>

    <section class="section">
      <h2>3. 服务端三件套 · 内存、调度、复用</h2>
      <p class="lead">
        KV cache 只把单条请求变快。服务端还要面对长度各异的请求、同时到达的并发、共享的 system prompt 和显存碎片。
        下面三个抽象就是 vLLM / SGLang 这类系统的骨架。
      </p>
      <div class="grid grid-3">
        <div v-for="p in servingPrimitives" :key="p.name" class="card primitive-card">
          <h3>{{ p.name }} <span class="tag">{{ p.tag }}</span></h3>
          <p class="desc">{{ p.desc }}</p>
          <pre class="code">{{ p.code }}</pre>
          <p class="hint"><CodeRef :value="p.file" base="llm_infer/" tiny /></p>
        </div>
      </div>
    </section>

    <section class="section">
      <h2>4. 加速、压缩与约束</h2>
      <p class="lead">
        cache 和调度搭起骨架, 下面这些模块分别压计算、压存储、控采样、约束输出。
        它们大多能叠加, 但每一种都有自己的正确性边界: 分页、调度、前缀复用、投机解码是精确的 (输出逐 token 不变),
        量化、attention sink、稀疏注意力是有损的, 要看误差。
      </p>
      <div class="card" style="padding: 0; overflow-x: auto;">
        <table class="infer-table">
          <thead>
            <tr>
              <th>模块</th>
              <th>优化对象</th>
              <th>和主线的关系</th>
              <th>原始代码</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="m in inferModules" :key="m.id">
              <td class="axis">{{ m.name }}</td>
              <td>{{ m.concept }}</td>
              <td>{{ m.link }}</td>
              <td class="mono small"><RepoLink :path="m.file" tiny /></td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <section class="section">
      <h2>5. full_engine · 把模块接成服务主循环</h2>
      <p class="lead">
        <RepoLink path="llm_infer/full_engine/engine.py" label="full_engine/engine.py" tiny /> 是最值得对着源码读的一页。
        它不是完整的 vLLM, 但把调度 (m03)、真分页 KV pool (m02)、前缀复用 (m04)、分块 prefill (m06)、抢占和采样 (m10)
        串在同一条控制流上, 并且断言 greedy 输出与朴素生成逐 token 相同 —— 哪怕 9 个 block 的小 pool 逼出了 4 次抢占。
      </p>
      <div class="grid grid-2" style="gap: 16px;">
        <div class="card">
          <h3>Engine.step <span class="tag">混合 batch</span></h3>
          <pre class="code">{{ engineStepCode }}</pre>
          <p class="hint">
            没有"prefill 步"和"decode 步"之分: 同一个 batch 里 n&gt;1 的 prefill chunk 和 n=1 的 decode 混跑,
            序列把 prompt 算追平了才有 logits 可采。
          </p>
        </div>
        <div class="card">
          <h3>接纳新请求时发生什么 <span class="tag">资源账本</span></h3>
          <pre class="code">{{ prefillCode }}</pre>
          <p class="hint">
            命中前缀的 token 是真的跳过了前向 —— KV 就在全局分页 pool 里, 经页表读回即可。
            demo 里 305 个待算 token = 233 个真前向 + 72 个前缀命中, 这笔账必须对得上。
          </p>
        </div>
      </div>
      <div class="card compare-card">
        <table class="infer-table">
          <thead>
            <tr>
              <th>真实服务概念</th>
              <th>本仓库对应实现</th>
              <th>读代码时抓住什么</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="r in engineMap" :key="r.concept">
              <td class="axis">{{ r.concept }}</td>
              <td class="mono small"><CodeRef :value="r.file" base="llm_infer/" tiny /></td>
              <td>{{ r.focus }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <!-- 本章挂载的实验台 (data/labmap/*.js) 与章末自测 (data/quiz/*.js), 没配置时不渲染 -->

    <LabMount />

    <QuizCard />


    <ChapterNav
      :prev="{ ...prevChapter, hint: '微调 / 对齐产出要部署的模型或适配器' }"
      :next="{ name: 'infer-kv-memory', label: '下一章 · KV 与缓存内存', hint: '从少算旧 token 开始拆推理优化' }"
    />
  </div>
</template>

<script setup>
import LabMount from '@/components/LabMount.vue'
import QuizCard from '@/components/QuizCard.vue'
import ChapterIntro from '@/components/ChapterIntro.vue'
import ChapterNav from '@/components/ChapterNav.vue'
import EvolutionChain from '@/components/EvolutionChain.vue'
import CodeRef from '@/components/CodeRef.vue'
import RepoLink from '@/components/RepoLink.vue'
import { inferModules, learningPath } from '@/data/models.js'

// 上一章从 learningPath 取, 不手写编号 (别的阶段加章后手写的 "4.4" 会过期)
const prevItem = learningPath[learningPath.findIndex((x) => x.route === 'infer') - 1]
const prevChapter = { name: prevItem.route, label: `上一章 · ${prevItem.label}` }

const inferChain = [
  {
    name: 'No Cache',
    year: 'baseline',
    pain: '每步都重算完整 prefix, 已生成越长越慢。',
    fix: '作为 m01 对照组, 用来证明 cache 路径输出必须一致。',
    color: 'var(--text-dim)',
  },
  {
    name: 'KV Cache',
    year: 'm01',
    pain: '注意力里的旧 K/V 每步都重复算。',
    fix: 'prefill 保存 K/V, decode_step 只追加新 token。',
    color: 'var(--left)',
  },
  {
    name: 'Paged KV',
    year: 'm02',
    pain: '请求长度不同, 连续 KV 内存容易碎片化。',
    fix: 'BlockManager 用 block_table 管逻辑页到物理块的映射。',
    color: 'var(--accent)',
  },
  {
    name: 'Scheduler',
    year: 'm03',
    pain: '请求随时进出, 静态 batch 会浪费大量空槽。',
    fix: '每步重组 batch, prefill chunk 与 decode 混批, block 不够就抢占。',
    color: 'var(--eye)',
  },
  {
    name: 'Reuse',
    year: 'm04-m06',
    pain: '相同 system prompt 和长 prompt 反复 prefill。',
    fix: 'prefix/radix cache 复用 KV, chunked prefill 减少长输入阻塞。',
    color: 'var(--right)',
  },
  {
    name: 'Engine',
    year: 'full',
    pain: '算法模块还不是服务。',
    fix: 'Engine.add_request / step / generate 串成 mini-vLLM 控制流。',
    color: 'var(--warn)',
  },
]

const noCacheCode = `ids = list(prompt_ids)
for _ in range(max_new):
    logits, _ = lm.prefill(np.array(ids))
    next_id = argmax(logits[-1])
    ids.append(next_id)`

const cacheCode = `logits, kv_cache = lm.prefill(prompt_ids)
next_id = argmax(logits[-1])
ids.append(next_id)

for _ in range(max_new - 1):
    logits, kv_cache = lm.decode_step(next_id, kv_cache)
    next_id = argmax(logits)
    ids.append(next_id)`

const servingPrimitives = [
  {
    name: 'BlockManager',
    tag: 'PagedAttention',
    desc: '把 KV pool 切成固定 block, 每条序列持有一张 block table。',
    file: 'llm_infer/m02_paged_attention/block_manager.py',
    code: `for blk in shared:          # 前缀命中的 block
    share_block(blk)        # ref_count += 1
n_new = blocks_needed(n_tokens) - len(shared)
table = list(shared) + [pop_free() for _ in range(n_new)]
block_tables[seq_id] = table`,
  },
  {
    name: 'Scheduler',
    tag: 'continuous',
    desc: '每步重组 batch = [(seq, n)]: running 先各拿 1 个 token, 剩余预算切给 prefill chunk; block 不够就抢占。',
    file: 'llm_infer/m03_continuous_batching/scheduler.py',
    code: `budget = max_batch_tokens
batch = schedule_running(budget)   # decode: n = 1
budget -= sum(n for _, n in batch)
if not just_preempted:
    batch += admit(budget)         # prefill chunk
return batch                       # [(seq, n)]`,
  },
  {
    name: 'PrefixCache',
    tag: 'reuse',
    desc: '完整 block 的 token 和父 hash 形成链式 hash, 命中后共享物理 block。',
    file: 'llm_infer/m04_prefix_cache/prefix_cache.py',
    code: `for i in range((len(ids) - 1) // bs):   # 至少留 1 个 token 真算
    parent = sha1(parent + block_i)     # 链式 hash
    blk = hash_to_block.get(parent)
    if blk is None: break
    hits.append(blk)`,
  },
]

const engineStepCode = `def step(self):
    batch = self.scheduler.schedule()   # [(seq, n)]
    tokens = []
    for seq, n in batch:
        end = seq.num_computed + n
        logits = self.runner.run(
            seq.all_ids[:end], table(seq), seq.num_computed)
        done = end == seq.num_tokens    # 追平了才采样
        tokens.append(sample(logits) if done else None)
    return self.scheduler.postprocess(batch, tokens)`

const prefillCode = `# Scheduler._admit: 接纳新请求
hits, n_hit = prefix_cache.match_prefix(ids)
n = min(len(ids) - n_hit, budget)   # 只吃一个 chunk
if not bm.can_allocate(len(ids), hits):
    break
bm.allocate(seq_id, len(ids), hits) # 命中块直接进页表
seq.num_computed = n_hit            # 命中的不用前向
batch.append((seq, n))`

const engineMap = [
  { concept: '请求入口', file: 'Engine.add_request', focus: 'prompt encode 后进入 waiting 队列, 每条请求绑定 SamplingParams' },
  { concept: '每步组 batch', file: 'm03_continuous_batching/scheduler.py', focus: 'schedule() 返回 [(seq, n)]: decode 优先, 剩余 token 预算给 prefill chunk; block 不够时 recompute 式抢占' },
  { concept: '真分页前向', file: 'full_engine/model_runner.py', focus: 'run() 只算 ids[start_pos:], 其余 KV 经 block_table 从全局 pool 读回' },
  { concept: '显存账本', file: 'BlockManager', focus: 'allocate(shared=…) / ensure_capacity / free / share_block 维护引用计数; free_list 顺序即 LRU 顺序' },
  { concept: '重复前缀', file: 'PrefixCache', focus: '命中的完整 block 共享引用, 未命中部分继续分配并注册' },
  { concept: '采样策略', file: 'm10_sampling/samplers.py', focus: 'rep penalty、temperature、top-k/top-p/min-p 最后作用在 logits 上' },
]
</script>

<style scoped>
table.infer-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
table.infer-table th {
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
table.infer-table td {
  padding: 10px 14px;
  border-bottom: 1px solid var(--border);
  vertical-align: top;
}
table.infer-table .axis {
  color: var(--text);
  font-weight: 600;
  white-space: nowrap;
}
table.infer-table .small {
  color: var(--text-muted);
  font-size: 12px;
}
.primitive-card pre.code {
  min-height: 142px;
}
.hint {
  margin-top: 10px;
  font-size: 12px;
  color: var(--text-dim);
  line-height: 1.6;
}
.compare-card {
  margin-top: 16px;
  padding: 0;
  overflow-x: auto;
}
</style>
