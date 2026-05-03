<template>
  <div>
    <h1 class="page-title">推理与部署优化 · 从能生成到高吞吐服务</h1>
    <p class="page-subtitle">
      <RepoLink path="llm_infer/" label="llm_infer/" tiny /> 把训练好的自回归模型放进服务环境:
      同样是不断生成下一个 token, 但目标从"loss 下降"变成"首 token 快、吞吐高、显存稳、输出可控"。
    </p>

    <ChapterIntro
      tldr="推理优化的主线是少算、少搬、少等、少浪费。KV cache 少算, PagedAttention 少碎片, continuous batching 少空转, prefix cache 少重复 prefill。"
      question="为什么训练时最贵的是反向, 推理时最贵的却常常是 KV cache、调度和内存带宽?"
      :goals="[
        '理解 KV cache / paged attention / prefix cache 各自省什么',
        '看懂连续批处理 / chunked prefill 怎么把 GPU 喂饱',
        '能把 mini-vLLM 的主循环对照到真实 vLLM/SGLang 上',
      ]"
      :codes="[
        { path: 'llm_infer/core/' },
        { path: 'llm_infer/full_engine/' },
        { path: 'llm_infer/m01_kv_cache/' },
        { path: 'llm_infer/m03_continuous_batching/' },
      ]"
      :prereq="{ name: 'finetune-runs', label: '阶段 4.4 · 训练脚本与落盘' }"
      :next-step="{ name: 'infer-kv-memory', label: '阶段 5.1 · KV 与缓存内存' }"
    />

    <section class="section">
      <h2>1. 推理优化的递进关系</h2>
      <p class="lead">
        先把逐步重算改成增量 decode, 再把 KV 显存做成可分配资源, 然后让请求动态组成 batch。
        后面的 prefix cache、投机解码、量化、结构化输出都挂在这个服务主循环上。
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
        没有 cache 时, 第 t 步要把 prefix+已生成的 t 个 token 全部 prefill 一遍。
        有 cache 后, prefill 只跑一次, decode 只给新 token 追加 K/V。
      </p>
      <div class="grid grid-2" style="gap: 16px;">
        <div class="card">
          <h3>朴素路径 <span class="tag">重复 prefill</span></h3>
          <pre class="code">{{ noCacheCode }}</pre>
          <p class="hint">
            成本随已生成长度增长。demo 会把这条路径和 cache 路径的输出 ids 对齐,
            确认优化没有改变生成结果。
          </p>
        </div>
        <div class="card">
          <h3>增量路径 <span class="tag">KV cache</span></h3>
          <pre class="code">{{ cacheCode }}</pre>
          <p class="hint">
            首步 prefill 保存每层 K/V, 后续 <code class="inline">decode_step</code>
            只处理一个新 token。对应 <RepoLink path="llm_infer/m01_kv_cache/demo.py" label="llm_infer/m01_kv_cache/demo.py" tiny />。
          </p>
        </div>
      </div>
    </section>

    <section class="section">
      <h2>3. 服务端三件套 · 内存、调度、复用</h2>
      <p class="lead">
        KV cache 让单请求变快, 但服务端还要处理多请求、多长度、共享前缀和显存碎片。
        这三件套是 vLLM/SGLang 类系统的核心抽象。
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
        cache 和调度解决服务骨架, 下面这些模块分别压缩计算、压缩存储、控制采样和约束输出。
        它们通常可以叠加, 但每一种都会引入自己的正确性边界。
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
        <RepoLink path="llm_infer/full_engine/engine.py" label="full_engine/engine.py" tiny /> 是最值得对照原始代码读的一页:
        它不追求完整 vLLM, 但把 add_request、prefill、decode、block manager、prefix cache 和 sampling 串在同一条控制流里。
      </p>
      <div class="grid grid-2" style="gap: 16px;">
        <div class="card">
          <h3>Engine.step <span class="tag">阶段切换</span></h3>
          <pre class="code">{{ engineStepCode }}</pre>
          <p class="hint">
            prefill 优先保证新请求尽快拿到首 token; waiting 清空后, running 请求进入 decode continuous batching。
          </p>
        </div>
        <div class="card">
          <h3>prefill 中发生什么 <span class="tag">资源账本</span></h3>
          <pre class="code">{{ prefillCode }}</pre>
          <p class="hint">
            这段代码把"命中前缀、分配新 block、注册 cache、采样首 token"放在一起,
            是理解 mini-vLLM 的入口。
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

    <ChapterNav
      :prev="{ name: 'finetune-runs', label: '阶段 4.4 · 训练脚本与落盘', hint: 'SFT / LoRA / DPO 产出要部署的模型或适配器' }"
      :next="{ name: 'infer-kv-memory', label: '阶段 5.1 · KV 与缓存内存', hint: '从少算旧 token 开始拆推理优化' }"
    />
  </div>
</template>

<script setup>
import ChapterIntro from '@/components/ChapterIntro.vue'
import ChapterNav from '@/components/ChapterNav.vue'
import EvolutionChain from '@/components/EvolutionChain.vue'
import CodeRef from '@/components/CodeRef.vue'
import RepoLink from '@/components/RepoLink.vue'
import { inferModules } from '@/data/models.js'

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
    fix: 'waiting / running 队列分开, prefill 和 decode 动态组批。',
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
    code: `table = []
for _ in range(n_blocks):
    blk = free_list.popleft()
    ref_count[blk] = 1
    table.append(blk)
block_tables[seq_id] = table`,
  },
  {
    name: 'Scheduler',
    tag: 'continuous',
    desc: 'waiting 请求先 prefill, running 请求 decode, 显存不够时 preempt。',
    file: 'llm_infer/m03_continuous_batching/scheduler.py',
    code: `if waiting:
    picked = pick_prefill_batch(token_budget)
    return picked, Stage.PREFILL

return list(running), Stage.DECODE`,
  },
  {
    name: 'PrefixCache',
    tag: 'reuse',
    desc: '完整 block 的 token 和父 hash 形成链式 hash, 命中后共享物理 block。',
    file: 'llm_infer/m04_prefix_cache/prefix_cache.py',
    code: `h = SHA1(parent_hash || token_block)
blk = hash_to_block.get(h)
if blk is not None:
    hits.append(blk)
    parent_hash = h`,
  },
]

const engineStepCode = `def step(self):
    self.stats_step += 1
    if self.waiting:
        return self._step_prefill()
    return self._step_decode()`

const prefillCode = `hits, n_hit_tokens = prefix_cache.match_prefix(seq.prompt_ids)
new_blocks_needed = n_blocks_needed - len(hits)

for blk in hits:
    bm.share_block(blk)
for _ in range(new_blocks_needed):
    allocate_new_block()

logits, kv_cache = lm.prefill(ids_arr)
tok_id = sample(logits[-1], params, history=seq.prompt_ids)`

const engineMap = [
  { concept: '请求入口', file: 'Engine.add_request', focus: 'prompt encode 后进入 waiting 队列, 每条请求绑定 SamplingParams' },
  { concept: '首 token 延迟', file: 'Engine._step_prefill', focus: 'prefill 优先, 同时查询 prefix cache 和 block capacity' },
  { concept: '吞吐', file: 'Engine._step_decode', focus: 'running 中每条序列每步 decode 一个 token, 完成后释放 block' },
  { concept: '显存账本', file: 'BlockManager', focus: 'allocate / append / free / share_block 维护 block 引用计数' },
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
