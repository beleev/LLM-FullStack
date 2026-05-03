<template>
  <div>
    <h1 class="page-title">规模化训练 · 把单机 loop 拆成分布式系统</h1>
    <p class="page-subtitle">
      <RepoLink path="llm_train/" label="llm_train/" tiny /> 用 numpy 把真实训练框架里最常见的机制压成小张量演示。
      重点不是模拟 GPU, 而是看清楚: 同一个 <code class="inline">loss → grad → update</code>
      如何在多卡、低精度、显存受限和故障恢复下仍然保持数学等价。
    </p>

    <ChapterIntro
      tldr="规模化训练不是换一个 optimizer, 而是把 batch、层、状态、激活、精度和通信拆开管理。所有 demo 都在证明: 分布式路径必须和单机基线对齐。"
      question="什么时候该切 batch, 什么时候该切矩阵, 什么时候该切优化器状态?"
      :goals="[
        '看懂 DDP / TP / PP / ZeRO 各自切的是 batch / 矩阵 / 层 / 状态',
        '知道为什么混合精度需要 loss scaling、grad clip',
        '能把 m01..m10 的功能挂回 full_loop 里的对应阶段',
      ]"
      :codes="[
        { path: 'llm_train/core/' },
        { path: 'llm_train/full_loop/' },
        { path: 'llm_train/m01_gradient_accumulation/' },
        { path: 'llm_train/m05_zero_fsdp/' },
      ]"
      :prereq="{ name: 'diffusion', label: '阶段 2.5 · 架构家族收束' }"
      :next-step="{ name: 'train-batch-ddp', label: '阶段 3.1 · batch 与 DDP' }"
    />

    <section class="section">
      <h2>1. 训练技术的依赖链</h2>
      <p class="lead">
        先把大 batch 等价拆开, 再扩到多副本, 再切层内矩阵和层间流水线。
        当模型状态也放不下时, 才进入 ZeRO/FSDP。混合精度、checkpoint 和稳定性是贯穿所有并行方式的保护层。
      </p>
      <EvolutionChain
        title="从单机循环到 full_loop"
        subtitle="每一步只解决一个新的瓶颈, 但都保留单机训练的语义。"
        :steps="trainChain"
      />
    </section>

    <section class="section">
      <h2>2. 六个切分维度</h2>
      <p class="lead">
        读训练框架时先问"它切的是哪一种东西"。下面这张表把名词压回具体张量和通信原语。
      </p>
      <div class="card" style="padding: 0; overflow-x: auto;">
        <table class="train-table">
          <thead>
            <tr>
              <th>维度</th>
              <th>解决什么</th>
              <th>核心等价</th>
              <th>原始代码</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="a in axes" :key="a.name">
              <td class="axis">{{ a.name }}</td>
              <td>{{ a.problem }}</td>
              <td>{{ a.equiv }}</td>
              <td class="mono small"><RepoLink :path="`llm_train/${a.file}`" :label="a.file" tiny /></td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <section class="section">
      <h2>3. 四个关键等价 · 对照原始代码</h2>
      <p class="lead">
        这些 demo 的写法都很直接: 先算一个 dense / single-process baseline, 再算并行版本,
        最后用 <code class="inline">assert</code> 验证输出或梯度一致。
      </p>

      <div class="grid grid-2" style="gap: 16px;">
        <div v-for="c in codeCards" :key="c.title" class="card">
          <h3>{{ c.title }} <span class="tag">{{ c.tag }}</span></h3>
          <p class="desc">{{ c.desc }}</p>
          <pre class="code">{{ c.code }}</pre>
          <p class="hint"><strong>看代码:</strong> <CodeRef :value="c.file" base="llm_train/" tiny /></p>
        </div>
      </div>
    </section>

    <section class="section">
      <h2>4. 从 llm_basic 到 full_loop</h2>
      <p class="lead">
        <RepoLink path="llm_basic/train.py" label="llm_basic/train.py" tiny /> 的主循环只有 get_batch、forward、loss、backward、update。
        <RepoLink path="llm_train/full_loop/demo.py" label="llm_train/full_loop/demo.py" tiny /> 没有改变这条语义, 只是把每一步展开成工程动作。
      </p>
      <div class="grid grid-2" style="gap: 16px;">
        <div class="card">
          <h3>单机最小循环 <span class="tag">llm_basic</span></h3>
          <pre class="code">{{ basicLoop }}</pre>
        </div>
        <div class="card">
          <h3>分布式训练一步 <span class="tag">full_loop</span></h3>
          <pre class="code">{{ distributedLoop }}</pre>
        </div>
      </div>
      <div class="card compare-card">
        <table class="train-table">
          <thead>
            <tr>
              <th>单机动作</th>
              <th>full_loop 中的展开</th>
              <th>为什么必要</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="r in loopMap" :key="r.simple">
              <td class="axis">{{ r.simple }}</td>
              <td>{{ r.full }}</td>
              <td>{{ r.why }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <section class="section">
      <h2>5. 模块索引 · 运行时看什么</h2>
      <p class="lead">
        每个模块都会打印"现象 → 数字 → 结论"。先跑单模块, 再跑
        <code class="inline">python -m llm_train.run_all</code> 看完整学习顺序
        (<RepoLink path="llm_train/run_all.py" label="llm_train/run_all.py" tiny />)。
      </p>
      <div class="card" style="padding: 0; overflow-x: auto;">
        <table class="train-table">
          <thead>
            <tr>
              <th>模块</th>
              <th>要点</th>
              <th>和前面知识的关系</th>
              <th>原始代码</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="m in trainModules" :key="m.id">
              <td class="axis">{{ m.name }}</td>
              <td>{{ m.concept }}</td>
              <td>{{ m.link }}</td>
              <td class="mono small"><RepoLink :path="m.file" tiny /></td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <ChapterNav
      :prev="{ name: 'diffusion', label: '阶段 2.5 · 扩散生成', hint: '模型结构已经看完, 下一步是把训练规模做上去' }"
      :next="{ name: 'train-batch-ddp', label: '阶段 3.1 · batch 与 DDP', hint: '先拆 batch, 再同步多卡梯度' }"
    />
  </div>
</template>

<script setup>
import ChapterIntro from '@/components/ChapterIntro.vue'
import ChapterNav from '@/components/ChapterNav.vue'
import EvolutionChain from '@/components/EvolutionChain.vue'
import CodeRef from '@/components/CodeRef.vue'
import RepoLink from '@/components/RepoLink.vue'
import { trainModules } from '@/data/models.js'

const trainChain = [
  {
    name: 'Accum',
    year: 'm01',
    pain: 'batch 想变大, 激活显存先爆。',
    fix: '拆 micro-batch, 梯度按样本数加权累积, 更新频率降低但数学等价。',
    color: 'var(--left)',
  },
  {
    name: 'DDP',
    year: 'm02',
    pain: '单卡吞吐不够。',
    fix: '每卡算本地 grad, all-reduce mean 后所有副本同步更新。',
    color: 'var(--accent)',
  },
  {
    name: 'TP / PP',
    year: 'm03-m04',
    pain: '单层太宽或层数太多, 一张卡放不下。',
    fix: 'TP 切矩阵宽度, PP 切层, 通信分别发生在层内和 stage 间。',
    color: 'var(--eye)',
  },
  {
    name: 'ZeRO',
    year: 'm05',
    pain: 'DDP 每卡都存完整参数、梯度、Adam 状态。',
    fix: 'reduce-scatter 梯度, 本地更新 shard, 必要时 all-gather 参数。',
    color: 'var(--right)',
  },
  {
    name: 'AMP + Guard',
    year: 'm06-m10',
    pain: '低精度更快但容易下溢、溢出和坏 step。',
    fix: 'loss scaling、fp32 master、grad clip、warmup/cosine、NaN 检测。',
    color: 'var(--warn)',
  },
  {
    name: 'Full Loop',
    year: 'full',
    pain: '单个技巧不等于可恢复的训练系统。',
    fix: '把数据切分、累积、同步、裁剪、分片更新和 checkpoint 串成主循环。',
    color: 'var(--left)',
  },
]

const axes = [
  { name: 'batch', problem: '吞吐和有效 batch size', equiv: 'micro grad 加权和 = full batch grad', file: 'm01_gradient_accumulation/demo.py' },
  { name: 'replica', problem: '多卡并行处理不同样本', equiv: 'all_reduce_mean(local_grads) = dense_grads', file: 'm02_data_parallel/demo.py' },
  { name: 'matrix', problem: '单层宽度太大', equiv: '切 W1/W2 后 concat / all-reduce 得到 dense MLP', file: 'm03_tensor_parallel/demo.py' },
  { name: 'layers', problem: '层数太多和流水线吞吐', equiv: 'micro-batch 排程减少 bubble', file: 'm04_pipeline_parallel/demo.py' },
  { name: 'states', problem: '参数、梯度、优化器状态占显存', equiv: 'shard update 后 all-gather 与 dense update 一致', file: 'm05_zero_fsdp/demo.py' },
  { name: 'precision', problem: '显存、带宽、tensor core 吞吐', equiv: 'scaled_grad / scale 恢复小梯度', file: 'm06_mixed_precision/demo.py' },
  { name: 'activation', problem: '反向需要存中间激活', equiv: '不存激活, 反向前重算 forward', file: 'm07_activation_checkpointing/demo.py' },
  { name: 'recovery', problem: '长训练中断和坏 step', equiv: '恢复模型、优化器、数据游标、RNG 后轨迹一致', file: 'm08_checkpoint_resume/demo.py' },
]

const codeCards = [
  {
    title: '梯度累积',
    tag: 'mean reduction',
    desc: 'micro loss 本地已经取均值, 所以累加时还要乘 micro_size / full_size。',
    file: 'llm_train/m01_gradient_accumulation/demo.py',
    code: `accum_grads = zeros_like(full_grads)
for xb, yb in micros:
    loss, grads = model.loss_and_grads(xb, yb)
    add_inplace(accum_grads, grads, scale=len(xb) / len(x))

assert max_abs_diff(full_grads, accum_grads) < 1e-6`,
  },
  {
    title: 'DDP 梯度同步',
    tag: 'all-reduce',
    desc: '每个 replica 只看一片 batch, 但同步后的梯度必须等价于整 batch 直接反向。',
    file: 'llm_train/m02_data_parallel/demo.py',
    code: `local_grads = []
for rank, replica in enumerate(replicas):
    xb = split(x, world_size)[rank]
    _, grads = replica.loss_and_grads(xb, yb)
    local_grads.append(grads)

synced = all_reduce_mean(local_grads)
replica.apply_grads(synced, lr=0.1)`,
  },
  {
    title: '张量并行',
    tag: 'Megatron pattern',
    desc: 'W1 按列切, W2 按行切。前向 partial_outs 求和, 反向再把各 shard 梯度拼回 dense 形状。',
    file: 'llm_train/m03_tensor_parallel/demo.py',
    code: `W1_shards = split(W1, world, axis=1)
W2_shards = split(W2, world, axis=0)

h_shards = [relu(x @ W1_s) for W1_s in W1_shards]
partial  = [h @ W2_s for h, W2_s in zip(h_shards, W2_shards)]
tp_out   = all_reduce_sum(partial)[0] + b2`,
  },
  {
    title: 'ZeRO / FSDP',
    tag: 'state sharding',
    desc: '核心动作不是神秘优化器, 而是梯度先 reduce-scatter, 每张卡只更新自己那片参数。',
    file: 'llm_train/m05_zero_fsdp/demo.py',
    code: `grad_shards  = reduce_scatter_sum(grad_per_rank, axis=0)
param_shards = split(param, world, axis=0)
updated      = [p - lr * g for p, g in zip(param_shards, grad_shards)]

gathered = all_gather(updated, axis=0)[0]
assert allclose(dense_updated, gathered)`,
  },
]

const basicLoop = `x, y = get_batch(...)
logits, cache = transformer_forward(W, x)
loss, dlogits = cross_entropy_forward_backward(logits, y)
grads = transformer_backward(dlogits, cache)
W, opt_state = adam_step(W, grads, opt_state, lr=LR)`

const distributedLoop = `x_ranks = split(x, world)
for rank in ranks:
    for xb, yb in micro_batches(rank):
        loss, grads = replica.loss_and_grads(xb, yb)
        accum += unscale(scale(grads)) * micro_weight

synced = all_reduce_mean(local_grads)
clipped = clip_by_global_norm(synced)
zero_style_sgd_step(replica0, clipped)
broadcast_params_to_replicas()`

const loopMap = [
  { simple: 'get_batch', full: 'batch 先按 data-parallel rank 切, rank 内再切 micro-batch', why: '同时控制吞吐和峰值显存' },
  { simple: 'backward', full: '本地反向后先累积, 再 all-reduce 或 reduce-scatter', why: '多卡梯度必须形成同一个全局更新方向' },
  { simple: 'optimizer.step', full: '先 unscale、clip, 再 ZeRO-style shard update', why: '低精度和大模型状态都需要额外保护' },
  { simple: 'checkpoint', full: '保存 step、model、optimizer/data cursor/RNG 等状态', why: '长训练必须可恢复且恢复后轨迹一致' },
]
</script>

<style scoped>
table.train-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
table.train-table th {
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
table.train-table td {
  padding: 10px 14px;
  border-bottom: 1px solid var(--border);
  vertical-align: top;
}
table.train-table .axis {
  color: var(--text);
  font-weight: 600;
  white-space: nowrap;
}
table.train-table .small {
  color: var(--text-muted);
  font-size: 12px;
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
