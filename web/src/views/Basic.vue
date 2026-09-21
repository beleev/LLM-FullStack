<template>
  <div>
    <h1 class="page-title">最小可跑闭环 · 用 numpy 把 Transformer 写穿</h1>
    <p class="page-subtitle">
      <RepoLink path="llm_basic/" label="llm_basic/" tiny /> 不装 PyTorch, 只把 numpy 当一个会广播的数组容器,
      把 forward / backward / Adam / 采样一行不落地摊开。
      <strong>它只干一件事</strong>: 让你看清梯度怎么流过模型。之后再读
      <RepoLink path="llm_models/" label="llm_models/" tiny /> 里的 PyTorch 代码, 你会知道 autograd 在背后替你做了什么。
    </p>

    <ChapterIntro
      tldr="现代 LLM 的训练循环就 5 行: get_batch → forward → loss → backward → adam_step。用 numpy 摊开这 5 行, Transformer 里就没有黑盒了。"
      question="不许用 autograd, 你能只凭链式法则把 RMSNorm / softmax / 残差的反向写对吗?"
      :goals="[
        '不用 PyTorch, 27 秒把一个 45,568 参数的 GPT 训到 val_loss 1.98',
        '报得出 ids [B,T] 到 logits [B,T,V] 每一步的形状和 cache',
        '一个反向算错时, 知道 gradcheck 是怎么把它揪出来的',
      ]"
      :codes="[
        { path: 'llm_basic/model.py' },
        { path: 'llm_basic/optim.py' },
        { path: 'llm_basic/train.py' },
        { path: 'llm_basic/sample.py' },
        { path: 'llm_basic/gradcheck.py' },
      ]"
      :prereq="{ name: 'home', label: '主线总览' }"
      :next-step="{ name: 'basic-data', label: '阶段 1.1 — 数据与 tokenizer' }"
    />

    <!-- ── 1. 整体闭环 ─────────────────────────────────────────── -->
    <section class="section">
      <h2>1. 一张图看完整闭环</h2>
      <p class="lead">
        左边是数据 / 训练 / 采样三条管道, 右边是撑起它们的 7 个函数对。
        <strong>每个算子都是 forward + backward 成对写的</strong>:
        forward 把反向要用的中间量塞进 cache, backward 按逆序取出来用。
      </p>

      <div class="grid grid-2" style="gap: 16px;">
        <div class="card">
          <h3>三条管道 <span class="tag">主入口</span></h3>
          <table class="pipeline">
            <tbody>
              <tr>
                <td class="pp-name mono"><RepoLink path="llm_basic/prepare.py" label="prepare.py" tiny /></td>
                <td class="pp-desc">下载 Tiny Shakespeare → 字符级编码 → <RepoLink path="llm_basic/train.bin" label="train.bin" tiny /> / <RepoLink path="llm_basic/val.bin" label="val.bin" tiny /> / <RepoLink path="llm_basic/meta.npz" label="meta.npz" tiny /></td>
              </tr>
              <tr>
                <td class="pp-name mono"><RepoLink path="llm_basic/train.py" label="train.py" tiny /></td>
                <td class="pp-desc">5 行循环: get_batch → forward → loss → backward → adam_step (2000 步 27 秒, val_loss 4.15 → 1.98)</td>
              </tr>
              <tr>
                <td class="pp-name mono"><RepoLink path="llm_basic/sample.py" label="sample.py" tiny /></td>
                <td class="pp-desc">加载 ckpt → 自回归生成 (temperature + top-k)</td>
              </tr>
              <tr>
                <td class="pp-name mono"><RepoLink path="llm_basic/gradcheck.py" label="gradcheck.py" tiny /></td>
                <td class="pp-desc">中心差分算出的数值梯度 vs 手写解析梯度, 7 个算子逐元素核对 (不到 1 秒)</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div class="card">
          <h3>7 个函数对 <span class="tag"><RepoLink path="llm_basic/model.py" label="model.py" tiny /></span></h3>
          <table class="pairs">
            <tbody>
              <tr v-for="p in pairs" :key="p.name">
                <td class="pair-name mono">{{ p.name }}</td>
                <td class="pair-role">{{ p.role }}</td>
                <td class="pair-key mono">{{ p.key }}</td>
              </tr>
            </tbody>
          </table>
          <p class="hint">右边一栏是这个算子反向最容易写错的地方, 后面逐个拆。</p>
        </div>
      </div>
    </section>

    <!-- ── 2. 模型结构 + 数据流 ───────────────────────────────────── -->
    <section class="section">
      <h2>2. 模型结构 · 默认 1 层 1 头, 但麻雀俱全</h2>
      <p class="lead">
        故意做到最小: 默认单层 (<code class="inline">--n-layer</code> 可加层, 加层只是一个 for 循环) + 单头 + ReLU MLP + 学得式位置编码。
        骨架和现代 LLaMA 一模一样 — <strong>差的只是零件</strong>, 不是结构。
      </p>

      <div class="card">
        <pre class="code">{{ modelDiagram }}</pre>
        <p class="hint">
          形状记号: <code class="inline">B=batch, T=seq_len, D=dim, H=MLP 中间维, V=vocab</code>。
          完整代码见 <RepoLink path="llm_basic/model.py:transformer_forward" label="llm_basic/model.py:transformer_forward" tiny />;
          想逐步看形状和 cache, 去 <router-link :to="{ name: 'basic-forward' }" class="dt-link">阶段 1.2 的形状流水线实验台</router-link>。
        </p>
      </div>
    </section>

    <!-- ── 3. 反向传播的 4 个最容易写错的点 ─────────────────────── -->
    <section class="section">
      <h2>3. 反向四大坑 · 看清梯度怎么流</h2>
      <p class="lead">
        这四件事 autograd 都替你处理了, 而且处理得悄无声息。手写一遍才知道它在做什么 —
        也才知道自己写错时为什么不会报错。
      </p>

      <div class="grid grid-2" style="gap: 16px;">
        <div v-for="g in gotchas" :key="g.title" class="card">
          <h3>{{ g.title }} <span class="tag">{{ g.tag }}</span></h3>
          <p class="desc" style="margin-bottom: 10px;">{{ g.intuition }}</p>
          <pre class="code">{{ g.formula }}</pre>
          <p class="hint" style="margin-top: 8px;">
            <strong>代码位置:</strong> <RepoLink :path="`llm_basic/${g.where}`" :label="g.where" tiny />
          </p>
        </div>
      </div>
    </section>

    <!-- ── 4. 训练循环 + Adam ───────────────────────────────────── -->
    <section class="section">
      <h2>4. 训练循环 · 5 行代码, 60 行 Adam</h2>
      <p class="lead">
        看完这一节, PyTorch 里的 <code class="inline">loss.backward(); opt.step()</code> 对你就不再是两个动词。
      </p>

      <div class="grid grid-2" style="gap: 16px;">
        <div class="card">
          <h3><RepoLink path="llm_basic/train.py" label="train.py" tiny /> 主循环 <span class="tag">5 行而已</span></h3>
          <pre class="code">{{ trainLoopCode }}</pre>
          <p class="hint">
            没有 <code class="inline">zero_grad()</code>: 每步返回的是新 dict, 旧梯度自然不会残留。
            没有 <code class="inline">requires_grad</code>: 反向是手写的, 不需要谁去标记。
            PyTorch 做的就是这 4 件事的工程化版本。
          </p>
        </div>

        <div class="card">
          <h3>Adam · 4 步 + 2 个修正 <span class="tag"><RepoLink path="llm_basic/optim.py" label="optim.py" tiny /></span></h3>
          <pre class="code">{{ adamCode }}</pre>
          <div class="adam-why">
            <p><strong>为什么不用 SGD?</strong></p>
            <p class="hint">
              Transformer 各层的梯度尺度差好几个数量级 (lm_head 远大于 embedding), 单一学习率压不住:
              大到能让平缓方向动起来, 陡的方向就发散了。Adam 除以 √v̂ 把每个坐标归一化,
              每步位移约等于 lr, 与坡度无关。核心就
              <RepoLink path="llm_basic/optim.py:adam_step" label="adam_step" tiny /> 里那十来行。
            </p>
          </div>
        </div>
      </div>
    </section>

    <!-- ── 5. gradcheck ───────────────────────────────────────── -->
    <section class="section">
      <h2>5. gradcheck · 用数值梯度反查解析梯度</h2>
      <p class="lead">
        手写反向最难的不是写, 是<strong>知道自己写对了</strong>。转置写反、sum 错维度不会让程序崩,
        loss 也照样下降, 只是降得慢一点。gradcheck 拿最朴素的有限差分逐元素核对一遍, 这些 bug 才现形 —
        把 <code class="inline">rmsnorm_backward</code> 的耦合项删掉, 它立刻报 dx 相对误差 3.11e-01。
      </p>

      <div class="card">
        <pre class="code">{{ gradcheckCode }}</pre>
        <p class="hint">
          判定为什么是相对 + 绝对的组合? 纯绝对容差对大梯度太松, 纯相对容差对接近 0 的梯度太严
          (抽样常落在 batch 里没出现过的那行 tok_emb 上, 两个 1e-10 的数相对误差能很大)。所以用
          <code class="inline">atol + rtol · max(|g_a|, |g_n|)</code>。
          eps 取 1e-5 是 float64 下 U 形误差曲线的谷底: 更小舍入误差吃掉信号, 更大又跑出截断误差 —
          阶段 1.3 的实验台可以亲手拖这条曲线。
          gradcheck.py 先逐算子全元素检查 (7 个算子相对误差 1e-11 ~ 2e-10), 再对 n_layer = 1 和 2 做端到端抽样检查。
        </p>
      </div>
    </section>

    <!-- ── 6. 简化 vs 现代 ─────────────────────────────────────── -->
    <section class="section">
      <h2>6. 这版本省了什么 · 引出阶段 2</h2>
      <p class="lead">
        每一行的"真模型怎么做"都对应 <RepoLink path="llm_models/" label="llm_models/" tiny />
        下的一个组件。阶段 2 的每一章, 就是回来把这张表的某一行补上。
      </p>
      <div class="card" style="padding: 0; overflow-x: auto;">
        <table class="diff-table">
          <thead>
            <tr>
              <th>简化项</th>
              <th>llm_basic 怎么做</th>
              <th>真模型 (llm_models) 怎么做</th>
              <th>下一站章节</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="d in diffs" :key="d.topic">
              <td class="dt-topic">{{ d.topic }}</td>
              <td class="mono small">{{ d.basic }}</td>
              <td class="mono small">{{ d.modern }}</td>
              <td>
                <router-link v-if="d.route" :to="{ name: d.route }" class="dt-link">
                  {{ d.routeLabel }} →
                </router-link>
                <RepoLink v-else-if="d.file" :path="d.file" :label="d.routeLabel" tiny />
                <span v-else class="muted small">{{ d.routeLabel }}</span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <!-- 本章挂载的实验台 (data/labmap/*.js) 与章末自测 (data/quiz/*.js), 没配置时不渲染 -->

    <LabMount />

    <QuizCard />


    <ChapterNav
      :prev="{ name: 'home', label: '主线总览', hint: '回到六阶段地图' }"
      :next="{ name: 'basic-data', label: '阶段 1.1 · 数据与 tokenizer', hint: '先把 input.txt 变成可复现的训练张量' }"
    />
  </div>
</template>

<script setup>
import LabMount from '@/components/LabMount.vue'
import QuizCard from '@/components/QuizCard.vue'
import ChapterIntro from '@/components/ChapterIntro.vue'
import ChapterNav from '@/components/ChapterNav.vue'
import RepoLink from '@/components/RepoLink.vue'

const pairs = [
  { name: 'embedding',   role: 'token / pos 查表',             key: '重复 id 要 np.add.at 累加' },
  { name: 'linear',      role: 'y = x @ W + b',                key: 'dW = x.T @ dy (先把 batch 拍平)' },
  { name: 'rmsnorm',     role: '按行能量归一 + 缩放',          key: '多一个耦合项 — 归一化的副作用' },
  { name: 'attention',   role: '单头 causal: softmax(QKᵀ)·V',  key: 'softmax 反向: a·(da − Σa·da)' },
  { name: 'mlp',         role: '升到 H 过 ReLU 再降回 D',      key: 'ReLU 反向要知道哪些位置曾是负的' },
  { name: 'block',       role: 'Pre-LN + Attn + MLP + 两个残差', key: '每个残差处 dx 都要加上捷径那份' },
  { name: 'transformer', role: '拼起来, 输出 logits',           key: 'CE 合并后 dlogits = (p − onehot)/N' },
]

const modelDiagram = `ids (B, T)
  │
  ▼
tok_emb[V,D] + pos_emb[T_max,D]            加法 → (B, T, D)
  │
  ▼
┌─ Block ──────────────────────────────┐
│  x → RMSNorm → Attn(单头, causal) ─┐ │   ← 残差 1
│                                    + │
│  h → RMSNorm → MLP(ReLU) ──────────┐ │   ← 残差 2
│                                    + │
└──────────────────────────────────────┘
  │
  ▼
RMSNorm
  │
  ▼
lm_head[D,V] → logits (B, T, V)
  │
  ▼
softmax + cross-entropy → loss

默认 V=65, D=64, H=128, T=64, n_layer=1  →  45,568 个参数
(2 层就是 78,656; 层数直接从参数名 block_{i}_* 数出来)`

const trainLoopCode = `for step in range(MAX_ITERS):
    x, y = get_batch(train_data, BATCH_SIZE, SEQ_LEN, rng)

    # 1. forward: 顺便把所有 cache 串起来
    logits, cache = transformer_forward(W, x)

    # 2. loss + dlogits (fused: dlogits=(probs-onehot)/N)
    loss, dlogits = cross_entropy_forward_backward(logits, y)

    # 3. backward: 沿 cache 倒推所有参数的梯度
    grads = transformer_backward(dlogits, cache)

    # 4. Adam: 拿 grads 更新 W, 不就地修改 (返回新 dict)
    W, opt_state = adam_step(W, grads, opt_state, lr=LR)`

const adamCode = `# t  ← t + 1
m = β1·m + (1-β1)·g           # 一阶矩 (动量)
v = β2·v + (1-β2)·g²          # 二阶矩 (自适应学习率)

# 早期 m, v 偏小, 偏置修正:
m̂ = m / (1 - β1ᵗ)
v̂ = v / (1 - β2ᵗ)

# 更新
W ← W - lr · m̂ / (√v̂ + eps)`

const gradcheckCode = `# 对每个参数 W[k] 的每个元素 i:
g_analytic = grads[k].flat[i]

# 数值梯度: 中心差分
W[k].flat[i] += eps;  loss_plus,  _ = forward_loss()
W[k].flat[i] -= 2·eps; loss_minus, _ = forward_loss()
W[k].flat[i] += eps   # 还原
g_numeric = (loss_plus - loss_minus) / (2·eps)

# 判定: |g_a - g_n| < atol + rtol · max(|g_a|, |g_n|)`

const gotchas = [
  {
    title: 'embedding 的反向',
    tag: 'np.add.at',
    intuition: '同一个 token id 在一个 batch 里会出现很多次, 每次都贡献一份梯度。写成 dW[ids] += dout, numpy 的花式索引只保留最后一次 — 高频 token 的梯度就这样悄悄缩水了, 而且不报错。',
    formula: 'dW = zeros((V, D))\n# 重复索引必须累加, 不是覆盖\nnp.add.at(dW, ids, dout)',
    where: 'model.py: embedding_backward',
  },
  {
    title: 'RMSNorm 的耦合项',
    tag: '归一化的副作用',
    intuition: '归一化让每行的能量恒定, 于是调大一个 x_i 就会挤小其它 x_j。dx 因此不只是 dy·g/rms, 还得减掉这份互相挤压。把第二项删掉跑 gradcheck, 相对误差立刻从 1e-10 跳到 3.11e-01。',
    formula: 'dx_i = (g_i / rms) · dy_i\n     - x_i / (D · rms³) · Σ_j (dy_j · g_j · x_j)\n              ↑ 删掉这一项, 误差 3.11e-01',
    where: 'model.py: rmsnorm_backward',
  },
  {
    title: 'softmax 的 Jacobian',
    tag: '不能逐元素算',
    intuition: 'softmax 的每个输出都依赖全部输入, 本地导数是一整个矩阵。好在乘上上游梯度后能化简成一行, 而且只用到输出 a — 所以 cache 里存的是 attn, 不是 scores。',
    formula: '设 a = softmax(s),  da 是上游梯度\nds_i = a_i · (da_i - Σ_j a_j · da_j)\n            ↑ 减去整行的加权和',
    where: 'model.py: attention_backward',
  },
  {
    title: '残差的反向 = 两路相加',
    tag: '梯度高速公路',
    intuition: '加法把一份梯度原样复制给两条路。回到 x 时要把"绕过子层的"和"穿过子层的"加起来 — 那个原样的 dout 就是梯度不会在层层相乘中消失的原因。',
    formula: 'h = x + Attn(RMSNorm(x))\nout = h + MLP(RMSNorm(h))\n\ndh = dout + 穿过 MLP 回来的那份\ndx = dh   + 穿过 Attn 回来的那份',
    where: 'model.py: block_backward',
  },
]

const diffs = [
  { topic: '位置编码',  basic: '学得式 pos_emb, 卡死在 T_max=64', modern: 'RoPE (旋转, 乘在 Q/K 上)',        route: 'position', routeLabel: '阶段 2.2' },
  { topic: '注意力',    basic: '单头, head_dim = D',        modern: 'MHA / GQA / MLA / DSA',               route: 'attention', routeLabel: '阶段 2.1' },
  { topic: 'FFN 激活',  basic: 'ReLU + 普通两层',           modern: 'GELU → SwiGLU (门控 + 三个 linear)',  route: 'blocks', routeLabel: '阶段 2.3' },
  { topic: '层数',      basic: '默认 1 层 (--n-layer 可调)',   modern: '几十层 + 各种 Block 变体',            route: 'blocks', routeLabel: '阶段 2.3' },
  { topic: '分词',      basic: '字符级 vocab=65 (bpe.py 另演示)', modern: 'BPE / SentencePiece, 词表几万',    route: 'basic-data', routeLabel: '阶段 1.1' },
  { topic: 'FFN 形态',  basic: '稠密 MLP',                  modern: 'MoE: 一组小 FFN + router top-k',       route: 'moe', routeLabel: '阶段 2.4' },
  { topic: '推理',      basic: '每步重算整段 forward',      modern: 'KV cache: 只算新 token 的 Q, 复用 K/V',  route: 'infer-kv-memory', routeLabel: '阶段 5.1' },
  { topic: '优化器',    basic: '裸 Adam (三个开关默认全关)', modern: 'AdamW + warmup + cosine + grad clip',  routeLabel: '参考 trainer.py', file: 'llm_models/training/trainer.py' },
  { topic: '精度',      basic: '全 float64 (gradcheck 需要)', modern: 'bf16 / fp16 混合精度 + loss scaling',  route: 'train', routeLabel: '阶段 3' },
]
</script>

<style scoped>
table.pipeline,
table.pairs,
table.diff-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
table.pipeline td,
table.pairs td {
  padding: 8px 0;
  border-bottom: 1px dashed var(--border);
  vertical-align: top;
}
table.pipeline tr:last-child td,
table.pairs tr:last-child td { border-bottom: none; }

.pp-name { width: 130px; color: var(--accent); font-size: 12.5px; }
.pp-desc { color: var(--text-muted); font-size: 12.5px; line-height: 1.6; }
.pair-name { width: 110px; color: var(--left); font-size: 12.5px; }
.pair-role { color: var(--text); font-size: 12.5px; padding-right: 12px; }
.pair-key  { color: var(--text-dim); font-size: 11.5px; text-align: right; min-width: 220px; }

.hint {
  margin-top: 10px;
  font-size: 12px;
  color: var(--text-dim);
  line-height: 1.6;
}
.adam-why { margin-top: 14px; padding-top: 12px; border-top: 1px dashed var(--border); }
.adam-why p { margin-bottom: 4px; font-size: 13px; }

table.diff-table th {
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
table.diff-table td {
  padding: 10px 14px;
  border-bottom: 1px solid var(--border);
}
table.diff-table .dt-topic { color: var(--text); font-weight: 500; width: 110px; }
table.diff-table .small { font-size: 12px; }
table.diff-table .muted { color: var(--text-dim); }
.dt-link {
  font-size: 12px;
  color: var(--accent);
  border-bottom: 1px dashed var(--accent);
}
.dt-link:hover { text-decoration: none; border-bottom-style: solid; }
</style>
