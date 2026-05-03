<template>
  <div>
    <h1 class="page-title">最小可跑闭环 · 用 numpy 把 Transformer 写穿</h1>
    <p class="page-subtitle">
      <code class="inline">llm_basic/</code> 不依赖 PyTorch, 用 numpy 当"会广播的容器",
      把 forward / backward / Adam / 采样全部肉眼可见地展开。
      <strong>它的任务只有一个</strong>: 让你看清楚梯度是怎么流过模型的 — 以后再看
      llm_models 里的 PyTorch 代码, 你会知道每一行 autograd 在背后做了什么。
    </p>

    <ChapterIntro
      tldr="所有现代 LLM 训练循环 = 5 行代码: get_batch → forward → loss → backward → adam_step。把这 5 行用 numpy 摊开, 你就会看到 Transformer 没有黑盒。"
      question="如果不让你用 autograd, 你能否只凭链式法则把 RMSNorm / softmax / 残差的反向写对?"
      :goals="[
        '不用 PyTorch, 也能在 2 分钟内把 Transformer 训出降低的 loss',
        '亲手摸到 forward / backward / Adam / 采样的每一步',
        '看懂一个反向算错时, gradcheck 是怎么把它揪出来的',
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
        左边是数据 / 训练 / 采样三条管道, 右边是支撑它们的 7 个核心函数对。
        <strong>每个函数对都是 forward + backward 成对出现</strong> —
        forward 把"反向需要的中间量"塞进 cache, backward 直接取用。
      </p>

      <div class="grid grid-2" style="gap: 16px;">
        <div class="card">
          <h3>三条管道 <span class="tag">主入口</span></h3>
          <table class="pipeline">
            <tbody>
              <tr>
                <td class="pp-name mono">prepare.py</td>
                <td class="pp-desc">下载 Tiny Shakespeare → 字符级编码 → <code class="inline">train.bin / val.bin / meta.npz</code></td>
              </tr>
              <tr>
                <td class="pp-name mono">train.py</td>
                <td class="pp-desc">5 行循环: get_batch → forward → loss → backward → adam_step (~2 分钟跑 2000 步, val_loss 4.17 → 1.9)</td>
              </tr>
              <tr>
                <td class="pp-name mono">sample.py</td>
                <td class="pp-desc">加载 ckpt → 自回归生成 (temperature + top-k)</td>
              </tr>
              <tr>
                <td class="pp-name mono">gradcheck.py</td>
                <td class="pp-desc">数值梯度 vs 解析梯度, 逐参数验证 backward 写对了 (~1 秒)</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div class="card">
          <h3>7 个函数对 <span class="tag">model.py 全貌</span></h3>
          <table class="pairs">
            <tbody>
              <tr v-for="p in pairs" :key="p.name">
                <td class="pair-name mono">{{ p.name }}</td>
                <td class="pair-role">{{ p.role }}</td>
                <td class="pair-key mono">{{ p.key }}</td>
              </tr>
            </tbody>
          </table>
          <p class="hint">关键栏 = 反向最容易出错的那一项, 后面会逐个拆解。</p>
        </div>
      </div>
    </section>

    <!-- ── 2. 模型结构 + 数据流 ───────────────────────────────────── -->
    <section class="section">
      <h2>2. 模型结构 · 1 层 1 头, 但麻雀俱全</h2>
      <p class="lead">
        刻意做到极简: 单层 + 单头 + ReLU MLP + 学得式位置编码。骨架与现代 LLaMA
        完全一致 — <strong>差别只在零件选择</strong>, 不在结构。
      </p>

      <div class="card">
        <pre class="code">{{ modelDiagram }}</pre>
        <p class="hint">
          形状: <code class="inline">B=batch, T=seq_len, D=dim, V=vocab</code>。
          完整代码见 <code class="inline">llm_basic/model.py:transformer_forward</code>。
        </p>
      </div>
    </section>

    <!-- ── 3. 反向传播的 4 个最容易写错的点 ─────────────────────── -->
    <section class="section">
      <h2>3. 反向四大坑 · 看清梯度怎么流</h2>
      <p class="lead">
        autograd 帮你把这些都默认处理了。但"知道它在做什么"才是看懂大模型训练的前提。
      </p>

      <div class="grid grid-2" style="gap: 16px;">
        <div v-for="g in gotchas" :key="g.title" class="card">
          <h3>{{ g.title }} <span class="tag">{{ g.tag }}</span></h3>
          <p class="desc" style="margin-bottom: 10px;">{{ g.intuition }}</p>
          <pre class="code">{{ g.formula }}</pre>
          <p class="hint" style="margin-top: 8px;">
            <strong>代码位置:</strong> <code class="inline">{{ g.where }}</code>
          </p>
        </div>
      </div>
    </section>

    <!-- ── 4. 训练循环 + Adam ───────────────────────────────────── -->
    <section class="section">
      <h2>4. 训练循环 · 5 行代码, 60 行 Adam</h2>
      <p class="lead">
        这一节是让你彻底看懂 PyTorch 里 <code class="inline">loss.backward(); opt.step()</code>
        到底等价于什么。
      </p>

      <div class="grid grid-2" style="gap: 16px;">
        <div class="card">
          <h3>train.py 主循环 <span class="tag">5 行而已</span></h3>
          <pre class="code">{{ trainLoopCode }}</pre>
          <p class="hint">
            没有 <code class="inline">zero_grad()</code> — 因为我们每步都返回新 dict;
            没有 <code class="inline">requires_grad</code> — 因为我们手写了反向。
            等你回去看 PyTorch, 会发现它做的就是这 4 件事的工程化版本。
          </p>
        </div>

        <div class="card">
          <h3>Adam · 4 步 + 2 个修正 <span class="tag">optim.py</span></h3>
          <pre class="code">{{ adamCode }}</pre>
          <div class="adam-why">
            <p><strong>为什么不用 SGD?</strong></p>
            <p class="hint">
              Transformer 各层梯度尺度差异大 (lm_head 远大于 embedding),
              SGD 单一学习率压不住。Adam 用二阶矩 √v̂ 给每个参数自适应缩放,
              小学习率也能稳定收敛。代码就 30 行 (<code class="inline">optim.py</code>)。
            </p>
          </div>
        </div>
      </div>
    </section>

    <!-- ── 5. gradcheck ───────────────────────────────────────── -->
    <section class="section">
      <h2>5. gradcheck · 用数值梯度反查解析梯度</h2>
      <p class="lead">
        手写反向最难的不是写, 是<strong>验证写对了</strong>。gradcheck 用最朴素的有限差分
        把每个参数都核对一遍 — 任何 transpose 写反、sum 维度搞错, 都会立刻报警。
      </p>

      <div class="card">
        <pre class="code">{{ gradcheckCode }}</pre>
        <p class="hint">
          为什么用相对+绝对组合? 单纯绝对容差对大梯度太松, 单纯相对容差对接近 0
          的梯度太严。所以用 <code class="inline">atol + rtol · max(|g_a|, |g_n|)</code>。
          eps 取 1e-4 是经验最优 — 太小数值误差吃掉信号, 太大又跑出二阶项。
        </p>
      </div>
    </section>

    <!-- ── 6. 简化 vs 现代 ─────────────────────────────────────── -->
    <section class="section">
      <h2>6. 这版本省了什么 · 引出阶段 2</h2>
      <p class="lead">
        下面每一行的"真模型怎么做"对应 <code class="inline">llm_models/</code>
        下的某个组件。看懂了这张表, 阶段 2 的每个章节都是在补一行。
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
                <span v-else class="muted small">{{ d.routeLabel }}</span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <ChapterNav
      :prev="{ name: 'home', label: '主线总览', hint: '回到六阶段地图' }"
      :next="{ name: 'basic-data', label: '阶段 1.1 · 数据与 tokenizer', hint: '先把 input.txt 变成可复现的训练张量' }"
    />
  </div>
</template>

<script setup>
import ChapterIntro from '@/components/ChapterIntro.vue'
import ChapterNav from '@/components/ChapterNav.vue'

const pairs = [
  { name: 'embedding',   role: 'token / pos lookup',          key: 'np.add.at 处理重复索引' },
  { name: 'linear',      role: 'y = x @ W + b',                key: 'dW = x.T @ dy (要 flatten batch)' },
  { name: 'rmsnorm',     role: '按行能量归一 + 缩放',          key: '反向带"耦合项" — 标准化的副作用' },
  { name: 'attention',   role: '单头 causal: softmax(QK^T)·V', key: 'softmax 反向: a·(da - Σa·da)' },
  { name: 'mlp',         role: '两层 + ReLU',                  key: '残差让 dx 走两条路相加' },
  { name: 'block',       role: 'Pre-LN + Attn + MLP + 残差',   key: '每个残差节点 dx 都要 + 一份' },
  { name: 'transformer', role: '组合一切, 输出 logits',         key: 'fused CE: dlogits=(probs-onehot)/N' },
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

默认: D=64, hidden=128, T=64  →  约 45K 参数`

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
    intuition: '同一个 token id 可能在 batch 里出现多次, 每次都贡献一份梯度 — 必须累加, 不是覆盖。',
    formula: 'dW = zeros((V, D))\nnp.add.at(dW, ids, dout)   # 用 add.at 处理重复索引',
    where: 'model.py: embedding_backward',
  },
  {
    title: 'RMSNorm 的耦合项',
    tag: '标准化副作用',
    intuition: '标准化让每行总能量恒为常数 — 改一个 x_i 会"挤压"其它 x_j。所以 dx 不是简单地 dy·g/rms, 还多一项耦合修正。',
    formula: 'dx_i = (g_i / rms) · dy_i\n     - x_i / (D · rms³) · Σ_j (dy_j · g_j · x_j)\n              ↑ 这一项就是耦合修正',
    where: 'model.py: rmsnorm_backward',
  },
  {
    title: 'softmax 的 Jacobian',
    tag: '不能逐元素',
    intuition: 'softmax 输出每个分量都依赖所有输入 — 所以反向是个"全连"的耦合, 但有个非常优雅的封闭式。',
    formula: '设 a = softmax(s),  da 是上游梯度\nds_i = a_i · (da_i - Σ_j a_j · da_j)\n            ↑ 减去整体加权和',
    where: 'model.py: attention_backward',
  },
  {
    title: '残差的反向 = 两路相加',
    tag: '梯度高速公路',
    intuition: 'out = h + m  ⇒  dh = dout, dm = dout — 一份梯度, 两条路径各拿一份, 最后回到 x 的 dx 要把"绕过 MLP 的"和"穿过 MLP 的"两路加起来。',
    formula: 'h = x + Attn(x)\nout = h + MLP(h)\n\ndh  = dout\ndm  = dout\ndx  = dout (绕过 Attn) + Attn_backward(dout, ...) (穿过 Attn)',
    where: 'model.py: block_backward',
  },
]

const diffs = [
  { topic: '位置编码',  basic: '学得式 pos_emb',            modern: 'RoPE (旋转, 乘在 Q/K 上)',           route: 'position', routeLabel: '阶段 2.2' },
  { topic: '注意力',    basic: '单头, head_dim = D',        modern: 'MHA / GQA / MLA / DSA',               route: 'attention', routeLabel: '阶段 2.1' },
  { topic: 'FFN 激活',  basic: 'ReLU + 普通两层',           modern: 'GELU → SwiGLU (门控 + 三个 linear)',  route: 'blocks', routeLabel: '阶段 2.3' },
  { topic: '层数',      basic: '1 层',                      modern: 'N 层 (一个 for 循环就够)',            route: 'blocks', routeLabel: '阶段 2.3' },
  { topic: 'FFN 形态',  basic: '稠密 MLP',                  modern: 'MoE: 一组小 FFN + router top-k',       route: 'moe', routeLabel: '阶段 2.4' },
  { topic: '推理',      basic: '每步重算整段 forward',      modern: 'KV cache: 只算新 token 的 Q, 复用 K/V',  route: 'infer-kv-memory', routeLabel: '阶段 5.1' },
  { topic: '优化器',    basic: '裸 Adam',                   modern: 'AdamW + warmup + cosine + grad clip',  routeLabel: '(参考 llm_models/training/trainer.py)' },
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
