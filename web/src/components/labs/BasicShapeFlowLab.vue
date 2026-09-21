<!--
  形状流水线实验台 (对应 llm_basic/model.py:transformer_forward)。
  只讲一件事: 一个张量从 ids [B,T] 走到 logits [B,T,V] 的路上, 每一步形状怎么变、
  往 cache 里塞了什么、吃掉多少参数和激活。点任意一行看细节, 拖滑杆看总账跟着变。
-->
<template>
  <LabFrame
    title="形状流水线 — ids [B,T] 到 logits [B,T,V] 的一路"
    sub="每一行是 model.py 里真实的一步, 条的宽度就是输出最后一维 (D / T / H / V) 的相对大小。点任意一行: 右侧告诉你它算什么、形状怎么变、往 cache 里存了什么、为什么反向非要这个值。拖 B/T/D/V/层数, 看参数和激活的总账怎么走。"
    module="llm_basic/model.py"
    run="cd llm_basic && python train.py --max-iters 300 --out /tmp/c.npz"
    :challenge="{
      ask: '只把 T 从 64 拖到 256, 其它不动。参数量涨多少? 激活字节涨多少? 为什么长上下文的账主要不记在参数表里?',
      answer: '参数里只有 pos_emb 跟着 T 长: 64×64=4,096 → 256×64=16,384, 总量 45,568 → 57,856, 涨 27%。激活从 21.0 MB 涨到 120 MB, 是 5.7 倍 —— 因为 scores / mask / attn 这三张 [B,T,T] 按 T² 长, 占激活的比例从 14% 升到 40%。参数表里根本没有 T²; 长上下文贵在激活和注意力的 T² 上。这也是为什么后面会有 KV cache、FlashAttention、activation checkpoint 这一串技术, 而不是去压参数。',
    }"
  >
    <template #controls>
      <div class="row">
        <button type="button" @click="preset(32, 64, 64, 65, 1)">llm_basic 默认 (B32 T64 D64 V65)</button>
        <button type="button" @click="preset(32, 64, 64, 650, 1)">词表 ×10</button>
        <button type="button" @click="preset(32, 256, 64, 65, 1)">序列 ×4</button>
      </div>
      <LabSlider v-model="B" label="batch B" :min="1" :max="64" />
      <LabSlider v-model="T" label="序列长 T (= pos_emb 行数)" :min="8" :max="256" :step="8" />
      <LabSlider v-model="D" label="模型维 D (H 取 2D)" :min="16" :max="256" :step="16" />
      <LabSlider v-model="V" label="词表 V" :min="16" :max="1024" />
      <LabSlider v-model="L" label="层数 n_layer" :min="1" :max="6" />
    </template>

    <div class="flow">
      <template v-for="r in rows" :key="r.id">
        <p v-if="r.id === 'n1'" class="loopline mono">for i in range(n_layer): ↓ 下面 13 步重复 {{ L }} 次</p>
        <button
          type="button" class="op" :class="{ sel: sel === r.id, blk: r.blk }"
          :aria-pressed="sel === r.id" @click="sel = r.id"
        >
          <span class="nm mono">{{ r.name }}</span>
          <span class="track"><span class="bar" :class="r.tone" :style="{ width: r.w + '%' }" /></span>
          <span class="sh mono">{{ r.shape }}</span>
          <span class="pp mono">{{ r.params ? '+' + fmtNum(r.params * (r.blk ? L : 1)) : '' }}</span>
        </button>
        <p v-if="r.id === 'res2'" class="loopline mono">↑ 进出都是 [B,T,D], 所以加层只是这个 for 循环</p>
      </template>
    </div>

    <template #stats>
      <div class="kv">
        <span>参数合计</span>
        <b :class="total.params === 45568 ? 'good' : ''">{{ total.params.toLocaleString() }}</b>
      </div>
      <div class="kv"><span>激活 (float64)</span><b>{{ fmtBytes(total.act * 8) }}</b></div>
      <div class="kv"><span>激活 ÷ 参数</span><b :class="ratio > 100 ? 'bad' : ''">{{ ratio.toFixed(0) }}×</b></div>
      <div class="kv">
        <span>[B,T,T] 占激活</span>
        <b :class="ttShare > 0.3 ? 'bad' : ''">{{ (ttShare * 100).toFixed(0) }}%</b>
      </div>

      <div class="detail">
        <b class="mono">{{ cur.full }}</b>
        <p class="one">{{ cur.one }}</p>
        <p class="sline mono">{{ cur.shapeIn }} → {{ cur.shape }}</p>
        <p class="kv2"><span class="k">cache</span><span class="mono">{{ cur.cache }}</span></p>
        <p class="why">{{ cur.why }}</p>
        <p class="kv2"><span class="k">参数</span><span class="mono">{{ cur.params ? `${cur.pf} = ${(cur.params * (cur.blk ? L : 1)).toLocaleString()}` : '0 (纯计算, 没有权重)' }}</span></p>
      </div>

      <div class="mask">
        <h4>causal mask <span class="mono">{{ T }}×{{ T }}</span></h4>
        <svg viewBox="0 0 100 100" role="img" :aria-label="`${T} 乘 ${T} 的因果掩码`">
          <rect x="0" y="0" width="100" height="100" class="mk-open" />
          <polygon :points="maskPoints" class="mk-cut" />
          <template v-if="T <= 32">
            <line v-for="i in T + 1" :key="'h' + i" x1="0" x2="100" :y1="(i - 1) * cell" :y2="(i - 1) * cell" class="mk-grid" />
            <line v-for="i in T + 1" :key="'v' + i" y1="0" y2="100" :x1="(i - 1) * cell" :x2="(i - 1) * cell" class="mk-grid" />
          </template>
        </svg>
        <p class="lab-note">
          行 = query i, 列 = key j。右上角 (j &gt; i, 未来) 填 −inf, softmax 后权重为 0。
          格子按 T² 长, 有效的只有一半多一点: {{ fmtNum(T * (T + 1) / 2) }} / {{ fmtNum(T * T) }}。
        </p>
      </div>
      <p class="lab-note">
        <strong>全模型只有这里, 不同位置之间才互相说话。</strong>
        embedding / RMSNorm / MLP / lm_head 都是对每个位置各算各的 —— 把 [B,T,D] 拍平成 [B·T, D] 结果一模一样。
      </p>
      <p class="lab-note">
        默认那一列 (V=65, D=64, H=128, T=64, n_layer=1) 的参数合计必须是 45,568 —
        和 <code class="inline">python train.py</code> 开头打印的数字一致。
      </p>
    </template>
  </LabFrame>
</template>

<script setup>
import { computed, ref } from 'vue'
import LabFrame from '@/components/lab/LabFrame.vue'
import LabSlider from '@/components/lab/LabSlider.vue'
import { fmtBytes, fmtNum } from '@/utils/labmath.js'

const B = ref(32), T = ref(64), D = ref(64), V = ref(65), L = ref(1)
const sel = ref('attn')
const preset = (b, t, d, v, l) => { B.value = b; T.value = t; D.value = d; V.value = v; L.value = l }
const cell = computed(() => 100 / T.value)
// 被屏蔽区 (j > i) 的边界: T 不大时画成真正的阶梯, 对角线那一格才不会被误涂
const maskPoints = computed(() => {
  const c = cell.value, n = T.value
  if (n > 32) return `${c},0 100,0 100,${100 - c}`
  const pts = [`${c},0`]
  for (let i = 1; i < n; i++) pts.push(`${i * c},${i * c}`, `${(i + 1) * c},${i * c}`)
  return pts.concat(`100,${100 - c}`, '100,0').join(' ')
})

// 每一行 = model.py 里真实的一步。blk: 属于 block 内部, 参数和激活要乘 n_layer
// last: 输出最后一维 (决定条的宽度); e: 输出元素数
const rows = computed(() => {
  const b = B.value, t = T.value, d = D.value, v = V.value, h = 2 * d
  const bt = b * t, btd = bt * d, btt = bt * t, bth = bt * h, btv = bt * v
  const list = [
    { id: 'ids', name: 'ids', full: 'ids (输入)', shapeIn: '文本', shape: `[${b},${t}]`, last: t, e: bt, tone: 'i',
      one: '一批 token id, int64。get_batch 随机挑 B 个起点, 各切 T 个 token。',
      cache: '—', why: 'ids 本身就是下一步 embedding 的 cache。', params: 0 },
    { id: 'tok', name: 'tok_emb 查表', full: 'embedding_forward(ids, tok_emb)', shapeIn: `[${b},${t}]`, shape: `[${b},${t},${d}]`, last: d, e: btd, tone: 'd',
      one: 'W[ids]: 每个 id 取出表里对应的一行 D 维向量。',
      cache: '(ids, W.shape)',
      why: '反向只要知道每个位置取的是哪一行, 就能用 np.add.at 把 dout 散射回 [V,D]。W 本身用不上, 所以不存。',
      params: v * d, pf: 'V×D' },
    { id: 'pos', name: '+ pos_emb', full: 'h = tok + embedding_forward(pos_ids, pos_emb)', shapeIn: `[${b},${t},${d}]`, shape: `[${b},${t},${d}]`, last: d, e: btd, tone: 'd',
      one: '每行 pos_ids 都是 0..T−1, 查位置表再加到 tok 上 —— 位置信息就这样进了残差流。',
      cache: '(pos_ids, W.shape); 加法本身不存',
      why: 'h = tok + pos 的反向是把同一个 dh 原样发给两张表, 不需要任何中间值。',
      params: t * d, pf: 'T_max×D' },

    { id: 'n1', name: 'RMSNorm', full: 'rmsnorm_forward(x, block_i_norm1_g)', shapeIn: `[${b},${t},${d}]`, shape: `[${b},${t},${d}]`, last: d, e: btd, tone: 'd', blk: 1,
      one: '每行除以自己的 rms = √(mean(x²)+ε), 再逐维乘 g。',
      cache: '(x, g, rms)',
      why: '反向的耦合项 x·Σⱼ(dyⱼ gⱼ xⱼ)/(D·rms³) 三样全要。只存输出算不出来 —— 归一化让一行里每个元素的梯度互相牵连。',
      params: d, pf: 'D' },
    { id: 'qkv', name: 'Q,K,V = x@W', full: 'linear_forward(n1, Wq/Wk/Wv, None) ×3', shapeIn: `[${b},${t},${d}]`, shape: `3×[${b},${t},${d}]`, last: d, e: 3 * btd, tone: 'd', blk: 1,
      one: '同一个 n1 分别乘三张 [D,D] 权重, 得到查询、键、值。单头, 所以 head_dim 就等于 D。',
      cache: '三份 (x, W)',
      why: 'dW = xᵀ@dout 要 x, dx = dout@Wᵀ 要 W。三条支路共用同一个 x, 所以反向要把三路 dx 加起来。',
      params: 3 * d * d, pf: '3×D×D' },
    { id: 'scores', name: 'Q @ Kᵀ / √D', full: 'scores = Q @ K.transpose(0,2,1) * scale', shapeIn: `[${b},${t},${d}]`, shape: `[${b},${t},${t}]`, last: t, e: btt, tone: 'tt', blk: 1,
      one: '每个 query 和每个 key 做点积。除以 √D 是因为点积方差正比于 D, 不缩放 softmax 会饱和。',
      cache: 'Q, K (存在 attention 的 cache 里)',
      why: 'dQ = ds@K、dK = dsᵀ@Q, 两边互相需要对方的前向值。这也是全模型唯一形状里出现两个 T 的地方。',
      params: 0 },
    { id: 'mask', name: 'causal mask', full: 'np.where(causal_mask(T), -inf, scores)', shapeIn: `[${b},${t},${t}]`, shape: `[${b},${t},${t}]`, last: t, e: btt, tone: 'tt', blk: 1,
      one: '上三角 (j > i, 也就是未来) 填 −inf。有了它, 一次前向才等于并行做了 T 道 next-token 题。',
      cache: '不存 — mask 由 T 现场生成',
      why: 'causal_mask(T) 是纯函数, 反向压根用不到: 被屏蔽的位置 attn = 0, softmax 反向自带因子 a, 梯度自动为 0, 不用再 mask 一次。',
      params: 0 },
    { id: 'attn', name: 'softmax', full: 'attn = softmax(scores, axis=-1)', shapeIn: `[${b},${t},${t}]`, shape: `[${b},${t},${t}]`, last: t, e: btt, tone: 'tt', blk: 1,
      one: '每行归一成一个概率分布, 和为 1: 位置 i 该从哪些位置取信息。',
      cache: 'attn (softmax 的输出)',
      why: '反向 ds = a ⊙ (da − Σⱼ aⱼ daⱼ) 只用到输出 a, 用不上输入 scores。所以 cache 存的是 attn 而不是 scores —— 反向要什么, forward 才存什么。',
      params: 0 },
    { id: 'ctx', name: 'attn @ V', full: 'ctx = attn @ V', shapeIn: `[${b},${t},${t}]`, shape: `[${b},${t},${d}]`, last: d, e: btd, tone: 'd', blk: 1,
      one: '按注意力权重把 V 的各行加权平均。T² 在这里塌回 D。',
      cache: 'attn, V',
      why: 'dV = attnᵀ@dctx 要 attn, dattn = dctx@Vᵀ 要 V。',
      params: 0 },
    { id: 'ao', name: '@ Wo', full: 'linear_forward(ctx, Wo, None)', shapeIn: `[${b},${t},${d}]`, shape: `[${b},${t},${d}]`, last: d, e: btd, tone: 'd', blk: 1,
      one: '把 attention 的输出投影回残差流的维度。',
      cache: '(ctx, Wo)', why: '和其它 linear 一样: dW 要输入, dx 要权重。',
      params: d * d, pf: 'D×D' },
    { id: 'res1', name: '+ 残差 1', full: 'h = x + a', shapeIn: `[${b},${t},${d}]`, shape: `[${b},${t},${d}]`, last: d, e: btd, tone: 'd', blk: 1,
      one: '子层输出加回主干。主干上全程只有加法, 没有任何非线性。',
      cache: '不存',
      why: '加法的反向是把 dout 原样发给两条路: dx = dout(捷径) + dx_from_n1(穿过 attention 回来的)。那个原样的 dout 就是梯度不消失的原因。',
      params: 0 },
    { id: 'n2', name: 'RMSNorm', full: 'rmsnorm_forward(h, block_i_norm2_g)', shapeIn: `[${b},${t},${d}]`, shape: `[${b},${t},${d}]`, last: d, e: btd, tone: 'd', blk: 1,
      one: '进 MLP 之前再归一一次。pre-norm: 归一化在子层里面, 主干保持干净。',
      cache: '(h, g, rms)', why: '同 norm1: 耦合项要输入、gain 和 rms 三样。',
      params: d, pf: 'D' },
    { id: 'mlp1', name: '@ W1 + b1', full: 'h1 = linear_forward(n2, mlp_W1, mlp_b1)', shapeIn: `[${b},${t},${d}]`, shape: `[${b},${t},${h}]`, last: h, e: bth, tone: 'h', blk: 1,
      one: '升维到 H。这里 H = 2D (代码里 D=64 → H=128)。',
      cache: '(n2, W1)', why: 'db = Σ dout 不用存任何东西, dW 要输入, dx 要 W。',
      params: d * h + h, pf: 'D×H + H' },
    { id: 'relu', name: 'ReLU', full: 'a1 = relu_forward(h1)', shapeIn: `[${b},${t},${h}]`, shape: `[${b},${t},${h}]`, last: h, e: bth, tone: 'h', blk: 1,
      one: '负数截成 0。全模型的非线性就这一处 (再加一个 softmax)。',
      cache: 'h1 (ReLU 的输入)',
      why: '反向 dout * (x > 0) 必须知道哪些位置当初是负的。这份 cache 和输出一样大, 是 block 里最胖的两张之一。',
      params: 0 },
    { id: 'mlp2', name: '@ W2 + b2', full: 'm = linear_forward(a1, mlp_W2, mlp_b2)', shapeIn: `[${b},${t},${h}]`, shape: `[${b},${t},${d}]`, last: d, e: btd, tone: 'd', blk: 1,
      one: '降维回 D, 准备加回主干。',
      cache: '(a1, W2)', why: '同上。', params: h * d + d, pf: 'H×D + D' },
    { id: 'res2', name: '+ 残差 2', full: 'out = h + m', shapeIn: `[${b},${t},${d}]`, shape: `[${b},${t},${d}]`, last: d, e: btd, tone: 'd', blk: 1,
      one: '第二个残差。进 block 是 [B,T,D], 出 block 还是 [B,T,D] —— 所以堆层不用重新推导。',
      cache: '不存', why: '和残差 1 一样: 两条路各拿一份 dout。', params: 0 },

    { id: 'nf', name: 'RMSNorm (final)', full: 'rmsnorm_forward(h, norm_f_g)', shapeIn: `[${b},${t},${d}]`, shape: `[${b},${t},${d}]`, last: d, e: btd, tone: 'd',
      one: '投到词表之前最后归一一次。',
      cache: '(h, g, rms)', why: '同前两次 RMSNorm。', params: d, pf: 'D' },
    { id: 'head', name: '@ lm_head', full: 'logits = linear_forward(h, lm_head, None)', shapeIn: `[${b},${t},${d}]`, shape: `[${b},${t},${v}]`, last: v, e: btv, tone: 'v',
      one: '把每个位置的 D 维向量投影成 V 个分数。每个位置都在猜下一个 token。',
      cache: '(h, lm_head)',
      why: 'dW = hᵀ@dlogits, dh = dlogits@lm_headᵀ。把 V 拖大就能看见: 这一条的宽度会压过整个主干, 参数和激活都是。',
      params: d * v, pf: 'D×V' },
    { id: 'ce', name: 'cross-entropy', full: 'cross_entropy_forward_backward(logits, y)', shapeIn: `[${b},${t},${v}]`, shape: `loss + dlogits`, last: v, e: 0, tone: 'v',
      one: `全程在 log 域算 −mean(log p[y]), 对 N = B·T = ${bt.toLocaleString()} 个位置求平均; dlogits 的形状和 logits 一样, 还是 [${b},${t},${v}]。`,
      cache: '不存 — forward 和 backward 合成了一个函数',
      why: 'softmax 和 CE 分开写时中间的 1/p[y] 会爆炸, 合起来正好约掉: dlogits = (p − onehot)/N。这就是反向的起点, 下一章从它开始倒走。',
      params: 0 },
  ]
  const mx = Math.max(d, 2 * d, v, t)
  return list.map((r) => ({ ...r, w: Math.max(4, (r.last / mx) * 100) }))
})

const total = computed(() => rows.value.reduce(
  (a, r) => ({ params: a.params + r.params * (r.blk ? L.value : 1), act: a.act + r.e * (r.blk ? L.value : 1) }),
  { params: 0, act: 0 },
))
const ratio = computed(() => (total.value.act * 8) / (total.value.params * 8 || 1))
const ttShare = computed(() => {
  const tt = rows.value.filter((r) => r.tone === 'tt').reduce((s, r) => s + r.e * L.value, 0)
  return tt / (total.value.act || 1)
})
const cur = computed(() => rows.value.find((r) => r.id === sel.value) || rows.value[0])
</script>

<style scoped>
.flow { display: flex; flex-direction: column; gap: 2px; min-width: 0; }
.loopline { font-size: 10.5px; color: var(--text-dim); padding: 4px 0 2px 8px; }

.op {
  display: grid; grid-template-columns: minmax(0, 104px) minmax(24px, 1fr) minmax(0, 98px) minmax(0, 52px);
  gap: 8px; align-items: center; width: 100%;
  padding: 4px 6px; text-align: left; background: none; border: 1px solid transparent;
  border-radius: var(--radius-sm); cursor: pointer;
}
.op:hover { background: var(--bg-elev); }
.op.blk { border-left: 2px solid var(--accent-soft); }
.op.sel { background: var(--accent-soft); border-color: var(--accent); }
.nm { font-size: 11px; color: var(--text); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.track { height: 12px; background: var(--bg-elev); border-radius: 2px; overflow: hidden; }
.bar { display: block; height: 100%; border-radius: 2px; }
.bar.i { background: var(--text-dim); }
.bar.d { background: var(--accent); }
.bar.tt { background: var(--warn); }
.bar.h { background: var(--left); }
.bar.v { background: var(--right); }
.sh { font-size: 10.5px; color: var(--text-muted); text-align: right; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
.pp { font-size: 10px; color: var(--text-dim); text-align: right; overflow: hidden; white-space: nowrap; }

.mask { display: grid; gap: 6px; }
.mask h4 { font-size: 12px; color: var(--text-muted); font-weight: 500; }
.mask svg { width: 126px; height: 126px; }
.mk-open { fill: var(--accent-soft); stroke: var(--border-strong); }
.mk-cut { fill: color-mix(in srgb, var(--danger) 28%, transparent); }
.mk-grid { stroke: var(--border); stroke-width: 0.4; }

.detail { border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 8px 10px; display: grid; gap: 6px; }
.detail > b { color: var(--text); font-size: 11.5px; word-break: break-all; }
.detail p { line-height: 1.65; }
.detail .kv2 { display: grid; grid-template-columns: 34px 1fr; gap: 6px; }
.detail .k { color: var(--text-dim); }
.detail .sline { color: var(--accent); font-size: 11.5px; }
.detail .why { color: var(--text-muted); border-left: 2px solid var(--border-strong); padding-left: 8px; }

@media (max-width: 720px) {
  .op { grid-template-columns: minmax(0, 88px) minmax(20px, 1fr) minmax(0, 84px) minmax(0, 44px); gap: 5px; }
}
</style>
