<template>
  <div>
    <h1 class="page-title">任务适配 · 从 SFT / LoRA 到 DPO / GRPO / 蒸馏</h1>
    <p class="page-subtitle">
      预训练给了你一个只会接龙的模型。微调要回答的是:
      <strong>怎么用尽量少的数据和算力, 把它拨到你手上这个任务上</strong>。
      <RepoLink path="llm_finetune/" label="llm_finetune/" tiny /> 把这条路上的 11 种方法都写成了能跑的最小实现。
      本页先立三根主柱 —— 全参 SFT、LoRA、不要 RM 的偏好对齐 (DPO); QLoRA / DoRA、SimPO / ORPO、奖励模型、
      GRPO 及其变体 (DAPO · Dr.GRPO · GSPO)、离线与 on-policy 蒸馏各有专章, 每章都带一个能拖的实验台。
    </p>

    <ChapterIntro
      tldr="SFT 教模型按指令回答, LoRA 把可训参数压到百分之零点几, DPO 把 RLHF 的三个阶段塌缩成一个分类 loss。每一步都是上一步在「数据 / 参数 / 流程」上的减法。"
      question="同样的偏好数据, 为什么 PPO 要同时养 4 个模型, 而 DPO 只要 2 个加一行 logsigmoid?"
      :goals="[
        '说出 SFT / LoRA / DPO 各自要什么数据、动哪些参数、每步几次前向',
        '算清 LoRA 为什么能把可训参数压到 2r/d',
        '讲明白 DPO 是怎么把 RM + PPO 合成一个分类 loss 的',
      ]"
      :codes="[
        { path: 'llm_finetune/methods/sft.py' },
        { path: 'llm_finetune/methods/lora.py' },
        { path: 'llm_finetune/methods/dpo.py' },
        { path: 'llm_finetune/utils/param_utils.py' },
      ]"
      :prereq="prevChapter"
      :next-step="{ name: 'finetune-sft', label: '下一章 · SFT 数据与 loss' }"
    />

    <!-- ── 1. 三阶段 alignment 总览 ───────────────────────────── -->
    <section class="section">
      <h2>1. 三阶段 alignment 与 DPO 的塌缩</h2>
      <p class="lead">
        ChatGPT (2022) 之后, 把通用 LM 变成能用的助手, 标准流程是
        <strong>SFT → Reward Model → PPO</strong>。三步各要一次训练, PPO 那步还要同时养 4 个模型。
        DPO (2023) 把后两步合成一个"对偏好对的分类 loss", RM 和 RL 都不用了。
      </p>

      <EvolutionChain
        title="演进 · 每一步都是流程上的简化"
        subtitle="SFT 教模型「怎么回答」, RM 学「人类更喜欢哪个」, PPO 在 RM 信号下试探性优化。DPO 把后两者合并。"
        :steps="alignSteps"
      />
    </section>

    <!-- ── 2. SFT ──────────────────────────────────────────── -->
    <section class="section">
      <h2>2. SFT · 在 prompt 上 mask 掉 loss</h2>
      <p class="lead">
        和预训练的代码差别只有一行: 把 prompt 位置的 labels 改成
        <code class="inline">-100</code>, cross-entropy 就会跳过它们。
        这一行最容易写错的是边界 —— labels 已经左移一格, 该盖的是前 P−1 格, 不是前 P 格。
        盖多一格, loss 曲线看不出异常, 但模型永远学不会"回答该怎么开头":
        同配置实测留出集 exact-match 从 1.000 掉到 0.000。
        <router-link :to="{ name: 'finetune-sft' }">SFT 一章的实验台</router-link>可以亲手把这一格点出来。
      </p>

      <div class="grid grid-2" style="gap: 16px;">
        <div class="card">
          <h3>labels 怎么构造 <span class="tag">关键</span></h3>
          <pre class="code">{{ sftLabels }}</pre>
          <p class="hint">
            为什么要 mask prompt? prompt 是人写的, 模型不该把"用户怎么提问"也学成自己要生成的东西。
            mask 不省显存也不省计算 (前向照样要算完整条序列), 它只是把监督信号全部押在 response 上。
          </p>
        </div>

        <div class="card">
          <h3>SFTLoss 就是预训练那个 loss <span class="tag">代码对照</span></h3>
          <pre class="code">{{ sftLoss }}</pre>
          <p class="hint">
            仓库里没有第二个类: <code class="inline">SFTLoss</code> 是
            <code class="inline">StandardLMLoss</code> 的别名, 一行赋值。
            SFT 与预训练的差别 100% 在 labels 里, 不在 loss 里 —— 这也是本章最该记住的一句。
            对应代码: <RepoLink path="llm_finetune/methods/sft.py" label="llm_finetune/methods/sft.py" tiny />
          </p>
        </div>
      </div>
    </section>

    <!-- ── 3. LoRA ─────────────────────────────────────────── -->
    <section class="section">
      <h2>3. LoRA · 给冻结的 W 加一对低秩补丁</h2>
      <p class="lead">
        论文 (Hu et al., 2021) 的关键观察: 微调带来的增量 ΔW <strong>本征秩很低</strong>,
        一对 r 维矩阵就装得下。训练时只更新这对矩阵, 推理时合并回去, 零额外开销。
        注意 LoRA 买到的是什么: 显存、存储、多租户可插拔。<strong>不是收敛速度</strong> ——
        本仓库同一基座各训 300 步, 全参留出集 EM 0.809, LoRA(r=8) 只有 0.352。
      </p>

      <div class="grid grid-2" style="gap: 16px;">
        <div class="card">
          <h3>数学形式 <span class="tag">3 行</span></h3>
          <pre class="code">{{ loraMath }}</pre>
          <p class="hint">
            <strong>无害启动</strong>: B 全零 ⇒ 第 0 步 BA = 0 ⇒ forward 输出与原模型逐位相同。
            微调起点就是预训练终点, 不会一上来先把已有能力砸坏。
            为什么是 B 而不是 A? 置零哪个都能让 BA = 0, 但被置零的那个第 1 步就有梯度, 另一个要等第 2 步;
            两个都置零则梯度永远是 0。
          </p>
        </div>

        <div class="card">
          <h3>参数量 · 凭什么省 99.6% <span class="tag">直观对比</span></h3>
          <table class="cmp">
            <thead>
              <tr>
                <th>方案</th>
                <th class="num">可训练参数</th>
                <th>注释</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>全参微调</td>
                <td class="num mono">d_in × d_out</td>
                <td class="muted">4096² ≈ 16M (一层 attn 投影)</td>
              </tr>
              <tr class="hl">
                <td>LoRA r=8</td>
                <td class="num mono">r × (d_in + d_out)</td>
                <td class="muted">8 × 8192 ≈ 65K  (≈ 0.4%)</td>
              </tr>
            </tbody>
          </table>
          <p class="hint">
            r 可调: 任务越难、数据越多, r 就要越大。α 一般取 2r 或 r,
            <code class="inline">scale = α/r</code> 让你改 r 之后不必重调学习率。
            另外 LoRA 常常要比全参高 3~10 倍的 lr: B 从 0 起步, ΔW 的有效步长本来就小。
          </p>
        </div>
      </div>

      <div class="card" style="margin-top: 16px;">
        <h3>LoRALinear · 一张图看清训练 / 合并两条路径</h3>
        <pre class="code">{{ loraDiagram }}</pre>
        <div class="lora-points">
          <div class="point">
            <strong>训练时</strong>
            <span class="muted">forward 走双分支, 只反传 A / B 的梯度;
            base.requires_grad = False。</span>
          </div>
          <div class="point">
            <strong>推理时</strong>
            <span class="muted">调 <code class="inline">merge()</code> 把 (α/r)·BA
            加进 base.weight, 之后 forward 走单分支 — 与原 nn.Linear 同速。</span>
          </div>
          <div class="point">
            <strong>落盘只有适配器</strong>
            <span class="muted"><code class="inline">get_lora_state_dict</code> 只抽出 A、B 两个矩阵 —
            本仓库 76 KB, 整模型 389 KB。基座共享, 适配器单独分发。</span>
          </div>
        </div>
        <p class="hint">
          代码: <RepoLink path="llm_finetune/methods/lora.py:LoRALinear" label="llm_finetune/methods/lora.py:LoRALinear / apply_lora / merge_lora_weights / get_lora_state_dict" tiny />
        </p>
      </div>

      <div class="card" style="margin-top: 16px;">
        <h3>注入哪些层 · target_modules <span class="tag">LLaMA 经验</span></h3>
        <table class="cmp">
          <thead>
            <tr><th>策略</th><th>注入</th><th>取舍</th></tr>
          </thead>
          <tbody>
            <tr><td>最小化</td><td class="mono">w_q, w_v</td><td class="muted">论文起点, 性能/参数比最优</td></tr>
            <tr class="hl"><td>推荐 (本仓库默认)</td><td class="mono">w_q, w_k, w_v, w_o</td><td class="muted">attention 全投影; 实战甜点</td></tr>
            <tr><td>全注入</td><td class="mono">attn 全部 + ffn 全部</td><td class="muted">QLoRA 论文的做法, 效果最好但 adapter 参数翻倍</td></tr>
          </tbody>
        </table>
        <p class="hint">
          <code class="inline">apply_lora</code> 用 <code class="inline">named_modules</code>
          匹配属性名的最后一段 (例如 "w_q"), 跟层在第几块、叫什么路径无关 —
          所以同一份代码能注进 LLaMA / Mistral / 任何沿用同名属性的模型。
        </p>
        <p class="hint">
          LoRA 的两个直系后代各有专章:
          <router-link :to="{ name: 'finetune-qlora' }">QLoRA</router-link>
          把冻结的基座再压到 4 bit (NF4 分位数码本 + 每 block 一个 absmax scale, 约 4.5 bit/参数;
          本仓库整模型 389 KB → 59 KB, 对应
          <RepoLink path="llm_finetune/methods/qlora.py" label="llm_finetune/methods/qlora.py" tiny />);
          <router-link :to="{ name: 'finetune-dora' }">DoRA</router-link>
          把权重拆成"幅度 × 方向", 低秩更新只管转方向, 长度单独学一个标量。
        </p>
      </div>
    </section>

    <!-- ── 4. DPO ──────────────────────────────────────────── -->
    <section class="section">
      <h2>4. DPO · 把偏好学习写成一行 logsigmoid</h2>
      <p class="lead">
        DPO 的洞见: KL 约束下的策略提升问题有<strong>解析最优解</strong>, 把它代回 Bradley-Terry,
        loss 就退化成"chosen 比 rejected 领先多少"的二分类。RM 和 PPO 都不用了。
        代价也要说清楚: 它只优化差值, 所以
        <router-link :to="{ name: 'finetune-dpo' }">chosen 自己的概率可能一路往下掉</router-link>
        (实测 log π(chosen) −4.03 → −4.18, 贪心 EM 0.332 → 0.137)。
      </p>

      <div class="grid grid-2" style="gap: 16px;">
        <div class="card">
          <h3>核心公式 <span class="tag">Bradley-Terry</span></h3>
          <pre class="code">{{ dpoMath }}</pre>
          <p class="hint">
            <strong>π_θ</strong>: 被微调的 policy, 起点是 SFT 终态;
            <strong>π_ref</strong>: 冻结的 reference, 一般就是 SFT 终态的副本;
            <strong>β</strong>: KL 约束强度, 0.1~0.5。β 越大, σ 越早饱和, policy 越贴着 ref 不动 —— 它不是"用力程度"。
            第 0 步 policy = ref, 两个 log-ratio 都是 0, loss 恰好是 −log σ(0) = ln 2 = 0.6931。
            初始 loss 不是这个数, 基本就是 ref 没对齐或 mask 写错了。
          </p>
        </div>

        <div class="card">
          <h3>每步训练做什么 <span class="tag">2 次 LM 前向</span></h3>
          <pre class="code">{{ dpoStep }}</pre>
          <p class="hint">
            chosen 和 rejected 拼成一个 2B 条的 batch, policy 前向一次、ref 前向一次 —— 每步 2 次 LM 前向,
            常驻权重两份 (本仓库 778 KB)。ref 前向不带梯度、不存激活, 所以实测每步只慢约 35%, 不是 100%。
            ref 用 <code class="inline">no_grad</code> + <code class="inline">eval()</code>:
            前者省激活显存, 后者关掉 dropout, 让 log π_ref 是个确定函数, 随机性不会污染偏好梯度。
            代码: <RepoLink path="llm_finetune/methods/dpo.py:DPOLoss" label="llm_finetune/methods/dpo.py:DPOLoss / PairwiseForward" tiny />
          </p>
        </div>
      </div>

      <div class="card" style="margin-top: 16px;">
        <h3>compute_sequence_logprobs · 序列级 log-prob 怎么算</h3>
        <pre class="code">{{ seqLogprob }}</pre>
        <p class="hint">
          为什么不直接用 <code class="inline">F.cross_entropy(reduction='sum')</code>?
          那个会把 batch 里所有样本的 NLL 加成一个数, 逐样本的粒度就没了。
          DPO 要的是 <strong>每个样本各自的 Σ log p</strong>, 才能逐对做 (chosen − rejected) 差分。
          注意求和范围包含<strong>第一个回复 token</strong>: prompt mask 多盖一位就丢掉了 log p(y₁|x),
          而 chosen 与 rejected 的 y₁ 恰恰可能不同。
        </p>
      </div>
    </section>

    <!-- ── 5. 三柱对照 ─────────────────────────────────────── -->
    <section class="section">
      <h2>5. 三柱并排对照</h2>
      <p class="lead">同一组 (model, data) 三种用法, 一张表看完。</p>
      <div class="card" style="padding: 0; overflow-x: auto;">
        <table class="three">
          <thead>
            <tr>
              <th>维度</th>
              <th class="col-sft">SFT</th>
              <th class="col-lora">LoRA</th>
              <th class="col-dpo">DPO</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="row in pillars" :key="row.dim">
              <td class="dim">{{ row.dim }}</td>
              <td class="mono">{{ row.sft }}</td>
              <td class="mono">{{ row.lora }}</td>
              <td class="mono">{{ row.dpo }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <!-- ── 6. 决策树 ─────────────────────────────────────────── -->
    <section class="section">
      <h2>6. 该用哪个 · 简单决策</h2>
      <div class="card">
        <ol class="decision">
          <li v-for="d in decisions" :key="d.q">
            <span class="q">{{ d.q }}</span>
            <span class="a">→ {{ d.a }}</span>
          </li>
        </ol>
        <p class="hint">
          一条常见的实战路线: <strong>base → LoRA-SFT → 解锁基座 → DPO</strong>。
          先用 LoRA 在指令数据上廉价预热, 再用 DPO 在偏好对上对齐。
          这张表只给方向; 想把"每步几次前向、常驻几份权重、落盘什么"一起摆出来比,
          去<router-link :to="{ name: 'finetune-runs' }">训练脚本与落盘一章的选型计算器</router-link>。
        </p>
      </div>
    </section>

    <!-- 本章挂载的实验台 (data/labmap/*.js) 与章末自测 (data/quiz/*.js), 没配置时不渲染 -->

    <LabMount />

    <QuizCard />


    <ChapterNav
      :prev="{ ...prevChapter, hint: '先理解训练系统如何组合成完整主循环' }"
      :next="{ name: 'finetune-sft', label: '下一章 · SFT 数据与 loss', hint: '从数据和 labels mask 开始拆微调' }"
    />
  </div>
</template>

<script setup>
import LabMount from '@/components/LabMount.vue'
import QuizCard from '@/components/QuizCard.vue'
import ChapterIntro from '@/components/ChapterIntro.vue'
import ChapterNav from '@/components/ChapterNav.vue'
import EvolutionChain from '@/components/EvolutionChain.vue'
import RepoLink from '@/components/RepoLink.vue'
import { learningPath } from '@/data/models.js'

// 上一章从 learningPath 取, 不手写编号 (训练阶段加章后手写的 "3.5" 会过期; 与 Infer.vue 同一写法)
const prevItem = learningPath[learningPath.findIndex((x) => x.route === 'finetune') - 1]
const prevChapter = { name: prevItem.route, label: `上一章 · ${prevItem.label}` }

const alignSteps = [
  { name: 'SFT',        year: '2022',
    pain: '裸预训练 LM 不会按指令回答, 只会接龙。',
    fix:  '在 (instruction, response) 配对上做 next-token CE, 但 prompt 区域 mask -100。',
    color: 'var(--left)' },
  { name: 'Reward Model', year: '2022',
    pain: 'SFT 只能模仿示例, 学不到「人更喜欢哪个回答」。',
    fix:  '收集 (prompt, A, B) 偏好对, 训一个二分类 reward model 给任意回答打分。',
    color: 'var(--eye)' },
  { name: 'PPO',        year: '2022',
    pain: 'RM 知道分数, 但模型不知怎么提分 — 还要避免漂得太远。',
    fix:  '用 PPO 在 RM 信号下试探, KL 项约束不离开 ref 太远。需要同时维持 4 个模型。',
    color: 'var(--right)' },
  { name: 'DPO',        year: '2023',
    pain: 'PPO 复杂、显存高、训练不稳; RM 还要单独训一阶段。',
    fix:  '解析推导得到 -log σ(β·(logπ_θ - logπ_ref) 差) — 直接在偏好对上做分类, 跳过 RM 与 RL。',
    color: 'var(--accent)' },
]

const sftLabels = `# 例: x = [<bos>] [Q1] [Q2] [Q3] [A1] [A2] [<eos>], prompt 长 P = 4
idx    = x[:-1] = [<bos>] [Q1]  [Q2]  [Q3] [A1] [A2]
labels = x[1:]  = [-100 ] [-100][-100][A1] [A2] [<eos>]
                   ↑ 只 mask 前 P-1 = 3 格  ↑ 看到 Q3 要预测 A1: 这一格必须留

# pad 也置 -100`

const sftLoss = `# llm_finetune/methods/sft.py 的全部内容 (去掉 docstring):
from llm_models.training.loss import StandardLMLoss

SFTLoss = StandardLMLoss        # ← 别名, 没有新实现

# StandardLMLoss 里就是一句:
#   F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1),
#                   ignore_index=-100)     # prompt / pad 在这里被跳过`

const loraMath = `# 原层
y = W x                                  # W ∈ ℝ^{d_out × d_in}, 冻结

# LoRA 注入
y = W x + (α/r) · B A x
        其中 A ∈ ℝ^{r × d_in},  B ∈ ℝ^{d_out × r},  r ≪ min(d_in, d_out)

# 初始化
A ~ Kaiming 正态           # 与 nn.Linear 默认对齐
B = 0                      # 训练第 1 步 BA=0, 起点 = 预训练终点 ("无害启动")`

const loraDiagram = `              ┌── base (W, b) ──────────►─┐
              │      [冻结]                │
  x ──┬──────┤                              ├──► y
      │      ↓                              │
      └─► dropout → A^T (r) → B^T ──────►─►─┘
                            ↑
                        (α/r) 缩放

# 推理 merge():  W ← W + (α/r)·BA  →  跳过 A/B 分支
# 训练 unmerge(): W ← W - (α/r)·BA  →  恢复双分支`

const dpoMath = `L_DPO = - E_{(x, y_w, y_l)}  log σ(
            β · [ log π_θ(y_w|x)/π_ref(y_w|x)
                - log π_θ(y_l|x)/π_ref(y_l|x) ]
        )

# 直觉
# - 提高 chosen 的 log-prob, 降低 rejected 的 log-prob
# - 但任何变化都「相对 reference」度量, 防止策略漂移
# - σ 把无界 logit 差压到 (0, 1), 形成稳定二分类`

const dpoStep = `# PairwiseForward.with_frozen_copy(policy) 包住的一步
# chosen 与 rejected 拼成一个 2B 条的 batch, 一次前向算完
both = cat([chosen.input_ids, rejected.input_ids])   # [2B, T]

p_logits = model(both)                 # 1) policy 前向, 要梯度
with torch.no_grad():
    r_logits = ref(both)               # 2) ref 前向, 不要梯度、不存激活

loss = DPOLoss(beta).compute(
    {"policy_chosen_logits": ..., "ref_chosen_logits": ..., ...},
    {"chosen_labels": ..., "rejected_labels": ...},
)["total_loss"]
loss.backward();  optimizer.step()     # 没有 DPOTrainer, 走通用 Trainer`

const seqLogprob = `def compute_sequence_logprobs(logits, labels, ignore=-100):
    log_p_full = F.log_softmax(logits, dim=-1)            # [B, T, V]
    valid      = labels != ignore                         # [B, T]

    safe_labels = labels.masked_fill(~valid, 0)            # 让 gather 合法
    token_logp  = log_p_full.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)

    return (token_logp * valid.float()).sum(dim=-1)        # [B]  ← 逐样本求和`

const pillars = [
  { dim: '数据形态',     sft: '(instruction, response)',  lora: '(instruction, response)',  dpo: '(prompt, chosen, rejected)' },
  { dim: '可训练参数',   sft: '100% 模型',                lora: '2r/d; d=4096, r=8 → 0.39%', dpo: '100% policy, ref 冻结' },
  { dim: 'loss',         sft: '带 -100 mask 的 CE',       lora: '同 SFT (只是参数更少)',    dpo: '-log σ(β·logit_diff)' },
  { dim: '需要 ref?',    sft: '否',                       lora: '否',                       dpo: '是 (deepcopy + freeze + eval)' },
  { dim: '需要 RM?',     sft: '否',                       lora: '否',                       dpo: '否 (DPO 的核心收益)' },
  { dim: '每步 LM 前向', sft: '1',                        lora: '1',                        dpo: '2 (chosen+rejected 拼 2B 条, policy 1 次 + ref 1 次)' },
  { dim: '常驻权重',     sft: '1 份 (389 KB)',            lora: '1 份冻结 + 适配器',         dpo: '2 份 (778 KB)' },
  { dim: '落盘',         sft: '一份完整权重',             lora: '适配器 (本仓库 76 KB) + 共享基座', dpo: '一份完整权重' },
]

const decisions = [
  { q: '只有 (问, 答), 显卡够大?',                    a: '全参 SFT — 上限最高, 同步数下也最快' },
  { q: '只有 (问, 答), 显卡紧?',                      a: 'LoRA SFT — 可训参数 2r/d, 适配器好切换' },
  { q: '连冻结的基座都塞不进显存?',                   a: 'QLoRA — 基座 NF4 约 4.5 bit/参数, 只训 adapter' },
  { q: '同样的 r, 想再往全参靠一点?',                 a: 'DoRA — 低秩只管方向, 长度另学一个标量' },
  { q: '有 SFT 模型, 又拿到了 (chosen, rejected)?',   a: 'DPO — 跳过 RM 与 PPO, 一行 loss' },
  { q: '有偏好对, 但不想再养一个 ref?',               a: 'SimPO / ORPO — 前向次数和常驻权重都减半' },
  { q: '连 SFT 阶段都想省掉?',                        a: 'ORPO — loss 里的 NLL 项就是 SFT' },
  { q: '答案能被程序判对错 (数学 / 代码)?',            a: 'GRPO 系 (DAPO · Dr.GRPO · GSPO) — 在线采样 + 可验证奖励' },
  { q: '有一个强 teacher, 想把能力压进小模型?',        a: '离线蒸馏起步, on-policy 蒸馏收尾' },
]
</script>

<style scoped>
table.cmp,
table.three {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
table.cmp th, table.cmp td,
table.three th, table.three td {
  padding: 9px 12px;
  border-bottom: 1px solid var(--border);
  text-align: left;
  vertical-align: middle;
}
table.cmp th, table.three th {
  background: var(--bg-elev);
  color: var(--text-muted);
  font-size: 11px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.7px;
}
table.cmp tr.hl, table.three tr:hover { background: var(--bg-elev); }
table.cmp .num { text-align: right; }
table.cmp .muted, table.three .muted { color: var(--text-dim); }

table.three th.col-sft  { color: var(--left); }
table.three th.col-lora { color: var(--accent); }
table.three th.col-dpo  { color: var(--eye); }
table.three .dim { color: var(--text-muted); width: 130px; }

.lora-points {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
  margin-top: 14px;
}
@media (max-width: 960px) {
  .lora-points { grid-template-columns: 1fr; }
}
.point {
  padding: 10px 12px;
  background: var(--bg-elev);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.point strong { font-size: 12px; color: var(--text); }
.point .muted { color: var(--text-muted); font-size: 12px; line-height: 1.55; }

.hint {
  margin-top: 10px;
  font-size: 12px;
  color: var(--text-dim);
  line-height: 1.6;
}

.decision {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.decision li {
  display: flex;
  gap: 12px;
  padding: 10px 12px;
  background: var(--bg-elev);
  border-radius: var(--radius-sm);
  border-left: 3px solid var(--accent);
  font-size: 13px;
}
.decision .q { color: var(--text); flex: 1; }
.decision .a { color: var(--accent); font-family: "SF Mono", Menlo, monospace; font-size: 12.5px; }
</style>
