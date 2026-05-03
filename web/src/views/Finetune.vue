<template>
  <div>
    <h1 class="page-title">任务适配 · SFT / LoRA / DPO</h1>
    <p class="page-subtitle">
      微调的核心问题是: <strong>用尽量少的数据和算力, 把一个通用 base model 拨到具体任务上</strong>。
      <RepoLink path="llm_finetune/" label="llm_finetune/" tiny /> 用三个最小可跑的实现, 串起从"全参 SFT"到
      "PEFT (LoRA)"再到"无 RM 的偏好对齐 (DPO)"的完整脉络。
    </p>

    <ChapterIntro
      tldr="SFT 把语言模型变对话模型, LoRA 把全参微调压到 0.5%, DPO 把 RLHF 三阶段塌缩成一个分类 loss。每一步都是上一步在「数据 / 参数 / 流程」上的简化。"
      question="同样的偏好数据, 为什么 PPO 要维持 4 个模型, 而 DPO 只需要 2 个 + 一行 logsigmoid?"
      :goals="[
        '区分 SFT / LoRA / DPO 三类方法的输入数据形态和 loss',
        '理解 LoRA 把可训参数压到 0.5% 的核心数学',
        '知道 DPO 为什么能用一个分类 loss 替换 PPO 三阶段',
      ]"
      :codes="[
        { path: 'llm_finetune/methods/sft.py' },
        { path: 'llm_finetune/methods/lora.py' },
        { path: 'llm_finetune/methods/dpo.py' },
        { path: 'llm_finetune/utils/param_utils.py' },
      ]"
      :prereq="{ name: 'train-collectives-loop', label: '阶段 3.5 · 通信与 full_loop' }"
      :next-step="{ name: 'finetune-sft', label: '阶段 4.1 · SFT 数据与 loss' }"
    />

    <!-- ── 1. 三阶段 alignment 总览 ───────────────────────────── -->
    <section class="section">
      <h2>1. 三阶段 alignment 与 DPO 的塌缩</h2>
      <p class="lead">
        ChatGPT (2022) 之后, 把通用 LM 变成可用助手的标准流程是
        <strong>SFT → Reward Model → PPO</strong>。DPO (2023) 给了一条捷径 —
        把后两步合成"基于偏好对的分类 loss", 跳过 RM 与 RL。
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
        和预训练唯一的代码差别只有一行: 把 prompt 区域的 labels 改成
        <code class="inline">-100</code>, 让 cross-entropy 跳过这些位置。
      </p>

      <div class="grid grid-2" style="gap: 16px;">
        <div class="card">
          <h3>labels 怎么构造 <span class="tag">关键</span></h3>
          <pre class="code">{{ sftLabels }}</pre>
          <p class="hint">
            为什么要 mask prompt? 因为 prompt 来自人 — 模型不该把"用户问题"也学成
            自己要生成的内容。只在 response 上算 CE = "教模型怎么回答", 而不是
            "教模型怎么提问"。
          </p>
        </div>

        <div class="card">
          <h3>SFTLoss 实际上等价于什么 <span class="tag">代码对照</span></h3>
          <pre class="code">{{ sftLoss }}</pre>
          <p class="hint">
            技术上和预训练的 <code class="inline">StandardLMLoss</code> 完全一样
            (都是带 -100 的 cross_entropy)。单独命名是为了:
            <strong>1)</strong> 教学叙事清晰; <strong>2)</strong> 留扩展位
            (NEFTune 噪声 / focal-style 加权); <strong>3)</strong> 与 DPOLoss 命名对仗。
            对应代码: <RepoLink path="llm_finetune/methods/sft.py:SFTLoss" label="llm_finetune/methods/sft.py:SFTLoss" tiny />
          </p>
        </div>
      </div>
    </section>

    <!-- ── 3. LoRA ─────────────────────────────────────────── -->
    <section class="section">
      <h2>3. LoRA · 给冻结的 W 加一对低秩补丁</h2>
      <p class="lead">
        论文 (Hu et al., 2021) 的关键观察: 微调引起的增量 ΔW <strong>本征秩很低</strong>,
        可以用一对 r 维矩阵表达。训练时只更新这对矩阵, 推理时合并回去, 零额外开销。
      </p>

      <div class="grid grid-2" style="gap: 16px;">
        <div class="card">
          <h3>数学形式 <span class="tag">3 行</span></h3>
          <pre class="code">{{ loraMath }}</pre>
          <p class="hint">
            <strong>无害启动</strong>: B 全零初始化 ⇒ 训练第 1 步 BA=0 ⇒
            forward 输出与原模型完全一致。这是 LoRA 最关键的设计:
            微调起点 = 预训练终点, 不会一上来就破坏已有能力。
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
            r 可调: 任务越难/数据越大 → r 越大。α 用 2r 或 r,
            <code class="inline">scale = α/r</code> 让你改 r 不必重调学习率。
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
            <strong>落盘只 ~MB</strong>
            <span class="muted"><code class="inline">get_lora_state_dict</code> 仅抽出
            A, B 两个矩阵 — 部署时基座共享, 适配器单独存。</span>
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
            <tr><td>全注入</td><td class="mono">attn 全部 + ffn 全部</td><td class="muted">类 QLoRA, 极致效果但参数翻倍</td></tr>
          </tbody>
        </table>
        <p class="hint">
          <code class="inline">apply_lora</code> 用 <code class="inline">named_modules</code>
          匹配最后一段属性名 (例如 "w_q"), 与具体层路径无关 —
          所以同一份代码能注入到 LLaMA / Mistral / 任何沿用同名属性的模型。
        </p>
      </div>
    </section>

    <!-- ── 4. DPO ──────────────────────────────────────────── -->
    <section class="section">
      <h2>4. DPO · 把偏好学习写成一行 logsigmoid</h2>
      <p class="lead">
        DPO 的洞见: KL 约束的策略提升问题有<strong>解析最优解</strong>, 把它代回原优化目标后,
        loss 退化成"chosen vs rejected 的 logit 差"的二分类。RM 和 PPO 都不需要了。
      </p>

      <div class="grid grid-2" style="gap: 16px;">
        <div class="card">
          <h3>核心公式 <span class="tag">Bradley-Terry</span></h3>
          <pre class="code">{{ dpoMath }}</pre>
          <p class="hint">
            <strong>π_θ</strong>: 被微调的 policy (起点 = SFT 终态);
            <strong>π_ref</strong>: 冻结的 reference (一般就是 SFT 终态的副本);
            <strong>β</strong>: KL 约束强度, 0.1~0.5。
            β 越大 → log_sigmoid 输入越陡 → 模型更保守贴近 ref。
          </p>
        </div>

        <div class="card">
          <h3>每步训练做什么 <span class="tag">4 次前向</span></h3>
          <pre class="code">{{ dpoStep }}</pre>
          <p class="hint">
            为什么 ref 用 <code class="inline">no_grad</code> + <code class="inline">eval()</code>?
            前者省激活显存, 后者关掉 dropout 让 log π_ref 是确定函数。
            否则随机性会污染偏好梯度。
            代码: <RepoLink path="llm_finetune/methods/dpo.py:DPOLoss" label="llm_finetune/methods/dpo.py:DPOLoss / DPOTrainer.train_step" tiny />
          </p>
        </div>
      </div>

      <div class="card" style="margin-top: 16px;">
        <h3>compute_sequence_logprobs · 序列级 log-prob 怎么算</h3>
        <pre class="code">{{ seqLogprob }}</pre>
        <p class="hint">
          为什么不直接用 <code class="inline">F.cross_entropy(reduction='sum')</code>?
          那个会把 batch 里所有样本的 NLL 求和, 失去逐样本粒度。
          DPO 需要 <strong>每个样本独立的 Σ log p</strong>, 才能 element-wise 做
          (chosen - rejected) 差分。
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
          一个常见的实战 pipeline: <strong>base → LoRA-SFT → 解锁基座 → DPO</strong>。
          先用 LoRA 在指令数据上廉价预热, 再用 DPO 在偏好对上做对齐。
        </p>
      </div>
    </section>

    <ChapterNav
      :prev="{ name: 'train-collectives-loop', label: '阶段 3.5 · 通信与 full_loop', hint: '先理解训练系统如何组合成完整主循环' }"
      :next="{ name: 'finetune-sft', label: '阶段 4.1 · SFT 数据与 loss', hint: '从数据和 labels mask 开始拆微调' }"
    />
  </div>
</template>

<script setup>
import ChapterIntro from '@/components/ChapterIntro.vue'
import ChapterNav from '@/components/ChapterNav.vue'
import EvolutionChain from '@/components/EvolutionChain.vue'
import RepoLink from '@/components/RepoLink.vue'

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

const sftLabels = `# 例:
input  = [<bos>] [Q1] [Q2] [Q3] [A1] [A2] [<eos>]
labels = [-100 ] [-100][-100][-100][A1] [A2] [<eos>]
         ↑ prompt 全部忽略           ↑ 只对 response 算 loss

# pad 也置 -100`

const sftLoss = `class SFTLoss(LossComputer):
    def compute(self, logits, labels, **kw):
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,        # ← prompt 这样被跳过
        )
        return {"total_loss": loss, "sft_loss": loss}`

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

const dpoStep = `# DPOTrainer.train_step
chosen, rejected = batch.chosen, batch.rejected

# 2 次 policy 前向 (要梯度)
p_chosen   = model(chosen.input_ids)
p_rejected = model(rejected.input_ids)

# 2 次 ref 前向 (no_grad + eval)
with torch.no_grad():
    r_chosen   = ref_model(chosen.input_ids)
    r_rejected = ref_model(rejected.input_ids)

loss = DPOLoss().compute(
    {"policy_chosen_logits": p_chosen,  ...},
    {"chosen_labels": ..., "rejected_labels": ...},
)["total_loss"]
loss.backward();  optimizer.step()`

const seqLogprob = `def compute_sequence_logprobs(logits, labels, ignore=-100):
    log_p_full = F.log_softmax(logits, dim=-1)            # [B, T, V]
    valid      = labels != ignore                         # [B, T]

    safe_labels = labels.masked_fill(~valid, 0)            # 让 gather 合法
    token_logp  = log_p_full.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)

    return (token_logp * valid.float()).sum(dim=-1)        # [B]  ← 逐样本求和`

const pillars = [
  { dim: '数据形态',     sft: '(instruction, response)',  lora: '(instruction, response)',  dpo: '(prompt, chosen, rejected)' },
  { dim: '可训练参数',   sft: '100% 模型',                lora: '0.4% 左右 (A, B)',         dpo: '100% policy, ref 冻结' },
  { dim: 'loss',         sft: '带 -100 mask 的 CE',       lora: '同 SFT (只是参数更少)',    dpo: '-log σ(β·logit_diff)' },
  { dim: '需要 ref?',    sft: '否',                       lora: '否',                       dpo: '是 (deepcopy + freeze + eval)' },
  { dim: '需要 RM?',     sft: '否',                       lora: '否',                       dpo: '否 (DPO 的核心收益)' },
  { dim: '一步前向次数', sft: '1',                        lora: '1',                        dpo: '4 (policy/ref × chosen/rejected)' },
  { dim: '部署成本',     sft: '一份完整权重',             lora: '~MB 适配器 + 共享基座',     dpo: '一份完整权重' },
]

const decisions = [
  { q: '只有 (问, 答) 数据, 显卡足够大?',           a: '全参 SFT — 上限最高' },
  { q: '只有 (问, 答) 数据, 显卡紧?',                a: 'LoRA SFT — 0.5% 参数, 适配器易切换' },
  { q: '已有 SFT 模型, 现在拿到 (chosen, rejected)?', a: 'DPO — 跳过 RM 与 PPO, 一行 loss' },
  { q: '想做风格 / 角色微调, 需要快速切换?',         a: 'LoRA — 多个适配器共享基座' },
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
