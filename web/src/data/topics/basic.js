// 阶段 1 · llm_basic: 四个章节的完整页面定义 (不新增章节, 覆盖 models.js 里的同名页)。
export default {
  stage: 'basic',
  chapters: [],
  pages: {
    'basic-data': {
      widgets: ['BpeLab'],
      title: '数据与 tokenizer · 把文本变成可训练张量',
      subtitle: '读完你能说清: 1,115,394 个字符怎么变成 train.bin 里的 1,003,854 个 token, 以及每个 batch 是从哪儿切出来的。',
      tldr: 'prepare.py 跑一次, 把 input.txt 固化成 train.bin / val.bin / meta.npz; 之后训练只 memmap 读二进制, 词表再也不会变。',
      question: '既然 BPE 更省 token, 为什么这里训练用的是最笨的字符级 tokenizer?',
      code: 'llm_basic/{prepare.py,tokenizer.py,bpe.py,input.txt,train.bin,val.bin,meta.npz}',
      points: [
        {
          title: '词表只建一次, 之后谁都不许改',
          body: '这里的 “训练 tokenizer” 就是一行 sorted(set(text)): 65 个字符按字典序排好, id 就定死了。排序是为了换台机器也得到同一张表。词表存进 meta.npz, train.py 和 sample.py 共用 —— 否则今天存的 ckpt 明天解码出来是乱码。',
        },
        {
          key: true,
          title: '训练走字符级, BPE 是旁边的独立演示',
          body: '训练、采样、自带的 ckpt.npz 全部基于 tokenizer.py 的 65 个字符。bpe.py 只读 input.txt, 不接训练流水线。这么定是为了让注意力和梯度当主角, 不是因为 BPE 不重要: 跑一下 python bpe.py 就看得见, 300 次合并后同一句话字符级要 60 个 token, BPE 只要 24 个, 压缩率 2.50 字符/token。上面的实验台把这个合并过程一步步摊开了。',
        },
        {
          title: 'y 就是 x 右移一位',
          body: 'get_batch 随机挑 32 个起点, 各切 64 个 token 当 x, 整体往后挪一格当 y。于是一条长度 64 的序列一次就提供 64 道 next-token 题 —— 前提是因果 mask 挡住了答案。',
        },
      ],
      links: [
        { from: 'input.txt', to: 'prepare.py', body: '1,115,394 个字符 → 65 个 id → 按顺序 90/10 切分。不随机抽, 随机抽会让 val 的句子在 train 里找到上下文。' },
        { from: 'train.bin', to: 'train.py:get_batch', body: 'vocab 65 < 256, 一个 token 一字节, 文件大小就是 token 数; memmap 进来, 切到哪段读哪段。' },
        { from: 'meta.npz', to: 'sample.py', body: 'itos 把生成的 id 还原成字符。词表跟着 ckpt 一起走, 是训练和推理之间的契约。' },
      ],
      sourceRows: [
        { concept: 'vocab 构造', code: 'prepare.py', takeaway: 'chars = sorted(set(text)); stoi 是它的反查表, 整个 “tokenizer 训练” 就这两行。' },
        { concept: '训练/验证切分', code: 'prepare.py', takeaway: '先按位置切 90/10 再落盘, val 才是模型完全没见过的连续文本。' },
        { concept: 'uint8 存储', code: 'prepare.py', takeaway: 'assert vocab_size < 256 —— 越过这条线就得换 uint16, 文件直接翻倍。' },
        { concept: 'batch 对齐', code: 'train.py:get_batch', takeaway: 'x = data[i:i+T], y = data[i+1:i+T+1]; 起点最多取到 len−T−1, 留一位给 y 的末尾。' },
        { concept: 'BPE 训练 (独立 demo)', code: 'bpe.py:train_bpe', takeaway: '统计相邻对 → 合并最高频的那对 → 重复 num_merges 次。纯贪心, 一个梯度都没有。' },
        { concept: 'BPE 编码', code: 'bpe.py:encode', takeaway: '按训练时的合并先后顺序重放规则; 顺序错了, 同一个词会切成不同 token。' },
      ],
      snippetTitle: 'next-token 数据形态',
      snippet: `text:  "hello"
ids:   [h, e, l, l, o]

x:     [h, e, l, l]
y:     [e, l, l, o]

# 位置 i 看到 x[:i+1], 要猜出 y[i] —— 这就是 next-token prediction`,
      source: ['llm_basic/bpe.py:train_bpe'],
      run: 'cd llm_basic && python prepare.py',
    },

    'basic-forward': {
      title: 'forward 与形状流 · 从 ids 到 logits',
      subtitle: '读完你能报出每一步的形状, 并说出每个 forward 往 cache 里塞了什么、为什么非塞不可。',
      tldr: 'ids [B,T] 一路走成 logits [B,T,V]: embedding → n_layer 个 Pre-LN block → final norm → lm_head, 每一步顺手把反向要用的中间量装进 cache 带回来。',
      question: '为什么 forward 不能只返回 logits, 还要拖着一长串 cache?',
      code: 'llm_basic/model.py:{embedding_forward,rmsnorm_forward,attention_forward,block_forward,transformer_forward}',
      points: [
        {
          key: true,
          title: 'cache 是手写的 autograd tape',
          body: '反向要用到前向算过的中间值 —— RMSNorm 的 rms、softmax 的输出 attn、每个 linear 的输入 x。torch 靠 autograd 在背后偷偷记下来, 这里每个 forward 自己打包成一个 tuple 返回。哪个值没进 cache, 对应的 backward 就写不出来; 反过来, 反向用不上的东西一律不存 (比如 softmax 存的是输出 attn, 不是输入 scores)。',
        },
        {
          title: '形状只有四个字母',
          body: 'B=batch, T=序列长, D=模型维, V=词表。主干上所有张量都是 [B,T,D], 只有两处例外: 注意力里的 [B,T,T], 和最后的 logits [B,T,V]。读代码先把矩阵乘法两边的形状对上, 数值细节留到第二遍。',
        },
        {
          title: '加层只是一个 for 循环',
          body: '每个 block 进去 [B,T,D], 出来还是 [B,T,D], 所以堆层不需要任何新推导: transformer_forward 里就是 for i in range(num_layers(W))。层数甚至不是配置项, 而是从参数名 block_{i}_norm1_g 数出来的 —— 没有 n_layer 字段的旧 ckpt.npz 因此照样加载。',
        },
      ],
      links: [
        { from: 'learned pos_emb', to: 'RoPE', body: '这里位置是一张 [T_max, D] 的表直接加上去, T 被 T_max=64 卡死; 阶段 2 改成旋转 Q/K, 不占参数也不卡长度。' },
        { from: '单头 attention', to: 'MHA / GQA / MLA', body: '这里 head_dim 就等于 D。分头只是把 D 切成几段各算各的, 形状账一模一样。' },
        { from: 'cache 全存', to: 'activation checkpoint', body: 'cache 就是显存大头。大模型显存不够时会丢掉一部分 cache, 反向时重算 —— 拿时间换空间。' },
      ],
      sourceRows: [
        { concept: 'embedding', code: 'model.py:embedding_forward', takeaway: 'W[ids] 查表; cache 只存 ids 和 W 的形状, 不存 W —— 反向只需要知道取了哪几行。' },
        { concept: 'RMSNorm', code: 'model.py:rmsnorm_forward', takeaway: 'cache 是 (x, g, rms) 三件套, 反向的耦合项三样全要。' },
        { concept: '缩放点积', code: 'model.py:attention_forward', takeaway: 'scale = 1/√D; 不缩放的话点积方差正比于 D, softmax 直接饱和。' },
        { concept: 'causal mask', code: 'model.py:causal_mask', takeaway: '上三角 (j > i) 填 −inf, softmax 后权重为 0。一次前向因此等于并行做了 T 道题。' },
        { concept: '残差', code: 'model.py:block_forward', takeaway: 'h = x + a 没有 cache —— 加法的反向是把 dout 原样发给两条路。' },
        { concept: '层循环', code: 'model.py:num_layers', takeaway: '数参数名里以 _norm1_g 结尾的有几个, 就是几层。' },
        { concept: '语言模型头', code: 'model.py:transformer_forward', takeaway: '[B,T,D] @ [D,V] → [B,T,V]; V 一大, 这一步的参数和激活都会压过整个主干。' },
      ],
      snippetTitle: '主干形状',
      snippet: `ids[B,T]
  → tok_emb + pos_emb        [B,T,D]
  for i in range(n_layer):               # 默认 1, --n-layer 可调
      → RMSNorm → Attention  [B,T,D]  + residual
      → RMSNorm → MLP        [B,T,D]  + residual
  → final RMSNorm            [B,T,D]
  → lm_head                  [B,T,V]

# 默认 V=65, D=64, H=128, T=64, n_layer=1 → 45,568 个参数`,
      source: ['llm_basic/model.py:transformer_forward'],
      run: 'cd llm_basic && python train.py --max-iters 300 --eval-interval 100 --out /tmp/c.npz',
    },

    'basic-backward': {
      title: '手写 backward · cache、链式法则与 gradcheck',
      subtitle: '读完你能指着任意一个 *_backward, 说出它从 cache 里取了什么、为什么必须取, 以及写错了怎么被抓出来。',
      tldr: '反向就是把 forward 倒着走一遍: 从 dlogits = (p − onehot)/N 出发, 每个 backward 从 cache 取中间量、算出 dx 往前递。写错不会报错, 所以 gradcheck 必须跑。',
      question: '把 np.add.at 写成 +=, loss 照样往下掉 —— 那我怎么知道反向写错了?',
      code: 'llm_basic/model.py:*_backward · llm_basic/gradcheck.py',
      points: [
        {
          key: true,
          title: '手写反向的 bug 不会报错, 只会让模型悄悄变笨',
          body: '转置写反、sum 错维度、softmax 漏一项, 程序都不会崩, loss 也照样下降, 只是降得慢一点。所以要有一个 “不可能写错” 的参照: 中心差分 (f(w+ε) − f(w−ε)) / 2ε。把 rmsnorm_backward 的耦合项故意删掉, 逐算子检查立刻报 dx 相对误差 3.11e-01 —— 这就是 gradcheck.py 存在的全部理由。',
        },
        {
          title: '一个张量被用了几次, 梯度就是几路之和',
          body: 'x 同时喂给 Q/K/V 三条支路, 反向要 dx_q + dx_k + dx_v; 残差 out = x + f(x), 反向是捷径那份加上穿过子层回来的那份; 同一个 token id 在 batch 里出现多次, embedding 梯度必须 np.add.at 累加, 写成 dW[ids] += dout 只会加最后一次。check_ops 故意用含重复 id 的输入 [[0,1,1],[5,1,0]] 来考这一点。',
        },
        {
          title: 'gradcheck 分两级, 定位方式完全不同',
          body: '逐算子: 拿一个随机张量 R 当 dout (令 L = Σ(out ⊙ R), 于是 dL/dout 正好是 R), 对每个输入的每个元素做差分, 7 个算子的相对误差落在 1e-11 ~ 2e-10, 哪个算子错一目了然。端到端: 整模型 + CE, n_layer = 1 和 2 各跑一遍, 每个参数抽 8 个位置, 判定用 atol + rtol·max(|g_a|,|g_n|) —— 抽样常落在真实梯度≈0 的位置, 纯相对误差会误报。',
        },
      ],
      links: [
        { from: 'softmax backward', to: '所有注意力变体', body: 'ds = a ⊙ (da − Σ a·da) 这一行, 从 MHA 到 FlashAttention 每换一次实现都要重新写对一次。' },
        { from: 'gradcheck', to: 'llm_train 的等价性断言', body: '先算一个 dense baseline, 再断言并行版本与它一致 —— 和这里拿数值梯度当参照是同一个套路。' },
        { from: 'RMSNorm backward', to: '深层训练稳定性', body: '归一化让一行里的元素梯度互相牵连; 这个耦合项正是它能压住深层 Transformer 数值尺度的原因。' },
      ],
      sourceRows: [
        { concept: 'CE 的梯度', code: 'model.py:cross_entropy_forward_backward', takeaway: 'dlogits = (p − onehot)/N; 每行加起来恰好为 0, gradcheck.py 里有这条断言。' },
        { concept: 'embedding backward', code: 'np.add.at(dW, ids, dout)', takeaway: '花式索引的 += 遇到重复 id 只加一次, 必须用 add.at。' },
        { concept: 'softmax backward', code: 'model.py:attention_backward', takeaway: 'ds = a * (da − Σ a·da); 被 mask 的位置 a = 0, 带一个因子 a 就自动归零, 不用再 mask 一次。' },
        { concept: 'RMSNorm 耦合项', code: 'model.py:rmsnorm_backward', takeaway: 'dx = c/rms − x·s/(D·rms³); 第二项来自 rms 依赖整行 x。' },
        { concept: '数值梯度', code: 'gradcheck.py:numeric_grad', takeaway: '(f(w+ε) − f(w−ε))/2ε, 截断误差 O(ε²); 就地改、算完还原。' },
        { concept: '逐算子检查', code: 'gradcheck.py:check_op', takeaway: '随机张量 R 当 dout, 任何算子都能单独验, 不用搭整个模型。' },
        { concept: '多层反向', code: 'model.py:transformer_backward', takeaway: 'reversed(range(n_layer)) 逐层回传; tok_emb 和 pos_emb 最后收到同一个 dh。' },
      ],
      snippetTitle: '反向入口 + 两级 gradcheck',
      snippet: `loss, dlogits = cross_entropy_forward_backward(logits, y)
grads = transformer_backward(dlogits, cache)   # 内部: for i in reversed(range(n_layer))

# 1) 逐算子: 用随机 R 当 dout, 全元素差分
out, cache = op_forward(x);  analytic = op_backward(R, cache)
numeric = numeric_grad(lambda: (op_forward(x)[0] * R).sum(), x)   # 中心差分, eps=1e-5
assert norm(analytic - numeric) / max(norm(analytic), norm(numeric)) < 1e-6

# 2) 端到端: n_layer = 1, 2 各一遍, 每个参数抽 8 个位置
assert abs(g_a - g_n) <= atol + rtol * max(abs(g_a), abs(g_n))    # atol=1e-7, rtol=1e-4`,
      source: ['llm_basic/gradcheck.py:numeric_grad', 'llm_basic/gradcheck.py:check_op'],
      run: 'cd llm_basic && python gradcheck.py',
    },

    'basic-optim-sample': {
      title: 'Adam 与采样 · 训练后如何生成文本',
      subtitle: '读完你能说清 Adam 每步到底走多远, 以及生成为什么只能一个 token 一个 token 往外挤。',
      tldr: 'Adam 用一阶矩定方向、二阶矩按坐标归一化步长, 每步位移大约就是 lr; 采样时不再算 loss, 只取最后一个位置的 logits, 抽一个 token 接上去再来一遍。',
      question: '训练时 64 个位置一次算完, 生成时为什么只能一个一个来?',
      code: 'llm_basic/{optim.py,sample.py,train.py}',
      points: [
        {
          key: true,
          title: '除以 √v̂ = 每个参数有自己的学习率',
          body: 'Transformer 各层的梯度尺度差几个数量级, SGD 的单一 lr 被最陡的方向卡死 (超过 2/κ 就发散), 平缓方向于是几乎不动。Adam 把更新量归一成 m̂/√v̂, 每步位移约等于 lr, 跟这个方向的坡度基本无关。m、v 从 0 起步前几步被低估, 所以除以 1 − βᵗ 做偏置修正: 第 1 步正好得到 sign(g), 每个元素恰好移动 lr。',
        },
        {
          title: '训练并行, 生成串行, 这条依赖链省不掉',
          body: '训练时完整答案 y 已知 (teacher forcing), 因果 mask 让 T 个位置互不偷看, 一次前向算完所有 CE。生成时第 t+1 个 token 依赖刚抽出来的第 t 个, 没法并行。能省的只是重复计算: sample.py 每生成 1 个 token 就把整段重算一遍, 按它的注释是整体 O(T³); 阶段 5 的 KV cache 干掉的就是这部分重复。',
        },
        {
          title: '三个开关默认全关, 开了也不一定更好',
          body: 'optim.py 里 AdamW 解耦衰减、全局范数裁剪、warmup+cosine 默认都不启用, 跑的就是原始 Adam + 恒定 lr。README 里有实测: 2 层 + AdamW(0.1) + clip(1.0) + cosine, 78,656 参数跑 2000 步得到 val_loss 2.045, 比默认 1 层的 1.981 还差 —— 步数这么少时 cosine 过早把 lr 压到 3e-5, 多出来那层还没学起来。这里要学的是这些开关怎么实现, 不是开了就赢。',
        },
      ],
      links: [
        { from: 'Adam', to: 'llm_train 的稳定性工具', body: 'warmup、cosine、grad clip、混合精度的 loss scaling, 都是在这个更新式外面加保护层。' },
        { from: 'sample.py', to: 'llm_infer 的采样', body: 'temperature 和 top-k 只是对 logits 的后处理, 不碰参数; 推理章节接着加 top-p、min-p、重复惩罚。' },
        { from: '每步重算整段前向', to: 'KV cache', body: '第 t 步和第 t+1 步的前 t 个 K/V 完全一样, 缓存下来就不用重算。' },
      ],
      sourceRows: [
        { concept: '一阶矩', code: 'optim.py:adam_step', takeaway: 'm = β₁m + (1−β₁)g; β₁=0.9 相当于对最近约 1/(1−β₁) = 10 步做滑动平均。' },
        { concept: '二阶矩', code: 'optim.py:adam_step', takeaway: 'v = β₂v + (1−β₂)g²; 除以 √v̂ 后每个坐标的步长都被归一化到 lr 量级。' },
        { concept: '偏置修正', code: 'optim.py:adam_step', takeaway: 'm/(1−β₁ᵗ)、v/(1−β₂ᵗ); optim.py 的自检断言第 1 步更新恰好是 lr·sign(g)。' },
        { concept: 'AdamW 与 L2 的区别', code: 'optim.py:adam_step', takeaway: 'λW 加在 update 上而不是加进 g, 因此不被 1/√v 缩放; 1 维的 gain/bias 不衰减。' },
        { concept: '全局范数裁剪', code: 'optim.py:clip_grad_norm', takeaway: '所有参数拼成一个长向量算 ‖g‖, 整体同乘 min(1, max_norm/‖g‖) —— 逐参数裁会改变梯度方向。' },
        { concept: 'warmup + cosine', code: 'optim.py:cosine_lr', takeaway: '前 warmup 步线性升到 max_lr, 之后半个余弦降到 min_lr。' },
        { concept: '自回归循环', code: 'sample.py:generate', takeaway: '取 logits[0, -1] → 除 temperature → top-k 过滤 → 抽样 → append → 再 forward。' },
        { concept: '滑动窗口', code: 'sample.py:generate', takeaway: 'pos_emb 只有 T_max = 64 行, 上下文超长时只保留最后 64 个 token。' },
      ],
      snippetTitle: 'Adam 一步 + 采样一轮',
      snippet: `# optim.py: adam_step
m = beta1 * m + (1 - beta1) * g
v = beta2 * v + (1 - beta2) * g * g
update = (m / (1 - beta1**t)) / (sqrt(v / (1 - beta2**t)) + eps)
if weight_decay and w.ndim >= 2:       # AdamW: 加在 update 上, 不进 m/v
    update = update + weight_decay * w
W = w - lr * update

# sample.py: 每生成一个 token, 整段重来一遍
for _ in range(max_new_tokens):
    ctx    = ids[-T_max:]                          # pos_emb 只有 T_max 行
    logits = transformer_forward(W, ctx)[0, -1]    # 只要最后一个位置
    probs  = softmax(top_k_filter(logits / temperature, k))
    ids.append(sample(probs))`,
      source: ['llm_basic/optim.py:adam_step', 'llm_basic/sample.py:generate'],
      run: 'cd llm_basic && python sample.py "ROMEO:" --max-new 120 --temperature 0.8',
    },
  },
}
