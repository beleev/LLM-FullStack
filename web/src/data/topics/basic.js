// 阶段 1 · llm_basic: 不新增章节, 只按字段覆盖 models.js 里的旧文字 (BPE 的定位、n_layer 循环、逐算子 gradcheck) 并补真源码。
export default {
  stage: 'basic',
  chapters: [],
  pages: {
    'basic-data': {
      tldr: 'prepare.py 把原始文本固化成 train.bin / val.bin / meta.npz; tokenizer.py 定义字符级 encode/decode 的可逆映射。bpe.py 是一个独立的 BPE 演示, 不参与训练流水线。',
      code: 'llm_basic/{prepare.py,tokenizer.py,bpe.py,input.txt,train.bin,val.bin,meta.npz}',
      points: [
        { title: '可复现数据', body: '原始文本只处理一次, 之后训练直接 mmap / np.fromfile 读二进制 ids, 避免每次运行重新构造 vocab。' },
        { title: '训练用字符级, BPE 单独演示', body: '训练 / 采样 / ckpt 全部基于字符级 tokenizer.py (vocab=65), 让注意力、loss、梯度成为主角。bpe.py 是只依赖 input.txt 的独立 demo: 从 256 个字节出发反复合并最高频相邻对 —— 上面的实验台演示的就是它。' },
        { title: 'batch 是连续片段', body: 'get_batch 从长 token 序列中切 [T] 片段, y 是 x 右移一位; 这就是 next-token prediction。' },
      ],
      sourceRows: [
        { concept: 'vocab 构造', code: 'prepare.py', takeaway: '把 unique chars 排序, 得到稳定 stoi/itos。' },
        { concept: '训练/验证切分', code: 'prepare.py', takeaway: '先切数据, 再保存 train.bin / val.bin, 避免验证集泄漏。' },
        { concept: 'batch 对齐', code: 'train.py:get_batch', takeaway: 'x = data[i:i+T], y = data[i+1:i+T+1]。' },
        { concept: 'BPE 训练 (独立 demo)', code: 'bpe.py:train_bpe', takeaway: '统计相邻对 → 合并最高频 → 重复 num_merges 次; 纯贪心, 没有梯度。' },
        { concept: 'BPE 编码', code: 'bpe.py:encode', takeaway: '按 “合并的先后顺序” 重放规则; 解码查表拼字节, 永远无损。' },
      ],
      source: ['llm_basic/bpe.py:train_bpe'],
    },
    'basic-forward': {
      tldr: 'Transformer forward = token/position embedding + n_layer 个 Pre-LN block (一个 for 循环) + final norm + lm_head。cache 保存反向需要的中间量。',
      points: [
        { title: '结构极简但完整', body: '默认 1 层、单头、ReLU MLP 是教学简化 (--n-layer 可加层); 残差、norm、causal attention、lm_head 都保留。' },
        { title: '形状先行', body: 'B/T/D/V 四个维度贯穿全章。读代码时先检查矩阵乘法两边形状, 再看数值细节。' },
        { title: 'cache 是手写 autograd tape', body: '每个 forward 返回反向需要的输入、权重、归一化统计或 softmax 概率; 多层时每层一份, 按顺序存进 blk_caches。' },
      ],
      sourceRows: [
        { concept: 'embedding', code: 'embedding_forward', takeaway: 'tok_emb[x] + pos_emb[:T] 得到 [B,T,D]。' },
        { concept: 'causal attention', code: 'attention_forward', takeaway: 'QK^T 加 causal mask 后 softmax, 再乘 V。' },
        { concept: '层循环', code: 'model.py:num_layers', takeaway: '层数从参数名 block_{i}_norm1_g 数出来, 旧的单层 ckpt 也能直接加载。' },
        { concept: '语言模型头', code: 'transformer_forward', takeaway: '最后投影到 vocab 维, logits 形状 [B,T,V]。' },
      ],
      snippet: `ids[B,T]
  → tok_emb + pos_emb        [B,T,D]
  for i in range(n_layer):               # 默认 1, --n-layer 可调
      → RMSNorm → Attention  [B,T,D]  + residual
      → RMSNorm → MLP        [B,T,D]  + residual
  → final RMSNorm
  → lm_head                  [B,T,V]`,
      source: ['llm_basic/model.py:transformer_forward'],
    },
    'basic-backward': {
      tldr: '每个 forward 都有配套 backward; 残差梯度相加, softmax/RMSNorm 有耦合项, embedding 要累加重复 token。gradcheck 先逐算子、再端到端 (n_layer = 1 和 2) 验证。',
      points: [
        { title: '反向按图倒走', body: '从 cross-entropy 给出的 dlogits = (p − onehot)/N 开始, 按 forward 的逆序逐层传播; 多层就是 for i in reversed(range(n_layer))。' },
        { title: '重复索引要累加', body: 'embedding 的同一个 token 可能出现多次, 梯度必须加到同一行参数上 (np.add.at)。check_ops 故意用含重复 id 的输入来考这一点。' },
        { title: 'gradcheck 分两级', body: '逐算子: L = Σ(out ⊙ R), 对每个输入的每个元素做中心差分, 哪个算子错一目了然; 端到端: 整个模型 + CE 抽样检查。ε = 1e-5、float64 —— 原因见上面的 ε 实验台。' },
      ],
      sourceRows: [
        { concept: 'embedding backward', code: 'np.add.at(dW, ids, dout)', takeaway: '处理重复 token id 的梯度累加。' },
        { concept: 'softmax backward', code: 'attention_backward', takeaway: 'ds = a * (da - sum(a * da))。' },
        { concept: '数值梯度', code: 'gradcheck.py:numeric_grad', takeaway: '(f(w+ε) − f(w−ε)) / 2ε, 截断误差 O(ε²)。' },
        { concept: '逐算子检查', code: 'gradcheck.py:check_op', takeaway: '随机张量 R 充当 dout: L = Σ(out ⊙ R) ⇒ dL/dout = R, 任何算子都能单独验。' },
        { concept: '多层反向', code: 'model.py:transformer_backward', takeaway: 'reversed(range(n_layer)) 逐层回传; tok_emb 和 pos_emb 收到同一个 dh。' },
      ],
      snippet: `loss, dlogits = cross_entropy_forward_backward(logits, y)
grads = transformer_backward(dlogits, cache)   # 内部: for i in reversed(range(n_layer))

# 1) 逐算子: 用随机 R 当 dout
out, cache = op_forward(x);  analytic = op_backward(R, cache)
numeric = numeric_grad(lambda: (op_forward(x)[0] * R).sum(), x)   # 中心差分, eps=1e-5
assert norm(analytic - numeric) / max(norm(analytic), norm(numeric)) < 1e-6

# 2) 端到端: n_layer = 1, 2 各一遍, 每个参数抽样若干位置
assert abs(g_a - g_n) <= atol + rtol * max(abs(g_a), abs(g_n))`,
      source: ['llm_basic/gradcheck.py:numeric_grad', 'llm_basic/gradcheck.py:check_op'],
    },
  },
}
