// 阶段 1 · llm_basic 自测题: 每章 3 题以上, 考"为什么这么写", 错误选项都是常见误解
export default {
  basic: [
    {
      q: '每个算子都写成 xxx_forward / xxx_backward 一对, forward 多返回的那个 cache 是干什么的?',
      options: ['加速下一次 forward', '保存优化器状态', '装着反向要用的中间量, 相当于手写的 autograd tape', '推理时的 KV cache'],
      answer: 2,
      why: '反向要用到前向算过的输入、softmax 概率、归一化统计。PyTorch 靠 autograd 偷偷记下来, 这里必须自己装进 cache, 再按逆序取用。',
    },
    {
      q: '这个阶段为什么全程用 float64, 而不是训练常用的 float32 / bf16?',
      options: ['gradcheck 的有限差分需要足够精度, float32 下误差谷底过不了 1e-6 的线', 'float64 训练出的模型更好', 'numpy 不支持 float32', '为了省显存'],
      answer: 0,
      why: '中心差分的总误差 ≈ ε² + δ/ε。float32 的 δ≈6e-8, 最低误差只到 1e-5 量级, 分不清 “实现正确” 和 “有个小 bug”。',
    },
    {
      q: '训练一步的正确顺序是?',
      options: ['backward → forward → 更新', 'forward → Adam 更新 → backward', '采样 → forward → backward', 'forward 得 loss → backward 得梯度 → Adam 更新参数'],
      answer: 3,
      why: '梯度是 loss 对参数的导数, 必须先有 loss; 优化器只消费梯度, 不关心它是怎么算出来的。',
    },
  ],
  'basic-data': [
    {
      q: '训练流水线实际用的是哪种 tokenizer? bpe.py 是什么角色?',
      options: ['训练用 BPE, tokenizer.py 已废弃', '训练用字符级 tokenizer.py (vocab=65); bpe.py 是只依赖 input.txt 的独立演示', '两者混用', 'BPE 已从仓库删除'],
      answer: 1,
      why: 'prepare / train / sample 和自带的 ckpt.npz 都基于 65 个字符的词表, 为的是让注意力和梯度当主角; bpe.py 单独演示 “反复合并最高频相邻对”。',
    },
    {
      q: '字符级 tokenizer 相比 BPE, 最主要的代价是什么?',
      options: ['无法无损解码', '同一段文本的 token 序列长得多, 而注意力是 O(T²)', '词表太大放不进内存', '不能处理空格'],
      answer: 1,
      why: 'bpe.py 的实测: 同一句话字符级要 60 个 token, 300 次合并后的 BPE 只要 24 个。序列短了, 模型也不用自己学拼写。',
    },
    {
      q: 'get_batch 里 y 为什么是 x 右移一位?',
      options: ['为了数据增强', '为了对齐 position embedding', '为了避免验证集泄漏', '因为训练目标是 next-token prediction: 位置 i 的标签就是第 i+1 个 token'],
      answer: 3,
      why: '因果 mask 保证位置 i 只看得到 ≤ i 的 token, 所以一条长度 T 的序列一次前向就提供了 T 个训练样本。',
    },
  ],
  'basic-forward': [
    {
      q: '把 --n-layer 从 1 改成 2, transformer_forward 要改哪里?',
      options: ['什么都不用改: block_forward 在 for 循环里按 block_{i}_ 前缀取参数, 每层的 cache 依次进列表', '要再手写一份第二层的 forward', '要改 lm_head 的形状', '要改 position embedding'],
      answer: 0,
      why: '每个 block 进去出来都是 [B,T,D], 残差流形状不变, 所以堆层只是一个循环; 层数甚至是从参数名 block_{i}_norm1_g 数出来的。',
    },
    {
      q: 'attention_forward 的 cache 里存的是 softmax 之后的 attn, 而不是 softmax 之前的 scores。为什么?',
      options: ['attn 更省内存', 'scores 里有 −inf, 存不下来', 'softmax 的反向 ds = a ⊙ (da − Σ a·da) 只用到输出 a, 用不上输入', '为了方便打印调试'],
      answer: 2,
      why: 'forward 存什么, 完全由 backward 要什么决定。这顺带解释了为什么被 mask 的位置不用再 mask 一次: 因子 a = 0 让梯度自动归零。',
    },
    {
      q: 'logits 的形状是 [B,T,V]。训练时 T 个位置的 loss 为什么能一次算完?',
      options: ['因为 batch 够大', '因为用了 RMSNorm', '因为 causal mask 让位置 i 看不到 i 之后的 token, 每个位置都是一道独立合法的预测题', '因为 lm_head 与 embedding 共享权重'],
      answer: 2,
      why: '没有 causal mask, 位置 i 会直接 “看到答案”; 有了它, 一次前向等于并行做了 T 道 next-token 题。',
    },
    {
      q: '默认配置 V=65, D=64, H=128, T=64, n_layer=1 一共 45,568 个参数。把 T 从 64 拖到 256, 参数和激活分别怎么变?',
      options: ['两者都按 T 线性增长', '参数只有 pos_emb 跟着长 (+27%), 激活涨几倍, 因为 scores/mask/attn 是 [B,T,T]', '参数按 T² 增长, 激活不变', '两者都不变'],
      answer: 1,
      why: '参数表里根本没有 T², 只有 pos_emb 的行数是 T_max。长上下文的账记在激活和注意力的 T² 上 —— 上面的形状流水线里可以直接拖出来看。',
    },
  ],
  'basic-backward': [
    {
      q: 'softmax + cross-entropy 合并后, 对 logits 的梯度是?',
      options: ['p − onehot(y) (再除以位置数 N)', '−1/p[y]', 'diag(p) − ppᵀ', 'onehot(y) − log p'],
      answer: 0,
      why: 'CE 的本地导数 −1/p[y] 和 softmax 的雅可比 diag(p)−ppᵀ 一相乘, 1/p[y] 正好被约掉, 结果既简单又数值稳定 —— 所以代码把两者合成一个函数。',
    },
    {
      q: 'gradcheck 的中心差分里, ε 为什么不是越小越好?',
      options: ['ε 小了计算慢', 'ε 小了截断误差变大', 'numpy 不支持太小的数', 'ε 太小时 f(w+ε) − f(w−ε) 被浮点舍入误差淹没, 误差按 δ/ε 上升'],
      answer: 3,
      why: '总误差 ≈ ε² (截断) + δ/ε (舍入), 是一条 U 形曲线; float64 下谷底在 1e-5 附近, 所以 gradcheck.py 取 eps=1e-5。',
    },
    {
      q: 'embedding_backward 为什么必须用 np.add.at(dW, ids, dout), 不能写 dW[ids] += dout?',
      options: ['add.at 更快', '同一个 token id 在 batch 里出现多次时, 花式索引的 += 只保留最后一次, 梯度会丢', '为了支持 float64', '为了处理 padding'],
      answer: 1,
      why: '这类 bug 不报错, loss 照样下降, 只是高频 token 的梯度被严重低估 —— 逐算子 gradcheck 故意用含重复 id 的输入 [[0,1,1],[5,1,0]] 来抓它。',
    },
  ],
  'basic-optim-sample': [
    {
      q: '在病态损失上 (某个方向比另一个陡 κ 倍), 纯 SGD 的学习率被什么卡住?',
      options: ['被最平的方向: lr 必须大于 1/κ', '被最陡的方向: lr 必须小于 2/κ, 于是平缓方向每步几乎不动', '被 batch size', '被参数个数'],
      answer: 1,
      why: '陡方向每步乘 (1 − lr·κ), 绝对值超过 1 就发散; 满足它之后, 平方向每步只缩 (1 − lr), 慢得要命。',
    },
    {
      q: 'Adam 用 m̂/√v̂ 更新, 最关键的性质是?',
      options: ['保证收敛到全局最优', '不需要学习率', '梯度不需要反向传播', '每个参数的步长约等于 lr, 与该方向梯度的绝对大小基本无关'],
      answer: 3,
      why: '除以 √v̂ 等于给每个坐标单独归一化, 相当于每个参数有自己的学习率; 偏差修正则补回 m、v 从 0 起步被拖小的那部分。',
    },
    {
      q: 'README 实测: 2 层 + AdamW + clip + cosine 跑 2000 步, val_loss 2.045, 比默认 1 层的 1.981 还差。最该得出的结论是?',
      options: ['AdamW 和 cosine 是错的实现', '这些开关不是 “打开就更好”: 2000 步太短, cosine 过早把 lr 压到 3e-5, 多出来那层还没学起来', '2 层模型永远不如 1 层', '应该把 weight_decay 调到 1.0'],
      answer: 1,
      why: '这些开关要学的是怎么实现、各自解决什么问题。规模、步数、调度必须配套, 否则加容量反而拖慢收敛。',
    },
    {
      q: '为什么训练时所有位置并行算, 生成时却只能一个 token 一个 token 来?',
      options: ['训练时完整答案已知 (teacher forcing); 生成时第 t+1 个 token 依赖刚采样出来的第 t 个', '生成时没有 GPU', '生成时 batch 只能是 1', '因为 temperature 采样不可并行'],
      answer: 0,
      why: '这条依赖链是算法固有的, 省不掉; 能省的只是重复计算 —— 这就是阶段 5 的 KV cache。temperature / top-k 只是对 logits 的后处理。',
    },
  ],
}
