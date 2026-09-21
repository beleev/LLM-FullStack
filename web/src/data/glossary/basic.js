// 阶段 1 · llm_basic 术语。number 取自代码里的公式或实验台里真算出来的数。
const t = (term, aka, oneliner, number, route) => ({ term, aka, stage: 'basic', oneliner, number, route })

export default [
  t('字符级 tokenizer', 'char-level', '语料里出现过的每个字符一个 id; 训练流水线实际用的就是它', 'vocab = 65', 'basic-data'),
  t('BPE', 'Byte Pair Encoding', '从 256 个字节出发, 反复合并最高频相邻对; 本仓库里是独立 demo', '词表 = 256 + 合并次数', 'basic-data'),
  t('next-token prediction', '下一词预测', 'y 是 x 右移一位; 一条长 T 的序列提供 T 个训练样本', 'y = data[i+1 : i+T+1]', 'basic-data'),
  t('cache', '手写 autograd tape', 'forward 顺手存下反向要用的中间量, backward 按逆序取用', 'forward → (out, cache)', 'basic-forward'),
  t('Pre-LN 残差', 'pre-norm residual', 'x + f(norm(x)): 主干是纯加法, 梯度直通到底', 'd(x+f)/dx = 1 + f′', 'basic-forward'),
  t('causal mask', '因果掩码', '位置 i 只能看 ≤ i 的位置, 训练时 T 个位置才能并行', '上三角填 −inf', 'basic-forward'),
  t('n_layer 循环', '堆层', '每个 block 输入输出同形, 加层只是一个 for; 层数从参数名数出来', '默认 1, --n-layer 可调', 'basic-forward'),
  t('链式法则', 'chain rule', '本节点梯度 = 上游梯度 × 本地导数, 从 loss 的 1 出发逐层倒推', '∂L/∂x = Wᵀ·∂L/∂z', 'basic-backward'),
  t('softmax+CE 梯度', 'p − onehot', '两者合并后 1/p[y] 被约掉, 又简单又数值稳定', 'dlogits = (p − onehot)/N', 'basic-backward'),
  t('softmax 雅可比', 'Jacobian', '每个输出依赖所有输入, 本地导数是矩阵, 这就是 “耦合项”', 'J = diag(p) − ppᵀ', 'basic-backward'),
  t('np.add.at', '重复索引累加', '同一 token 在 batch 里出现多次, embedding 梯度必须累加', 'dW[ids] += 会丢梯度', 'basic-backward'),
  t('gradcheck', '数值梯度检查', '用中心差分核对手写反向: 先逐算子全元素, 再端到端抽样', '逐算子 rel err ≈ 1e-10', 'basic-backward'),
  t('中心差分', 'central difference', '两侧各挪 ε 再相减; 误差 = ε² 截断 + δ/ε 舍入, 呈 U 形', 'float64 谷底 ε ≈ 1e-5', 'basic-backward'),
  t('Adam', '自适应矩估计', '一阶矩定方向, 二阶矩按坐标归一化步长: 每个参数自己的学习率', '每步位移 ≈ lr', 'basic-optim-sample'),
  t('偏差修正', 'bias correction', 'm、v 从 0 起步被拖小, 除以 1 − βᵗ 补回来', 'm̂ = m / (1 − β₁ᵗ)', 'basic-optim-sample'),
  t('条件数', 'condition number κ', '最陡与最平方向的曲率比; SGD 的 lr 被最陡方向卡死', 'SGD 稳定要求 lr < 2/κ', 'basic-optim-sample'),
  t('temperature', '采样温度', 'softmax(z/T): T<1 更尖更确定, T>1 更平更随机', 'T → 0 即贪心', 'basic-optim-sample'),
  t('top-k 采样', 'top-k sampling', '只在概率最高的 k 个 token 里采样, 砍掉长尾里的胡话', 'k = 1 即贪心', 'basic-optim-sample'),
]
