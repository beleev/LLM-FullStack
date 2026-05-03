"""
训练配置模块
================

定义训练超参数的不可变数据类 (frozen dataclass)。

为什么用 frozen dataclass？
- 配置一旦构造完成就不应被修改，避免训练中途意外篡改导致难以复现的 bug
- 不可变对象天然线程安全，便于多 worker 共享
- 与 "代码风格中的 immutability 原则" 保持一致

设计要点：
- 学习率调度采用 "linear warmup + cosine decay"，这是 Transformer 训练的事实标准：
  warmup 缓解 Adam 在初期对二阶动量估计不准带来的发散，
  cosine 衰减末期接近 0，便于模型收敛到平坦极小值。
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    """
    训练配置 (不可变)

    各字段含义与设计动机：
        learning_rate:    学习率。3e-4 是 Karpathy 戏称的 "Adam 默认最佳学习率"，
                          对中小规模 Transformer 通常无需精调。
        weight_decay:     AdamW 的解耦权重衰减系数。注意：理想情况下应只对
                          矩阵权重衰减，对 LayerNorm/bias 等小参数应排除
                          (避免过度正则损伤表达能力)；本教学实现为简洁起见
                          统一作用于所有参数。
        max_grad_norm:    梯度裁剪阈值。Transformer 训练初期容易出现梯度尖峰
                          (尤其是注意力 softmax 饱和时)，裁剪可避免一步崩坏
                          整个模型；常见经验值为 1.0。
        num_steps:        总训练步数 (本教学库为快速演示，仅训练几十步)。
        warmup_steps:     学习率从 0 线性升到 learning_rate 所需的步数。
                          warmup 是 Adam-类优化器训练 Transformer 的关键稳定剂。
        batch_size:       每步样本数。受 GPU 显存制约时，可结合梯度累积模拟大 batch。
        seq_len:          输入序列长度。决定了注意力计算量 (O(seq_len^2))。
        aux_loss_weight:  MoE 负载均衡 (auxiliary) loss 的权重。
                          权重过大会损伤主任务，过小则路由会坍塌到少数专家；
                          DeepSeek/Switch Transformer 经验取值约 0.01。
        audio_loss_weight: Qwen2.5-Omni 中 Talker (音频) 分支 loss 的权重，
                          相对于 Thinker (文本) loss 的相对重要性。
        log_interval:     每多少步打印一次训练指标。
        seed:             随机种子，保证实验可复现。
    """
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    num_steps: int = 50
    warmup_steps: int = 10
    batch_size: int = 2
    seq_len: int = 32
    aux_loss_weight: float = 0.01
    audio_loss_weight: float = 0.5
    log_interval: int = 10
    seed: int = 42

    def get_lr_lambda(self, total_steps: int):
        """
        构造 "线性 warmup + 余弦退火" 的 lr 缩放函数。

        为什么这样设计？
            - Warmup 阶段 (step < warmup_steps)：返回值 step / warmup，
              让学习率从 0 线性升到 1.0。Adam 在前几步动量估计噪声大，
              直接用大学习率容易发散；warmup 可以平稳过渡。
            - 余弦退火阶段：返回 0.5 * (1 + cos(pi * progress))，
              progress 从 0 升到 1，函数值从 1 平滑下降到 0。
              相比线性衰减，cosine 在末期下降更慢，便于精细收敛；
              且无需额外的 step decay 超参，工程简洁。

        Args:
            total_steps: 总训练步数 (含 warmup)。

        Returns:
            lr_lambda(step) -> float：可直接传入 LambdaLR 调度器。
        """
        warmup = self.warmup_steps

        def lr_lambda(step: int) -> float:
            # warmup 阶段：线性从 0 升到 1
            if step < warmup:
                return step / max(1, warmup)
            # 退火阶段：余弦从 1 降到 0
            progress = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return lr_lambda
