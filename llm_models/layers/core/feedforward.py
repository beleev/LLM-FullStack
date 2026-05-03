"""
前馈网络 (FFN) 模块 — Transformer 中除注意力外的另一半参数

包含:
- FeedForward:        ReLU 激活的两层 FFN  (Vaswani et al., 2017 原始 Transformer)
- GeLUFeedForward:    GELU 激活的两层 FFN  (BERT 2018 / GPT-2/3 风格)
- SwiGLUFeedForward:  SwiGLU 门控 FFN     (Shazeer 2020; PaLM/LLaMA/Qwen/DeepSeek 风格)

演进与设计动机:
    1. ReLU FFN (2017): 简单有效，但负半轴梯度恒 0，存在 "dying neuron"
    2. GELU FFN (2018-): 平滑近似 x·Φ(x)，BERT/GPT 系都用，避免硬截断
    3. SwiGLU FFN (2022-): 引入门控分支，PaLM 论文实证比 ReLU/GELU 高 0.5+ 点 PPL

SwiGLU 关键洞察:
    门控分支 Swish(W_gate · x) 学习 "应该让哪些维度通过"，
    内容分支 W_up · x 学习 "通过的内容是什么"，
    二者逐元素相乘后再降维。代价是参数量增加 50%，故 d_ff 通常按 2/3 缩减
    以保持总参数量与 ReLU-FFN 接近。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """
    前馈网络 (FFN) — 原始 Transformer 版本 (ReLU)

    结构: Linear -> ReLU -> Linear
    公式: FFN(x) = W2 · ReLU(W1·x + b1) + b2

    每层 Transformer 都有这样一个独立的 FFN。它的作用通常被解释为
    "key-value memory"：W1 行向量是 key (匹配输入)，W2 列向量是 value (写出内容)。
    d_ff = 4·d_model 是实证最优带宽，参数主要集中在这里 (>~70% LLM 参数在 FFN)。

    Args:
        d_model: 输入/输出维度
        d_ff: 中间隐藏层维度 (经验值 4·d_model)
    """

    def __init__(self, d_model: int, d_ff: int):
        super(FeedForward, self).__init__()

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量 [batch, seq_len, d_model]

        Returns:
            输出张量 [batch, seq_len, d_model]
        """
        return self.fc2(self.relu(self.fc1(x)))


class GeLUFeedForward(nn.Module):
    """
    GELU 前馈网络 — BERT / GPT-2/3 风格

    结构: Linear -> GELU -> Linear
    公式: FFN(x) = W2 · GELU(W1·x + b1) + b2

    GELU 公式: GELU(x) = x · Φ(x)，Φ 为标准正态 CDF
    直觉解释: 把 ReLU 的 "0/1 硬开关" 换成 "按高斯概率软开关"，
            既保留稀疏性又有平滑梯度，避免 ReLU 的 dying neuron 问题。

    Args:
        d_model: 输入/输出维度
        d_ff: 中间隐藏层维度 (经验值 4·d_model)
    """

    def __init__(self, d_model: int, d_ff: int):
        super(GeLUFeedForward, self).__init__()

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU 前馈网络 — LLaMA / Qwen / DeepSeek 等现代 LLM 标配

    出处: Shazeer 2020 "GLU Variants Improve Transformer"，PaLM 论文广泛验证。

    公式: FFN(x) = W_down · ( Swish(W_gate·x) ⊙ (W_up·x) )
         其中 Swish(x) = x · sigmoid(x)，亦称 SiLU。

    门控 (GLU) 思想:
        把单条 "linear -> activation" 通路拆成两条并行的 linear，
        一条 (gate) 经 Swish 后作为门控信号，一条 (up) 作为内容信号，
        相乘后再降维。让网络自行学习 "哪些通道该开/关、开多大"。

    工程权衡:
        - 三个矩阵 (gate / up / down) 比标准 FFN 的两个多 50% 参数，
          故同模型预算下通常把 d_ff 缩到 ~(2/3) · 4 · d_model 来对齐参数量
        - 现代 LLM 普遍 bias=False (节省参数 + 更稳的归一化数值)

    Args:
        d_model: 输入/输出维度
        d_ff: 中间隐藏层维度 (推荐 int(8/3 · d_model) 来保持总参数量与 ReLU-FFN 一致)
        bias: 是否使用偏置 (现代 LLM 通常 False)
    """

    def __init__(self, d_model: int, d_ff: int, bias: bool = False):
        super(SwiGLUFeedForward, self).__init__()

        # Gate 投影: 生成门控信号
        self.w_gate = nn.Linear(d_model, d_ff, bias=bias)
        # Up 投影: 生成特征值
        self.w_up = nn.Linear(d_model, d_ff, bias=bias)
        # Down 投影: 降回模型维度
        self.w_down = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x):
        """
        前向传播

        数据流:
            x ──┬── w_gate ── Swish ──┐
                │                      ⊙ (逐元素乘) ── w_down ── output
                └── w_up    ──────────┘

        Args:
            x: 输入张量 [batch, seq_len, d_model]

        Returns:
            输出张量 [batch, seq_len, d_model]
        """
        # F.silu 即 Swish: x * sigmoid(x)，PyTorch 内置融合实现
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        # 逐元素门控相乘，再降维回 d_model
        return self.w_down(gate * up)
