"""
前馈网络 (FFN) — Transformer 里 attention 之外的另一半 (LLM 约 2/3 的参数在这里)

是什么: 逐位置的两层 MLP, 常被解释为 key-value memory (W1 的行匹配输入, W2 的列写出内容)。
演进:  ReLU (2017, 负半轴梯度恒 0 → dying neuron)
      → GELU (BERT/GPT, x·Φ(x), 把硬开关换成按高斯概率的软开关)
      → SwiGLU (PaLM/LLaMA, 多一条门控分支: "让哪些通道过" 与 "过什么内容" 分开学)
关键数字: 两矩阵 FFN 取 d_ff = 4·d; SwiGLU 有三个矩阵, 取 d_ff ≈ 8/3·d 让参数量持平 (3·8/3 = 2·4)。
读代码时盯住: SwiGLU 的 `gate * up` —— 逐元素相乘就是 "门控"。
"""

import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """FFN(x) = W2 · ReLU(W1·x + b1) + b2  (Vaswani et al., 2017)"""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):                                   # [B, T, D]
        return self.fc2(self.relu(self.fc1(x)))             # [B, T, D] -> [B, T, d_ff] -> [B, T, D]


class GeLUFeedForward(nn.Module):
    """FFN(x) = W2 · GELU(W1·x + b1) + b2,  GELU(x) = x·Φ(x)  (BERT / GPT-2/3)"""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):                                   # [B, T, D]
        return self.fc2(F.gelu(self.fc1(x)))                # [B, T, D] -> [B, T, d_ff] -> [B, T, D]


class SwiGLUFeedForward(nn.Module):
    """
    FFN(x) = W_down · ( SiLU(W_gate·x) ⊙ (W_up·x) )      (Shazeer 2020; LLaMA / Qwen / DeepSeek)

        x ──┬── w_gate ── SiLU ──┐
            │                     ⊙ ── w_down ── out
            └── w_up ────────────┘

    Args:
        d_ff: 推荐 ≈ 8/3·d_model (见文件头)
        bias: 现代 LLM 普遍 False
    """

    def __init__(self, d_model: int, d_ff: int, bias: bool = False):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=bias)   # 门: 每个通道开多大
        self.w_up = nn.Linear(d_model, d_ff, bias=bias)     # 内容
        self.w_down = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x):                                   # [B, T, D]
        gate = F.silu(self.w_gate(x))                       # [B, T, d_ff]  SiLU(x) = x·sigmoid(x)
        return self.w_down(gate * self.w_up(x))             # [B, T, d_ff] -> [B, T, D]
