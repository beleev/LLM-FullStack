"""
权重初始化 — 让 "初始 loss ≈ ln V" 成立的那一行

问题: nn.Embedding 默认 N(0,1)。一旦 lm_head 与 embedding 共享权重 (weight tying),
      初始 logits = h · Eᵀ 的标准差 ≈ ‖h‖ ≈ sqrt(D), softmax 极度尖锐,
      初始 CE 高达几十上百, 而不是均匀猜测的 ln V。
解法: GPT-2 起的惯例 —— 所有 Linear / Embedding 用 N(0, 0.02²), bias 置 0。
      此时初始 logits std ≈ 0.02·sqrt(D) ≪ 1, softmax 近似均匀, CE ≈ ln V。

读代码时盯住: std。它是全库唯一的初始化超参。
"""

import torch.nn as nn


def init_weights(model: nn.Module, std: float = 0.02) -> nn.Module:
    """对 model 内所有 Linear / Embedding 做 N(0, std²) 初始化, 返回 model 便于链式调用。

    只碰 Linear / Embedding: RMSNorm 的 γ、Mamba 的 A_log、MoE 的 routing_bias 等
    有各自语义的参数保持原样。tied 权重是同一个 Parameter, 重复初始化无害。
    """
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=std)
    return model
