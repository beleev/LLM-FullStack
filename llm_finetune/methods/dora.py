"""
DoRA — Weight-Decomposed Low-Rank Adaptation (Liu et al., 2024)

是什么: 把权重拆成 "每行的长度 m" × "方向 V/‖V‖", LoRA 只改方向, 长度单独用一个向量学。
解决什么: 论文观察到全参微调常常 "方向变很多、长度几乎不变" (或反过来), 而 LoRA 的 ΔW = BA 把两者绑在一起按比例变;
          拆开后低秩支路只需表达方向变化, 同样的 r 更接近全参微调。
核心公式:  W' = m ⊙ (W + (α/r)BA) / ‖W + (α/r)BA‖_row        m 初始化为 ‖W‖_row ⇒ 第 0 步 W' = W
           比 LoRA 每层只多 d_out 个参数。
读代码时盯住: `norm.detach()` —— 论文 §4.3: 反传时把分母当常数, 省掉一整份 [d_out, d_in] 的梯度显存, 精度几乎不变。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_finetune.methods.lora import LoRALinear


class DoRALinear(LoRALinear):
    def __init__(self, base: nn.Module, r: int = 8, alpha: float = 16, dropout: float = 0.0) -> None:
        super().__init__(base, r=r, alpha=alpha, dropout=dropout)
        # 名字带 "lora_": 复用 mark_only_lora_as_trainable / get_lora_state_dict
        self.lora_magnitude = nn.Parameter(base.weight.detach().norm(dim=1))       # [d_out]

    def merged_weight(self) -> torch.Tensor:
        direction = self.base.weight + self.delta()                  # V  [d_out, d_in] (未归一化)
        norm = direction.norm(dim=1, keepdim=True).detach()          # [d_out, 1] 当常数: 梯度只经分子和 m
        return self.lora_magnitude.unsqueeze(1) * direction / norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 归一化是对**整行权重**做的, 没法像 LoRA 那样拆成两次小 matmul: 每步都要显式构造 W'
        return F.linear(self.lora_dropout(x), self.merged_weight(), self.base.bias)
