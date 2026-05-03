"""
SFT — Supervised Fine-Tuning (全参监督微调)
=================================================

历史背景:
    SFT 是 GPT-3.5 → InstructTune → ChatGPT (2022) 公开的 alignment 三阶段中的
    第一阶段。把"语言模型"变成"对话模型"的最朴素办法: 收集
    (instruction, response) 配对数据, 用标准下一 token 预测 loss 训练。

与 pretraining 的关键差异: **loss masking on prompt**
    预训练阶段, 模型对每个位置都计算交叉熵 (除了 pad)。
    SFT 阶段, 我们希望模型 **只学习 "回复" 部分如何生成**, 而不要把
    prompt (用户问题) 也当成生成目标 —— 因为问题来自人, 不是模型该自己生成的。

    实现技巧: 在构造 labels 时, 将 prompt 区域设为 -100;
    PyTorch 的 ``cross_entropy(ignore_index=-100)`` 会跳过这些位置, 等价于
    "只在 response 上算 loss"。

    例:  input  = [<bos>] [Q1] [Q2] [Q3] [A1] [A2] [<eos>]
         labels = [-100 ] [-100][-100][-100][A1] [A2] [<eos>]
                  ↑ prompt 全部忽略           ↑ 只对 response 算 loss

为什么仍需要包装一个 SFTLoss 类?
    技术上 SFTLoss 与 ``llm_models.training.StandardLMLoss`` 完全等价 (都是
    带 -100 ignore 的交叉熵)。把它单独命名:
      1. 教学叙事更清晰 — finetune 章节使用 finetune 命名
      2. 留出未来扩展空间 — 比如未来可加入 NEFTune 噪声 / focal-style 加权
      3. 与 ``DPOLoss`` 形成命名对仗
"""

from typing import Any, Dict

import torch
import torch.nn.functional as F

from llm_models.training.loss import LossComputer


class SFTLoss(LossComputer):
    """
    监督微调损失 — 标准 next-token 交叉熵, 但语义上明确依赖 prompt-masking。

    使用约定:
        - 输入 ``model_output`` 为 logits, 形状 [B, T, V]
        - ``labels`` 形状 [B, T], prompt 部分必须置为 -100, response 部分填真实 token id
        - pad 也置为 -100

    Args:
        ignore_index: 默认 -100 (PyTorch cross_entropy 的默认 ignore 值)。
                      留作可调以兼容某些自定义数据 pipeline。

    Example:
        >>> loss_fn = SFTLoss()
        >>> logits = model(input_ids)              # [B, T, V]
        >>> out = loss_fn.compute(logits, labels)  # labels prompt 区为 -100
        >>> out["total_loss"].backward()
    """

    def __init__(self, ignore_index: int = -100) -> None:
        self.ignore_index = ignore_index

    def compute(
        self,
        model_output: Any,
        labels: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # 与 StandardLMLoss 一致: 把 (B, T, V) 展平为 (B*T, V) 以走融合 kernel,
        # 比 Python 循环快得多。
        logits = model_output
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=self.ignore_index,
        )
        # sft_loss 与 total_loss 此处相同, 双键便于日志接口与其它 LossComputer 对齐。
        return {"total_loss": loss, "sft_loss": loss}
