"""
知识蒸馏 (off-policy, forward KL) — Hinton et al., 2015

是什么: student 除了学硬标签, 还在**同一批数据**的每个位置上模仿 teacher 的整个输出分布。
解决什么: 硬标签每个位置只有 log V 比特; teacher 的分布还带着 "其它答案各有多合理" —— 对有多个合理答案的数据,
          一条软标签 ≈ 许多条采样出来的硬标签, 方差小得多。
核心公式:  L = α·CE(student, y) + (1−α)·T²·KL( p_T^teacher ‖ p_T^student ),   p_T = softmax(z / T)
           ×T²: softmax(z/T) 的梯度自带 1/T², 不补回来的话调 T 会顺带改变两项的相对权重。
读代码时盯住: `mask` —— KD 项和 CE 项必须用**同一个** mask (label = −100 的位置都不算), 否则 prompt / pad 位置也在被蒸馏。
forward KL = mode-covering: teacher 有质量的地方 student 都得有。数据来自 teacher / 数据集而不是 student 自己,
所以 student 从没在**自己会走到的前缀**上被训练过 → 对照 on_policy_distill.py。
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_models.training.loss import LossComputer
from llm_finetune.utils.param_utils import freeze_module


class TeacherStudent(nn.Module):
    """和 dpo.PairwiseForward 同一个扩展点: 把 "student 前向 + 冻结 teacher 前向" 包成一个 Module 交给通用 Trainer。"""

    def __init__(self, student: nn.Module, teacher: nn.Module) -> None:
        super().__init__()
        self.student = student
        freeze_module(teacher)
        teacher.eval()
        self._teacher = (teacher,)                # tuple: 不注册为子模块 → 不进 optimizer, 不被 .train() 影响

    def forward(self, idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        with torch.no_grad():                     # teacher 只提供目标
            teacher_logits = self._teacher[0](idx)
        return {"student": self.student(idx), "teacher": teacher_logits}      # [B, T, V] ×2


class DistillLoss(LossComputer):
    """temperature: 越大 teacher 分布越平, 非最大项越显眼; alpha: 硬标签 CE 的权重。"""

    def __init__(self, temperature: float = 2.0, alpha: float = 0.3) -> None:
        if temperature <= 0 or not 0.0 <= alpha <= 1.0:
            raise ValueError("需要 temperature > 0 且 0 ≤ alpha ≤ 1")
        self.temperature, self.alpha = temperature, alpha

    def compute(self, model_output: Dict[str, torch.Tensor], labels: torch.Tensor,
                **kwargs) -> Dict[str, torch.Tensor]:
        s, t, T = model_output["student"], model_output["teacher"], self.temperature
        ce = F.cross_entropy(s.reshape(-1, s.size(-1)), labels.reshape(-1), ignore_index=-100)

        mask = labels != -100                                                        # [B, T]
        log_p_s = F.log_softmax(s / T, dim=-1)                                       # [B, T, V]
        log_p_t = F.log_softmax(t / T, dim=-1)
        kl_tok = (log_p_t.exp() * (log_p_t - log_p_s)).sum(dim=-1)                   # [B, T] 逐位置 KL(teacher‖student)
        kd = (kl_tok * mask).sum() / mask.sum() * T * T                              # 只在被监督的位置上平均

        return {"total_loss": self.alpha * ce + (1 - self.alpha) * kd,
                "ce_loss": ce.detach(), "kd_loss": kd.detach()}
