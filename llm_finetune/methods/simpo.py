"""
SimPO — Simple Preference Optimization (Meng et al., 2024)

是什么: 去掉 reference model 的 DPO。隐式奖励直接取 "平均每 token 的 log-prob"。
解决什么: (1) DPO 要常驻一份 ref 并每步多一次前向; (2) DPO 的奖励 log π/π_ref 与**生成时**用的指标 (平均 log-prob) 不一致,
          且 Σ_t 形式天然偏爱短回复 / 惩罚长回复 —— 长度归一化后两者统一。
核心公式:  r(x,y) = (β/|y|)·log π_θ(y|x)        L = − log σ( r(x,y_w) − r(x,y_l) − γ )
           γ > 0 是目标 margin: 不仅要 chosen 赢, 还要赢出 γ 这么多。
读代码时盯住: `average=True` 和 γ —— 去掉它们就退化成 "没有 ref 的 DPO", 会无约束地压低 rejected。
代价: 没有 ref 当锚, 只靠 lr / 步数控制漂移; β 比 DPO 大一个量级 (2~10), 因为平均 log-prob 的数值小得多。
"""

from typing import Dict

import torch
import torch.nn.functional as F

from llm_models.training.loss import LossComputer
from llm_finetune.methods.dpo import _preference_metrics, compute_sequence_logprobs


class SimPOLoss(LossComputer):
    def __init__(self, beta: float = 2.0, gamma: float = 1.0) -> None:
        self.beta, self.gamma = beta, gamma

    def compute(self, model_output: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor],
                **kwargs) -> Dict[str, torch.Tensor]:
        # [B] 长度归一化的奖励; 与 DPO 唯一的输入差别: model_output 里没有 ref_*
        chosen_reward = self.beta * compute_sequence_logprobs(model_output["chosen"], labels["chosen"], average=True)
        rejected_reward = self.beta * compute_sequence_logprobs(model_output["rejected"], labels["rejected"], average=True)
        loss = -F.logsigmoid(chosen_reward - rejected_reward - self.gamma).mean()
        return {
            "total_loss": loss,
            "logp_chosen": (chosen_reward / self.beta).detach().mean(),
            **_preference_metrics(chosen_reward, rejected_reward),
        }
