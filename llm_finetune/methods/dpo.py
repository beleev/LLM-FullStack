"""
DPO — Direct Preference Optimization (Rafailov et al., 2023)

是什么: 直接在偏好对 (x, y_w, y_l) 上做二分类, 不训 reward model, 不跑 RL。
解决什么: RLHF = SFT → RM → PPO, 要同时养 4 个模型; DPO 证明 KL 约束下的最优策略满足
          r(x,y) = β·log π(y|x)/π_ref(y|x) + const, 代回 Bradley-Terry 就得到一个纯监督 loss。
核心公式:  L = − log σ( β·[ (log π_θ(y_w|x) − log π_ref(y_w|x)) − (log π_θ(y_l|x) − log π_ref(y_l|x)) ] )
           log π(y|x) = Σ_{t∈回复} log p(y_t | x, y_<t)   —— 包含第一个回复 token y_1 (见 data/tasks.make_labels)
读代码时盯住: `PairwiseForward` —— 它是本章接入通用 Trainer 的扩展点: "一步要跑几次前向" 被包进一个 nn.Module,
              Trainer 的循环 (clip / step / scheduler) 一行不用复制。
代价: 每步多一次 ref 前向 + 常驻一份 ref 权重 (SimPO / ORPO 把它去掉, 见 simpo.py / orpo.py)。
"""

import copy
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_models.training.loss import LossComputer
from llm_finetune.utils.param_utils import freeze_module


def compute_sequence_logprobs(logits: torch.Tensor, labels: torch.Tensor,
                              average: bool = False) -> torch.Tensor:
    """
    logits [B, T, V], labels [B, T] (已左移; prompt / pad = -100) → [B] 回复段 Σ_t log p(y_t)。
    average=True 时除以回复长度 (SimPO / ORPO 用)。
    """
    valid = labels != -100                                            # [B, T]
    safe = labels.masked_fill(~valid, 0)                              # gather 不接受 -100, 先塞个合法下标, 下面再乘 0
    token_logp = F.log_softmax(logits, dim=-1).gather(-1, safe.unsqueeze(-1)).squeeze(-1)   # [B, T]
    total = (token_logp * valid).sum(dim=-1)                          # [B]
    return total / valid.sum(dim=-1) if average else total


class PairwiseForward(nn.Module):
    """
    Trainer 只会 `model(**batch)` 一次。偏好类方法每步要对 chosen / rejected (以及可选的冻结 ref) 各跑一次 ——
    把这些前向包成一个 Module 的 forward, Trainer 就不用改。DPO / SimPO / ORPO / Reward Model 共用。

    forward 的形参名 = PreferenceDataGenerator 的 key; 返回 {"chosen", "rejected"[, "ref_chosen", "ref_rejected"]},
    值是被包模型的原始输出 (LM: logits [B,T,V]; RM: 分数 [B])。
    """

    def __init__(self, model: nn.Module, ref: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.model = model
        if ref is not None:
            freeze_module(ref)
            ref.eval()
        # 放进 tuple 而不是直接赋值: 不注册为子模块 → 不进 optimizer, 也不会被 Trainer 的 model.train() 切回训练态
        self._ref = (ref,)

    @classmethod
    def with_frozen_copy(cls, model: nn.Module) -> "PairwiseForward":
        """DPO 标准做法: ref = 训练开始那一刻的 policy (SFT 终态) 的冻结副本。"""
        return cls(model, ref=copy.deepcopy(model))

    def forward(self, chosen_input_ids, rejected_input_ids,
                chosen_attention_mask=None, rejected_attention_mask=None) -> Dict[str, torch.Tensor]:
        ids = torch.cat([chosen_input_ids, rejected_input_ids])                       # [2B, T] 拼一起只前向一次
        mask = None if chosen_attention_mask is None else torch.cat([chosen_attention_mask, rejected_attention_mask])
        out = dict(zip(("chosen", "rejected"), self.model(ids, mask).chunk(2)))
        ref = self._ref[0]
        if ref is not None:
            with torch.no_grad():                                                     # ref 不更新, 也不用存激活
                out.update(zip(("ref_chosen", "ref_rejected"), ref(ids, mask).chunk(2)))
        return out


def _preference_metrics(chosen_reward: torch.Tensor, rejected_reward: torch.Tensor) -> Dict[str, torch.Tensor]:
    """偏好类 loss 共用的监控量; 输入是任何意义下的 "隐式奖励" [B]。"""
    c, r = chosen_reward.detach(), rejected_reward.detach()
    return {"reward_margin": (c - r).mean(), "accuracy": (c > r).float().mean()}


class DPOLoss(LossComputer):
    """beta: KL 约束强度。越大越贴近 ref (同样的 log-ratio 差更快让 σ 饱和, 梯度更早消失)。"""

    def __init__(self, beta: float = 0.1) -> None:
        self.beta = beta

    def compute(self, model_output: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor],
                **kwargs) -> Dict[str, torch.Tensor]:
        logp = {k: compute_sequence_logprobs(v, labels[k.replace("ref_", "")]) for k, v in model_output.items()}
        # 隐式奖励 r̂ = β·(log π_θ − log π_ref); ref 项在 no_grad 下算出, 本来就没有梯度
        chosen_reward = self.beta * (logp["chosen"] - logp["ref_chosen"])             # [B]
        rejected_reward = self.beta * (logp["rejected"] - logp["ref_rejected"])       # [B]
        loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()                  # logsigmoid: σ 饱和时不会 log(0)
        return {
            "total_loss": loss,
            # chosen 的 log-prob 本身可能**下降** (DPO 只保证差值变大) —— 训练时值得单独盯
            "logp_chosen": logp["chosen"].detach().mean(),
            **_preference_metrics(chosen_reward, rejected_reward),
        }
