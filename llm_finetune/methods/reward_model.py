"""
Reward Model — RLHF 的第二步 (InstructGPT, 2022)

是什么: LM 主干 + 一个标量头, 读完 (prompt, response) 给出一个分数 r(x, y)。
解决什么: 人只能稳定地标 "A 比 B 好", 给不出绝对分; 在线 RL (PPO / GRPO / best-of-N) 却需要一个可调用的标量奖励。
核心公式:  P(y_w ≻ y_l) = σ(r_w − r_l)        L = − log σ(r_w − r_l)        (Bradley-Terry; 只有分差有意义)
           r(x, y) = value_head( h[最后一个**非 pad** token] )
读代码时盯住: `last = attention_mask.sum(1) − 1` —— 右 pad 时 h[:, −1] 是 pad 位置的隐状态, 读它等于给 pad 打分。
与 DPO 的关系: 同一份偏好数据、同一个 Bradley-Terry; DPO 把 r 写成 β·log π/π_ref 从而跳过这一步。
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_models.training.loss import LossComputer


class RewardModel(nn.Module):
    """
    backbone: 任何 `forward(idx, attention_mask) → [B, T, V]` 且最后一层叫 `lm_head` 的 LM (本库 LLaMA 满足)。

    取隐状态的办法: 把 lm_head 换成 Identity, backbone 的 forward 就直接返回 ln_f 之后的 h [B, T, D] ——
    只依赖公开的 forward 约定, 不用手抄一遍主干 (mask / RoPE / cache 怎么变都不受影响)。
    注意这会**原地改掉**传进来的 backbone: RM 拿走它的所有权, 还要当 policy 用就先 deepcopy。
    """

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        backbone.lm_head = nn.Identity()          # embedding 权重不受影响 (tied 的只是同一个 Parameter 的引用)
        self.backbone = backbone
        self.value_head = nn.Linear(backbone.d_model, 1, bias=False)
        nn.init.normal_(self.value_head.weight, std=0.01)             # 小初始化: 起步时 r_w − r_l ≈ 0, loss ≈ ln 2

    def forward(self, input_ids: torch.Tensor,                        # [B, T] 右 pad
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:   # → [B]
        h = self.backbone(input_ids, attention_mask)                  # [B, T, D]
        scores = self.value_head(h).squeeze(-1)                       # [B, T] 每个前缀一个分
        if attention_mask is None:
            return scores[:, -1]
        last = attention_mask.long().sum(dim=1) - 1                   # [B] 最后一个真 token 的下标 (右 pad 假设)
        return scores.gather(1, last.unsqueeze(1)).squeeze(1)         # [B]


class BradleyTerryLoss(LossComputer):
    """model_output = PairwiseForward(RewardModel) 的输出 {"chosen": [B], "rejected": [B]}; labels 不用。"""

    def compute(self, model_output: Dict[str, torch.Tensor], labels=None, **kwargs) -> Dict[str, torch.Tensor]:
        r_w, r_l = model_output["chosen"], model_output["rejected"]
        loss = -F.logsigmoid(r_w - r_l).mean()
        return {
            "total_loss": loss,
            "reward_margin": (r_w - r_l).detach().mean(),
            "accuracy": (r_w > r_l).float().mean(),
        }
