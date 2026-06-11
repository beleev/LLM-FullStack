"""
Reward Model — 奖励模型 (RLHF 三阶段的第二步)
=================================================

历史背景:
    InstructGPT / ChatGPT (2022) 的对齐配方是三阶段:
        SFT → **Reward Model (RM)** → PPO
    人类标注的是 "A 回复比 B 回复好" 这种**相对偏好** (绝对打分太难对齐标注员),
    RM 的任务是把相对偏好蒸馏成一个**绝对标量分数** r(x, y), 之后的 RL 阶段
    才有可优化的奖励信号。

结构 (业界标准做法):
    把 LLM 的 lm_head ([D] → [V]) 换成 value head ([D] → [1]):
        r(x, y) = value_head( h_最后一个token )
    读完整条 (prompt, response) 后, 最后位置的 hidden state 聚合了全部信息,
    它的标量投影就是这条回复的分数。backbone 通常从 SFT 模型初始化。

损失 (Bradley-Terry 偏好模型):
    人类只说了 "chosen 比 rejected 好", 于是最大化:
        P(chosen ≻ rejected) = σ(r_chosen - r_rejected)
        L = -log σ(r_chosen - r_rejected)
    只关心分差, 不关心绝对值 —— 所以 RM 的分数没有客观量纲, 只有序关系。

与 DPO 的关系 (对照 methods/dpo.py):
    DPO 把 "训 RM + RL" 两步解析地合并成一步; 但显式 RM 仍是
    GRPO / PPO / 拒绝采样 / Best-of-N 等在线方法的前置依赖。
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_models.models.language_models.llama import LLaMA
from llm_models.utils.masks import combine_causal_and_padding_mask


class RewardModel(nn.Module):
    """
    LLaMA backbone + 标量 value head。

    forward 返回每条序列的标量分数 [B] (取最后一个位置的 hidden 投影)。

    实现说明:
        复用传入的 LLaMA 的 embedding / layers / ln_f, 但**不走 lm_head**。
        这正是业界做法: RM 与 policy 共享同一套骨架, 只换输出头。

    Args:
        backbone: LLaMA 实例 (通常应从 SFT checkpoint 初始化)。
    """

    def __init__(self, backbone: LLaMA) -> None:
        super().__init__()
        self.backbone = backbone
        # 标量打分头: 业界常用无 bias 的线性层, 初始化小一点让训练初期分差温和
        self.value_head = nn.Linear(backbone.d_model, 1, bias=False)
        nn.init.normal_(self.value_head.weight, std=0.01)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [B, T]  prompt + response 拼接后的完整序列
        Returns:
            rewards: [B]  每条序列一个标量分
        """
        B, T = input_ids.shape
        bb = self.backbone

        # 与 LLaMA.forward 相同的主干前向, 唯一区别是最后不过 lm_head
        x = bb.token_embedding(input_ids) * math.sqrt(bb.d_model)
        causal = bb._causal_mask(T)
        mask = combine_causal_and_padding_mask(causal, attention_mask)
        for layer in bb.layers:
            x = layer(x, mask=mask, rope=bb.rope)
        h = bb.ln_f(x)                       # [B, T, D]

        scores = self.value_head(h).squeeze(-1)   # [B, T] 每个前缀都有一个分
        return scores[:, -1]                      # 读完全文后的最终分 [B]


def bradley_terry_loss(
    reward_chosen: torch.Tensor,
    reward_rejected: torch.Tensor,
) -> torch.Tensor:
    """
    L = -log σ(r_chosen - r_rejected)

    用 F.logsigmoid 而非 log(sigmoid(...)): 分差很大时 σ 饱和到 1,
    log(1-ε) 会数值下溢, logsigmoid 内部做了稳定化。
    """
    return -F.logsigmoid(reward_chosen - reward_rejected).mean()
