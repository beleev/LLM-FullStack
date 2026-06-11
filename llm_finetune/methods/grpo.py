"""
GRPO — Group Relative Policy Optimization (DeepSeek, 2024)
===========================================================

历史背景:
    经典 RLHF 用 PPO 做策略优化, 需要同时维护 4 个模型:
        policy + reference + reward model + **value model (critic)**
    其中 critic 与 policy 同尺寸, 训练它既贵又不稳。

    GRPO (DeepSeekMath 2024 提出, DeepSeek-R1 2025 发扬光大) 的关键一步:
    **用"组内相对分"替掉 critic**。对同一个 prompt 采样一组 G 条回复,
    用组内均值当 baseline:

        A_i = (r_i - mean(r_1..G)) / std(r_1..G)

    比组里平均水平好的回复 → 正优势 → 提高它的概率; 反之压低。
    不需要学习任何 value 网络, RL 的内存/工程复杂度直接砍半。

    配合**可验证奖励** (数学答案对不对、代码过不过测试, 即 RLVR),
    连 reward model 都可以省掉 —— 这正是 R1 的训练配方。

目标函数 (教学版, 单步更新, 重要性比率 = 1):
    L = - E_i [ Â_i · (1/|o_i|) Σ_t log π_θ(o_i,t) ]  +  β · KL(π_θ || π_ref)

    KL 用 k3 无偏估计 (Schulman): exp(q-p) - (q-p) - 1, 其中
    q = log π_ref, p = log π_θ。逐 token 估计, 恒非负, 方差小。

    真实 GRPO 对同一批样本做多个 epoch, 需要 PPO 式 clip 的重要性比率;
    教学版每批只更新一次, 此时比率恒为 1, clip 不起作用, 故省略。
"""

import copy
from typing import Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_finetune.utils.param_utils import freeze_module


def completion_logprobs(
    model: nn.Module,
    sequences: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    """
    计算 completion 段逐 token 的 log π(o_t | 前文)。

    Args:
        sequences:  [B, P+C]  prompt + completion 完整序列
        prompt_len: P
    Returns:
        logp: [B, C]  completion 每个 token 的对数概率
    """
    logits = model(sequences[:, :-1])                  # 位置 t 预测 t+1
    logp_full = F.log_softmax(logits, dim=-1)
    targets = sequences[:, 1:]                          # 对齐的 next-token
    token_logp = logp_full.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [B, P+C-1]
    return token_logp[:, prompt_len - 1 :]              # 只留 completion 段 (C 个)


class GRPOTrainer:
    """
    最小 GRPO 训练器 (教学版)。

    每个 step:
        1. 取一批 prompt, 每个 prompt 用当前 policy 采样 G 条 completion
        2. 用 reward_fn (规则可验证 / RM 皆可) 给每条 completion 打分
        3. 组内归一化得到优势 Â_i —— GRPO 的全部精髓就在这一行
        4. loss = -(Â · mean_t logπ) + β·KL(policy ‖ ref), 一次梯度更新

    Args:
        policy:      被优化的 LLaMA (会被更新)
        reward_fn:   (sequences [N, P+C], prompt_len) -> rewards [N], 越大越好
        group_size:  G, 每个 prompt 的采样条数 (R1 用 16~64, 教学用 8)
        beta:        KL 惩罚系数 (防止策略跑飞, 离 ref 太远)
        lr:          学习率
        max_new:     completion 长度 C
        temperature: 采样温度 (>0 才有组内多样性, 优势才有差异)
    """

    def __init__(
        self,
        policy: nn.Module,
        reward_fn: Callable[[torch.Tensor, int], torch.Tensor],
        group_size: int = 8,
        beta: float = 0.04,
        lr: float = 1e-4,
        max_new: int = 8,
        temperature: float = 1.0,
    ) -> None:
        self.policy = policy
        self.ref = copy.deepcopy(policy)     # 冻结的参考策略 (KL 的锚点)
        freeze_module(self.ref)
        self.ref.eval()

        self.reward_fn = reward_fn
        self.group_size = group_size
        self.beta = beta
        self.max_new = max_new
        self.temperature = temperature
        self.optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)

    @torch.no_grad()
    def _sample_group(self, prompts: torch.Tensor) -> torch.Tensor:
        """每个 prompt 复制 G 份再批量采样, 返回 [B*G, P+C]。"""
        B, P = prompts.shape
        expanded = prompts.repeat_interleave(self.group_size, dim=0)   # [B*G, P]
        self.policy.eval()
        seqs = self.policy.generate(
            expanded, max_new_tokens=self.max_new, temperature=self.temperature
        )
        # generate 在 inference_mode 下运行, 返回的 inference tensor 不能参与
        # 后续带梯度的前向 (embedding 会拒绝保存), clone 成普通张量
        return seqs.clone()

    def step(self, prompts: torch.Tensor) -> Dict[str, float]:
        B, P = prompts.shape
        G = self.group_size

        # ---- 1) 采样一组 completion ----
        seqs = self._sample_group(prompts)                  # [B*G, P+C]

        # ---- 2) 打分 ----
        rewards = self.reward_fn(seqs, P).float()           # [B*G]

        # ---- 3) 组内相对优势: GRPO 的核心, 没有 critic ----
        grouped = rewards.view(B, G)
        adv = (grouped - grouped.mean(dim=1, keepdim=True)) / (
            grouped.std(dim=1, keepdim=True) + 1e-4
        )
        adv = adv.view(B * G)                               # [B*G]

        # ---- 4) 策略梯度 + KL 惩罚 ----
        self.policy.train()
        logp = completion_logprobs(self.policy, seqs, P)    # [B*G, C] 带梯度
        with torch.no_grad():
            logp_ref = completion_logprobs(self.ref, seqs, P)

        # k3 估计: exp(q-p) - (q-p) - 1 >= 0, 逐 token
        log_ratio = logp_ref - logp
        kl = (log_ratio.exp() - log_ratio - 1).mean()

        pg_loss = -(adv.detach() * logp.mean(dim=1)).mean()
        loss = pg_loss + self.beta * kl

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        return {
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std()),
            "kl": float(kl),
            "pg_loss": float(pg_loss),
        }


def make_region_reward(vocab_size: int) -> Callable[[torch.Tensor, int], torch.Tensor]:
    """
    教学用的**可验证奖励** (RLVR 思想的最小版):
        reward = completion 中落在"答案区" [vocab/2, vocab) 的 token 比例。

    规则完全确定、可程序化验证 —— 对应真实场景中的"数学答案正确 / 单元测试通过"。
    没有 reward model, 也就没有 reward hacking 模型可钻的空子。
    """

    def reward_fn(sequences: torch.Tensor, prompt_len: int) -> torch.Tensor:
        completion = sequences[:, prompt_len:]
        return (completion >= vocab_size // 2).float().mean(dim=1)

    return reward_fn
