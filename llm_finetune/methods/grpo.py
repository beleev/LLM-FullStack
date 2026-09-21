"""
GRPO 及其变体 (DAPO / Dr.GRPO / GSPO) — 在线 RL, 组内相对优势替代 critic (DeepSeekMath 2024 → R1 2025)

是什么: 同一个 prompt 采 G 条回复, 用组内均值当 baseline: A_i = (r_i − mean_G r) / std_G r。
解决什么: PPO 要再养一个和 policy 一样大的 value 网络来估 baseline; 组内均值是免费的 Monte-Carlo baseline。
核心公式 (一批 rollout 上做 μ 个 epoch 的更新, 所以需要重要性比率 + 裁剪):
    ρ_t = π_θ(o_t) / π_old(o_t)        J = E[ min( ρ_t·A, clip(ρ_t, 1−ε_low, 1+ε_high)·A ) ] − β·KL(π_θ‖π_ref)
    第 1 个 epoch θ = θ_old ⇒ ρ ≡ 1, 裁剪不起作用 (但梯度不为 0: ∇ρ = ∇log π); 从第 2 个 epoch 起 ρ ≠ 1。
变体只是 GRPOConfig 里的开关 (见 VARIANTS):
    DAPO    ε_high > ε_low (给低概率 token 更大上涨空间, 防熵塌缩) + 丢掉组内奖励全相同的 prompt (A≡0, 没有梯度)
            + 按 token 而不是按序列平均 (长回复里的 token 不再被稀释)
    Dr.GRPO 去掉 ÷std (它给太难 / 太易的题加权) 与 ÷|o_i| (它让答错的长回复每 token 受罚更轻 → 越错越长)
    GSPO    ρ 取序列级几何平均 exp(mean_t log ρ_t): 一条回复整体裁剪, 噪声远小于逐 token 比率
读代码时盯住: `old_logp` (采样那一刻的策略, no_grad) 与 `temperature` —— 采样用 π^{1/T}, 算 log-prob 也必须用 π^{1/T},
              否则 ρ 的分母不是真正的行为策略, 第 1 个 epoch 就已经 off-policy。
为什么不接通用 Trainer: 数据由**当前**策略采样 (生成器得拿到模型), 且一批数据要更新 μ 次 —— "取 batch → 算 loss → 更新一次" 的约定不成立。
"""

import copy
from dataclasses import dataclass, replace
from typing import Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_finetune.data.tasks import completion_mask
from llm_finetune.utils.param_utils import freeze_module


@dataclass(frozen=True)
class GRPOConfig:
    group_size: int = 8              # G
    max_new: int = 9                 # C, 每条回复最多采多少 token
    temperature: float = 1.0
    lr: float = 3e-4
    inner_epochs: int = 2            # μ: 每批 rollout 更新几次
    clip_low: float = 0.2            # ε_low
    clip_high: float = 0.2           # ε_high
    beta: float = 0.0                # KL 惩罚; RLVR 场景 (DAPO / Dr.GRPO) 普遍设 0, 连 ref 都不用养
    std_norm: bool = True            # A 是否 ÷ std
    loss_agg: str = "seq_mean"       # "seq_mean": 先序列内平均再跨序列 | "token_mean": 全体 token 一起平均 | "fixed_len": Σ_t ÷ C
    seq_ratio: bool = False          # True = GSPO 的序列级 ρ
    dynamic_sampling: bool = False   # True = 丢掉组内奖励全相同的 prompt


VARIANTS: Dict[str, dict] = {
    "grpo": {},
    "dapo": dict(clip_high=0.28, dynamic_sampling=True, loss_agg="token_mean"),
    "dr_grpo": dict(std_norm=False, loss_agg="fixed_len"),
    # 序列级 ρ 是 C 个 token 的几何平均, 波动小得多 → 裁剪区间要相应收窄 (论文用 3e-4 量级)
    "gspo": dict(seq_ratio=True, clip_low=0.05, clip_high=0.05),
}


def make_config(variant: str = "grpo", **overrides) -> GRPOConfig:
    return replace(GRPOConfig(), **{**VARIANTS[variant], **overrides})


def group_advantages(rewards: torch.Tensor, group_size: int, std_norm: bool) -> torch.Tensor:
    """rewards [B·G] → A [B·G]。同组 (同一个 prompt) 的 G 条互为 baseline。"""
    g = rewards.view(-1, group_size)                                  # [B, G]
    adv = g - g.mean(dim=1, keepdim=True)
    if std_norm:
        adv = adv / (g.std(dim=1, keepdim=True) + 1e-4)
    return adv.view(-1)


def aggregate(per_token: torch.Tensor, mask: torch.Tensor, how: str) -> torch.Tensor:
    """per_token, mask [N, C] → 标量。三种写法只差 "每个 token 的权重"。"""
    if how == "seq_mean":                                             # token 权重 = 1/(N·|o_i|): 长回复里的 token 被稀释
        return ((per_token * mask).sum(1) / mask.sum(1)).mean()
    if how == "token_mean":                                           # token 权重 = 1/Σ|o|: 人人相同
        return (per_token * mask).sum() / mask.sum()
    if how == "fixed_len":                                            # token 权重 = 1/(N·C): 人人相同, 且与本批长度无关
        return (per_token * mask).sum(1).mean() / mask.size(1)
    raise ValueError(how)


def completion_logprobs(model: nn.Module, seqs: torch.Tensor, prompt_len: int,
                        temperature: float = 1.0) -> torch.Tensor:
    """seqs [N, P+C] → [N, C]: 回复段每个 token 在 π^{1/T} 下的 log-prob。"""
    logits = model(seqs[:, :-1]) / temperature                        # [N, P+C-1, V]  位置 t 的输出预测 token t+1
    logp = F.log_softmax(logits, dim=-1).gather(-1, seqs[:, 1:].unsqueeze(-1)).squeeze(-1)   # [N, P+C-1]
    return logp[:, prompt_len - 1:]                                   # 第一个回复 token 由位置 P−1 (prompt 末尾) 预测


class GRPOTrainer:
    """reward_fn(prompts [N, P], completions [N, C]) → [N]; 规则 verifier 或 reward model 均可。"""

    def __init__(self, policy: nn.Module, reward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 config: GRPOConfig = GRPOConfig()) -> None:
        self.policy, self.reward_fn, self.cfg = policy, reward_fn, config
        self.ref = None
        if config.beta > 0:                                           # 只有用 KL 惩罚时才需要常驻一份 ref
            self.ref = copy.deepcopy(policy)
            freeze_module(self.ref)
            self.ref.eval()
        self.optimizer = torch.optim.AdamW(policy.parameters(), lr=config.lr)

    def step(self, prompts: torch.Tensor) -> Dict[str, float]:
        cfg, P = self.cfg, prompts.size(1)

        # ---- 1) rollout: 每个 prompt 采 G 条 ----
        expanded = prompts.repeat_interleave(cfg.group_size, dim=0)               # [N=B·G, P]
        # generate 内部是 inference_mode (并在结束时恢复 train/eval 状态); clone 成普通张量才能参与带梯度的前向
        seqs = self.policy.generate(expanded, cfg.max_new, temperature=cfg.temperature).clone()
        completions = seqs[:, P:]                                                 # [N, C]
        mask = completion_mask(completions).float()                               # [N, C] EOS 之后不算回复

        # ---- 2) 判分 + 组内优势 ----
        rewards = self.reward_fn(expanded, completions).float()                   # [N]
        adv = group_advantages(rewards, cfg.group_size, cfg.std_norm)             # [N]
        informative = rewards.view(-1, cfg.group_size).std(dim=1) > 0             # [B] 组内奖励有差异
        metrics = {
            "reward": float(rewards.mean()),
            "zero_adv_frac": 1 - float(informative.float().mean()),               # 这些 prompt 的 A ≡ 0, 贡献不了梯度
            "length": float(mask.sum(1).mean()),
        }
        if cfg.dynamic_sampling:
            # ponytail: 只过滤不补采 (DAPO 原版会继续采样直到凑满 batch); 要保持 batch 恒定时在这里加循环
            keep = informative.repeat_interleave(cfg.group_size)
            seqs, mask, adv = seqs[keep], mask[keep], adv[keep]
        metrics["used_zero_adv_frac"] = float((adv == 0).float().mean()) if len(adv) else 0.0
        if len(adv) == 0 or not informative.any():
            return {**metrics, "clip_frac": 0.0, "ratio_dev_epoch1": 0.0, "ratio_dev_last": 0.0, "skipped": 1.0}

        # ---- 3) 行为策略的 log-prob: 参数还没动, 用与采样相同的温度算一遍并冻结 ----
        with torch.no_grad():
            old_logp = completion_logprobs(self.policy, seqs, P, cfg.temperature)             # [N, C]
            ref_logp = completion_logprobs(self.ref, seqs, P, cfg.temperature) if self.ref else None

        # ---- 4) 同一批 rollout 上更新 μ 次 ----
        self.policy.train()
        A = adv.unsqueeze(1)                                                      # [N, 1] 一条回复的所有 token 共用一个 A
        for epoch in range(cfg.inner_epochs):
            logp = completion_logprobs(self.policy, seqs, P, cfg.temperature)     # [N, C] 带梯度
            log_ratio = logp - old_logp                                           # [N, C]
            if cfg.seq_ratio:                                                     # GSPO: 序列内取平均 → 几何平均比率
                log_ratio = ((log_ratio * mask).sum(1) / mask.sum(1)).unsqueeze(1).expand_as(logp)
            ratio = log_ratio.exp()
            clipped = ratio.clamp(1 - cfg.clip_low, 1 + cfg.clip_high)
            surrogate = torch.min(ratio * A, clipped * A)                         # 悲观下界: 只在 "对自己有利的方向" 截断
            per_token = -surrogate
            if ref_logp is not None:
                d = ref_logp - logp                                               # k3 估计: e^d − d − 1 ≥ 0, 无偏且方差小
                per_token = per_token + cfg.beta * (d.exp() - d - 1)
            loss = aggregate(per_token, mask, cfg.loss_agg)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()

            with torch.no_grad():
                dev = float(((ratio - 1).abs() * mask).max())
                if epoch == 0:
                    metrics["ratio_dev_epoch1"] = dev                             # 恒为 0: 还没更新过
                is_clipped = ((ratio * A) != surrogate) & (mask > 0)              # min 选中了被截断的那一支
                metrics["clip_frac"] = float(is_clipped.float().sum() / mask.sum())
                metrics["ratio_dev_last"] = dev
                metrics["log_ratio_std"] = float(log_ratio[mask > 0].std())
        return metrics
