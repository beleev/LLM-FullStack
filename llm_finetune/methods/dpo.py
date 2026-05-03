"""
DPO — Direct Preference Optimization (Rafailov et al., 2023)
=============================================================

历史背景:
    经典 RLHF (InstructGPT / ChatGPT) 走三阶段:
        SFT → Reward Model → PPO
    PPO 实现复杂、显存占用高、训练不稳定 (要同时维持 4 个模型)。

    DPO 的洞见: 把 RLHF 的优化目标 (KL-约束的策略提升) 解析地变形,
    可以直接写成一个 **基于偏好对的分类 loss**, 完全跳过 reward model 与
    强化学习, 只需要 (prompt, chosen, rejected) 三元组即可。

数学形式 (Bradley-Terry 偏好模型 + KL 约束最优解):
    L_DPO = -E_{(x, y_w, y_l)} log σ(
                β · [ log π_θ(y_w|x) / π_ref(y_w|x)
                    - log π_θ(y_l|x) / π_ref(y_l|x) ]
            )
    其中:
        π_θ:   被微调的 policy (一般初始化为 SFT 后的模型)
        π_ref: 冻结的 reference (通常就是 SFT 终态的副本)
        y_w:   人类偏好的 chosen 回复
        y_l:   被 reject 的回复
        β:     KL 约束强度, 0.1~0.5 区间, 越大越保守

直觉:
    - 提高 chosen 的 log-prob, 降低 rejected 的 log-prob
    - 但任何变化都要相对 reference 度量, 防止策略漂移过远
    - σ 把无界的 logit 差压缩到 (0, 1), 形成稳定的二分类问题

关键工程细节:
    1. **序列级 log-prob**: log π(y|x) = Σ_t log p(y_t | y_<t, x), 是
       teacher-forcing 一次前向后, 把 logits 与 labels 对齐求 NLL 之和。
       prompt 部分用 -100 mask, 保证只统计 response。
    2. **batch 拼接**: chosen 与 rejected 拼到同一 batch 一次前向, 节省 50% 显存。
    3. **reference 冻结 + eval mode**: 防止 dropout / batchnorm 引入随机扰动,
       保证 log π_ref 是确定函数。
    4. β 的实际作用: 大 β → log_sigmoid 输入更陡, 模型更保守贴近 ref;
       小 β → 允许更大幅度地偏好 chosen, 但也更容易过拟合到偏好数据。
"""

import copy
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_models.training.config import TrainingConfig
from llm_models.training.data import SyntheticDataGenerator
from llm_models.training.loss import LossComputer
from llm_models.training.trainer import Trainer
from llm_finetune.utils.param_utils import freeze_module

Metric = Union[torch.Tensor, float]


def compute_sequence_logprobs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    计算每个序列的 ``Σ_t log p(y_t | y_<t)`` (按 ignore_index 屏蔽 prompt/pad)。

    与 ``F.cross_entropy`` 的 reduction='sum' 不同:
        - cross_entropy 平均/求和后无法区分样本, 适合训练 loss
        - 这里需要 **逐样本** 累计的 log-prob, 才能让 DPO 对 chosen vs rejected
          做 element-wise 差分

    Args:
        logits: [B, T, V]  模型输出
        labels: [B, T]    目标 token id, prompt/pad 位置为 ignore_index
    Returns:
        log_probs: [B]    每个序列的 response 部分对数似然之和
    """
    # 把 logits 转为 log p, 内存上多一份 [B, T, V] 张量, 但比手写 logsumexp + gather
    # 更可读且数值更稳 (PyTorch 的 log_softmax 已做 max-shift)
    log_probs_full = F.log_softmax(logits, dim=-1)

    # mask 标记哪些位置参与求和: True = 有效 response token
    valid = labels != ignore_index  # [B, T]

    # gather 时需要合法索引; 把无效位置 (-100) 暂时替换为 0, 后面 mask 抹掉
    safe_labels = labels.masked_fill(~valid, 0)

    # gather 出每个位置选中 token 的 log p, [B, T]
    token_logp = log_probs_full.gather(
        dim=-1, index=safe_labels.unsqueeze(-1)
    ).squeeze(-1)

    # 屏蔽无效位置后按 batch 求和, 得到 [B]
    return (token_logp * valid.float()).sum(dim=-1)


class DPOLoss(LossComputer):
    """
    Direct Preference Optimization 损失。

    与其它 LossComputer 的差异:
        其它 loss 只需要 model_output + labels;
        DPO 需要 (policy_logits, ref_logits) 两路 + (chosen_labels, rejected_labels) 两批。
        因此 ``compute`` 的入参约定与基类一致, 但内部对 dict 字段做了细化。

    输入约定 (model_output 是一个 dict):
        {
          "policy_chosen_logits":   [B, T, V],
          "policy_rejected_logits": [B, T, V],
          "ref_chosen_logits":      [B, T, V],
          "ref_rejected_logits":    [B, T, V],
        }
    labels 是一个 dict:
        {
          "chosen_labels":   [B, T],  prompt -100 mask
          "rejected_labels": [B, T],
        }

    Args:
        beta: KL 约束强度。常用 0.1; 越大越保守。
    """

    def __init__(self, beta: float = 0.1) -> None:
        self.beta = beta

    def compute(
        self,
        model_output: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # 1) 计算 policy 与 reference 在 chosen / rejected 上的序列 log-prob
        policy_chosen_logp = compute_sequence_logprobs(
            model_output["policy_chosen_logits"], labels["chosen_labels"]
        )
        policy_rejected_logp = compute_sequence_logprobs(
            model_output["policy_rejected_logits"], labels["rejected_labels"]
        )
        # ref 已经在 DPOTrainer 里 no_grad 前向, 这里 detach 再保险一道
        ref_chosen_logp = compute_sequence_logprobs(
            model_output["ref_chosen_logits"], labels["chosen_labels"]
        ).detach()
        ref_rejected_logp = compute_sequence_logprobs(
            model_output["ref_rejected_logits"], labels["rejected_labels"]
        ).detach()

        # 2) DPO 公式核心: 两个 log-ratio 的差
        chosen_logratio = policy_chosen_logp - ref_chosen_logp
        rejected_logratio = policy_rejected_logp - ref_rejected_logp
        logits = self.beta * (chosen_logratio - rejected_logratio)

        # 3) -log σ(z); 直接用 F.logsigmoid 避免 σ 饱和导致 log(0)
        loss = -F.logsigmoid(logits).mean()

        # 4) 监控指标: chosen / rejected 的隐式奖励 (rewards) 与胜率
        # rewards = β · (logπ_θ - logπ_ref), 论文用此衡量"偏好幅度"
        with torch.no_grad():
            chosen_reward = self.beta * chosen_logratio
            rejected_reward = self.beta * rejected_logratio
            margin = (chosen_reward - rejected_reward).mean()
            accuracy = (chosen_reward > rejected_reward).float().mean()

        return {
            "total_loss": loss,
            "dpo_loss": loss.detach(),
            "reward_chosen": chosen_reward.mean(),
            "reward_rejected": rejected_reward.mean(),
            "reward_margin": margin,
            "accuracy": accuracy,
        }


class DPOTrainer(Trainer):
    """
    DPO 专用训练器, 在 ``Trainer`` 基础上扩展 "双前向 (policy + ref)" 流程。

    与基类的差异点:
        - 多持有一个 ``ref_model``, 训练前自动深拷贝 + 冻结 + eval()
        - ``train_step`` 重写: 在 chosen 上 policy/ref 各前向一次, 再在 rejected
          上重复, 拼装成 DPOLoss 期望的 dict
        - 不传 lm labels 给 model, 而是把 labels 转给 loss_computer

    显存优化:
        默认实现把 chosen / rejected **顺序**前向, 实现简单。
        生产中可把 (B,T) chosen 与 (B,T) rejected concat 成 (2B,T) 一次前向,
        显存换吞吐。教学版本保持顺序, 易调试。

    Args:
        model:          policy 模型 (SFT 终态)。会被 DPO 训练。
        config:         TrainingConfig
        data_generator: ``PreferenceDataGenerator`` 类型, 每步产出 chosen/rejected
        loss_computer:  ``DPOLoss`` 实例
        ref_model:      可选; None 时自动 deepcopy(model) 并冻结。
                        若已有独立的 SFT checkpoint 可传入复用。
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        data_generator: SyntheticDataGenerator,
        loss_computer: LossComputer,
        ref_model: Optional[nn.Module] = None,
    ) -> None:
        super().__init__(model, config, data_generator, loss_computer)

        # 没传 ref_model 就深拷贝 policy 作为参考; 这是 DPO 论文的标准做法
        # (假设 policy 起点 = SFT 终态, ref 也用同一个 SFT 终态)
        if ref_model is None:
            ref_model = copy.deepcopy(model)
        freeze_module(ref_model)
        ref_model.eval()
        self.ref_model = ref_model

    def train_step(self) -> Dict[str, Metric]:
        """
        DPO 单步训练: 4 次前向 (policy chosen, policy rejected, ref chosen, ref rejected)。

        流程:
            1. 取 batch (含 chosen_input_ids, rejected_input_ids, 与对应 labels)
            2. policy 在 chosen / rejected 上各前向一次 (要梯度)
            3. ref 在 chosen / rejected 上各前向一次 (no_grad, 节省激活显存)
            4. 把 4 路 logits + 2 组 labels 喂给 DPOLoss
            5. 反向 / 裁剪 / step / scheduler / zero_grad
        """
        self.model.train()
        self.ref_model.eval()

        batch = self.data_generator.generate_batch()

        # 取出 labels (DPO 数据生成器约定: chosen_labels / rejected_labels)
        chosen_labels = batch.pop("chosen_labels")
        rejected_labels = batch.pop("rejected_labels")
        chosen_input_ids = batch.pop("chosen_input_ids")
        rejected_input_ids = batch.pop("rejected_input_ids")

        # ---- 2) policy 双前向 ----
        policy_chosen_logits = self.model(chosen_input_ids)
        policy_rejected_logits = self.model(rejected_input_ids)

        # ---- 3) ref 双前向 (no_grad: ref 不更新, 也不需要保存激活) ----
        with torch.no_grad():
            ref_chosen_logits = self.ref_model(chosen_input_ids)
            ref_rejected_logits = self.ref_model(rejected_input_ids)

        # ---- 4) 计算 DPO loss ----
        loss_dict = self.loss_computer.compute(
            model_output={
                "policy_chosen_logits": policy_chosen_logits,
                "policy_rejected_logits": policy_rejected_logits,
                "ref_chosen_logits": ref_chosen_logits,
                "ref_rejected_logits": ref_rejected_logits,
            },
            labels={
                "chosen_labels": chosen_labels,
                "rejected_labels": rejected_labels,
            },
        )

        # ---- 5) 反向 + 裁剪 + 更新 ----
        loss_dict["total_loss"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm,
        )
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        self._global_step += 1

        metrics: Dict[str, Metric] = dict(loss_dict)
        metrics["grad_norm"] = grad_norm
        metrics["lr"] = self.scheduler.get_last_lr()[0]
        return metrics
