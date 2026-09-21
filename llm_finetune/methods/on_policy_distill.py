"""
On-policy 蒸馏 (reverse KL) — GKD (Agarwal et al., 2024) / MiniLLM (Gu et al., 2024); Qwen3、Thinking Machines 2025 的配方

是什么: student **自己采样**回复, teacher 在这些回复的每个 token 位置上给出分布, student 逐 token 最小化 KL(student ‖ teacher)。
解决什么: off-policy 蒸馏 (distill.py) 只在 teacher / 数据集的前缀上训练; student 推理时一旦走偏, 就进入从没被训练过的前缀
          (exposure bias)。on-policy 直接在 student 会走到的状态上纠正它 —— 像 RL 一样 on-policy, 又像蒸馏一样每个 token 都有稠密信号。
核心公式:  y ~ π_s(·|x) (no_grad 采样),   L = (1/|y|) Σ_t KL( π_s(·|x, y_<t) ‖ π_teacher(·|x, y_<t) )
           reverse KL = mode-seeking: student 在 teacher 认为不可能的地方放质量会被重罚, 但**漏掉** teacher 的某个模式不受罚。
           forward KL (distill.py) 正相反 = mode-covering。
读代码时盯住: 采样不回传梯度 (不是 REINFORCE), 梯度只来自对**整个词表**解析求和的逐位置 KL —— 所以方差远小于 RL。
为什么不接通用 Trainer: 与 GRPO 同理, 训练数据由当前 student 现场生成。
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_finetune.data.tasks import completion_mask


def token_kl(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    """逐位置 KL(p‖q): [..., V] ×2 → [...], 对词表解析求和。"""
    log_p, log_q = F.log_softmax(logits_p, dim=-1), F.log_softmax(logits_q, dim=-1)
    return (log_p.exp() * (log_p - log_q)).sum(dim=-1)


def on_policy_distill_step(student: nn.Module, teacher: nn.Module, prompts: torch.Tensor,
                           max_new: int, optimizer: torch.optim.Optimizer) -> Dict[str, float]:
    P = prompts.size(1)
    # 1) student 自己采样; generate 在 inference_mode 下运行, clone 成普通张量才能再做带梯度的前向
    seqs = student.generate(prompts, max_new, temperature=1.0).clone()            # [B, P+C]
    mask = completion_mask(seqs[:, P:]).float()                                   # [B, C] EOS 之后不算

    # 2) 两个模型在**同一条 student 轨迹**上前向; 位置 P−1 … P+C−2 的输出预测回复的 C 个 token
    student.train()
    s_logits = student(seqs[:, :-1])[:, P - 1:]                                   # [B, C, V] 带梯度
    with torch.no_grad():                                                         # teacher 只当评分者
        t_logits = teacher(seqs[:, :-1])[:, P - 1:]                               # [B, C, V]

    # 3) reverse KL: student 在前 (期望在 student 自己的分布下取)
    loss = (token_kl(s_logits, t_logits) * mask).sum() / mask.sum()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
    optimizer.step()
    return {"reverse_kl": float(loss.detach())}
