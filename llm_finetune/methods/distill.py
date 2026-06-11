"""
Knowledge Distillation — 知识蒸馏 (Hinton et al., 2015)
========================================================

历史背景:
    大模型能力强但部署贵。蒸馏让小模型 (student) 不仅学习硬标签, 还模仿
    大模型 (teacher) 的**完整输出分布**。LLM 时代它无处不在:
        - DistilBERT (2019): 6 层学 12 层, 保留 97% 能力
        - 各家 "-mini / -flash / -turbo" 小模型普遍有蒸馏环节
        - DeepSeek-R1 (2025): 用 R1 的输出把推理能力蒸进 Qwen/Llama 小模型
          (R1 蒸馏版走的是数据蒸馏/序列蒸馏: 老师生成文本给学生做 SFT;
           本文件实现的是 logit 蒸馏, 两者思想一致 —— 学分布而非学标签)

为什么软标签比硬标签信息多 (dark knowledge):
    硬标签:  "正确答案是 token 42"                     → log V 比特
    软标签:  "42: 0.7, 17: 0.2, 99: 0.05, ..."        → 老师对相似 token 的
             相对排序、置信度全在分布里, 学生每个样本拿到 V 维监督

温度 T 的作用:
    softmax(z/T): T 越大分布越平, 非最大项的"暗知识"被放大。
    KL 项要乘 T^2: softmax(z/T) 的梯度自带 1/T^2 缩放, 不补偿的话
    调温度会顺带改变 KD 项与 CE 项的相对权重 (Hinton 论文的细节)。

损失:
    L = α · CE(student, 硬标签) + (1-α) · T² · KL( p_teacher^T ‖ p_student^T )
"""

from typing import Dict

import torch
import torch.nn.functional as F


class DistillLoss:
    """
    logit 蒸馏损失。

    与库内其它 LossComputer 不同, 它需要 teacher / student 两路 logits,
    因此不接入通用 Trainer, 由 run_finetune/distill 的训练循环直接调用
    (与 DPOTrainer 重写 train_step 是同一类问题的两种解法)。

    Args:
        temperature: 软化温度 T (常用 1~4; 越大暗知识越多, 信号也越弱)
        alpha:       硬标签 CE 的权重 (1-alpha 给蒸馏项)
    """

    def __init__(self, temperature: float = 2.0, alpha: float = 0.3) -> None:
        if temperature <= 0:
            raise ValueError("temperature 必须 > 0")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha 必须在 [0, 1]")
        self.temperature = temperature
        self.alpha = alpha

    def compute(
        self,
        student_logits: torch.Tensor,   # [B, T, V] 需要梯度
        teacher_logits: torch.Tensor,   # [B, T, V] 应当 detach / no_grad 得到
        labels: torch.Tensor,           # [B, T]    硬标签 (-100 跳过)
    ) -> Dict[str, torch.Tensor]:
        V = student_logits.size(-1)
        T = self.temperature

        # ---- 硬标签 CE (与普通 LM 训练相同) ----
        ce = F.cross_entropy(
            student_logits.reshape(-1, V),
            labels.reshape(-1),
            ignore_index=-100,
        )

        # ---- 软标签 KL: 两边都用温度 T 软化, 再乘 T^2 补偿梯度 ----
        kd = F.kl_div(
            F.log_softmax(student_logits / T, dim=-1).reshape(-1, V),
            F.softmax(teacher_logits.detach() / T, dim=-1).reshape(-1, V),
            reduction="batchmean",
        ) * (T * T)

        total = self.alpha * ce + (1.0 - self.alpha) * kd
        return {
            "total_loss": total,
            "ce_loss": ce.detach(),
            "kd_loss": kd.detach(),
        }


@torch.no_grad()
def soften_demo(logits: torch.Tensor, temperatures: tuple = (1.0, 2.0, 4.0)) -> None:
    """打印同一行 logits 在不同温度下的 top-3 概率 —— 直观看到"暗知识"被放大。"""
    for T in temperatures:
        probs = F.softmax(logits / T, dim=-1)
        top_p, top_i = probs.topk(3)
        items = ", ".join(f"tok{int(i)}: {float(p):.3f}" for p, i in zip(top_p, top_i))
        print(f"    T={T:<4} top-3: {items}")
