#!/usr/bin/env python
"""
Off-policy 蒸馏 (forward KL): teacher 学的是一个**有两种正确答案**的任务 (70% 排序 / 30% 照抄),
同样步数下比较 "只学采样出来的硬标签" 和 "学 teacher 的整个分布" —— 分别在 128 条固定数据 和 无限新数据 两种条件下。

    python -m llm_finetune.run_finetune.distill.train_distill
"""

from typing import Dict

import torch

from llm_models.models.language_models.llama import LLaMA
from llm_models.training.data import SyntheticDataGenerator

from llm_finetune import (
    DistillLoss, SeqTask, SFTLoss, TeacherStudent, compute_sequence_logprobs, count_parameters, make_labels, token_kl,
)
from llm_finetune.run_finetune.common import fit, make_model

MAJOR, MINOR, P_MAJOR = SeqTask("sort"), SeqTask("copy"), 0.7
TEACHER_STEPS, STUDENT_STEPS, LR = 800, 300, 3e-3


class MixtureData(SyntheticDataGenerator):
    """每条样本以 0.7 / 0.3 的概率取 sort / copy 作为回复 —— 同一个 prompt 有两个都 "对" 的答案。"""

    def __init__(self, batch_size: int = 64, fixed: bool = False) -> None:
        self.batch_size, self.fixed = batch_size, fixed

    def _sample(self) -> Dict[str, torch.Tensor]:
        prompts = MAJOR.sample_prompts(self.batch_size)
        pick_major = torch.rand(self.batch_size, 1) < P_MAJOR
        response = torch.where(pick_major, MAJOR.target(prompts), MINOR.target(prompts))      # [B, R]
        idx, labels = make_labels(torch.cat([prompts, response], dim=1), MAJOR.prompt_len)
        return {"idx": idx, "labels": labels}


def train_teacher() -> LLaMA:
    teacher = make_model(MAJOR)                                # 2 层, d=64
    fit(teacher, MixtureData(), SFTLoss(), TEACHER_STEPS, LR, log_interval=TEACHER_STEPS)
    return teacher.eval()


def make_student() -> LLaMA:
    return LLaMA(vocab_size=MAJOR.vocab_size, d_model=48, n_heads=4, num_kv_heads=2, num_layers=1, max_len=32)


@torch.no_grad()
def evaluate(student: LLaMA, teacher: LLaMA, n: int = 512) -> Dict[str, float]:
    """全部在留出 prompt 上。forward_kl 在 teacher 的样本上量, reverse_kl 在 student 自己的样本上量。"""
    state = torch.get_rng_state()
    torch.manual_seed(99)
    prompts = MAJOR.sample_prompts(n, "test")
    P, R = MAJOR.prompt_len, MAJOR.response_len
    own = student.generate(prompts, R, temperature=1.0)                       # [n, P+R] student 的样本
    ref = teacher.generate(prompts, R, temperature=1.0)                       # [n, P+R] teacher 的样本
    torch.set_rng_state(state)

    completion = own[:, P:]
    valid = (completion == MAJOR.target(prompts)).all(1) | (completion == MINOR.target(prompts)).all(1)
    out = {
        "valid": float(valid.float().mean()),                                 # student 采样出来的回复是两种正确答案之一
        "reverse_kl": float(token_kl(student(own[:, :-1])[:, P - 1:], teacher(own[:, :-1])[:, P - 1:]).mean()),
        "forward_kl": float(token_kl(teacher(ref[:, :-1])[:, P - 1:], student(ref[:, :-1])[:, P - 1:]).mean()),
    }
    for name, task in (("logp_major", MAJOR), ("logp_minor", MINOR)):         # student 给两种答案各分了多少概率
        idx, labels = make_labels(torch.cat([prompts, task.target(prompts)], dim=1), P)
        out[name] = float(compute_sequence_logprobs(student(idx), labels).mean())
    return out


def report(rows: Dict[str, Dict[str, float]]) -> None:
    print(f"\n{'':<22}{'样本合格率':>8}{'forward KL':>12}{'reverse KL':>12}{'logπ(sort 答案)':>16}{'logπ(copy 答案)':>16}")
    for name, m in rows.items():
        print(f"{name:<22}{m['valid']:>12.3f}{m['forward_kl']:>12.3f}{m['reverse_kl']:>12.3f}"
              f"{m['logp_major']:>16.2f}{m['logp_minor']:>16.2f}")


def main() -> None:
    torch.manual_seed(0)
    teacher = train_teacher()

    # ---- KD 项必须和 CE 项用同一个 mask: 把 teacher 在 prompt 位置的 logits 全打乱, loss 不应变化 ----
    batch = MixtureData(8).generate_batch()
    loss_fn = DistillLoss(temperature=2.0, alpha=0.3)
    s_logits, t_logits = torch.randn(8, batch["idx"].size(1), 16), teacher(batch["idx"]).detach()
    scrambled = torch.where((batch["labels"] == -100).unsqueeze(-1), torch.randn_like(t_logits), t_logits)
    kd = [float(loss_fn.compute({"student": s_logits, "teacher": t}, batch["labels"])["kd_loss"]) for t in (t_logits, scrambled)]
    assert abs(kd[0] - kd[1]) < 1e-6, "KD 项泄漏到了 label = -100 的位置"

    rows = {"teacher": evaluate(teacher, teacher)}
    for data_name, make_data in [("128 条固定", lambda: MixtureData(128, fixed=True)), ("无限新数据", MixtureData)]:
        for name, loss in [("硬标签", DistillLoss(1.0, alpha=1.0)), ("软标签 T=2", loss_fn)]:
            torch.manual_seed(1)
            student = make_student()
            fit(TeacherStudent(student, teacher), make_data(), loss, STUDENT_STEPS, LR, log_interval=STUDENT_STEPS)
            rows[f"{data_name} / {name}"] = evaluate(student, teacher)
    report(rows)
    print(f"参数量: teacher {count_parameters(teacher)['total']:,} → student {count_parameters(student)['total']:,}")

    # 数据有限时, 一条软标签顶得上同一前缀的许多条采样硬标签 → 留出集上更接近 teacher。
    # 数据无限时硬标签自己就能把分布采出来, 两者都卡在 student 的容量上 —— 差距消失 (表里如实列出, 不设断言)。
    assert rows["128 条固定 / 软标签 T=2"]["forward_kl"] < rows["128 条固定 / 硬标签"]["forward_kl"] - 0.05


if __name__ == "__main__":
    main()
